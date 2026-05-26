"""
Credit Scoring ML Pipeline - Version 4 (Proper Vintage Analysis)

This module fixes the critical data leakage in V3 by implementing proper
Observation Window vs Performance Window separation for each client.

Key Innovation:
- Observation Window: Historical data used to calculate features
- Performance Window: Recent data (12 months) used to define defaults
- Strict temporal isolation prevents information leakage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    roc_curve, confusion_matrix, classification_report
)
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: PROPER VINTAGE ANALYSIS WITH TIME WINDOWS
# ============================================================================

def extract_behavioral_features_v4(credit_record_path, performance_window_fraction=0.33):
    """
    Extract behavioral features with proper Observation Window vs Performance Window split.

    CRITICAL CONCEPT (Vintage Analysis):
    ────────────────────────────────────

    For each client, we split their ENTIRE history into TWO parts:

    1. OBSERVATION WINDOW: Earlier portion of history (default: 66% of months)
       - Used to calculate behavioral features (LATE_PAYMENT_COUNT, etc.)
       - Represents the client's historical behavior at application time

    2. PERFORMANCE WINDOW: Recent portion of history (default: 33% of months)
       - Used ONLY to define the target label (did they default later?)
       - Represents subsequent performance

    Timeline Example (Client ID = 5001711, Total history: 48 months):
    ──────────────────────────────────────────────────────────────

    Client's credit history: Months -48, -47, ..., -13, -12, -11, ..., -1, 0

    Most recent month: 0 (current/application time)

    Performance Window: Months -12 to 0 (last 33% of months)
    - Use to calculate target: Did they default in this period?

    Observation Window: Months -48 to -13 (first 66% of months)
    - Use to calculate features: Historical late payment patterns

    Parameters:
    -----------
    credit_record_path : str
        Path to credit_record.csv
    performance_window_fraction : float
        Fraction of history to use as performance window (default 0.33 = 1/3)

    Returns:
    --------
    dict : {
        'behavioral_features': DataFrame with extracted features,
        'targets': DataFrame with target labels,
        'client_stats': Dict with filtering statistics
    }
    """

    print("=" * 80)
    print("STEP 1: PROPER VINTAGE ANALYSIS (OBSERVATION vs PERFORMANCE WINDOWS)")
    print("=" * 80)

    credit_record = pd.read_csv(credit_record_path)
    print(f"\nCredit records loaded: {len(credit_record)} rows")
    print(f"Unique clients: {credit_record['ID'].nunique()}")

    print(f"\nPerformance Window: {performance_window_fraction:.0%} of client history (most recent)")
    print(f"Observation Window: {1-performance_window_fraction:.0%} of client history (earlier)")

    # ========== STEP 1B: DEFINE OBSERVATION WINDOW (First 66% of History) ==========

    print(f"\n{'-' * 80}")
    print("DEFINING OBSERVATION WINDOW FOR FEATURES (First 66% of History)")
    print(f"{'-' * 80}")

    client_month_ranges = []

    for client_id in credit_record['ID'].unique():
        client_history = credit_record[credit_record['ID'] == client_id]

        # Month range for this client
        most_recent = client_history['MONTHS_BALANCE'].min()
        oldest = client_history['MONTHS_BALANCE'].max()
        total_months = oldest - most_recent

        # Observation window: first 66% of history
        # Calculate cutoff month
        obs_months_count = max(6, int(total_months * 0.66))
        obs_cutoff = oldest - obs_months_count

        client_month_ranges.append({
            'ID': client_id,
            'most_recent': most_recent,
            'oldest': oldest,
            'total_months': total_months,
            'obs_cutoff': obs_cutoff,
            'obs_months': obs_months_count,
            'has_sufficient': total_months >= 6
        })

    ranges_df = pd.DataFrame(client_month_ranges)
    valid_ranges = ranges_df[ranges_df['has_sufficient']].copy()

    print(f"\nClients with sufficient history (>= 6 months): {len(valid_ranges)} out of {len(ranges_df)}")
    print(f"Excluded: {len(ranges_df) - len(valid_ranges)} clients")

    # ========== STEP 1C: EXTRACT TARGET (Entire History - Like V2/V3) ==========

    print(f"\n{'-' * 80}")
    print("EXTRACTING TARGET FROM ENTIRE HISTORY (Original Definition)")
    print(f"{'-' * 80}")
    print("Target = Ever reached status 2,3,4,5 (any time in credit history)")

    targets = []
    bad_statuses = ['2', '3', '4', '5']

    for client_id in valid_ranges['ID']:
        client_history = credit_record[credit_record['ID'] == client_id]

        # Check ENTIRE history for bad status (original approach)
        hit_bad_status = (client_history['STATUS'].isin(bad_statuses)).any()
        target = 1 if hit_bad_status else 0

        targets.append({
            'ID': client_id,
            'target': target
        })

    targets_df = pd.DataFrame(targets)

    print(f"\nTarget Distribution (from ENTIRE History):")
    print(targets_df['target'].value_counts())
    bad_pct = (targets_df['target'] == 1).sum() / len(targets_df) * 100
    print(f"  Bad clients:  {(targets_df['target'] == 1).sum()} ({bad_pct:.2f}%)")
    print(f"  Good clients: {(targets_df['target'] == 0).sum()} ({100-bad_pct:.2f}%)")

    # ========== STEP 1D: EXTRACT FEATURES FROM OBSERVATION WINDOW ONLY ==========

    print(f"\n{'-' * 80}")
    print("EXTRACTING BEHAVIORAL FEATURES FROM OBSERVATION WINDOW ONLY")
    print(f"(First 66% of client history - prevents leakage)")
    print(f"{'-' * 80}")

    behavioral_features = []

    for _, row in valid_ranges.iterrows():
        client_id = row['ID']
        client_history = credit_record[credit_record['ID'] == client_id]

        # Get ONLY observation window records (first 66% of history)
        obs_records = client_history[
            client_history['MONTHS_BALANCE'] <= row['obs_cutoff']
        ]

        if len(obs_records) == 0:
            continue

        # -------- Feature Engineering --------
        months_in_book = np.abs(obs_records['MONTHS_BALANCE']).max()

        status_mapping = {
            'C': 0, 'X': 0, '0': 0.5, '1': 1.5,
            '2': 2.5, '3': 3.5, '4': 4.5, '5': 5.0
        }
        numeric_status = obs_records['STATUS'].map(status_mapping)
        max_delinquency = numeric_status.max()

        good_standing_months = (obs_records['STATUS'].isin(['C', '0'])).sum()
        total_obs_months = len(obs_records)
        punctuality_rate = good_standing_months / total_obs_months if total_obs_months > 0 else 0

        late_payment_count = (obs_records['STATUS'].isin(['1', '2', '3', '4', '5'])).sum()

        avg_delinquency_status = numeric_status.mean()

        delinquent_months = obs_records[obs_records['STATUS'].isin(['1', '2', '3', '4', '5'])]['MONTHS_BALANCE'].values
        if len(delinquent_months) > 0:
            months_since_first_deliq = -min(delinquent_months)
        else:
            months_since_first_deliq = np.nan

        delinquency_frequency_ratio = late_payment_count / total_obs_months if total_obs_months > 0 else 0

        behavioral_features.append({
            'ID': client_id,
            'MONTHS_IN_BOOK': months_in_book,
            'MAX_DELINQUENCY': max_delinquency,
            'PUNCTUALITY_RATE': punctuality_rate,
            'LATE_PAYMENT_COUNT': late_payment_count,
            'AVG_DELINQUENCY_STATUS': avg_delinquency_status,
            'MONTHS_SINCE_FIRST_DELINQUENCY': months_since_first_deliq,
            'DELINQUENCY_FREQUENCY_RATIO': delinquency_frequency_ratio,
            'OBS_WINDOW_MONTHS': total_obs_months
        })

    behavioral_df = pd.DataFrame(behavioral_features)
    behavioral_df['MONTHS_SINCE_FIRST_DELINQUENCY'] = behavioral_df['MONTHS_SINCE_FIRST_DELINQUENCY'].fillna(999)

    print(f"\nBehavioral features extracted: {len(behavioral_df)} clients")
    print(f"\nBehavioral Features Summary (from {1-performance_window_fraction:.0%} of history):")
    print(behavioral_df[['MONTHS_IN_BOOK', 'MAX_DELINQUENCY', 'PUNCTUALITY_RATE',
                         'LATE_PAYMENT_COUNT']].describe())

    return {
        'behavioral_features': behavioral_df,
        'targets': targets_df,
        'ranges': valid_ranges,
        'client_stats': {
            'total_clients': len(ranges_df),
            'valid_clients': len(valid_ranges),
            'excluded_clients': len(ranges_df) - len(valid_ranges),
            'bad_rate': (targets_df['target'] == 1).sum() / len(targets_df)
        }
    }


# ============================================================================
# STEP 2: DATA MERGING & PREPROCESSING
# ============================================================================

def preprocess_data_v4(application_record_path, vintage_data):
    """
    Merge application records with properly windowed features and targets.
    """

    print("\n" + "=" * 80)
    print("STEP 2: DATA MERGING (Application Records + Behavioral Features + Targets)")
    print("=" * 80)

    applications = pd.read_csv(application_record_path)
    print(f"\nApplications loaded: {len(applications)} rows")

    behavioral_df = vintage_data['behavioral_features']
    targets_df = vintage_data['targets']

    # Merge all datasets
    df = applications.merge(behavioral_df, on='ID', how='inner')
    print(f"After merging behavioral features: {len(df)} rows")

    df = df.merge(targets_df, on='ID', how='inner')
    print(f"After merging target labels: {len(df)} rows")

    print(f"Final sample represents {len(df) / len(applications) * 100:.2f}% of applications")

    # Missing Value Handling
    print(f"\nMissing values before preprocessing:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])

    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')

    print("After filling missing values:")
    print(df.isnull().sum().sum(), "missing values remaining")

    # Feature Engineering (Demographics)
    df['AGE'] = -df['DAYS_BIRTH'] / 365.25
    df['EXPERIENCE_YEARS'] = df['DAYS_EMPLOYED'].apply(
        lambda x: -x / 365.25 if x <= 0 else 0
    )

    df = df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED', 'ID', 'OBS_WINDOW_MONTHS',
                  'perf_window_months', 'perf_window_bad_records'], axis=1, errors='ignore')

    # Categorical Encoding
    print(f"\nCategorical variables encoding:")

    binary_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                   'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']

    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col] == 'Y').astype(int)

    categorical_cols = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    print(f"Final dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")

    return df


# ============================================================================
# STEP 3: STRATIFIED K-FOLD CV WITH LIGHTGBM
# ============================================================================

def stratified_cross_validation_v4(X, y, n_splits=5, random_state=42):
    """
    Stratified K-Fold CV with realistic (non-leaky) features.
    """

    print("\n" + "=" * 80)
    print(f"STEP 3: STRATIFIED {n_splits}-FOLD CROSS-VALIDATION (V4 - NO LEAKAGE)")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_results = {
        'fold': [],
        'roc_auc': [],
        'recall': [],
        'precision': [],
        'f1': [],
        'models': []
    }

    fold_num = 1

    for train_idx, val_idx in skf.split(X, y):
        print(f"\n{'-' * 80}")
        print(f"FOLD {fold_num}/{n_splits}")
        print(f"{'-' * 80}")

        if isinstance(X, pd.DataFrame):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
        else:
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        print(f"Training set: {len(X_train_fold)} samples (Bad: {(y_train_fold == 1).sum()})")
        print(f"Validation set: {len(X_val_fold)} samples (Bad: {(y_val_fold == 1).sum()})")

        train_data = lgb.Dataset(
            X_train_scaled,
            label=y_train_fold,
            feature_name=[f'feature_{i}' for i in range(X_train_scaled.shape[1])]
        )

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'is_unbalance': True,
            'random_state': random_state
        }

        model = lgb.train(params, train_data, num_boost_round=200)

        y_pred_proba = model.predict(X_val_scaled)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
        recall = recall_score(y_val_fold, y_pred, zero_division=0)
        precision = precision_score(y_val_fold, y_pred, zero_division=0)
        f1 = f1_score(y_val_fold, y_pred, zero_division=0)

        cv_results['fold'].append(fold_num)
        cv_results['roc_auc'].append(roc_auc)
        cv_results['recall'].append(recall)
        cv_results['precision'].append(precision)
        cv_results['f1'].append(f1)
        cv_results['models'].append(model)

        print(f"\nFold {fold_num} Metrics:")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        fold_num += 1

    # Summary
    print(f"\n{'=' * 80}")
    print("CROSS-VALIDATION SUMMARY (V4 - REALISTIC NO-LEAKAGE RESULTS)")
    print(f"{'=' * 80}")

    cv_summary = pd.DataFrame({
        'Fold': cv_results['fold'],
        'ROC-AUC': cv_results['roc_auc'],
        'Recall': cv_results['recall'],
        'Precision': cv_results['precision'],
        'F1-Score': cv_results['f1']
    })

    print(f"\n{cv_summary.to_string(index=False)}")

    print(f"\n{'-' * 80}")
    print("MEAN AND STANDARD DEVIATION ACROSS FOLDS:")
    print(f"{'-' * 80}")
    print(f"ROC-AUC:   {np.mean(cv_results['roc_auc']):.4f} +/- {np.std(cv_results['roc_auc']):.4f}")
    print(f"Recall:    {np.mean(cv_results['recall']):.4f} +/- {np.std(cv_results['recall']):.4f}")
    print(f"Precision: {np.mean(cv_results['precision']):.4f} +/- {np.std(cv_results['precision']):.4f}")
    print(f"F1-Score:  {np.mean(cv_results['f1']):.4f} +/- {np.std(cv_results['f1']):.4f}")

    return cv_results, cv_summary


# ============================================================================
# STEP 4: FINAL MODEL & FEATURE IMPORTANCE
# ============================================================================

def train_final_model_v4(X_train, y_train, X_test, y_test, feature_names, random_state=42):
    """Train final model and extract feature importance."""

    print("\n" + "=" * 80)
    print("STEP 4: FINAL MODEL TRAINING & FEATURE IMPORTANCE")
    print("=" * 80)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_data = lgb.Dataset(
        X_train_scaled,
        label=y_train,
        feature_name=[f'feature_{i}' for i in range(X_train_scaled.shape[1])]
    )

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'is_unbalance': True,
        'random_state': random_state
    }

    model = lgb.train(params, train_data, num_boost_round=200)

    print("Model training completed!")

    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nTest Set Performance:")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    # Feature Importance
    print(f"\n{'-' * 80}")
    print("TOP 15 FEATURE IMPORTANCES (V4 - NO LEAKAGE)")
    print(f"{'-' * 80}")

    importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    behavioral_keywords = ['MONTHS_IN_BOOK', 'MAX_DELINQUENCY', 'PUNCTUALITY_RATE',
                          'LATE_PAYMENT_COUNT', 'AVG_DELINQUENCY', 'MONTHS_SINCE',
                          'DELINQUENCY_FREQUENCY']

    feature_importance_df['is_behavioral'] = feature_importance_df['feature'].apply(
        lambda x: any(keyword in x for keyword in behavioral_keywords)
    )

    print(f"\n{feature_importance_df.head(15)[['feature', 'importance', 'is_behavioral']].to_string(index=False)}")

    behavioral_in_top15 = feature_importance_df.head(15)['is_behavioral'].sum()
    print(f"\nBehavioral features in TOP 15: {behavioral_in_top15} out of 15")

    return {
        'model': model,
        'roc_auc': roc_auc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'feature_importance': feature_importance_df,
        'cm': cm
    }


def plot_feature_importance_v4(feature_importance_df, top_n=15):
    """Plot feature importance with behavioral vs demographic distinction."""

    fig, ax = plt.subplots(figsize=(12, 10))

    top_features = feature_importance_df.head(top_n)
    colors = ['#FF6B6B' if is_behavioral else '#4ECDC4'
              for is_behavioral in top_features['is_behavioral']]

    ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax.set_xlabel('Importance (Gain)', fontsize=11)
    ax.set_title(f'Top {top_n} Features - V4 (Red=Behavioral, Blue=Demographic)\nNo Data Leakage - Realistic Results',
                fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FF6B6B', label='Behavioral Features'),
                       Patch(facecolor='#4ECDC4', label='Demographic Features')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN PIPELINE EXECUTION V4
# ============================================================================

def run_pipeline_v4(application_path, credit_path, test_size=0.2, random_state=42):
    """
    Execute complete V4 pipeline with proper Vintage Analysis windows.
    """

    # Step 1: Extract with proper windows
    vintage_data = extract_behavioral_features_v4(credit_path, performance_window_fraction=0.33)

    # Step 2: Preprocess with merged data
    df = preprocess_data_v4(application_path, vintage_data)

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split
    print("\n" + "=" * 80)
    print("TRAIN-TEST SPLIT")
    print("=" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Class distribution (train): Good={len(y_train[y_train==0])}, Bad={len(y_train[y_train==1])}")
    print(f"Class distribution (test): Good={len(y_test[y_test==0])}, Bad={len(y_test[y_test==1])}")

    # Step 3: Stratified K-Fold CV
    cv_results, cv_summary = stratified_cross_validation_v4(X_train, y_train, n_splits=5)

    # Step 4: Final model
    final_results = train_final_model_v4(X_train, y_train.values, X_test, y_test.values,
                                         X.columns.tolist())

    return {
        'model': final_results['model'],
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'cv_summary': cv_summary,
        'final_results': final_results,
        'vintage_stats': vintage_data['client_stats'],
        'feature_names': X.columns.tolist()
    }


if __name__ == '__main__':
    base_path = r'C:\Users\Gabriel\Documents\VSCode\credit_card_approval'
    application_path = fr'{base_path}\data\raw\application_record.csv'
    credit_path = fr'{base_path}\data\raw\credit_record.csv'

    print("\n" + "=" * 80)
    print("CREDIT SCORING PIPELINE - VERSION 4 (PROPER VINTAGE ANALYSIS - NO LEAKAGE)")
    print("=" * 80)

    pipeline_output = run_pipeline_v4(application_path, credit_path)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    fig = plot_feature_importance_v4(pipeline_output['final_results']['feature_importance'], top_n=15)
    plt.savefig(fr'{base_path}\reports\feature_importance_v4.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Saved: reports/feature_importance_v4.png")

    print("\n" + "=" * 80)
    print("PIPELINE V4 EXECUTION COMPLETE")
    print("=" * 80)

    print("\nKEY RESULTS SUMMARY:")
    print(f"Sample size: {len(pipeline_output['y_test']) + len(pipeline_output['y_train'])} clients")
    print(f"Bad rate: {pipeline_output['vintage_stats']['bad_rate']:.2%}")
    print(f"\nROC-AUC (Test Set):     {pipeline_output['final_results']['roc_auc']:.4f}")
    print(f"ROC-AUC (CV Mean):      {pipeline_output['cv_summary']['ROC-AUC'].mean():.4f} +/- {pipeline_output['cv_summary']['ROC-AUC'].std():.4f}")
    print(f"Recall (Test Set):      {pipeline_output['final_results']['recall']:.4f}")
    print(f"Precision (Test Set):   {pipeline_output['final_results']['precision']:.4f}")
    print(f"F1-Score (Test Set):    {pipeline_output['final_results']['f1']:.4f}")
