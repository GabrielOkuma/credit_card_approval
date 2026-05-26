"""
Credit Scoring ML Pipeline - Version 3 (Feature Engineering)

This module implements advanced behavioral feature engineering from credit history:
1. Extract temporal and behavioral patterns from credit_record.csv
2. Calculate aggregated statistics (punctuality, delinquency patterns)
3. Integrate with existing pipeline
4. Evaluate impact on model performance using 5-Fold Stratified CV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    roc_curve, precision_recall_curve, auc, confusion_matrix
)
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: VINTAGE ANALYSIS - Label Construction (Reused from V2)
# ============================================================================

def create_target_labels(credit_record_path):
    """Create target labels from credit history."""

    print("=" * 70)
    print("STEP 1: VINTAGE ANALYSIS - Label Construction")
    print("=" * 70)

    credit_record = pd.read_csv(credit_record_path)
    print(f"\nCredit records loaded: {len(credit_record)} rows")
    print(f"Unique clients: {credit_record['ID'].nunique()}")

    print("\nStatus distribution in credit history:")
    print(credit_record['STATUS'].value_counts().sort_index())

    bad_statuses = ['2', '3', '4', '5']

    target_labels = (
        credit_record[credit_record['STATUS'].isin(bad_statuses)]
        .groupby('ID')
        .size()
        .reset_index(name='_bad_flag')
    )
    target_labels['target'] = 1
    target_labels = target_labels[['ID', 'target']]

    all_ids = credit_record['ID'].unique()
    all_clients = pd.DataFrame({'ID': all_ids})
    target_labels = all_clients.merge(target_labels, on='ID', how='left')
    target_labels['target'] = target_labels['target'].fillna(0).astype(int)

    print(f"\nTarget distribution:")
    print(target_labels['target'].value_counts())

    return target_labels


# ============================================================================
# STEP 2: ADVANCED BEHAVIORAL FEATURE ENGINEERING
# ============================================================================

def extract_behavioral_features(credit_record_path):
    """
    Extract temporal and behavioral features from credit history.

    This is the key innovation in V3: Instead of just a binary target,
    we extract rich behavioral signals that capture payment patterns.

    Features:
    - MONTHS_IN_BOOK: Total credit history length (depth of history)
    - MAX_DELINQUENCY: Worst status ever reached (severity of worst episode)
    - PUNCTUALITY_RATE: % of months in good standing (reliability metric)
    - LATE_PAYMENT_COUNT: Number of delinquent months (frequency metric)
    - AVG_DELINQUENCY_STATUS: Average delinquency level (persistence metric)
    - MONTHS_SINCE_FIRST_DELINQUENCY: Recency of delinquency
    - DELINQUENCY_FREQUENCY_RATIO: % of months with any delinquency

    Returns:
    --------
    pd.DataFrame : Behavioral features indexed by client ID
    """

    print("\n" + "=" * 70)
    print("STEP 2: ADVANCED BEHAVIORAL FEATURE ENGINEERING")
    print("=" * 70)

    credit_record = pd.read_csv(credit_record_path)
    print(f"\nProcessing {len(credit_record)} credit records for {credit_record['ID'].nunique()} clients...")

    behavioral_features = []

    for client_id in credit_record['ID'].unique():
        client_history = credit_record[credit_record['ID'] == client_id].copy()

        # -------- Feature 1: MONTHS_IN_BOOK --------
        # Total length of credit history (absolute value of MONTHS_BALANCE)
        # Deeper history = more data about client = lower risk
        months_in_book = np.abs(client_history['MONTHS_BALANCE']).max()

        # -------- Feature 2: MAX_DELINQUENCY --------
        # Worst status ever reached (0=none/current, 1=minor late, 5=severe late)
        # Convert categorical statuses to numeric
        status_mapping = {
            'C': 0,  # Current/Paid off (good)
            'X': 0,  # No loan (neutral)
            '0': 0.5,  # 1-29 days late (minor)
            '1': 1.5,  # 30-59 days late (moderate)
            '2': 2.5,  # 60-89 days late (severe)
            '3': 3.5,  # 90-119 days late (very severe)
            '4': 4.5,  # 120-149 days late (critical)
            '5': 5.0   # >150 days late (worst)
        }

        numeric_status = client_history['STATUS'].map(status_mapping)
        max_delinquency = numeric_status.max()

        # -------- Feature 3: PUNCTUALITY_RATE --------
        # Percentage of months where client was in good standing (C or 0 status)
        # Higher = more reliable
        good_standing_statuses = ['C', '0']
        good_standing_months = (client_history['STATUS'].isin(good_standing_statuses)).sum()
        total_active_months = len(client_history)
        punctuality_rate = good_standing_months / total_active_months if total_active_months > 0 else 0

        # -------- Feature 4: LATE_PAYMENT_COUNT --------
        # Total number of months with any delinquency
        late_statuses = ['1', '2', '3', '4', '5']
        late_payment_count = (client_history['STATUS'].isin(late_statuses)).sum()

        # -------- Feature 5: AVG_DELINQUENCY_STATUS --------
        # Average delinquency severity across all months
        avg_delinquency_status = numeric_status.mean()

        # -------- Feature 6: MONTHS_SINCE_FIRST_DELINQUENCY --------
        # How long ago was the first delinquency? (recency)
        delinquent_months = client_history[client_history['STATUS'].isin(late_statuses)]['MONTHS_BALANCE'].values
        if len(delinquent_months) > 0:
            # Most recent month = 0, further back = more negative
            months_since_first_deliq = -min(delinquent_months)
        else:
            months_since_first_deliq = np.nan  # Never delinquent

        # -------- Feature 7: DELINQUENCY_FREQUENCY_RATIO --------
        # Percentage of months with any delinquency
        delinquency_frequency_ratio = late_payment_count / total_active_months if total_active_months > 0 else 0

        behavioral_features.append({
            'ID': client_id,
            'MONTHS_IN_BOOK': months_in_book,
            'MAX_DELINQUENCY': max_delinquency,
            'PUNCTUALITY_RATE': punctuality_rate,
            'LATE_PAYMENT_COUNT': late_payment_count,
            'AVG_DELINQUENCY_STATUS': avg_delinquency_status,
            'MONTHS_SINCE_FIRST_DELINQUENCY': months_since_first_deliq,
            'DELINQUENCY_FREQUENCY_RATIO': delinquency_frequency_ratio
        })

    behavioral_df = pd.DataFrame(behavioral_features)

    # Fill NaN in MONTHS_SINCE_FIRST_DELINQUENCY with 999 (never delinquent = very low risk)
    behavioral_df['MONTHS_SINCE_FIRST_DELINQUENCY'] = behavioral_df['MONTHS_SINCE_FIRST_DELINQUENCY'].fillna(999)

    print(f"\nBehavioral features extracted for {len(behavioral_df)} clients")
    print(f"\nBehavioral Features Summary:")
    print(behavioral_df.describe())

    return behavioral_df


# ============================================================================
# STEP 3: DATA MERGING & PREPROCESSING WITH NEW FEATURES
# ============================================================================

def preprocess_data_v3(application_record_path, target_labels, behavioral_features):
    """
    Merge application records, behavioral features, and target labels.

    Key difference from V2: Includes behavioral features from credit history.
    """

    print("\n" + "=" * 70)
    print("STEP 3: DATA MERGING & PREPROCESSING (WITH BEHAVIORAL FEATURES)")
    print("=" * 70)

    applications = pd.read_csv(application_record_path)
    print(f"\nApplications loaded: {len(applications)} rows")

    # Merge all datasets
    df = applications.merge(behavioral_features, on='ID', how='left')
    print(f"After merging behavioral features: {len(df)} rows")

    df = df.merge(target_labels, on='ID', how='inner')
    print(f"After merging target labels (inner join): {len(df)} rows")

    # Missing Value Handling
    print("\nMissing values before preprocessing:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])

    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')

    # For behavioral features, fill any remaining NaN with 0 (conservative approach)
    behavioral_cols = ['MONTHS_IN_BOOK', 'MAX_DELINQUENCY', 'PUNCTUALITY_RATE',
                       'LATE_PAYMENT_COUNT', 'AVG_DELINQUENCY_STATUS',
                       'MONTHS_SINCE_FIRST_DELINQUENCY', 'DELINQUENCY_FREQUENCY_RATIO']
    for col in behavioral_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print("After filling missing values:")
    print(df.isnull().sum().sum(), "missing values remaining")

    # Feature Engineering (Demographics)
    df['AGE'] = -df['DAYS_BIRTH'] / 365.25
    df['EXPERIENCE_YEARS'] = df['DAYS_EMPLOYED'].apply(
        lambda x: -x / 365.25 if x <= 0 else 0
    )
    print(f"\nDemographic features created - AGE: {df['AGE'].mean():.1f} years, EXPERIENCE: {df['EXPERIENCE_YEARS'].mean():.1f} years")

    df = df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED', 'ID'], axis=1)

    # Categorical Encoding
    print("\nCategorical variables encoding:")

    binary_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                   'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']

    for col in binary_cols:
        df[col] = (df[col] == 'Y').astype(int)
    print(f"Converted {len(binary_cols)} binary columns to 0/1")

    categorical_cols = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    print(f"One-Hot encoded {len(categorical_cols)} categorical columns")

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Total features: {len(df.columns) - 1} (excluding target)")
    print(f"New behavioral features: {len(behavioral_cols)}")
    print(f"Target distribution:\n{df['target'].value_counts()}")

    return df


# ============================================================================
# STEP 4: STRATIFIED K-FOLD CV WITH COMPARISON
# ============================================================================

def stratified_cross_validation_v3(X, y, n_splits=5, random_state=42):
    """
    Stratified K-Fold CV with LightGBM (same as V2 but returns more details).
    """

    print("\n" + "=" * 70)
    print(f"STEP 4: STRATIFIED {n_splits}-FOLD CROSS-VALIDATION (V3)")
    print("=" * 70)

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
        print(f"\n{'-' * 70}")
        print(f"FOLD {fold_num}/{n_splits}")
        print(f"{'-' * 70}")

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

        print(f"Training set: {len(X_train_fold)} samples")
        print(f"Validation set: {len(X_val_fold)} samples")

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
        recall = recall_score(y_val_fold, y_pred)
        precision = precision_score(y_val_fold, y_pred, zero_division=0)
        f1 = f1_score(y_val_fold, y_pred)

        cv_results['fold'].append(fold_num)
        cv_results['roc_auc'].append(roc_auc)
        cv_results['recall'].append(recall)
        cv_results['precision'].append(precision)
        cv_results['f1'].append(f1)
        cv_results['models'].append(model)

        print(f"\nFold {fold_num} Metrics:")
        print(f"  ROC-AUC:  {roc_auc:.4f}")
        print(f"  Recall:   {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        fold_num += 1

    # Summary
    print(f"\n{'=' * 70}")
    print("CROSS-VALIDATION SUMMARY (V3)")
    print(f"{'=' * 70}")

    cv_summary = pd.DataFrame({
        'Fold': cv_results['fold'],
        'ROC-AUC': cv_results['roc_auc'],
        'Recall': cv_results['recall'],
        'Precision': cv_results['precision'],
        'F1-Score': cv_results['f1']
    })

    print(f"\n{cv_summary.to_string(index=False)}")

    print(f"\n{'-' * 70}")
    print("MEAN AND STANDARD DEVIATION ACROSS FOLDS:")
    print(f"{'-' * 70}")
    print(f"ROC-AUC:   {np.mean(cv_results['roc_auc']):.4f} +/- {np.std(cv_results['roc_auc']):.4f}")
    print(f"Recall:    {np.mean(cv_results['recall']):.4f} +/- {np.std(cv_results['recall']):.4f}")
    print(f"Precision: {np.mean(cv_results['precision']):.4f} +/- {np.std(cv_results['precision']):.4f}")
    print(f"F1-Score:  {np.mean(cv_results['f1']):.4f} +/- {np.std(cv_results['f1']):.4f}")

    return cv_results, cv_summary


# ============================================================================
# STEP 5: FEATURE IMPORTANCE COMPARISON
# ============================================================================

def train_final_model_v3(X_train, y_train, X_test, y_test, random_state=42):
    """Train final model and extract feature importance."""

    print("\n" + "=" * 70)
    print("STEP 5: FINAL MODEL TRAINING & FEATURE IMPORTANCE")
    print("=" * 70)

    print("\nTraining LightGBM on full training set...")

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
    y_pred_default = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred_default)

    print(f"\nTest Set Performance:")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    return {
        'model': model,
        'scaler': scaler,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test,
        'roc_auc': roc_auc,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled
    }


def compare_feature_importance(model, feature_names, top_n=20):
    """
    Extract and display feature importance, highlighting behavioral features.
    """

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS (V3 with Behavioral Features)")
    print("=" * 70)

    importance = model.feature_importance(importance_type='gain')

    # Map feature indices back to names
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Identify behavioral features
    behavioral_feature_keywords = ['MONTHS_IN_BOOK', 'MAX_DELINQUENCY', 'PUNCTUALITY_RATE',
                                   'LATE_PAYMENT_COUNT', 'AVG_DELINQUENCY', 'MONTHS_SINCE',
                                   'DELINQUENCY_FREQUENCY']

    feature_importance_df['is_behavioral'] = feature_importance_df['feature'].apply(
        lambda x: any(keyword in x for keyword in behavioral_feature_keywords)
    )

    print(f"\nTop {top_n} Most Important Features:")
    print(feature_importance_df.head(top_n)[['feature', 'importance', 'is_behavioral']].to_string(index=False))

    # Count behavioral features in top N
    behavioral_in_top = feature_importance_df.head(top_n)['is_behavioral'].sum()
    print(f"\nBehavioral features in TOP {top_n}: {behavioral_in_top} out of {top_n}")
    print(f"Average importance (behavioral): {feature_importance_df[feature_importance_df['is_behavioral']]['importance'].mean():.0f}")
    print(f"Average importance (demographic): {feature_importance_df[~feature_importance_df['is_behavioral']]['importance'].mean():.0f}")

    return feature_importance_df


def plot_feature_importance_comparison(feature_importance_df, top_n=20):
    """
    Plot feature importance with behavioral features highlighted.
    """

    fig, ax = plt.subplots(figsize=(12, 10))

    top_features = feature_importance_df.head(top_n)

    # Color behavioral features differently
    colors = ['#FF6B6B' if is_behavioral else '#4ECDC4'
              for is_behavioral in top_features['is_behavioral']]

    ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values, fontsize=9)
    ax.set_xlabel('Importance (Gain)', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importances - V3 (Red=Behavioral, Blue=Demographic)', fontsize=12)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FF6B6B', label='Behavioral Features'),
                       Patch(facecolor='#4ECDC4', label='Demographic Features')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    return fig


def plot_cv_comparison(cv_summary_v2=None, cv_summary_v3=None):
    """
    Compare V2 vs V3 performance across folds.
    """

    if cv_summary_v2 is None or cv_summary_v3 is None:
        print("Skipping V2 vs V3 comparison (insufficient data)")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['ROC-AUC', 'Recall', 'Precision', 'F1-Score']
    v2_values = [cv_summary_v2[metric].mean() for metric in metrics]
    v3_values = [cv_summary_v3[metric].mean() for metric in metrics]

    # Bar chart comparison
    x = np.arange(len(metrics))
    width = 0.35

    axes[0, 0].bar(x - width/2, v2_values, width, label='V2 (Demographics)', color='steelblue')
    axes[0, 0].bar(x + width/2, v3_values, width, label='V3 (Behavioral)', color='coral')
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('V2 vs V3 Mean Performance')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3, axis='y')

    # ROC-AUC comparison
    axes[0, 1].plot(cv_summary_v2['Fold'], cv_summary_v2['ROC-AUC'], marker='o',
                   label='V2', linewidth=2, markersize=8)
    axes[0, 1].plot(cv_summary_v3['Fold'], cv_summary_v3['ROC-AUC'], marker='s',
                   label='V3', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].set_title('ROC-AUC Across Folds')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Recall comparison
    axes[1, 0].plot(cv_summary_v2['Fold'], cv_summary_v2['Recall'], marker='o',
                   label='V2', linewidth=2, markersize=8)
    axes[1, 0].plot(cv_summary_v3['Fold'], cv_summary_v3['Recall'], marker='s',
                   label='V3', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall Across Folds')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Improvement percentage
    improvements = [(v3_values[i] - v2_values[i]) / v2_values[i] * 100
                   for i in range(len(metrics))]
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    axes[1, 1].bar(metrics, improvements, color=colors_imp, alpha=0.7)
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].set_title('V3 Improvement over V2')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 1].grid(alpha=0.3, axis='y')

    # Add percentage labels on bars
    for i, (metric, imp) in enumerate(zip(metrics, improvements)):
        axes[1, 1].text(i, imp, f'{imp:+.2f}%', ha='center', va='bottom' if imp > 0 else 'top')

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN PIPELINE EXECUTION V3
# ============================================================================

def run_pipeline_v3(application_path, credit_path, test_size=0.2, random_state=42):
    """
    Execute the complete V3 ML pipeline with behavioral feature engineering.
    """

    # Step 1: Vintage Analysis
    target_labels = create_target_labels(credit_path)

    # Step 2: Behavioral Feature Engineering
    behavioral_features = extract_behavioral_features(credit_path)

    # Step 3: Preprocessing with new features
    df = preprocess_data_v3(application_path, target_labels, behavioral_features)

    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split
    print("\n" + "=" * 70)
    print("TRAIN-TEST SPLIT")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTrain set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Step 4: Stratified K-Fold CV
    cv_results, cv_summary = stratified_cross_validation_v3(X_train, y_train, n_splits=5)

    # Step 5: Final model and feature importance
    final_results = train_final_model_v3(X_train, y_train.values, X_test, y_test.values)

    # Extract and compare feature importance
    feature_importance_df = compare_feature_importance(final_results['model'], X.columns.tolist(), top_n=20)

    return {
        'model': final_results['model'],
        'scaler': final_results['scaler'],
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'y_pred_proba': final_results['y_pred_proba'],
        'cv_results': cv_results,
        'cv_summary': cv_summary,
        'feature_importance': feature_importance_df,
        'feature_names': X.columns.tolist(),
        'roc_auc': final_results['roc_auc'],
        'behavioral_features': behavioral_features
    }


if __name__ == '__main__':
    # Set paths
    base_path = r'C:\Users\Gabriel\Documents\VSCode\credit_card_approval'
    application_path = fr'{base_path}\data\raw\application_record.csv'
    credit_path = fr'{base_path}\data\raw\credit_record.csv'

    # Run V3 pipeline
    print("\n" + "=" * 70)
    print("CREDIT SCORING PIPELINE - VERSION 3 (FEATURE ENGINEERING)")
    print("=" * 70)

    pipeline_output = run_pipeline_v3(application_path, credit_path)

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    fig1 = plot_feature_importance_comparison(pipeline_output['feature_importance'], top_n=20)
    plt.savefig(fr'{base_path}\reports\feature_importance_v3.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Saved: reports/feature_importance_v3.png")

    print("\n" + "=" * 70)
    print("PIPELINE V3 EXECUTION COMPLETE")
    print("=" * 70)

    print("\nKEY METRICS SUMMARY:")
    print(f"ROC-AUC (Test Set): {pipeline_output['roc_auc']:.4f}")
    print(f"ROC-AUC (CV Mean):  {pipeline_output['cv_summary']['ROC-AUC'].mean():.4f}")
    print(f"ROC-AUC (CV Std):   {pipeline_output['cv_summary']['ROC-AUC'].std():.4f}")
