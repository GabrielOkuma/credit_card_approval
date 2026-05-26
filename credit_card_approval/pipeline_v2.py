"""
Credit Scoring ML Pipeline - Version 2 (Enhanced)

This module upgrades the credit scoring pipeline with:
1. LightGBM classifier with native imbalance handling
2. Stratified K-Fold Cross-Validation for robust evaluation
3. Threshold optimization and sensitivity analysis
4. Comprehensive comparison metrics across folds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, f1_score,
    confusion_matrix, classification_report, auc, recall_score, precision_score
)
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: VINTAGE ANALYSIS - Label Construction
# ============================================================================

def create_target_labels(credit_record_path):
    """
    Perform Vintage Analysis to construct target labels.

    Logic:
    - For each client (ID), examine their entire credit history
    - A client is marked as 'bad' (target=1) if they ever had:
      * Status 2 (60-89 days past due)
      * Status 3 (90-119 days past due)
      * Status 4 (120-149 days past due)
      * Status 5 (>150 days overdue/bad debt)
    - Otherwise, they are marked as 'good' (target=0)
    """

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
    print(f"Bad clients: {(target_labels['target'] == 1).sum()} ({(target_labels['target'] == 1).sum() / len(target_labels) * 100:.2f}%)")
    print(f"Good clients: {(target_labels['target'] == 0).sum()} ({(target_labels['target'] == 0).sum() / len(target_labels) * 100:.2f}%)")

    return target_labels


# ============================================================================
# STEP 2: DATA MERGING & PREPROCESSING
# ============================================================================

def preprocess_data(application_record_path, target_labels):
    """
    Merge application records with target labels and perform preprocessing.
    """

    print("\n" + "=" * 70)
    print("STEP 2: DATA MERGING & PREPROCESSING")
    print("=" * 70)

    applications = pd.read_csv(application_record_path)
    print(f"\nApplications loaded: {len(applications)} rows")

    df = applications.merge(target_labels, on='ID', how='inner')
    print(f"After merging with target labels: {len(df)} rows")

    # Missing Value Handling
    print("\nMissing values before preprocessing:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])

    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')

    print("After filling missing values:")
    print(df.isnull().sum().sum(), "missing values remaining")

    # Feature Engineering
    df['AGE'] = -df['DAYS_BIRTH'] / 365.25
    df['EXPERIENCE_YEARS'] = df['DAYS_EMPLOYED'].apply(
        lambda x: -x / 365.25 if x <= 0 else 0
    )
    print(f"\nAGE created - Mean: {df['AGE'].mean():.1f} years")
    print(f"EXPERIENCE_YEARS created - Mean: {df['EXPERIENCE_YEARS'].mean():.1f} years")

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
    print(f"Total features after encoding: {len(df.columns) - 1}")

    print(f"\nFinal preprocessed dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")

    return df


# ============================================================================
# STEP 3: STRATIFIED K-FOLD CROSS-VALIDATION WITH LIGHTGBM
# ============================================================================

def stratified_cross_validation(X, y, n_splits=5, random_state=42):
    """
    Perform Stratified K-Fold Cross-Validation with LightGBM.

    LightGBM handles class imbalance natively via is_unbalance=True parameter,
    avoiding the need for SMOTE during cross-validation.

    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    n_splits : int
        Number of folds for cross-validation
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    dict : Cross-validation results including metrics for each fold
    """

    print("\n" + "=" * 70)
    print(f"STEP 3: STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Storage for metrics
    cv_results = {
        'fold': [],
        'roc_auc': [],
        'recall': [],
        'precision': [],
        'f1': [],
        'models': [],
        'y_preds': [],
        'y_pred_probas': []
    }

    fold_num = 1

    for train_idx, val_idx in skf.split(X, y):
        print(f"\n{'-' * 70}")
        print(f"FOLD {fold_num}/{n_splits}")
        print(f"{'-' * 70}")

        # Split data
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

        # Scale features (only if not already scaled)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        print(f"Training set: {len(X_train_fold)} samples")
        print(f"Validation set: {len(X_val_fold)} samples")
        print(f"Class distribution (train): {pd.Series(y_train_fold).value_counts().to_dict()}")

        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X_train_scaled,
            label=y_train_fold,
            feature_name=[f'feature_{i}' for i in range(X_train_scaled.shape[1])]
        )

        # LightGBM parameters with native imbalance handling
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'is_unbalance': True,  # Native imbalance handling
            'random_state': random_state
        }

        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            valid_names=['training']
        )

        # Predict on validation set
        y_pred_proba = model.predict(X_val_scaled)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
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
        cv_results['y_preds'].append(y_pred)
        cv_results['y_pred_probas'].append(y_pred_proba)

        # Print fold metrics
        print(f"\nFold {fold_num} Metrics:")
        print(f"  ROC-AUC:  {roc_auc:.4f}")
        print(f"  Recall:   {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        fold_num += 1

    # Calculate and print cross-validation summary
    print(f"\n{'=' * 70}")
    print("CROSS-VALIDATION SUMMARY")
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
    print(f"ROC-AUC:   {np.mean(cv_results['roc_auc']):.4f} ± {np.std(cv_results['roc_auc']):.4f}")
    print(f"Recall:    {np.mean(cv_results['recall']):.4f} ± {np.std(cv_results['recall']):.4f}")
    print(f"Precision: {np.mean(cv_results['precision']):.4f} ± {np.std(cv_results['precision']):.4f}")
    print(f"F1-Score:  {np.mean(cv_results['f1']):.4f} ± {np.std(cv_results['f1']):.4f}")

    return cv_results, cv_summary


# ============================================================================
# STEP 4: FINAL MODEL TRAINING & THRESHOLD ANALYSIS
# ============================================================================

def train_final_model(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train final LightGBM model on complete training set.

    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    dict : Model and predictions
    """

    print("\n" + "=" * 70)
    print("STEP 4: FINAL MODEL TRAINING")
    print("=" * 70)

    print("\nTraining LightGBM on full training set...")

    # Create dataset
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=[f'feature_{i}' for i in range(X_train.shape[1])]
    )

    # LightGBM parameters
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

    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200
    )

    print("Model training completed!")

    # Predictions
    y_pred_proba = model.predict(X_test)
    y_pred_default = (y_pred_proba >= 0.5).astype(int)

    # Test set metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred_default)

    print(f"\nTest Set Performance (Threshold = 0.5):")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    return {
        'model': model,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test,
        'roc_auc': roc_auc
    }


def analyze_thresholds(y_test, y_pred_proba, thresholds=[0.3, 0.4, 0.5]):
    """
    Analyze model performance at different probability thresholds.

    Parameters:
    -----------
    y_test : np.ndarray
        True test labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    thresholds : list
        List of thresholds to evaluate

    Returns:
    --------
    pd.DataFrame : Performance metrics at each threshold
    """

    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("=" * 70)

    results = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results.append({
            'Threshold': threshold,
            'ROC-AUC': roc_auc,
            'Recall': recall,
            'Specificity': specificity,
            'Precision': precision,
            'F1-Score': f1,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn
        })

        print(f"\n{'-' * 70}")
        print(f"THRESHOLD = {threshold}")
        print(f"{'-' * 70}")
        print(f"ROC-AUC:      {roc_auc:.4f}")
        print(f"Recall:       {recall:.4f} (Sensitivity - Bad detection rate)")
        print(f"Specificity:  {specificity:.4f} (Good approval rate)")
        print(f"Precision:    {precision:.4f} (Bad prediction accuracy)")
        print(f"F1-Score:     {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP (Correct bad):    {tp:5d}")
        print(f"  FP (Good marked bad): {fp:5d}")
        print(f"  TN (Correct good):   {tn:5d}")
        print(f"  FN (Bad marked good): {fn:5d}")
        print(f"\nInterpretation:")
        print(f"  Correctly identified {tp} out of {tp + fn} bad clients (Recall: {recall:.1%})")
        print(f"  False rejected {fp} good clients out of {fp + tn} good applicants")
        print(f"  When we predict 'bad', we're correct {precision:.1%} of the time")

    threshold_df = pd.DataFrame(results)
    return threshold_df


# ============================================================================
# STEP 5: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot feature importance from LightGBM model.

    Parameters:
    -----------
    model : lgb.Booster
        Trained LightGBM model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    """

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(f"\nTop {top_n} Most Important Features:")
    print(feature_importance_df.head(top_n).to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = feature_importance_df.head(top_n)
    ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance (Gain)')
    ax.set_title(f'Top {top_n} Feature Importances - LightGBM')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()

    return fig, feature_importance_df


def plot_cv_metrics(cv_summary):
    """
    Visualize cross-validation metrics across folds.

    Parameters:
    -----------
    cv_summary : pd.DataFrame
        Summary metrics from cross-validation
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ['ROC-AUC', 'Recall', 'Precision', 'F1-Score']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
        values = cv_summary[metric].values
        folds = cv_summary['Fold'].values

        ax.plot(folds, values, marker='o', linewidth=2, markersize=8, color=color)
        ax.axhline(y=values.mean(), color=color, linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Mean: {values.mean():.4f}')
        ax.fill_between(folds, values.mean() - values.std(), values.mean() + values.std(),
                        alpha=0.2, color=color)

        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Across Folds')
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_xticks(folds)

    plt.tight_layout()
    return fig


def plot_roc_and_pr_curves(y_test, y_pred_proba, roc_auc):
    """
    Plot ROC curve and Precision-Recall curve.

    Parameters:
    -----------
    y_test : np.ndarray
        True test labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    roc_auc : float
        ROC-AUC score
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def run_pipeline_v2(application_path, credit_path, test_size=0.2, random_state=42):
    """
    Execute the complete upgraded ML pipeline.

    Parameters:
    -----------
    application_path : str
        Path to application_record.csv
    credit_path : str
        Path to credit_record.csv
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    dict : Complete pipeline outputs
    """

    # Step 1: Vintage Analysis
    target_labels = create_target_labels(credit_path)

    # Step 2: Preprocessing
    df = preprocess_data(application_path, target_labels)

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
    print(f"Class distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Class distribution (test): {pd.Series(y_test).value_counts().to_dict()}")

    # Step 3: Stratified Cross-Validation (pass unscaled data)
    cv_results, cv_summary = stratified_cross_validation(X_train, y_train, n_splits=5)

    # Scale features for final model training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 4: Train final model
    final_results = train_final_model(X_train_scaled, y_train.values, X_test_scaled, y_test.values)

    # Step 5: Threshold analysis
    threshold_df = analyze_thresholds(y_test.values, final_results['y_pred_proba'])

    # Step 6: Feature importance
    feature_importance_df = final_results['model'].feature_importance(importance_type='gain')

    return {
        'model': final_results['model'],
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'y_pred_proba': final_results['y_pred_proba'],
        'cv_results': cv_results,
        'cv_summary': cv_summary,
        'threshold_df': threshold_df,
        'feature_names': X.columns.tolist(),
        'roc_auc': final_results['roc_auc']
    }


if __name__ == '__main__':
    # Set paths
    base_path = r'C:\Users\Gabriel\Documents\VSCode\credit_card_approval'
    application_path = fr'{base_path}\data\raw\application_record.csv'
    credit_path = fr'{base_path}\data\raw\credit_record.csv'

    # Run pipeline
    pipeline_output = run_pipeline_v2(application_path, credit_path)

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # CV metrics
    fig1 = plot_cv_metrics(pipeline_output['cv_summary'])
    plt.savefig(fr'{base_path}\reports\cv_metrics_across_folds.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Saved: reports/cv_metrics_across_folds.png")

    # ROC and PR curves
    fig2 = plot_roc_and_pr_curves(pipeline_output['y_test'], pipeline_output['y_pred_proba'],
                                   pipeline_output['roc_auc'])
    plt.savefig(fr'{base_path}\reports\roc_pr_curves.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: reports/roc_pr_curves.png")

    # Feature importance
    fig3, fi_df = plot_feature_importance(pipeline_output['model'], pipeline_output['feature_names'])
    plt.savefig(fr'{base_path}\reports\feature_importance.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: reports/feature_importance.png")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
