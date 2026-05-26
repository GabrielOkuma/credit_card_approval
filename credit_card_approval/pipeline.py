"""
Credit Scoring ML Pipeline

This module implements a complete machine learning pipeline for credit scoring:
1. Vintage Analysis: Construct target labels from credit history
2. Data Merging & Preprocessing: Merge datasets and handle data quality issues
3. Handling Imbalanced Data: Apply SMOTE to balance classes
4. Modeling & Evaluation: Train and evaluate classifier with appropriate metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, f1_score,
    confusion_matrix, classification_report, auc
)
from imblearn.over_sampling import SMOTE
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
    - This approach assumes the observation window includes the entire history

    Parameters:
    -----------
    credit_record_path : str
        Path to credit_record.csv

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns [ID, target] where target is 0 or 1
    """

    print("=" * 70)
    print("STEP 1: VINTAGE ANALYSIS - Label Construction")
    print("=" * 70)

    # Load credit history
    credit_record = pd.read_csv(credit_record_path)
    print(f"\nCredit records loaded: {len(credit_record)} rows")
    print(f"Unique clients: {credit_record['ID'].nunique()}")

    # Display status distribution
    print("\nStatus distribution in credit history:")
    print(credit_record['STATUS'].value_counts().sort_index())

    # Define "bad" client: anyone with status 2, 3, 4, or 5 (severe delinquency)
    # Status 0 = 1-29 days past due (minor delinquency)
    # Status 1 = 30-59 days past due (moderate delinquency)
    # Status 2-5 = severe delinquency (60+ days overdue)
    # Status C = paid off (good)
    # Status X = no loan (neutral)

    bad_statuses = ['2', '3', '4', '5']

    # Group by ID and check if they ever had severe delinquency
    target_labels = (
        credit_record[credit_record['STATUS'].isin(bad_statuses)]
        .groupby('ID')
        .size()
        .reset_index(name='_bad_flag')
    )
    target_labels['target'] = 1  # Mark as bad
    target_labels = target_labels[['ID', 'target']]

    # Get all unique IDs from credit records and mark missing ones as good (0)
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

    Preprocessing steps:
    - Inner join on ID (keep only clients with credit history)
    - Handle missing values
    - Feature engineering: DAYS_BIRTH -> AGE, DAYS_EMPLOYED -> EXPERIENCE
    - Encode categorical variables

    Parameters:
    -----------
    application_record_path : str
        Path to application_record.csv
    target_labels : pd.DataFrame
        Target labels from vintage analysis

    Returns:
    --------
    pd.DataFrame
        Preprocessed feature matrix with target variable
    """

    print("\n" + "=" * 70)
    print("STEP 2: DATA MERGING & PREPROCESSING")
    print("=" * 70)

    # Load application records
    applications = pd.read_csv(application_record_path)
    print(f"\nApplications loaded: {len(applications)} rows")

    # Merge with target labels (inner join - keep only clients with credit history)
    df = applications.merge(target_labels, on='ID', how='inner')
    print(f"After merging with target labels: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # -------- Missing Value Handling --------
    print("\nMissing values before preprocessing:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])

    # Fill OCCUPATION_TYPE with 'Unknown'
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')

    print("After filling missing values:")
    print(df.isnull().sum().sum(), "missing values remaining")

    # -------- Feature Engineering --------

    # Convert DAYS_BIRTH to age in years
    df['AGE'] = -df['DAYS_BIRTH'] / 365.25
    print(f"\nAGE created from DAYS_BIRTH - Mean: {df['AGE'].mean():.1f} years")

    # Handle DAYS_EMPLOYED: negative values = employed, positive values = unemployed
    # Convert to years of experience, treating positive as 0 (currently unemployed)
    df['EXPERIENCE_YEARS'] = df['DAYS_EMPLOYED'].apply(
        lambda x: -x / 365.25 if x <= 0 else 0
    )
    print(f"EXPERIENCE_YEARS created - Mean: {df['EXPERIENCE_YEARS'].mean():.1f} years")

    # Drop original DAYS columns
    df = df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED', 'ID'], axis=1)

    # -------- Categorical Encoding --------
    print("\nCategorical variables encoding:")

    # Binary categorical variables (already Y/N) -> convert to 0/1
    binary_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                   'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']

    for col in binary_cols:
        df[col] = (df[col] == 'Y').astype(int)
    print(f"Converted {len(binary_cols)} binary columns to 0/1")

    # One-Hot Encode nominal categorical variables
    categorical_cols = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
    print(f"One-Hot encoded {len(categorical_cols)} categorical columns")
    print(f"Total features after encoding: {len(df.columns) - 1}")  # -1 for target

    # Display final dataset info
    print(f"\nFinal preprocessed dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")

    return df


# ============================================================================
# STEP 3: HANDLING IMBALANCED DATA
# ============================================================================

def handle_imbalance(X_train, y_train):
    """
    Apply SMOTE to balance the training set.

    SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic
    samples for the minority class, balancing the training data.

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target

    Returns:
    --------
    tuple : (X_train_balanced, y_train_balanced)
    """

    print("\n" + "=" * 70)
    print("STEP 3: HANDLING IMBALANCED DATA")
    print("=" * 70)

    print(f"\nClass distribution BEFORE SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({c/len(y_train)*100:.2f}%)")

    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"\nClass distribution AFTER SMOTE:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples ({c/len(y_train_balanced)*100:.2f}%)")

    return X_train_balanced, y_train_balanced


# ============================================================================
# STEP 4: MODELING & EVALUATION
# ============================================================================

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train a RandomForest classifier and evaluate with appropriate metrics
    for imbalanced data.

    Metrics used:
    - ROC-AUC: Measures ability to distinguish between classes
    - Precision-Recall Curve: Better for imbalanced data
    - F1-Score: Harmonic mean of precision and recall
    - Confusion Matrix: Shows true positives, false positives, etc.
    - Classification Report: Detailed per-class metrics

    Parameters:
    -----------
    X_train : np.ndarray
        Training features (balanced via SMOTE)
    y_train : np.ndarray
        Training target
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target

    Returns:
    --------
    dict : Dictionary containing model and performance metrics
    """

    print("\n" + "=" * 70)
    print("STEP 4: MODELING & EVALUATION")
    print("=" * 70)

    # Train RandomForest
    print("\nTraining RandomForest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("Model training completed!")

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Probability of class 1

    # -------- Evaluation Metrics --------
    print("\n" + "-" * 70)
    print("EVALUATION METRICS")
    print("-" * 70)

    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

    # F1-Score
    f1 = f1_score(y_test, y_pred)
    print(f"F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Additional metrics from confusion matrix
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"\nSensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")

    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good', 'Bad']))

    return {
        'model': clf,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'confusion_matrix': cm
    }


def plot_evaluation_metrics(y_test, y_pred_proba, results):
    """
    Create visualizations for model evaluation.

    Plots:
    - ROC Curve
    - Precision-Recall Curve
    - Confusion Matrix Heatmap
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = results['roc_auc']
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    axes[0, 1].plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(alpha=0.3)

    # Confusion Matrix Heatmap
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Good', 'Bad'],
                yticklabels=['Good', 'Bad'])
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    axes[1, 0].set_title('Confusion Matrix')

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': range(results['model'].n_features_in_),
        'importance': results['model'].feature_importances_
    }).nlargest(10, 'importance')

    axes[1, 1].barh(feature_importance['feature'].astype(str),
                    feature_importance['importance'])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 10 Feature Importances')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================

def run_pipeline(application_path, credit_path, test_size=0.2, random_state=42):
    """
    Execute the complete ML pipeline.

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
    dict : Dictionary containing model, data, and results
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

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Handle Imbalance
    X_train_balanced, y_train_balanced = handle_imbalance(X_train_scaled, y_train)

    # Step 4: Train and Evaluate
    results = train_and_evaluate(X_train_balanced, y_train_balanced, X_test_scaled, y_test)

    return {
        'model': results['model'],
        'scaler': scaler,
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'results': results,
        'feature_names': X.columns.tolist()
    }


if __name__ == '__main__':
    # Set paths
    base_path = r'C:\Users\Gabriel\Documents\VSCode\credit_card_approval'
    application_path = fr'{base_path}\data\raw\application_record.csv'
    credit_path = fr'{base_path}\data\raw\credit_record.csv'

    # Run pipeline
    pipeline_output = run_pipeline(application_path, credit_path)

    # Generate visualizations
    print("\nGenerating evaluation plots...")
    fig = plot_evaluation_metrics(
        pipeline_output['y_test'],
        pipeline_output['results']['y_pred_proba'],
        pipeline_output['results']
    )
    plt.savefig(fr'{base_path}\reports\model_evaluation.png', dpi=300, bbox_inches='tight')
    print("Plots saved to reports/model_evaluation.png")
