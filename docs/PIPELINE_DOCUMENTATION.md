# Credit Scoring ML Pipeline - Complete Documentation

## Overview
This project implements a comprehensive machine learning pipeline for credit scoring, predicting whether applicants are "good" or "bad" clients. The pipeline addresses the fundamental challenge of class imbalance in credit data through vintage analysis, SMOTE, and robust evaluation metrics.

---

## Architecture & Components

### 1. **Step 1: Vintage Analysis - Label Construction**

**Objective:** Create target labels from credit history using vintage analysis principles.

**Logic:**
- Examined credit history for 45,985 clients with 1,048,575 transaction records
- Classified clients as "bad" (target=1) if they ever reached:
  - Status 2: 60-89 days past due
  - Status 3: 90-119 days past due  
  - Status 4: 120-149 days past due
  - Status 5: >150 days overdue/bad debt

**Results:**
- Bad clients: 667 (1.45%)
- Good clients: 45,318 (98.55%)
- **Key insight:** Severe class imbalance necessitates specialized handling

### 2. **Step 2: Data Merging & Preprocessing**

**Data Integration:**
- Merged application_record.csv (438,557 rows) with credit_record.csv
- Inner join on ID → 36,457 matched clients with credit history

**Feature Engineering:**
- `DAYS_BIRTH` → `AGE` (years): Mean 43.7 years
- `DAYS_EMPLOYED` → `EXPERIENCE_YEARS` (years): Mean 6.0 years, handling positive anomalies (unemployed)

**Categorical Encoding:**
- Binary columns (7): Gender, car ownership, realty, mobile/work/phone/email → 0/1
- Nominal columns (5): Income type, education, family status, housing type, occupation
- One-Hot encoding → 47 total features (48 with target)

**Missing Value Handling:**
- OCCUPATION_TYPE: 11,323 missing values → filled with "Unknown" category

**Final Dataset:**
- Shape: (36,457, 48)
- Target distribution: 616 bad (1.69%), 35,841 good (98.31%)

### 3. **Step 3: Handling Class Imbalance with SMOTE**

**Problem:** Training set imbalance (98.31% good vs 1.69% bad) causes:
- Model bias toward majority class
- Poor minority class detection
- Unreliable performance on rare events (defaults)

**Solution: SMOTE (Synthetic Minority Over-sampling Technique)**
- Generates synthetic samples for minority class using k-nearest neighbors
- Creates balanced training set: 28,672 good + 28,672 bad (50-50)
- Test set remains untouched (1.69% bad) for realistic evaluation

**Implementation:**
```python
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

### 4. **Step 4: Modeling & Evaluation**

**Model Selection: RandomForest**
- 100 trees with max_depth=15
- Class weights balanced to handle residual imbalance
- Robust to feature scaling and feature interactions

**Evaluation Metrics (Imbalanced Data Focus):**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| ROC-AUC | 0.6481 | Moderate discrimination ability between classes |
| F1-Score | 0.1113 | Low due to few true positives (high FN) |
| Sensitivity (Recall) | 0.2114 | Catches 21.14% of bad clients |
| Specificity | 0.9556 | Correctly identifies 95.56% of good clients |
| Precision | 0.0756 | Only 7.56% of "bad" predictions correct |

**Confusion Matrix:**
```
              Predicted
           Good    Bad
Actual  Good [6851]  [318]
        Bad  [97]    [26]
```
- TN: 6,851 (correct good)
- FP: 318 (good marked as bad)
- FN: 97 (bad marked as good) ← Risk!
- TP: 26 (correct bad)

---

## Key Findings

### Model Performance Interpretation

**Strengths:**
- High specificity (95.56%): Few good applicants wrongly rejected
- Reasonable ROC-AUC for imbalanced data
- Avoids over-confidence through stratified train-test split

**Challenges:**
- Low sensitivity (21.14%): Misses 79% of bad clients (97 out of 123)
- High false negatives create credit risk
- Low precision: Many false alarms on good clients

### Top Predictive Features
The model identifies key risk indicators:
1. **Income variables** - Higher income → lower risk
2. **Age** - Older applicants → lower default risk
3. **Employment status** - Experience matters
4. **Phone ownership flags** - Contact reliability indicators
5. **Family status** - Marital status correlates with stability

---

## Recommendations & Next Steps

### 1. **Threshold Optimization**
Current threshold: 0.5 (default)
- Calculate Youden's J-statistic to find optimal threshold
- Balance sensitivity vs specificity based on business cost matrix
- Example: If cost of bad loan = 5x cost of rejected good loan, optimize accordingly

### 2. **Alternative Algorithms**
- **XGBoost/LightGBM**: Often better discrimination on imbalanced data
- **Ensemble methods**: Combine multiple models for robustness
- **Logistic Regression**: Baseline for comparison

### 3. **Feature Engineering**
From credit history:
- Average delinquency status per client
- Maximum delinquency reached
- Trend in status over time
- Frequency of late payments

From demographics:
- Income-to-family-size ratio
- Debt service capability
- Regional economic indicators

### 4. **Model Robustness**
- Implement k-fold cross-validation
- Stability analysis across time periods (temporal validation)
- Fairness audit: Check bias across protected attributes

### 5. **Production Deployment**
- Monitor model performance degradation
- Implement retraining pipeline when performance drifts
- Track false negative rate closely (credit risk metric)
- Create explainability layer for rejected applications

---

## Files Generated

### Code Files
- **`credit_card_approval/pipeline.py`** (500+ lines)
  - Modular functions for each pipeline step
  - Extensive documentation and logging
  - Reusable for new data batches
  
### Notebooks
- **`notebooks/credit_scoring_pipeline.ipynb`**
  - Interactive step-by-step execution
  - Visualization of each stage
  - Analysis and interpretation
  - Threshold optimization exploration

### Reports
- **`reports/model_evaluation.png`**
  - ROC Curve: Shows discrimination ability
  - Precision-Recall Curve: Focuses on minority class
  - Confusion Matrix Heatmap: Visual classification breakdown
  - Feature Importance: Top 10 predictive features

---

## Technical Stack

**Libraries Used:**
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Model training and preprocessing
- `imbalanced-learn (imblearn)`: SMOTE implementation
- `matplotlib/seaborn`: Visualization

**Key Design Decisions:**
1. **Inner join** on credit data: Only analyze clients with credit history
2. **SMOTE over undersampling**: Preserve information, don't discard good clients
3. **Stratified train-test split**: Maintain class distribution in evaluation
4. **RandomForest over deep learning**: Interpretability, no hyperparameter tuning needed initially

---

## Running the Pipeline

### Quick Start (Python Script)
```bash
cd c:\Users\Gabriel\Documents\VSCode\credit_card_approval
python credit_card_approval/pipeline.py
```
Outputs: Evaluation metrics printed to console + PNG saved to `reports/`

### Interactive (Jupyter Notebook)
```bash
jupyter notebook notebooks/credit_scoring_pipeline.ipynb
```
- Run cells sequentially
- Customize parameters and experiment
- Visualizations in-line

---

## Class Imbalance: Why This Matters

In credit scoring, **class imbalance is structural, not a problem to ignore:**
- Bad clients ARE rare in healthy portfolios (1-5%)
- Traditional accuracy is misleading (accuracy = 98.5% by always predicting "good")
- Cost asymmetry: Missing one bad loan is worse than rejecting one good applicant
- SMOTE helps the model learn minority class patterns, but threshold tuning is critical

---

## Conclusion

This pipeline successfully demonstrates:
1. ✅ Rigorous label construction via vintage analysis
2. ✅ Comprehensive data preprocessing and feature engineering
3. ✅ Industry-standard imbalance handling (SMOTE)
4. ✅ Appropriate evaluation metrics for imbalanced classification
5. ✅ Interpretable model with clear risk factors

**Current model status:** Good foundation with room for improvement in sensitivity. Next phase should focus on algorithm experimentation and threshold optimization based on business requirements.
