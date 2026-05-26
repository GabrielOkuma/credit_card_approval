# Credit Scoring ML Pipeline V2 - Quick Reference

## 🚀 What Changed

| Feature | V1 | V2 |
|---------|----|----|
| Algorithm | RandomForest | **LightGBM** |
| Imbalance Handling | SMOTE | **Native is_unbalance=True** |
| Validation | Single split | **5-Fold Stratified CV** |
| ROC-AUC | 0.6481 | **0.7355** (+13.5%) |
| Recall | 21.14% | **48.78%** (+130%) |

## 📊 Performance Summary

### Cross-Validation Results (5 Folds)
```
ROC-AUC:  0.7577 ± 0.0220  (very stable)
Recall:   0.5029 ± 0.0616  (catches ~50% of bad clients)
Precision: 0.1220 ± 0.0079 (precision fairly consistent)
F1-Score: 0.1961 ± 0.0144
```

### Test Set Results
```
ROC-AUC:       0.7355
Recall (0.5):  48.78%  (60 out of 123 bad clients)
Specificity:   92.30%  (6,617 out of 7,169 good clients approved)
Precision:     9.80%
```

## 🎯 Threshold Decision Guide

Choose threshold based on your business priorities:

| Threshold | Recall | Specificity | Use Case |
|-----------|--------|-------------|----------|
| **0.3** | 59.3% | 78.1% | Risk-averse (catch more defaults) |
| **0.4** | 51.2% | 87.7% | Balanced (good middle ground) |
| **0.5** | 48.8% | 92.3% | Volume-focused (minimize rejections) |

**Formula:** `Optimal = Cost_FP / (Cost_FP + Cost_FN)`

Example: If bad loan costs $5K and rejecting good costs $1K:
- Optimal threshold = 1/(1+5) = 0.167 (very aggressive)

## 📁 Files & Outputs

```
credit_card_approval/
├── pipeline_v2.py                    # Main code (700+ lines)
│   Functions:
│   - stratified_cross_validation()   # 5-fold CV with metrics
│   - train_final_model()             # Final LightGBM training
│   - analyze_thresholds()            # 3-threshold analysis
│   - plot_cv_metrics()               # CV visualization
│   - plot_feature_importance()       # Feature ranking
│   - plot_roc_and_pr_curves()        # Discrimination curves
│
├── reports/
│   ├── cv_metrics_across_folds.png   # Fold stability
│   ├── roc_pr_curves.png             # ROC & PR curves
│   ├── feature_importance.png        # Top 15 features
│   └── model_evaluation.png          # (old v1 results)
│
└── docs/
    ├── UPGRADE_SUMMARY_V2.md         # Full technical details
    ├── PIPELINE_DOCUMENTATION.md     # V1 documentation
    └── QUICKSTART.md                 # Getting started
```

## ⚡ Running the Pipeline

```bash
# Execute V2 pipeline
cd c:\Users\Gabriel\Documents\VSCode\credit_card_approval
python credit_card_approval/pipeline_v2.py

# Output: 
# - Console: Full metrics across 5 folds + threshold analysis
# - PNG files: cv_metrics_across_folds.png, roc_pr_curves.png, feature_importance.png
# - Runtime: ~3 minutes (includes 5-fold CV)
```

## 🔑 Key Findings

### Top Risk Factors (Feature Importance)
1. **Age** - Younger applicants are riskier
2. **Experience Years** - More experience = lower risk
3. **Income** - Higher income = lower risk
4. **Family Size** - Affects repayment capacity
5. **Marital Status** - Married = lower risk

### Model Stability
- **ROC-AUC variation** across folds: ±2.2% (very stable)
- **Recall variation** across folds: ±6.2% (moderate variation)
- **Conclusion:** Model reliably identifies discrimination patterns

### Imbalance Handling
- LightGBM's `is_unbalance=True` automatically adjusts weights
- No SMOTE needed (simpler, faster, cleaner)
- Avoids data leakage issues from synthetic oversampling

## 📈 Metrics Explanation

**ROC-AUC (Area Under Curve):** Measures discrimination ability
- 0.5 = random guessing
- 0.7 = good
- 0.8+ = excellent
- V2 achieves 0.7355 (good improvement)

**Recall (Sensitivity):** % of actual bad clients caught
- Higher = catch more defaults (but more false alarms)
- V2: 48.8% at threshold 0.5 (2x improvement over V1)

**Specificity:** % of good clients correctly approved
- Higher = fewer rejected good applicants
- V2: 92.3% at threshold 0.5 (maintain volume)

**Precision:** % of "bad" predictions that are actually bad
- Lower due to class imbalance (expected)
- V2: 9.8% at threshold 0.5 (still improved)

**F1-Score:** Harmonic mean of recall and precision
- Balances both metrics
- V2: 0.1633 (2.2x improvement over V1)

## 🔧 Code Highlights

### Stratified K-Fold CV Implementation
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    # Each fold maintains class distribution
    # Train LightGBM, evaluate, track metrics
    # Results: ROC-AUC, Recall, Precision, F1
```

### LightGBM with Native Imbalance Handling
```python
params = {
    'is_unbalance': True,  # Handle imbalance automatically
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
}

model = lgb.train(params, train_data, num_boost_round=200)
```

### Threshold Analysis
```python
# Evaluate at multiple thresholds
for threshold in [0.3, 0.4, 0.5]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    # Calculate: recall, specificity, precision, F1
    # Show confusion matrix
    # Provide business interpretation
```

## 💾 Production Deployment

```python
import pickle
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# Load trained artifacts
model = pickle.load(open('models/lightgbm_v2.pkl', 'rb'))
scaler = pickle.load(open('models/scaler_v2.pkl', 'rb'))

# Score new applicant
X_new = scaler.transform(applicant_data)
default_prob = model.predict(X_new)

# Apply business threshold (e.g., 0.4)
decision = "REJECT" if default_prob >= 0.4 else "APPROVE"
```

## 🎓 Learning Resources

- **LightGBM Docs:** https://lightgbm.readthedocs.io/
- **Cross-Validation:** scikit-learn StratifiedKFold
- **ROC/AUC:** scikit-learn ROC curve explanation
- **Feature Importance:** LightGBM gain-based importance

## ✅ Validation Checklist

- [x] LightGBM installed and working
- [x] 5-fold CV implemented and stable
- [x] Metrics tracked across all folds
- [x] Threshold analysis at 0.3, 0.4, 0.5
- [x] Feature importance extracted
- [x] Visualizations generated (4 PNG files)
- [x] Code production-ready with comments
- [x] Performance improved by 13.5% (ROC-AUC)
- [x] Recall improved by 130% (bad client detection)

## 🚨 Common Issues & Fixes

**Q: Module not found error for lightgbm?**
```bash
pip install lightgbm
```

**Q: Pipeline takes too long?**
- Current: ~3 minutes for 5-fold CV
- Faster: Reduce n_splits from 5 to 3, or skip CV for testing

**Q: How to use different threshold?**
- Modify `thresholds=[0.3, 0.4, 0.5]` parameter
- Or add custom threshold analysis code

**Q: How to save the model for production?**
```python
model.save_model('models/lightgbm_v2.pkl')
scaler_dict = {'means': scaler.mean_, 'scales': scaler.scale_}
```

## 📞 Next Steps

1. **Review threshold analysis** in output
2. **Calculate optimal threshold** for your business
3. **Pilot test** with threshold 0.4 (balanced)
4. **Monitor performance** monthly
5. **Retrain** as new data arrives
6. **Iterate** with feature engineering

---

**Status:** ✅ Complete and Tested  
**Version:** 2.0  
**Date:** May 25, 2026
