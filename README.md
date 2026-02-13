# Lead Conversion Modeling (Case Study)

End-to-end machine learning workflow to predict which marketing leads convert (`SOLD=1`) and prioritize outreach. The notebook covers data cleaning, feature engineering, modeling, evaluation, and decision-support outputs (lift and top-lead selection).

**Problem Solved**
Predict conversion for MAS leads so sales can focus on the highest-likelihood prospects, improving conversion efficiency and reducing wasted outreach.

**Impact At A Glance**
Best models achieved **~0.68 ROC AUC** and **~0.12 PR AUC** on a highly imbalanced dataset, with **~2.35x lift in the top 10%** of ranked leads, enabling targeted outreach with materially higher expected conversion rates.

## Data
Source file: `Jr DS Case Study -- data.csv.`

Key characteristics:
- ~70,395 rows, with ~4,250 positive conversions and ~66,145 non-conversions (~6% positive rate).
- Mix of demographic, marketing channel, device, timing, and verification features.
- Example key features used in modeling:
`AGE`, `URGENCY`, `SOURCE_GROUP`, `NETWORK`, `MEDIUM`, `DEVICE`, `SUBMITTED_PERIOD`, `KEYWORD_GROUP`, `MATCH_TYPE`, `BUYING_POWER_SCORE`, `NET_ASSET_VALUE`, `STATE`, `INPUT_PHONE_MATCHES`, `INPUT_EMAIL_MATCHES`, `ESTIMATED_INCOME`.

## Workflow (Complete)
1. **Environment & Reproducibility**
Set random seed, suppress noisy warnings, and ensure libraries (e.g., XGBoost) are installed for consistent runs.

2. **Data Cleaning & Feature Engineering**
Normalize column names, coerce numerics, cap extreme outliers (p99), and create robust missingness flags.
Engineer time-based features (hour, weekday/weekend, quarter), group income into bands/quantiles, and consolidate high-cardinality text fields (source/keyword) into stable categories.

3. **Exploratory Analysis**
Missingness profiling, correlation heatmaps, and conversion tables (with confidence intervals) to validate signal and leakage risks.

4. **Train/Test Split**
Stratified split (75/25) to preserve the ~6% conversion rate in both sets.

5. **Preprocessing Pipeline**
`ColumnTransformer` with imputation, scaling for numeric features, and one-hot encoding for categoricals.
All preprocessing is inside the pipeline to prevent leakage.

6. **Modeling**
Baselines and tuned models:
`DummyClassifier`, `LogisticRegression`, `RandomForestClassifier`, `XGBoost`.
Class imbalance handled via `scale_pos_weight`, `class_weight`, and SMOTE experiments.

7. **Evaluation**
ROC AUC, PR AUC, log-loss, accuracy, precision, recall, F1.
Threshold sweeps and **lift@10%** to align model choice with business targeting.

8. **Decision Support Outputs**
Top-lead selection utilities and lift tables to operationalize model ranking in outreach workflows.

## Results Summary (Test Set)
Top performing models:
- **XGBoost (tuned)**: ROC AUC **0.683**, PR AUC **0.118**, Recall **0.624**, Lift@10% **2.35**
- **Logistic (tuned)**: ROC AUC **0.681**, PR AUC **0.115**, Recall **0.649**, Lift@10% **2.38**

These results materially outperform the majority-class baseline (ROC AUC 0.50, PR AUC 0.06).

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, statsmodels, SHAP.

## Core Skills Demonstrated
Imbalanced classification, feature engineering, leakage-safe preprocessing, model tuning with cross-validation, metric selection aligned to business goals (PR AUC, lift), and production-ready ranking outputs.

## Repo Structure
`Untitled1.ipynb` - Full analysis and modeling workflow  
`Jr DS Case Study -- MAS data.csv` - Source dataset  
`Third party field definitions.xlsx` - Field metadata  
`Presentation.pptx` - Executive-level presentation  
`ConsumerAffairs_Logo.jpg`, `pic.avif` - Visual assets

## How To Run
1. Create a Python environment with:
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `seaborn`, `matplotlib`, `statsmodels`, `shap`, `imbalanced-learn`.
2. Open `Untitled1.ipynb` in Jupyter or VS Code and run all cells top-to-bottom.

## Notes
This project is intentionally designed to be business-facing: model choice and thresholds are evaluated based on **lift** and **precision-recall tradeoffs**, not just overall accuracy.

## Next Enhancements (Optional)
Calibrated probability outputs for better decision thresholds, richer feature attribution reporting, and automated scoring pipeline for new lead batches.
