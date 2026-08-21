# Loan Approval Prediction & Risk Analysis System

A complete, end-to-end machine-learning project that predicts whether a
loan application should be **Approved** or **Rejected**, and separately
estimates the applicant's **lending risk** (Low / Medium / High, 0-100
score) with an explainable-AI breakdown of the top contributing factors.

Built as a B.Tech AI/ML/Data Science project — beginner-friendly comments
throughout the code explain every step.

---

## 1. Project Architecture

```
loan-risk-prediction/
│
├── data/
│   ├── generate_dataset.py     # builds the synthetic dataset (see §2)
│   └── loan_data.csv           # generated dataset (created by running the script)
│
├── notebooks/                  # optional: put exploratory .ipynb work here
│
├── models/                     # created by train_model.py
│   ├── best_model_pipeline.joblib   # full sklearn Pipeline (preprocessing + classifier)
│   ├── model_comparison.csv         # metrics table for all candidate models
│   └── best_model_info.json         # which model won and why
│
├── src/
│   ├── data_preprocessing.py   # loading, cleaning, missing values, outliers, pipeline
│   ├── eda.py                  # exploratory data analysis + saved plots
│   ├── train_model.py          # trains & compares 6 models, saves the best
│   ├── predict.py              # single entry point: applicant -> full prediction report
│   ├── risk_analysis.py        # 0-100 risk score + Low/Medium/High logic
│   └── explainability.py       # SHAP / feature-importance explanations
│
├── app/
│   └── app.py                  # Streamlit frontend (7 pages, see §6)
│
├── database/
│   └── predictions.db          # SQLite DB of prediction history (created at runtime)
│
├── reports/                    # EDA plots saved here by eda.py
│
├── requirements.txt
└── README.md
```

**Data flow:**

```
raw CSV -> clean_data() -> ColumnTransformer (impute+encode+scale)
        -> train/test split -> train 6 models -> pick best by ROC-AUC/F1
        -> save full Pipeline (best_model_pipeline.joblib)
        -> predict.py loads it for new applicants
        -> risk_analysis.py blends model probability + rule-based factors
        -> explainability.py explains the prediction
        -> Streamlit app displays everything + logs to SQLite
```

---

## 2. Dataset Requirements & Recommended Public Datasets

This sandbox has no internet access, so `data/generate_dataset.py`
**generates a realistic synthetic dataset** with the same columns,
realistic relationships (credit score, DTI, repayment history genuinely
drive the label), and injected messiness (missing values, duplicates,
outliers) so every cleaning step in the pipeline has real work to do.

**For your actual submission, you should swap in a real dataset.**
Recommended public datasets (search on Kaggle):
- **"Loan Prediction Problem Dataset"** (Analytics Vidhya / Kaggle) — classic
  approve/reject dataset with income, credit history, education, etc.
- **"Loan-Approval-Prediction-Dataset"** (Kaggle) — includes CIBIL/credit
  score, asset values, and loan status.
- **LendingClub Loan Data** (Kaggle) — large, real-world peer-to-peer
  lending data with actual repayment/default outcomes; great for the risk
  component, but heavier to clean.

### Using a real dataset
1. Download a CSV and place it at `data/loan_data.csv`.
2. Rename its columns to match `NUMERIC_FEATURES` / `CATEGORICAL_FEATURES`
   in `src/data_preprocessing.py` (or edit those lists to match your columns).
3. Make sure your target column is called `loan_status` with values
   `"Approved"` / `"Rejected"` (or edit `TARGET_COLUMN` / the mapping in
   `get_feature_target()`).
4. Re-run `python src/train_model.py`. Nothing else needs to change.

---

## 3. Data Preprocessing (`src/data_preprocessing.py`)

- **Loading**: `pd.read_csv`.
- **Duplicate removal**: `drop_duplicates()`.
- **Dropped columns**: applicant ID (not predictive) and `gender` (a
  protected attribute — see §9 Ethics).
- **Outlier handling**: IQR-based capping (winsorizing) on income and
  loan amount — extreme values are **capped, not deleted**, so we don't
  systematically remove legitimate high-income/high-loan applicants.
- **Missing values**: handled *inside* the pipeline (median for numeric,
  most-frequent for categorical) so the exact same imputation applies to
  brand-new applicants at prediction time — not just to the training data.
- **Encoding**: One-Hot Encoding for categoricals (`handle_unknown="ignore"`
  so an unseen category at prediction time doesn't crash the app).
- **Scaling**: StandardScaler for numeric features (needed for Logistic
  Regression/SVM; harmless for tree models).
- **No data leakage**: all of the above (imputer means, one-hot
  categories, scaler statistics) are wrapped in a scikit-learn
  `ColumnTransformer` inside a `Pipeline`, which is fit **only on the
  training split**, never on the full dataset before splitting.

---

## 4. Exploratory Data Analysis (`src/eda.py`)

Run: `python src/eda.py` (run from the project root). Produces, in `reports/`:
1. Target class balance
2. Numeric feature distributions
3. Correlation heatmap (numeric features + target)
4. Boxplots/countplots of key relationships (credit score, DTI, income,
   employment status, repayment history, education vs. loan status)

It also prints/saves each numeric feature's correlation with approval —
useful for a viva slide.

---

## 5. Model Training & Comparison (`src/train_model.py`)

Trains **Logistic Regression, Decision Tree, Random Forest, Gradient
Boosting, SVM (RBF)**, and **XGBoost** (auto-skipped with a warning if
`xgboost` isn't installed in your environment), each inside a full
`Pipeline` (preprocessing + classifier).

- **Class imbalance**: handled with `class_weight="balanced"` (and
  `scale_pos_weight` for XGBoost) rather than SMOTE. This project uses
  class weighting because it requires no synthetic sample generation
  (simpler to explain in a viva, zero leakage risk) — swapping in SMOTE
  is straightforward if your instructor requires it: wrap
  `imblearn.over_sampling.SMOTE` into an `imblearn.pipeline.Pipeline`
  applied only to the training fold.
- **Evaluation**: Accuracy, Precision, Recall, F1-score, ROC-AUC, and a
  Confusion Matrix are computed for every model.
- **Model selection**: sorted by **ROC-AUC first, F1-score as tiebreaker**
  — explicitly *not* by raw accuracy, because a lending dataset can be
  imbalanced enough that "always reject" looks falsely accurate. See the
  Model Performance page in the app for a plain-language explanation of
  this choice.
- **Output**: the winning `Pipeline` is saved whole
  (`models/best_model_pipeline.joblib`), so preprocessing and the
  classifier are always applied together, consistently.

Run: `python src/train_model.py`

Typical result on the synthetic dataset (yours will vary slightly by
random seed and, especially, once you swap in a real dataset):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.82 | ~0.80 | ~0.81 | ~0.81 | ~0.90 |
| SVM (RBF) | ~0.81 | ~0.79 | ~0.82 | ~0.80 | ~0.89 |
| Gradient Boosting | ~0.81 | ~0.80 | ~0.80 | ~0.80 | ~0.89 |
| Random Forest | ~0.80 | ~0.78 | ~0.80 | ~0.79 | ~0.89 |
| Decision Tree | ~0.75 | ~0.74 | ~0.74 | ~0.74 | ~0.83 |

(XGBoost typically lands close to Gradient Boosting/Random Forest —
install it locally to see it in your own comparison table.)

---

## 6. Risk Scoring Methodology (`src/risk_analysis.py`)

The **Risk Score (0-100)** is explicitly **not** a mysterious black-box
number. It's a documented blend:

```
risk_score = round(100 × blended_default_probability)

blended_default_probability = 0.65 × P(default | ML model)
                             + 0.35 × rule_based_risk_index
```

- **P(default | ML model)** = `1 - P(Approved)` from the trained
  classifier's `predict_proba`. This is a **model-estimated probability**
  calibrated only as well as the training data and model allow — not a
  certified real-world default probability.
- **rule_based_risk_index** is a transparent, auditable weighted sum
  (0-1, higher = riskier) of six normalized underwriting factors:

  | Factor | Weight |
  |---|---|
  | Credit score | 30% |
  | Debt-to-income ratio | 22% |
  | Loan-to-income ratio | 18% |
  | Previous repayment history | 15% |
  | Existing loan count | 8% |
  | Employment stability (years employed / unemployed flag) | 7% |

  Blending in a transparent rule-based component alongside the ML score
  is a common, auditable pattern in real credit-risk systems — it keeps
  the number interpretable and defensible even when the underlying model
  is complex, and it's easy to explain and justify in a viva.

**Risk bands** (tune these against real outcomes if you use real data):
`0-33 = Low`, `34-66 = Medium`, `67-100 = High`.

---

## 7. Explainability (`src/explainability.py`)

- **Global**: native `feature_importances_` (tree models) or `|coef_|`
  (Logistic Regression/SVM with linear kernel) show which features matter
  most across the whole model.
- **Local (per-applicant)**: uses **SHAP** (`TreeExplainer` for tree
  models, `LinearExplainer` for linear models) when the `shap` package is
  installed. If it isn't available in your environment, the code
  automatically falls back to a transparent **perturbation method**: for
  each feature, it replaces the applicant's value with the population
  median/mode and measures how much the predicted approval probability
  moves — a simple, honest local explanation that never leaves the app
  broken.

---

## 8. Prediction Output Format

For every applicant (`src/predict.py` → `predict_applicant()`), the
system returns:

```json
{
  "loan_decision": "Approved",
  "approval_probability": 0.91,
  "default_risk_probability": 0.09,
  "risk_level": "Low",
  "risk_score": 15,
  "top_factors": [
    {"feature": "credit_score", "direction": "toward Approval", "abs_impact": 0.14},
    {"feature": "debt_to_income_ratio", "direction": "toward Approval", "abs_impact": 0.09},
    ...
  ]
}
```

---

## 9. Ethical Considerations, Fairness & Limitations

- **This is not a real credit-underwriting system.** It's an educational
  demonstration. No output from this project should be used to make an
  actual lending decision.
- **The model does not guarantee repayment.** All probabilities and risk
  scores are statistical estimates from historical/synthetic patterns —
  they describe correlation, not certainty, and not causation.
- **Protected attributes excluded.** `gender` is present in the raw
  dataset (for demographic realism) but is **deliberately dropped before
  modeling**. Real-world lending laws (e.g. the U.S. Equal Credit
  Opportunity Act, and equivalents elsewhere) generally prohibit using
  race, gender, religion, caste, marital status (in some jurisdictions),
  or similar protected attributes to make credit decisions. This project
  goes further and does not use them as model inputs at all.
- **Historical bias risk.** If trained on real historical lending data,
  a model can learn and perpetuate **existing societal biases** present
  in that history (e.g. if certain neighborhoods/groups were historically
  under-approved for reasons unrelated to creditworthiness). Always audit
  a real deployment for disparate impact across demographic groups before
  trusting it.
- **Explainability is descriptive, not causal.** SHAP/feature-importance
  values show what the model relied on, not proof that a factor actually
  *causes* default.
- **Decision support, not replacement.** The intended use is to *assist*
  a human loan officer with a data-backed second opinion, with a final
  human decision-maker in the loop — not full automation.

---

## 10. Instructions: Running the Project Locally

```bash
# 1. Clone/copy the project, then from the project root:
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# 2. Generate the dataset (skip this step if you've supplied your own
#    data/loan_data.csv following the schema in §2)
python data/generate_dataset.py

# 3. Explore the data (optional, saves plots to reports/)
python src/eda.py

# 4. Train and compare all models (saves the best pipeline to models/)
python src/train_model.py

# 5. Try a single prediction from the command line
python src/predict.py

# 6. Launch the web app
streamlit run app/app.py
```

The app will open at `http://localhost:8501`. Use the sidebar to move
between the Home, Application Form, Prediction, Risk Dashboard, History,
Model Performance, and "Why This Decision?" pages.

---

## 11. Sample Input & Output

**Sample applicant (also in `src/predict.py`'s `__main__` block):**

```python
{
  "age": 34, "married": "Yes", "dependents": "1", "education": "Graduate",
  "self_employed": "No", "employment_status": "Salaried", "years_employed": 6.5,
  "applicant_income": 5200, "coapplicant_income": 1800, "loan_amount": 180,
  "loan_term": 180, "credit_score": 712, "existing_loans": 1,
  "debt_to_income_ratio": 0.28, "previous_repayment_history": 1,
  "property_area": "Urban", "property_asset_value": 45000,
  "bank_account_years": 8.0, "avg_bank_balance": 12000
}
```

**Sample output (`python src/predict.py`):**

```
LOAN APPLICATION PREDICTION REPORT
Loan Decision           : Rejected
Approval Probability    : 9.9%
Default/Risk Probability: 90.1%
Risk Level              : High
Risk Score              : 73/100

Top factors influencing this decision:
  - loan_amount                    toward Rejection   (impact 0.771)
  - credit_score                   toward Approval    (impact 0.068)
  - years_employed                 toward Approval    (impact 0.031)
  - property_asset_value           toward Approval    (impact 0.025)
  - debt_to_income_ratio           toward Rejection   (impact 0.020)
  - avg_bank_balance               toward Approval    (impact 0.016)
```

(Numbers above are from the bundled synthetic dataset with a fixed
random seed — re-running after regenerating data, retraining, or with
your own dataset will naturally shift these values. It illustrates the
report format, not a fixed "correct" answer.)

---

## 12. Suggestions to Strengthen This as a Final-Year/Portfolio Project

1. **Swap in a real dataset** (Kaggle Loan Prediction or LendingClub) and
   re-tune the risk-score weights/thresholds against real outcomes.
2. **Hyperparameter tuning**: add `GridSearchCV`/`RandomizedSearchCV` or
   `Optuna` around the winning model for a stronger, defensible result.
3. **Calibration curve**: plot `sklearn.calibration.calibration_curve` to
   show how well `predict_proba` outputs match real observed default
   rates — strengthens the "this is a calibrated probability" claim.
4. **Fairness audit**: compute approval-rate and false-negative-rate
   parity across demographic subgroups (using a held-out demographic
   column *not* fed to the model) and discuss any disparities found.
5. **Cross-validation** instead of (or alongside) a single train/test
   split, to report metric variance, not just a point estimate.
6. **Deploy it**: containerize with Docker and deploy the Streamlit app
   (Streamlit Community Cloud, Render, or a small VM) for a live demo link.
7. **A/B the risk-blend weight** (currently 65/35 model/rules) and show
   how the Low/Medium/High distribution shifts — good viva material.
8. **Add unit tests** (`pytest`) for `risk_analysis.py` and
   `data_preprocessing.py` to demonstrate software-engineering rigor.
9. **API layer**: wrap `predict.py` in a small FastAPI service so the
   model can be called from other apps, not just the Streamlit UI.
10. **Time-based validation**: if using LendingClub-style data with loan
    issue dates, split train/test by time (not randomly) to simulate a
    realistic "train on the past, predict the future" deployment.

---

## 13. Possible Viva Questions & Answers

**Q1. Why did you split the risk score into "model probability" and
"rule-based index" instead of just using the ML probability directly?**
A: A pure ML probability can be hard to audit/defend ("the model just
said so"). Blending in a transparent, weighted rule-based index (documented
factor-by-factor) makes the score interpretable and defensible, mirroring
how real credit-risk systems combine statistical models with policy rules.

**Q2. How did you avoid data leakage?**
A: All preprocessing (imputation statistics, one-hot categories, scaling
parameters) is wrapped in a scikit-learn `Pipeline`/`ColumnTransformer`
fit only on the training split, never on the full dataset before the
train/test split. The exact same fitted transformations are then applied,
unchanged, to the test set and to any new applicant at prediction time.

**Q3. Why select the best model by ROC-AUC/F1 instead of accuracy?**
A: Loan datasets are often imbalanced. A model predicting the majority
class for every applicant can post high accuracy while being useless for
the minority class. ROC-AUC measures ranking quality across all decision
thresholds, and F1 balances precision and recall — both are more
informative than accuracy alone on imbalanced classification problems.

**Q4. How do you handle class imbalance?**
A: `class_weight="balanced"` for the scikit-learn classifiers (and
`scale_pos_weight` for XGBoost), which up-weights the minority class's
loss contribution during training. This was chosen over SMOTE to avoid
generating synthetic feature vectors that could introduce artifacts,
and because it requires no extra preprocessing pipeline step, though
SMOTE (via `imblearn`) is a reasonable alternative and is discussed in
§5.

**Q5. What does the Risk Score actually mean — is 73/100 "73% chance of
default"?**
A: Not exactly. It's `round(100 × blended_default_probability)`, where
`blended_default_probability` is a weighted combination (65% model,
35% rule-based index) of two [0,1] risk estimates. It correlates with
default risk and is useful for ranking/triaging applicants, but it is
not a certified, precisely-calibrated real-world default probability
without further calibration validation against real outcome data.

**Q6. How does the explainability (SHAP) output differ from feature
importance?**
A: Global feature importance (`feature_importances_`/`coef_`) tells you
which features matter most **across the whole model/dataset**. SHAP
values are **local** — they explain how much each feature pushed **one
specific applicant's** prediction up or down relative to a baseline,
which is what powers the "Why This Decision?" page.

**Q7. Why did you exclude gender from the model?**
A: Gender is a protected attribute under fair-lending laws in most
jurisdictions (e.g. ECOA in the US). Using it — or features that proxy
for it — to make credit decisions can be illegal and is ethically
inappropriate. We keep it in the raw synthetic dataset for demographic
realism but explicitly drop it before feature selection/modeling.

**Q8. What are the limitations of this system?**
A: (1) It's trained on synthetic/limited historical data, so it can't
capture real-world credit dynamics precisely; (2) probabilities are only
as well-calibrated as the training process, not certified; (3) it can
inherit historical bias if trained on biased real data; (4) SHAP/feature
importance is correlational, not causal; (5) it's meant to support, not
replace, a human underwriter's judgment.

**Q9. Why use a Pipeline object instead of separate preprocessing and
model-training scripts?**
A: A single `Pipeline` guarantees the exact preprocessing used during
training is reproduced, in the same order, at prediction time — for both
the test set and any brand-new applicant — eliminating a whole class of
train/serve skew bugs and making the saved `.joblib` file
self-contained (one artifact = one thing to version and deploy).

**Q10. How would you extend this to production?**
A: Add authentication, input validation/schema enforcement, model
monitoring for data/concept drift, a proper database (PostgreSQL) instead
of SQLite, a fairness/bias monitoring dashboard, model versioning
(e.g. MLflow), a REST API layer, logging/observability, and a human
review workflow with audit trails before any credit decision is acted on.

---

## 14. Technology Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn ·
SHAP · Joblib · Streamlit · SQLite
