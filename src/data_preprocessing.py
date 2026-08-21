"""
data_preprocessing.py
----------------------
Beginner-friendly walkthrough of data cleaning + a leakage-safe
scikit-learn Pipeline for preprocessing.

Key ideas explained in comments:
- We NEVER fit scalers/encoders/imputers on the full dataset before
  splitting. Everything that "learns" from data (means, categories,
  scaling factors) is wrapped in a Pipeline that gets fit ONLY on the
  training set, and applied unchanged to the test set. This avoids
  data leakage.
- Missing values and outliers are handled inside the pipeline (imputers)
  so the exact same logic applies at prediction time on brand-new
  applicants.
"""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DATA_PATH = "data/loan_data.csv"

# Columns we deliberately EXCLUDE from modeling.
# `gender` is a protected attribute; using it (or proxies for it) to
# decide creditworthiness is an ethical/legal red flag in most
# jurisdictions (e.g. ECOA in the US), so we drop it before modeling.
DROP_COLUMNS = ["applicant_id", "gender"]

TARGET_COLUMN = "loan_status"

NUMERIC_FEATURES = [
    "age", "years_employed", "applicant_income", "coapplicant_income",
    "loan_amount", "loan_term", "credit_score", "existing_loans",
    "debt_to_income_ratio", "property_asset_value", "bank_account_years",
    "avg_bank_balance",
]

CATEGORICAL_FEATURES = [
    "married", "dependents", "education", "self_employed",
    "employment_status", "previous_repayment_history", "property_area",
]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Step 1: Load the dataset."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2/3/4: Data cleaning.
      - Drop exact duplicate rows.
      - Drop columns we don't want as features (IDs, protected attributes).
      - Cap obvious outliers using the IQR rule (winsorizing) on a couple
        of especially skew-prone numeric columns. We CAP rather than
        DELETE outliers so we don't lose real (if extreme) applicants —
        deleting rows can bias a lending model against legitimately
        high-income/high-loan applicants.
    """
    before = len(df)
    df = df.drop_duplicates().copy()
    removed = before - len(df)
    print(f"Removed {removed} duplicate rows.")

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")

    for col in ["applicant_income", "loan_amount"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"Capped {n_outliers} outliers in '{col}' to [{lower:.1f}, {upper:.1f}]")

    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Step 7/8: Encode categoricals + scale numerics, all inside a
    ColumnTransformer so it can be fit on the TRAIN split only and then
    reused identically on the test split and on future new applicants.

    - Numeric features: median imputation (robust to outliers) + standard
      scaling (needed for Logistic Regression / SVM; tree models ignore
      scaling but it doesn't hurt them).
    - Categorical features: most-frequent imputation + one-hot encoding.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_feature_target(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = (df[TARGET_COLUMN] == "Approved").astype(int)  # 1 = Approved, 0 = Rejected
    return X, y


if __name__ == "__main__":
    df = load_raw_data()
    print("Raw shape:", df.shape)
    print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])
    df_clean = clean_data(df)
    print("Clean shape:", df_clean.shape)
    X, y = get_feature_target(df_clean)
    print("Feature matrix shape:", X.shape, "| Target distribution:\n", y.value_counts())
