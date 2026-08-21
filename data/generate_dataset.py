"""
generate_dataset.py
--------------------
Generates a realistic SYNTHETIC loan-application dataset so the whole
pipeline in this project can be run end-to-end without needing to download
anything.

WHY A SYNTHETIC DATASET?
This sandbox has no internet access, so a real dataset (e.g. Kaggle's
"Loan Prediction Problem Dataset" or "Loan-Approval-Prediction-Dataset",
or LendingClub's public loan data) cannot be downloaded here. The code
below builds a dataset with the SAME columns, realistic relationships,
and realistic noise, so you can:
  1. Run this whole project immediately, end-to-end, right now.
  2. Later swap in a real CSV (see README "Using a real dataset") with the
     same column names and every downstream script keeps working unchanged.

The label-generating logic below intentionally encodes real underwriting
relationships (higher DTI -> more risk, lower credit score -> more risk,
etc.) plus random noise, so the resulting ML problem is realistic and not
trivially separable.
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_loan_dataset(n_samples: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    education = rng.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.62, 0.38])
    self_employed = rng.choice(["Yes", "No"], size=n_samples, p=[0.18, 0.82])
    married = rng.choice(["Yes", "No"], size=n_samples, p=[0.65, 0.35])
    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.6, 0.4])  # kept for realism, NOT used in modeling
    dependents = rng.choice(["0", "1", "2", "3+"], size=n_samples, p=[0.55, 0.2, 0.15, 0.1])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=n_samples, p=[0.38, 0.38, 0.24])

    age = np.clip(rng.normal(38, 11, n_samples), 21, 70).round().astype(int)

    # Income correlated with education/employment
    base_income = rng.lognormal(mean=8.2, sigma=0.55, size=n_samples)
    edu_bonus = np.where(education == "Graduate", 1.25, 1.0)
    applicant_income = np.clip(base_income * edu_bonus, 1200, 60000).round(-1)

    has_coapplicant = rng.random(n_samples) < 0.55
    coapplicant_income = np.where(
        has_coapplicant,
        np.clip(rng.lognormal(mean=7.6, sigma=0.6, size=n_samples), 0, 30000).round(-1),
        0,
    )

    total_income = applicant_income + coapplicant_income

    credit_score = np.clip(rng.normal(650, 90, n_samples), 300, 900).round().astype(int)

    employment_status = rng.choice(
        ["Salaried", "Self-Employed", "Business Owner", "Unemployed"],
        size=n_samples, p=[0.55, 0.20, 0.15, 0.10],
    )

    years_employed = np.clip(rng.exponential(5, n_samples), 0, 35).round(1)
    years_employed = np.where(employment_status == "Unemployed", 0.0, years_employed)

    existing_loans = rng.poisson(0.7, n_samples).clip(0, 5)

    # Loan amount (in thousands) requested, loosely tied to income
    loan_amount = np.clip(
        (total_income / 1000) * rng.uniform(1.5, 6.0, n_samples), 10, 700
    ).round(1)

    loan_term_choices = np.array([12, 36, 60, 84, 120, 180, 240, 360])
    loan_term = rng.choice(loan_term_choices, size=n_samples)

    # Debt-to-income ratio: existing debt load relative to income
    debt_to_income = np.clip(
        rng.beta(2, 5, n_samples) * 0.9 + existing_loans * 0.02, 0.01, 0.85
    ).round(3)

    # Previous repayment history: 1 = good history, 0 = defaulted/missed payments before
    repayment_history_score = np.clip(
        rng.normal(0.75, 0.2, n_samples) - existing_loans * 0.03, 0, 1
    )
    previous_repayment_history = (repayment_history_score > 0.45).astype(int)

    property_asset_value = np.clip(rng.lognormal(mean=10.5, sigma=0.7, size=n_samples), 0, 3_000_000).round(-2)

    bank_account_years = np.clip(rng.exponential(6, n_samples), 0, 40).round(1)
    avg_bank_balance = np.clip(rng.lognormal(mean=8.5, sigma=1.0, size=n_samples), 0, 500000).round(-1)

    df = pd.DataFrame({
        "applicant_id": [f"APP{100000+i}" for i in range(n_samples)],
        "age": age,
        "gender": gender,
        "married": married,
        "dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "employment_status": employment_status,
        "years_employed": years_employed,
        "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income,
        "loan_amount": loan_amount,          # in thousands
        "loan_term": loan_term,               # in months
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "debt_to_income_ratio": debt_to_income,
        "previous_repayment_history": previous_repayment_history,  # 1=good, 0=poor
        "property_area": property_area,
        "property_asset_value": property_asset_value,
        "bank_account_years": bank_account_years,
        "avg_bank_balance": avg_bank_balance,
    })

    # ---------------------------------------------------------------
    # Build a latent "default risk" score from realistic underwriting
    # factors, then derive Loan_Status (approval) and add noise so the
    # problem isn't trivially separable (mirrors real-world messiness).
    # ---------------------------------------------------------------
    z = (
        -0.010 * (credit_score - 650)
        + 3.60 * debt_to_income
        + 1.55 * (loan_amount * 1000 / (total_income + 1) / 12).clip(0, 5)
        - 0.85 * previous_repayment_history
        + 0.28 * existing_loans
        - 0.06 * years_employed
        + np.where(employment_status == "Unemployed", 1.4, 0)
        + np.where(employment_status == "Self-Employed", 0.15, 0)
        - 0.00002 * property_asset_value.clip(0, 200000)
        - 0.000015 * avg_bank_balance
        + rng.normal(0, 0.9, n_samples)  # noise
    )
    default_prob = 1 / (1 + np.exp(-z))
    df["_default_probability_latent"] = default_prob  # kept only for inspection, not a feature

    loan_status = np.where(default_prob < 0.42, "Approved", "Rejected")
    df["loan_status"] = loan_status

    # ---- Inject realistic data-quality issues (so cleaning steps matter) ----
    n_missing = int(n_samples * 0.04)
    for col in ["credit_score", "self_employed", "loan_amount", "dependents", "property_asset_value"]:
        idx = rng.choice(n_samples, size=n_missing, replace=False)
        df.loc[idx, col] = np.nan

    # Duplicate a few rows
    dup_idx = rng.choice(n_samples, size=int(n_samples * 0.01), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # A few extreme outliers in income and loan_amount
    out_idx = rng.choice(df.index, size=8, replace=False)
    df.loc[out_idx[:4], "applicant_income"] = rng.uniform(300000, 900000, 4)
    df.loc[out_idx[4:], "loan_amount"] = rng.uniform(3000, 9000, 4)

    df = df.drop(columns=["_default_probability_latent"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_loan_dataset()
    out_path = "data/loan_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["loan_status"].value_counts(normalize=True))
