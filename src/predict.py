"""
predict.py
----------
Single entry point that takes ONE new applicant's raw data and returns
the full prediction package:
  - Loan Decision (Approved / Rejected)
  - Default/Risk Probability
  - Risk Level (Low / Medium / High)
  - Risk Score (0-100)
  - Top factors influencing the decision (explainability)

This is the module the Streamlit app calls. It is also runnable
standalone for testing / the sample I/O shown in the README.
"""
import json
import joblib
import pandas as pd

from data_preprocessing import (
    load_raw_data, clean_data, get_feature_target,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from risk_analysis import compute_risk_score
from explainability import explain_single_prediction, global_feature_importance

MODEL_PATH = "models/best_model_pipeline.joblib"

DECISION_THRESHOLD = 0.5  # probability of approval >= threshold -> Approved


def _load_background_data():
    """Small reference sample used by the explainability fallback method
    (typical/median values for 'what if this feature were typical?')."""
    df = clean_data(load_raw_data())
    X, _ = get_feature_target(df)
    return X


def predict_applicant(applicant: dict, pipeline=None, background_df=None, explain: bool = True) -> dict:
    """
    Parameters
    ----------
    applicant : dict
        Raw applicant fields matching NUMERIC_FEATURES + CATEGORICAL_FEATURES.
    pipeline : fitted sklearn Pipeline, optional (loaded from disk if not given)
    background_df : reference DataFrame for explainability, optional
    explain : whether to compute the top-factors explanation (slightly slower)

    Returns
    -------
    dict with decision, probabilities, risk score/level, and top factors.
    """
    if pipeline is None:
        pipeline = joblib.load(MODEL_PATH)
    if background_df is None:
        background_df = _load_background_data()

    row = pd.DataFrame([applicant])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    proba_approved = float(pipeline.predict_proba(row)[0, 1])
    proba_default = 1 - proba_approved
    decision = "Approved" if proba_approved >= DECISION_THRESHOLD else "Rejected"

    risk = compute_risk_score(model_default_probability=proba_default, applicant=applicant)

    result = {
        "loan_decision": decision,
        "approval_probability": round(proba_approved, 4),
        "default_risk_probability": round(proba_default, 4),
        "risk_level": risk["risk_level"],
        "risk_score": risk["risk_score"],
        "risk_breakdown": risk,
    }

    if explain:
        top_factors = explain_single_prediction(pipeline, row, background_df, top_n=6)
        result["top_factors"] = top_factors.to_dict(orient="records")

    return result


def print_prediction_report(applicant: dict, result: dict):
    print("=" * 60)
    print("LOAN APPLICATION PREDICTION REPORT")
    print("=" * 60)
    print(f"Loan Decision           : {result['loan_decision']}")
    print(f"Approval Probability    : {result['approval_probability']*100:.1f}%")
    print(f"Default/Risk Probability: {result['default_risk_probability']*100:.1f}%")
    print(f"Risk Level              : {result['risk_level']}")
    print(f"Risk Score              : {result['risk_score']}/100")
    print("\nTop factors influencing this decision:")
    for f in result.get("top_factors", []):
        print(f"  - {f['feature']:<30s} {f['direction']:<18s} (impact {f['abs_impact']:.3f})")
    print("\nNOTE: This is a statistical estimate from a machine-learning model")
    print("trained on historical/synthetic data. It is decision SUPPORT, not a")
    print("guarantee of repayment or a final lending decision. See README for")
    print("model limitations and fairness considerations.")
    print("=" * 60)


if __name__ == "__main__":
    sample_applicant = {
        "age": 34,
        "married": "Yes",
        "dependents": "1",
        "education": "Graduate",
        "self_employed": "No",
        "employment_status": "Salaried",
        "years_employed": 6.5,
        "applicant_income": 5200,
        "coapplicant_income": 1800,
        "loan_amount": 180,
        "loan_term": 180,
        "credit_score": 712,
        "existing_loans": 1,
        "debt_to_income_ratio": 0.28,
        "previous_repayment_history": 1,
        "property_area": "Urban",
        "property_asset_value": 45000,
        "bank_account_years": 8.0,
        "avg_bank_balance": 12000,
    }

    result = predict_applicant(sample_applicant)
    print_prediction_report(sample_applicant, result)
