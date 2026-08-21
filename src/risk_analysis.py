"""
risk_analysis.py
-----------------
Computes a 0-100 lending Risk Score and a Low / Medium / High Risk Level
for an applicant. This is a SEPARATE layer from the approve/reject
classifier, combining:

  (a) the ML model's own calibrated default-probability estimate, and
  (b) a transparent, rule-based adjustment using classic underwriting
      factors (credit score, DTI, loan-to-income, repayment history,
      employment stability).

HOW THE SCORE IS CALCULATED (be explicit about this — do not present it
as an unexplained "black box" number):

  risk_score = round(100 * blended_probability)

  blended_probability = 0.65 * P(default | model)
                       + 0.35 * rule_based_risk_index

  - P(default | model) = 1 - P(Approved) from the trained classifier's
    predict_proba output. This is a MODEL-ESTIMATED probability, not a
    guarantee, and its calibration quality depends on the training data.
  - rule_based_risk_index is a transparent weighted sum (0-1) of
    normalized underwriting factors, explained factor-by-factor in
    `compute_rule_based_risk_index()` below. Blending in a transparent
    rule-based component (rather than trusting the ML score alone) is a
    common, auditable practice in real credit-risk systems, and it keeps
    the score interpretable even if the underlying ML model is complex.

  A HIGHER score = HIGHER risk of default (0 = safest, 100 = riskiest).

RISK BANDS (thresholds are a reasonable, commonly-used starting point —
documented here so they can be tuned/justified, not left implicit):
  0-33   -> Low Risk
  34-66  -> Medium Risk
  67-100 -> High Risk
"""
import numpy as np
import pandas as pd

LOW_RISK_MAX = 33
MEDIUM_RISK_MAX = 66


def _normalize(value, lo, hi):
    """Clip and scale a value into [0, 1], where 1 = riskier."""
    return float(np.clip((value - lo) / (hi - lo), 0, 1))


def compute_rule_based_risk_index(applicant: dict) -> dict:
    """
    Transparent, weighted rule-based risk index in [0, 1].
    Each factor is normalized to [0,1] (1 = riskiest) then weighted.
    Weights are documented here so they can be defended in a viva or
    tuned against real underwriting guidelines.
    """
    credit_score = applicant.get("credit_score", 650)
    dti = applicant.get("debt_to_income_ratio", 0.3)
    loan_amount = applicant.get("loan_amount", 0)  # in thousands
    total_income = (applicant.get("applicant_income", 0) + applicant.get("coapplicant_income", 0)) / 1000
    repayment_history = applicant.get("previous_repayment_history", 1)  # 1 good, 0 poor
    existing_loans = applicant.get("existing_loans", 0)
    years_employed = applicant.get("years_employed", 0)
    employment_status = applicant.get("employment_status", "Salaried")

    # 1. Credit score risk (lower score = higher risk). Typical range 300-900.
    credit_risk = 1 - _normalize(credit_score, 300, 900)

    # 2. Debt-to-income ratio risk (higher DTI = higher risk). >0.5 is very risky.
    dti_risk = _normalize(dti, 0.0, 0.6)

    # 3. Loan amount relative to income (loan-to-income ratio)
    loan_to_income = loan_amount / max(total_income, 0.1)
    lti_risk = _normalize(loan_to_income, 0.5, 6.0)

    # 4. Previous repayment history (binary: poor history = max risk contribution)
    history_risk = 1.0 if repayment_history == 0 else 0.0

    # 5. Existing debt burden (more existing loans = more risk)
    existing_loans_risk = _normalize(existing_loans, 0, 4)

    # 6. Employment stability (years employed; unemployed applicants flagged separately)
    if employment_status == "Unemployed":
        employment_risk = 1.0
    else:
        employment_risk = 1 - _normalize(years_employed, 0, 10)

    weights = {
        "credit_score": 0.30,
        "debt_to_income": 0.22,
        "loan_to_income": 0.18,
        "repayment_history": 0.15,
        "existing_loans": 0.08,
        "employment_stability": 0.07,
    }
    components = {
        "credit_score": credit_risk,
        "debt_to_income": dti_risk,
        "loan_to_income": lti_risk,
        "repayment_history": history_risk,
        "existing_loans": existing_loans_risk,
        "employment_stability": employment_risk,
    }
    rule_based_index = sum(weights[k] * components[k] for k in weights)

    return {
        "rule_based_index": rule_based_index,
        "components": components,
        "weights": weights,
    }


def compute_risk_score(model_default_probability: float, applicant: dict, model_weight: float = 0.65) -> dict:
    """
    Combines the model's default probability with the rule-based index
    into a single, explainable 0-100 risk score + Low/Medium/High label.
    """
    rule_result = compute_rule_based_risk_index(applicant)
    rule_based_index = rule_result["rule_based_index"]

    blended_probability = (
        model_weight * model_default_probability
        + (1 - model_weight) * rule_based_index
    )
    blended_probability = float(np.clip(blended_probability, 0, 1))
    risk_score = round(blended_probability * 100)

    if risk_score <= LOW_RISK_MAX:
        risk_level = "Low"
    elif risk_score <= MEDIUM_RISK_MAX:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "model_default_probability": round(model_default_probability, 4),
        "rule_based_index": round(rule_based_index, 4),
        "blended_default_probability": round(blended_probability, 4),
        "component_breakdown": {
            k: round(v, 3) for k, v in rule_result["components"].items()
        },
        "component_weights": rule_result["weights"],
    }


if __name__ == "__main__":
    sample_applicant = {
        "credit_score": 580,
        "debt_to_income_ratio": 0.48,
        "loan_amount": 250,
        "applicant_income": 4000,
        "coapplicant_income": 0,
        "previous_repayment_history": 0,
        "existing_loans": 2,
        "years_employed": 1.2,
        "employment_status": "Self-Employed",
    }
    result = compute_risk_score(model_default_probability=0.55, applicant=sample_applicant)
    import json
    print(json.dumps(result, indent=2))
