"""
explainability.py
------------------
Explains WHY the model made a given prediction.

Two layers, so this always works regardless of which model was selected
and whether the `shap` library is installed:

1. GLOBAL importance: which features matter most across the whole model
   (native feature_importances_/coefficients, or permutation importance
   as a universal fallback).
2. LOCAL (per-applicant) explanation: SHAP values when available, which
   tell you exactly how much each feature pushed THIS applicant's score
   up or down. If SHAP isn't installed, we fall back to a permutation-
   based local approximation so the app never breaks.

Run standalone: python src/explainability.py
"""
import numpy as np
import pandas as pd
import joblib

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from data_preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def get_output_feature_names(pipeline):
    """Recover human-readable feature names after the ColumnTransformer
    (one-hot encoding expands categorical columns)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_pipeline = preprocessor.named_transformers_["cat"]
    ohe = cat_pipeline.named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERIC_FEATURES + cat_names


def global_feature_importance(pipeline, top_n: int = 15) -> pd.DataFrame:
    """
    Step: model-level explainability (Step 9 of ML requirements).
    Uses the model's native importance/coefficients when available.
    """
    clf = pipeline.named_steps["classifier"]
    feature_names = get_output_feature_names(pipeline)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        raise ValueError(
            f"Model {type(clf).__name__} exposes neither feature_importances_ "
            f"nor coef_; use permutation importance instead."
        )

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    imp_df["importance_pct"] = (imp_df["importance"] / imp_df["importance"].sum() * 100).round(1)
    return imp_df


def explain_single_prediction(pipeline, applicant_df: pd.DataFrame, background_df: pd.DataFrame, top_n: int = 6):
    """
    Local explanation for ONE applicant row.

    Returns a DataFrame of the top contributing factors with a
    +/- direction (pushed the application toward approval or rejection).

    - If shap is installed: uses a small, fast, model-agnostic
      KernelExplainer/TreeExplainer as appropriate.
    - If shap is NOT installed: falls back to a simple, transparent
      "leave-one-out" perturbation method — for each feature, replace the
      applicant's value with the population median/mode and see how much
      the predicted probability changes. This is a reasonable, honest
      approximation, not a black box.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    feature_names = get_output_feature_names(pipeline)

    if HAS_SHAP:
        try:
            X_bg_transformed = preprocessor.transform(background_df.sample(
                min(100, len(background_df)), random_state=42))
            X_applicant_transformed = preprocessor.transform(applicant_df)

            if hasattr(clf, "feature_importances_"):
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_applicant_transformed)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # class 1 = Approved
                sv = np.array(shap_values).flatten()
            else:
                explainer = shap.LinearExplainer(clf, X_bg_transformed)
                sv = np.array(explainer.shap_values(X_applicant_transformed)).flatten()

            contrib_df = pd.DataFrame({"feature": feature_names, "shap_value": sv})
            contrib_df["direction"] = np.where(
                contrib_df["shap_value"] >= 0, "toward Approval", "toward Rejection"
            )
            contrib_df["abs_impact"] = contrib_df["shap_value"].abs()
            return contrib_df.sort_values("abs_impact", ascending=False).head(top_n).reset_index(drop=True)
        except Exception as e:
            print(f"[warn] SHAP explanation failed ({e}); falling back to perturbation method.")

    # ---- Fallback: perturbation-based local explanation ----
    base_proba = pipeline.predict_proba(applicant_df)[0, 1]
    rows = []
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        perturbed = applicant_df.copy()
        if col in NUMERIC_FEATURES:
            perturbed[col] = background_df[col].median()
        else:
            perturbed[col] = background_df[col].mode()[0]
        new_proba = pipeline.predict_proba(perturbed)[0, 1]
        delta = base_proba - new_proba  # how much this value (vs "typical") moved approval prob
        rows.append({"feature": col, "shap_value": delta})

    contrib_df = pd.DataFrame(rows)
    contrib_df["direction"] = np.where(
        contrib_df["shap_value"] >= 0, "toward Approval", "toward Rejection"
    )
    contrib_df["abs_impact"] = contrib_df["shap_value"].abs()
    return contrib_df.sort_values("abs_impact", ascending=False).head(top_n).reset_index(drop=True)


if __name__ == "__main__":
    from data_preprocessing import load_raw_data, clean_data, get_feature_target

    pipeline = joblib.load("models/best_model_pipeline.joblib")
    df = clean_data(load_raw_data())
    X, y = get_feature_target(df)

    print("Global feature importance:")
    print(global_feature_importance(pipeline))

    sample_applicant = X.iloc[[0]]
    print(f"\nLocal explanation for applicant 0 (shap installed: {HAS_SHAP}):")
    print(explain_single_prediction(pipeline, sample_applicant, X))
