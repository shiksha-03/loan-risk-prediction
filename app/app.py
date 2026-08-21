"""
app.py
------
Streamlit frontend for the Loan Approval Prediction & Risk Analysis System.

Pages:
  1. Home / Dashboard
  2. Loan Application Form
  3. Loan Approval Prediction
  4. Risk Analysis Dashboard
  5. Prediction History
  6. Model Performance
  7. Explanation / "Why This Decision?"

Run from the project root:  streamlit run app/app.py
"""
import os
import sys
import sqlite3
import json
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Make src/ importable when running "streamlit run app/app.py" from project root
SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from data_preprocessing import (  # noqa: E402
    load_raw_data, clean_data, get_feature_target,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from predict import predict_applicant  # noqa: E402
from explainability import global_feature_importance  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_pipeline.joblib")
COMPARISON_PATH = os.path.join(PROJECT_ROOT, "models", "model_comparison.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "database", "predictions.db")

st.set_page_config(page_title="Loan Approval & Risk Analysis", page_icon="💳", layout="wide")


# ----------------------------------------------------------------------
# Database helpers (Prediction History)
# ----------------------------------------------------------------------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            applicant_json TEXT,
            loan_decision TEXT,
            approval_probability REAL,
            default_risk_probability REAL,
            risk_level TEXT,
            risk_score INTEGER
        )
    """)
    conn.commit()
    return conn


def save_prediction(applicant: dict, result: dict):
    conn = init_db()
    conn.execute(
        """INSERT INTO predictions
           (timestamp, applicant_json, loan_decision, approval_probability,
            default_risk_probability, risk_level, risk_score)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            json.dumps(applicant),
            result["loan_decision"],
            result["approval_probability"],
            result["default_risk_probability"],
            result["risk_level"],
            result["risk_score"],
        ),
    )
    conn.commit()
    conn.close()


def load_history() -> pd.DataFrame:
    conn = init_db()
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df


# ----------------------------------------------------------------------
# Cached resource loaders
# ----------------------------------------------------------------------
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_background_data():
    df = clean_data(load_raw_data())
    X, y = get_feature_target(df)
    return df, X, y


@st.cache_data
def get_model_comparison():
    if os.path.exists(COMPARISON_PATH):
        return pd.read_csv(COMPARISON_PATH)
    return None


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("💳 Loan Risk System")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home / Dashboard",
        "📝 Loan Application Form",
        "✅ Loan Approval Prediction",
        "⚠️ Risk Analysis Dashboard",
        "🕘 Prediction History",
        "📊 Model Performance",
        "❓ Why This Decision?",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Educational project. Predictions are statistical estimates from a "
    "model trained on synthetic/historical data — not a real lending "
    "decision or a guarantee of repayment."
)

model = get_model()
if model is None:
    st.sidebar.error("No trained model found. Run `python src/train_model.py` first.")

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
    st.session_state["last_applicant"] = None


# ----------------------------------------------------------------------
# PAGE 1: Home / Dashboard
# ----------------------------------------------------------------------
if page == "🏠 Home / Dashboard":
    st.title("💳 Loan Approval Prediction & Risk Analysis System")
    st.markdown(
        "A machine-learning system that predicts loan approval outcomes and "
        "estimates applicant lending risk, built for academic / portfolio demonstration."
    )

    df, X, y = get_background_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applications (dataset)", f"{len(df):,}")
    col2.metric("Historical Approval Rate", f"{(df['loan_status']=='Approved').mean()*100:.1f}%")
    col3.metric("Avg Credit Score", f"{df['credit_score'].mean():.0f}")
    col4.metric("Avg Debt-to-Income", f"{df['debt_to_income_ratio'].mean():.2f}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Loan Status Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.countplot(data=df, x="loan_status", hue="loan_status", palette="viridis", legend=False, ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Credit Score by Loan Status")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=df, x="loan_status", y="credit_score", ax=ax)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("How to use this app")
    st.markdown(
        "1. Go to **Loan Application Form** and enter applicant details.\n"
        "2. View the outcome on **Loan Approval Prediction**.\n"
        "3. Check **Risk Analysis Dashboard** for the 0-100 risk score breakdown.\n"
        "4. Open **Why This Decision?** to see the top factors that drove the result.\n"
        "5. Review past predictions under **Prediction History**.\n"
        "6. Compare model candidates under **Model Performance**."
    )


# ----------------------------------------------------------------------
# PAGE 2: Loan Application Form
# ----------------------------------------------------------------------
elif page == "📝 Loan Application Form":
    st.title("📝 Loan Application Form")
    st.caption("Enter applicant details. Fields mirror the model's training features.")

    with st.form("loan_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", 18, 80, 35)
            married = st.selectbox("Marital Status", ["Yes", "No"])
            dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            employment_status = st.selectbox(
                "Employment Status", ["Salaried", "Self-Employed", "Business Owner", "Unemployed"]
            )
            years_employed = st.number_input("Years Employed", 0.0, 45.0, 5.0, step=0.5)

        with c2:
            applicant_income = st.number_input("Applicant Monthly Income", 0, 200000, 5000, step=100)
            coapplicant_income = st.number_input("Co-applicant Monthly Income", 0, 200000, 0, step=100)
            loan_amount = st.number_input("Loan Amount Requested (in thousands)", 1.0, 5000.0, 150.0, step=5.0)
            loan_term = st.selectbox("Loan Term (months)", [12, 36, 60, 84, 120, 180, 240, 360], index=5)
            credit_score = st.slider("Credit Score", 300, 900, 680)
            existing_loans = st.number_input("Number of Existing Loans", 0, 10, 0)

        with c3:
            debt_to_income_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.30, step=0.01)
            previous_repayment_history = st.selectbox(
                "Previous Repayment History", ["Good (on time)", "Poor (missed/defaulted)"]
            )
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            property_asset_value = st.number_input("Property / Asset Value", 0, 5_000_000, 50000, step=1000)
            bank_account_years = st.number_input("Years with Bank Account", 0.0, 50.0, 5.0, step=0.5)
            avg_bank_balance = st.number_input("Average Bank Balance", 0, 1_000_000, 8000, step=500)

        submitted = st.form_submit_button("Submit & Predict", use_container_width=True)

    if submitted:
        applicant = {
            "age": age,
            "married": married,
            "dependents": dependents,
            "education": education,
            "self_employed": self_employed,
            "employment_status": employment_status,
            "years_employed": years_employed,
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "credit_score": credit_score,
            "existing_loans": existing_loans,
            "debt_to_income_ratio": debt_to_income_ratio,
            "previous_repayment_history": 1 if previous_repayment_history.startswith("Good") else 0,
            "property_area": property_area,
            "property_asset_value": property_asset_value,
            "bank_account_years": bank_account_years,
            "avg_bank_balance": avg_bank_balance,
        }

        if model is None:
            st.error("No trained model available. Run `python src/train_model.py` first.")
        else:
            _, X_bg, _ = get_background_data()
            result = predict_applicant(applicant, pipeline=model, background_df=X_bg, explain=True)
            st.session_state["last_result"] = result
            st.session_state["last_applicant"] = applicant
            save_prediction(applicant, result)
            st.success("Prediction generated! Go to **Loan Approval Prediction** or **Risk Analysis Dashboard** to view it.")


# ----------------------------------------------------------------------
# PAGE 3: Loan Approval Prediction
# ----------------------------------------------------------------------
elif page == "✅ Loan Approval Prediction":
    st.title("✅ Loan Approval Prediction")

    result = st.session_state.get("last_result")
    if result is None:
        st.info("No prediction yet. Fill out the **Loan Application Form** first.")
    else:
        decision = result["loan_decision"]
        color = "green" if decision == "Approved" else "red"
        st.markdown(f"## Loan Decision: :{color}[{decision}]")

        c1, c2, c3 = st.columns(3)
        c1.metric("Approval Probability", f"{result['approval_probability']*100:.1f}%")
        c2.metric("Default / Risk Probability", f"{result['default_risk_probability']*100:.1f}%")
        c3.metric("Risk Level", result["risk_level"])

        st.progress(result["approval_probability"], text="Approval probability")

        st.markdown("---")
        st.caption(
            "⚠️ This decision is a statistical estimate from a machine-learning model. "
            "It is decision SUPPORT for a human underwriter, not an automatic, final, "
            "or guaranteed lending decision."
        )


# ----------------------------------------------------------------------
# PAGE 4: Risk Analysis Dashboard
# ----------------------------------------------------------------------
elif page == "⚠️ Risk Analysis Dashboard":
    st.title("⚠️ Risk Analysis Dashboard")

    result = st.session_state.get("last_result")
    if result is None:
        st.info("No prediction yet. Fill out the **Loan Application Form** first.")
    else:
        risk = result["risk_breakdown"]
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Risk Score", f"{result['risk_score']} / 100")
            st.metric("Risk Level", result["risk_level"])
            level_color = {"Low": "green", "Medium": "orange", "High": "red"}[result["risk_level"]]
            st.markdown(f"### :{level_color}[{result['risk_level']} Risk]")

        with c2:
            st.subheader("How this score was built")
            st.markdown(
                f"- **Model-estimated default probability:** {risk['model_default_probability']*100:.1f}% "
                f"(weight: 65%)\n"
                f"- **Rule-based underwriting index:** {risk['rule_based_index']*100:.1f}% (weight: 35%)\n"
                f"- **Blended default probability:** {risk['blended_default_probability']*100:.1f}%\n"
                f"- **Risk Score = round(100 × blended probability) = {result['risk_score']}**"
            )

        st.markdown("---")
        st.subheader("Rule-based risk factor breakdown")
        comp_df = pd.DataFrame({
            "Factor": list(risk["component_breakdown"].keys()),
            "Risk Contribution (0=low,1=high)": list(risk["component_breakdown"].values()),
            "Weight": [risk["component_weights"][k] for k in risk["component_breakdown"].keys()],
        })
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=comp_df, y="Factor", x="Risk Contribution (0=low,1=high)", hue="Factor", legend=False, palette="rocket", ax=ax)
        ax.set_xlim(0, 1)
        st.pyplot(fig)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.info(
            "Risk bands: **0-33 = Low**, **34-66 = Medium**, **67-100 = High**. "
            "These thresholds are a starting point for this project and should be "
            "validated/tuned against real lending outcomes before any real-world use."
        )


# ----------------------------------------------------------------------
# PAGE 5: Prediction History
# ----------------------------------------------------------------------
elif page == "🕘 Prediction History":
    st.title("🕘 Prediction History")
    history_df = load_history()

    if history_df.empty:
        st.info("No predictions saved yet. Submit the Loan Application Form to create history.")
    else:
        st.dataframe(
            history_df[["timestamp", "loan_decision", "approval_probability",
                        "default_risk_probability", "risk_level", "risk_score"]],
            use_container_width=True, hide_index=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Decisions over time")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=history_df, x="loan_decision", hue="loan_decision", legend=False, ax=ax)
            st.pyplot(fig)
        with c2:
            st.subheader("Risk level distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=history_df, x="risk_level", order=["Low", "Medium", "High"],
                          hue="risk_level", legend=False, palette="rocket", ax=ax)
            st.pyplot(fig)

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download history as CSV", csv, "prediction_history.csv", "text/csv")


# ----------------------------------------------------------------------
# PAGE 6: Model Performance
# ----------------------------------------------------------------------
elif page == "📊 Model Performance":
    st.title("📊 Model Performance & Comparison")

    comparison_df = get_model_comparison()
    if comparison_df is None:
        st.warning("No comparison file found. Run `python src/train_model.py` first.")
    else:
        st.dataframe(comparison_df.round(4), use_container_width=True, hide_index=True)

        metric_to_plot = st.selectbox("Compare models by:", ["roc_auc", "f1_score", "accuracy", "precision", "recall"])
        fig, ax = plt.subplots(figsize=(8, 4))
        sorted_df = comparison_df.sort_values(metric_to_plot, ascending=False)
        sns.barplot(data=sorted_df, x=metric_to_plot, y="model", hue="model", legend=False, palette="crest", ax=ax)
        ax.set_xlim(0, 1)
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Why not just use accuracy?")
        st.markdown(
            "Loan datasets are often imbalanced (more approvals or more rejections). "
            "A model can score high **accuracy** by mostly predicting the majority class "
            "while still being poor at catching the minority class. This project selects "
            "the best model using **ROC-AUC** (ranking quality across all thresholds) with "
            "**F1-score** as a tiebreaker, which better reflect real predictive quality on "
            "imbalanced data."
        )

        st.markdown("---")
        st.subheader("Global feature importance (best model)")
        if model is not None:
            imp_df = global_feature_importance(model, top_n=12)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=imp_df, x="importance_pct", y="feature", hue="feature", legend=False, palette="mako", ax=ax)
            ax.set_xlabel("Relative importance (%)")
            st.pyplot(fig)


# ----------------------------------------------------------------------
# PAGE 7: Why This Decision?
# ----------------------------------------------------------------------
elif page == "❓ Why This Decision?":
    st.title("❓ Why This Decision?")

    result = st.session_state.get("last_result")
    applicant = st.session_state.get("last_applicant")

    if result is None:
        st.info("No prediction yet. Fill out the **Loan Application Form** first.")
    else:
        st.markdown(f"### Decision: **{result['loan_decision']}** (Risk Level: {result['risk_level']})")
        st.caption(
            "The chart below shows the factors that most influenced THIS specific "
            "applicant's prediction (a local explanation), using SHAP values when "
            "available, or a transparent perturbation-based approximation otherwise."
        )

        factors_df = pd.DataFrame(result["top_factors"])
        factors_df["signed_impact"] = factors_df.apply(
            lambda r: r["abs_impact"] if r["direction"] == "toward Approval" else -r["abs_impact"], axis=1
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = factors_df["signed_impact"].apply(lambda v: "#2ca02c" if v >= 0 else "#d62728")
        ax.barh(factors_df["feature"], factors_df["signed_impact"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Impact on approval probability (green = toward approval, red = toward rejection)")
        ax.invert_yaxis()
        st.pyplot(fig)

        st.dataframe(
            factors_df[["feature", "direction", "abs_impact"]].rename(
                columns={"feature": "Factor", "direction": "Direction", "abs_impact": "Impact magnitude"}
            ),
            use_container_width=True, hide_index=True,
        )

        with st.expander("View submitted applicant details"):
            st.json(applicant)

        st.warning(
            "This explanation shows correlational drivers learned from historical/synthetic "
            "data, not a causal guarantee. It should support a human reviewer's judgment, "
            "not replace it — and must not be used to justify decisions based on protected "
            "attributes (e.g. gender, religion, caste, race), which were excluded from "
            "training for fairness and legal-compliance reasons."
        )
