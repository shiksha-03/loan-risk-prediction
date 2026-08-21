"""
eda.py
------
Exploratory Data Analysis: distributions, correlations, and relationships
between features and the target (loan_status). Saves plots to reports/.

Run: python src/eda.py
"""
import os
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_preprocessing import load_raw_data, clean_data, NUMERIC_FEATURES

OUT_DIR = "reports"


def run_eda():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = clean_data(load_raw_data())
    sns.set_theme(style="whitegrid")

    # 1. Target distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="loan_status", palette="viridis")
    plt.title("Loan Status Distribution (class balance)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_target_distribution.png", dpi=120)
    plt.close()

    # 2. Numeric feature distributions
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, col in zip(axes.flatten(), NUMERIC_FEATURES):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="teal")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_numeric_distributions.png", dpi=120)
    plt.close()

    # 3. Correlation heatmap (numeric features + target)
    corr_df = df[NUMERIC_FEATURES].copy()
    corr_df["approved"] = (df["loan_status"] == "Approved").astype(int)
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (numeric features + target)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_correlation_heatmap.png", dpi=120)
    plt.close()

    # 4. Key relationships with target
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    sns.boxplot(data=df, x="loan_status", y="credit_score", ax=axes[0, 0])
    sns.boxplot(data=df, x="loan_status", y="debt_to_income_ratio", ax=axes[0, 1])
    sns.boxplot(data=df, x="loan_status", y="applicant_income", ax=axes[0, 2])
    sns.countplot(data=df, x="employment_status", hue="loan_status", ax=axes[1, 0])
    sns.countplot(data=df, x="previous_repayment_history", hue="loan_status", ax=axes[1, 1])
    sns.countplot(data=df, x="education", hue="loan_status", ax=axes[1, 2])
    for ax in axes.flatten():
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_key_relationships.png", dpi=120)
    plt.close()

    # Print top correlations with target as text summary
    corr_with_target = corr_df.corr()["approved"].drop("approved").sort_values(key=abs, ascending=False)
    print("Top correlations with loan approval:\n", corr_with_target)
    corr_with_target.to_csv(f"{OUT_DIR}/correlations_with_target.csv")

    print(f"\nEDA plots saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    run_eda()
