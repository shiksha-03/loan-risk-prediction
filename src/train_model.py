"""
train_model.py
---------------
Trains and compares Logistic Regression, Decision Tree, Random Forest,
XGBoost, Gradient Boosting, and SVM on the loan approval task, using a
single leakage-safe Pipeline (preprocessing + classifier) for each model.

- Class imbalance is handled with `class_weight="balanced"` (or
  `scale_pos_weight` for XGBoost) rather than oversampling, so no
  synthetic rows leak information between train/test splits.
- Model selection uses ROC-AUC and F1 (not raw accuracy), because in
  loan approval a model that just predicts the majority class can look
  "accurate" while being useless.
- The winning model's FULL pipeline (preprocessing + classifier) is
  saved with joblib so identical preprocessing is guaranteed at
  prediction time.

Run: python src/train_model.py
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from data_preprocessing import (
    load_raw_data, clean_data, get_feature_target, build_preprocessing_pipeline,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[warn] xgboost is not installed in this environment — it will be skipped. "
          "Install it via requirements.txt to include it in the comparison.")

MODELS_DIR = "models"
RANDOM_STATE = 42


def get_candidate_models(y_train):
    """Step 12: Define all candidate models, each with imbalance handling."""
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        )

    return models


def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(pd.Series(metrics).drop("model").round(4).to_string())
    print("Confusion Matrix [[TN FP] [FN TP]]:\n", cm)
    return metrics, cm


def main():
    print("Loading and cleaning data...")
    df = clean_data(load_raw_data())
    X, y = get_feature_target(df)

    # Step 10: Reproducible stratified train/test split (BEFORE any fitting)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Train class balance:\n{y_train.value_counts(normalize=True).round(3)}")

    preprocessor = build_preprocessing_pipeline()
    candidates = get_candidate_models(y_train)

    results = []
    fitted_pipelines = {}

    for name, clf in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        metrics, cm = evaluate_model(name, pipeline, X_test, y_test)
        metrics["train_time_sec"] = round(train_time, 2)
        results.append(metrics)
        fitted_pipelines[name] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        by=["roc_auc", "f1_score"], ascending=False
    ).reset_index(drop=True)

    print("\n\n================ MODEL COMPARISON (sorted by ROC-AUC then F1) ================")
    print(results_df.round(4).to_string(index=False))

    # Step 14: select best model by ROC-AUC primarily, F1 as tiebreaker —
    # NOT by raw accuracy, since accuracy can be misleading on imbalanced
    # lending data (a model that rejects everyone can still look "accurate").
    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\nSelected best model: {best_model_name}")

    # Step 15: Save trained model (full pipeline = preprocessing + classifier)
    import os
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_pipeline, f"{MODELS_DIR}/best_model_pipeline.joblib")

    results_df.to_csv(f"{MODELS_DIR}/model_comparison.csv", index=False)
    with open(f"{MODELS_DIR}/best_model_info.json", "w") as f:
        json.dump({
            "best_model_name": best_model_name,
            "metrics": results_df.iloc[0].to_dict(),
            "feature_columns": list(X.columns),
        }, f, indent=2, default=str)

    print(f"\nSaved best pipeline -> {MODELS_DIR}/best_model_pipeline.joblib")
    print(f"Saved comparison table -> {MODELS_DIR}/model_comparison.csv")

    return results_df, best_pipeline, best_model_name


if __name__ == "__main__":
    main()
