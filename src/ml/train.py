"""
Machine Learning Training Pipeline for 30-Day Churn Prediction.
Handles class imbalance, benchmarks baseline models, optimizes XGBoost,
fits SHAP explainers, and logs ONLY strictly measured metrics.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

from src.warehouse.db import WarehouseManager
from src.ml.features import prepare_features, FEATURE_COLUMNS

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_dataset_from_warehouse():
    """Load feature mart from database and join with ground truth labels."""
    wh = WarehouseManager()
    mart_query = "SELECT * FROM analytics_mart_churn_features;"
    mart_df = wh.read_query(mart_query)
    
    gt_path = os.path.join("data", "ground_truth_churn.csv")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth churn file not found: {gt_path}")
    gt_df = pd.read_csv(gt_path)
    
    merged = pd.merge(mart_df, gt_df, on="customer_id", how="inner")
    print(f"[ML Train] Loaded {len(merged)} customer records from gold feature mart.")
    return merged


def optimize_threshold(y_true, y_probs, cost_fp=20.0, cost_fn=300.0, reward_tp=250.0):
    """
    Find optimal decision threshold maximizing financial expected value.
    - cost_fp: Cost of giving unnecessary retention discount to non-churner (~$20)
    - cost_fn: Lost customer LTV (~$300)
    - reward_tp: Net retained customer value (~$250)
    """
    best_thresh = 0.5
    best_value = -float("inf")
    
    thresholds = np.linspace(0.1, 0.9, 81)
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        # Economic value calculation
        net_value = (tp * reward_tp) - (fp * cost_fp) - (fn * cost_fn)
        if net_value > best_value:
            best_value = net_value
            best_thresh = t
            
    return round(float(best_thresh), 3)


def train_and_evaluate():
    print("\n=======================================================")
    print("[START] ML TRAINING & BENCHMARKING PIPELINE")
    print("=======================================================\n")
    
    df = load_dataset_from_warehouse()
    X, feature_names = prepare_features(df)
    y = df["churn_30d"].values
    
    churn_rate = float(np.mean(y))
    print(f"[ML Train] Dataset Shape: {X.shape}, Churn Rate: {churn_rate:.2%}")
    
    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[ML Train] Train split: {len(X_train)} samples, Test split: {len(X_test)} samples")
    
    # Scale features for linear model baseline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Baseline Model: Logistic Regression
    print("\n--- 1. Training Baseline: Logistic Regression ---")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_roc = roc_auc_score(y_test, lr_probs)
    lr_pr_auc = average_precision_score(y_test, lr_probs)
    print(f"[Baseline LR] ROC-AUC: {lr_roc:.4f} | PR-AUC: {lr_pr_auc:.4f}")
    
    # 2. Challenger Model: Random Forest
    print("\n--- 2. Training Challenger: Random Forest ---")
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_roc = roc_auc_score(y_test, rf_probs)
    rf_pr_auc = average_precision_score(y_test, rf_probs)
    print(f"[Challenger RF] ROC-AUC: {rf_roc:.4f} | PR-AUC: {rf_pr_auc:.4f}")
    
    # 3. Champion Model: XGBoost with class-weight imbalance handling
    print("\n--- 3. Training Champion: XGBoost Classifier ---")
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = float(neg_count / pos_count)
    print(f"[ML Train] Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric=["logloss", "auc"],
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Test split evaluation
    y_probs = xgb_model.predict_proba(X_test)[:, 1]
    optimal_thresh = optimize_threshold(y_test, y_probs)
    y_pred = (y_probs >= optimal_thresh).astype(int)
    
    # Calculate STRICTLY MEASURED real test metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_probs))
    pr_auc = float(average_precision_score(y_test, y_probs))
    brier = float(brier_score_loss(y_test, y_probs))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print("\n=======================================================")
    print("[EVALUATION] MEASURED TEST METRICS (XGBoost Champion)")
    print("=======================================================")
    print(f"* ROC-AUC Score:          {roc_auc:.4f}")
    print(f"* PR-AUC (Avg Precision): {pr_auc:.4f}")
    print(f"* Optimal Threshold:      {optimal_thresh:.3f}")
    print(f"* Recall (Retention Catch):{rec:.4f} ({rec*100:.1f}% churners identified)")
    print(f"* Precision:              {prec:.4f}")
    print(f"* F1-Score:               {f1:.4f}")
    print(f"* Accuracy:               {acc:.4f}")
    print(f"* Brier Calibration Score:{brier:.4f}")
    print(f"* Confusion Matrix (TN, FP / FN, TP):\n  {cm}")
    print("=======================================================\n")
    
    # 4. Fit SHAP TreeExplainer
    print("--- 4. Computing SHAP Explainability Artifacts ---")
    explainer = shap.TreeExplainer(xgb_model)
    # Background sample for explanation speed
    shap_sample = X_test.iloc[:200]
    shap_values = explainer.shap_values(shap_sample)
    print(f"[SHAP] Generated explainability values for {len(shap_sample)} test profiles.")
    
    # 5. Export Model Artifacts
    model_path = os.path.join(ARTIFACTS_DIR, "xgboost_churn_model.joblib")
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
    explainer_path = os.path.join(ARTIFACTS_DIR, "shap_explainer.joblib")
    metrics_path = os.path.join(ARTIFACTS_DIR, "model_metrics.json")
    
    joblib.dump(xgb_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(explainer, explainer_path)
    
    metrics_data = {
        "model_name": "XGBoost Classifier",
        "dataset_samples": len(df),
        "test_samples": len(X_test),
        "class_imbalance": {
            "churn_rate": churn_rate,
            "scale_pos_weight": scale_pos_weight
        },
        "measured_metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "optimal_threshold": optimal_thresh,
            "recall": rec,
            "precision": prec,
            "f1_score": f1,
            "accuracy": acc,
            "brier_score": brier,
            "confusion_matrix": cm
        },
        "model_benchmarks": {
            "logistic_regression": {"roc_auc": round(float(lr_roc), 4), "pr_auc": round(float(lr_pr_auc), 4)},
            "random_forest": {"roc_auc": round(float(rf_roc), 4), "pr_auc": round(float(rf_pr_auc), 4)},
            "xgboost": {"roc_auc": round(float(roc_auc), 4), "pr_auc": round(float(pr_auc), 4)}
        },
        "features": feature_names
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
        
    print(f"[Artifacts] Saved champion model to: {model_path}")
    print(f"[Artifacts] Saved measured metrics to: {metrics_path}")
    return metrics_data


if __name__ == "__main__":
    train_and_evaluate()
