"""
Inference & Customer Risk Scoring Engine.
Calculates churn probability, risk tier categorization, local SHAP driver explanations,
and retention action recommendations.
"""

import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.warehouse.db import WarehouseManager
from src.ml.features import prepare_features, FEATURE_COLUMNS, FEATURE_LABELS_MAPPING

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "xgboost_churn_model.joblib")
EXPLAINER_PATH = os.path.join(ARTIFACTS_DIR, "shap_explainer.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "model_metrics.json")

_cached_model = None
_cached_explainer = None
_cached_metrics = None


def get_artifacts():
    global _cached_model, _cached_explainer, _cached_metrics
    if _cached_model is None and os.path.exists(MODEL_PATH):
        _cached_model = joblib.load(MODEL_PATH)
    if _cached_explainer is None and os.path.exists(EXPLAINER_PATH):
        _cached_explainer = joblib.load(EXPLAINER_PATH)
    if _cached_metrics is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            _cached_metrics = json.load(f)
    return _cached_model, _cached_explainer, _cached_metrics


def assign_risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return "CRITICAL"
    elif prob >= 0.45:
        return "HIGH"
    elif prob >= 0.25:
        return "MEDIUM"
    return "LOW"


def generate_retention_recommendation(top_driver: str, risk_tier: str, is_m2m: int) -> str:
    if risk_tier in ["LOW"]:
        return "No action required. Customer is healthy."
        
    if "unresolved" in top_driver.lower() or "ticket" in top_driver.lower() or "support" in top_driver.lower():
        return "Trigger immediate VIP Customer Success outreach to resolve pending support tickets within 24 hours."
    elif "velocity" in top_driver.lower() or "drop" in top_driver.lower() or "crash" in top_driver.lower():
        return "Re-engagement campaign: Offer interactive onboarding assistance or check device compatibility."
    elif "invoice" in top_driver.lower() or "delay" in top_driver.lower() or "payment" in top_driver.lower():
        return "Billing intervention: Proactively assist with auto-pay card update and waive recent late fees."
    elif is_m2m or "month" in top_driver.lower():
        return "Contract lock-in: Offer 15% discount for switching to an annual contract."
    else:
        return "Targeted loyalty perk: Provide free speed boost or premium feature trial for 30 days."


def score_feature_vector(row_df: pd.DataFrame) -> Dict[str, Any]:
    """Score a single feature row dataframe and calculate SHAP drivers."""
    model, explainer, metrics = get_artifacts()
    if model is None:
        raise RuntimeError("ML model artifact not found. Please train model first.")
        
    X, _ = prepare_features(row_df)
    prob = float(model.predict_proba(X)[0, 1])
    risk_tier = assign_risk_tier(prob)
    
    # Calculate SHAP local explanations
    top_drivers = []
    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(X)[0]
            # Pair feature names with SHAP contributions
            driver_pairs = []
            for col, val in zip(FEATURE_COLUMNS, shap_vals):
                driver_pairs.append({
                    "feature": col,
                    "label": FEATURE_LABELS_MAPPING.get(col, col.replace("_", " ").title()),
                    "shap_value": float(val),
                    "impact": "Increases Churn Risk" if val > 0 else "Decreases Churn Risk"
                })
            # Sort by absolute impact
            driver_pairs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            top_drivers = driver_pairs[:4]
        except Exception as e:
            print(f"[SHAP] Error explaining row: {e}")
            
    primary_driver = top_drivers[0]["feature"] if top_drivers else "general"
    is_m2m = int(row_df.get("is_month_to_month", pd.Series([0])).iloc[0])
    recommendation = generate_retention_recommendation(primary_driver, risk_tier, is_m2m)
    
    return {
        "churn_probability": round(prob, 4),
        "risk_tier": risk_tier,
        "is_churn_alert": bool(prob >= (metrics.get("measured_metrics", {}).get("optimal_threshold", 0.35) if metrics else 0.35)),
        "primary_churn_driver": top_drivers[0]["label"] if top_drivers else "Behavioral Drop",
        "top_drivers": top_drivers,
        "recommended_action": recommendation
    }


def predict_customer_risk(customer_id: str) -> Dict[str, Any]:
    """Look up customer in warehouse feature mart and score risk."""
    wh = WarehouseManager()
    query = f"SELECT * FROM analytics_mart_churn_features WHERE customer_id = '{customer_id}' LIMIT 1;"
    df = wh.read_query(query)
    
    if df.empty:
        return {"error": f"Customer ID '{customer_id}' not found in warehouse."}
        
    scoring = score_feature_vector(df)
    scoring["customer_id"] = customer_id
    scoring["contract_type"] = df["contract_type"].iloc[0]
    scoring["monthly_charges"] = float(df["monthly_charges"].iloc[0])
    scoring["tenure_months"] = int(df["tenure_months"].iloc[0])
    scoring["total_tickets"] = int(df["total_tickets"].iloc[0])
    scoring["unresolved_tickets"] = int(df["unresolved_tickets_count"].iloc[0])
    return scoring


def batch_score_all_customers() -> pd.DataFrame:
    """Score all customers in the feature mart and return sorted high-risk table."""
    wh = WarehouseManager()
    query = "SELECT * FROM analytics_mart_churn_features;"
    df = wh.read_query(query)
    
    if df.empty:
        return pd.DataFrame()
        
    model, _, metrics = get_artifacts()
    X, _ = prepare_features(df)
    probs = model.predict_proba(X)[:, 1]
    
    df["churn_probability"] = np.round(probs, 4)
    df["risk_tier"] = [assign_risk_tier(p) for p in probs]
    df["expected_revenue_loss"] = np.round(df["monthly_charges"] * df["churn_probability"], 2)
    
    df_sorted = df.sort_values(by="churn_probability", ascending=False).reset_index(drop=True)
    return df_sorted
