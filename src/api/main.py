"""
FastAPI Production Serving Layer for Churn Intelligence.
Exposes endpoints for customer risk scoring, SHAP explainability, batch ranking,
data quality metrics, and on-demand ETL execution.
"""

import os
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.warehouse.db import WarehouseManager
from src.ml.predict import (
    predict_customer_risk, score_feature_vector,
    batch_score_all_customers, get_artifacts
)
from src.etl.pipeline import ChurnETLPipeline
from src.api.schemas import (
    HealthResponse, CustomerRiskResponse, CustomPredictRequest,
    CustomPredictResponse, PipelineTriggerResponse
)

app = FastAPI(
    title="Customer Churn Prediction & Retention Intelligence API",
    description="Serves automated churn risk scores, SHAP driver explanations, and retention recommendations.",
    version="1.0.0"
)

# Enable CORS for frontend / dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Verify API health, model availability, and database warehouse connection."""
    wh = WarehouseManager()
    model, _, _ = get_artifacts()
    return {
        "status": "healthy",
        "service": "churn_prediction_api",
        "model_loaded": (model is not None),
        "warehouse_connected": (wh.engine is not None),
        "warehouse_type": wh.db_type
    }


@app.get("/api/v1/customers/{customer_id}/risk", response_model=CustomerRiskResponse)
def get_customer_risk(customer_id: str):
    """Retrieve 30-day churn probability, risk tier, top SHAP drivers, and suggested retention action."""
    result = predict_customer_risk(customer_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/api/v1/predict", response_model=CustomPredictResponse)
def predict_custom_features(payload: CustomPredictRequest):
    """Real-time scoring for custom feature payloads or 'what-if' retention simulations."""
    df_payload = pd.DataFrame([payload.model_dump()])
    result = score_feature_vector(df_payload)
    return result


@app.get("/api/v1/retention/high-risk")
def get_high_risk_customers(
    limit: int = Query(50, ge=1, le=500),
    tier: Optional[str] = Query(None, description="Filter by risk tier: CRITICAL, HIGH, MEDIUM, LOW")
):
    """Get ranked list of customers most likely to churn in the next 30 days, sorted by expected revenue loss."""
    df_scored = batch_score_all_customers()
    if df_scored.empty:
        raise HTTPException(status_code=404, detail="No customer features found in warehouse.")
        
    if tier:
        df_scored = df_scored[df_scored["risk_tier"] == tier.upper()]
        
    records = df_scored[[
        "customer_id", "contract_type", "monthly_charges", "tenure_months",
        "total_tickets", "unresolved_tickets_count", "usage_drop_velocity",
        "churn_probability", "risk_tier", "expected_revenue_loss"
    ]].head(limit).to_dict(orient="records")
    
    return {
        "total_at_risk_returned": len(records),
        "customers": records
    }


@app.get("/api/v1/metrics")
def get_model_evaluation_metrics():
    """Returns strictly measured model evaluation metrics from test set."""
    _, _, metrics = get_artifacts()
    if not metrics:
        raise HTTPException(status_code=404, detail="Model metrics not yet computed. Please run ML training.")
    return metrics


@app.post("/api/v1/pipeline/trigger", response_model=PipelineTriggerResponse)
def trigger_etl_pipeline(background_tasks: BackgroundTasks):
    """Trigger on-demand automated ETL execution to refresh warehouse tables."""
    try:
        pipeline = ChurnETLPipeline()
        report = pipeline.run()
        return {
            "status": "SUCCESS",
            "message": "ETL pipeline executed and warehouse tables refreshed.",
            "execution_report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ETL pipeline execution failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
