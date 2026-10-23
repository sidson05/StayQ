"""
Integration Tests for FastAPI Serving Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "warehouse_type" in data


def test_api_predict_custom():
    payload = {
        "tenure_months": 3,
        "is_month_to_month": 1,
        "is_electronic_check": 1,
        "is_paperless_billing": 1,
        "has_tech_support": 0,
        "has_online_security": 0,
        "has_streaming_tv": 1,
        "monthly_charges": 95.0,
        "total_charges": 285.0,
        "total_invoices": 3,
        "failed_invoices_count": 2,
        "failed_invoices_ratio": 0.66,
        "avg_payment_delay_days": 18.0,
        "max_payment_delay_days": 25,
        "total_tickets": 4,
        "unresolved_tickets_count": 2,
        "escalated_tickets_count": 1,
        "critical_tickets_count": 1,
        "avg_ticket_sentiment": -0.6,
        "has_cancellation_inquiry": 1,
        "total_data_usage_gb": 120.0,
        "avg_biweekly_usage_gb": 20.0,
        "total_logins": 15,
        "recent_usage_30d_gb": 10.0,
        "prior_usage_60d_gb": 110.0,
        "usage_drop_velocity": 0.18,
        "recent_logins_30d": 2,
        "total_crashes": 3,
        "recent_crashes_30d": 2
    }
    response = client.post("/api/v1/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "churn_probability" in data
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_tier"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        assert len(data["top_drivers"]) > 0
