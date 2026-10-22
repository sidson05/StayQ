"""
Unit Tests for Machine Learning Feature Engineering and Inference.
"""

import pytest
import pandas as pd
import numpy as np
from src.ml.features import prepare_features, FEATURE_COLUMNS
from src.ml.predict import assign_risk_tier, generate_retention_recommendation


def test_prepare_features_structure():
    mock_mart = pd.DataFrame({
        "customer_id": ["CUST-10001"],
        "is_month_to_month": [1],
        "tenure_months": [6],
        "monthly_charges": [70.0],
        "total_charges": [420.0],
        "usage_drop_velocity": [0.25],
        "total_tickets": [3],
        "unresolved_tickets_count": [2]
    })
    
    X, cols = prepare_features(mock_mart)
    assert len(cols) == len(FEATURE_COLUMNS)
    assert X.shape == (1, len(FEATURE_COLUMNS))
    assert not X.isnull().values.any()
    assert np.isfinite(X.values).all()


def test_risk_tier_assignment():
    assert assign_risk_tier(0.85) == "CRITICAL"
    assert assign_risk_tier(0.55) == "HIGH"
    assert assign_risk_tier(0.30) == "MEDIUM"
    assert assign_risk_tier(0.12) == "LOW"


def test_retention_recommendation_logic():
    rec_ticket = generate_retention_recommendation("unresolved_tickets_count", "HIGH", 1)
    assert "VIP Customer Success" in rec_ticket or "support" in rec_ticket.lower()

    rec_low = generate_retention_recommendation("any", "LOW", 0)
    assert "No action required" in rec_low
