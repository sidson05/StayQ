"""
Feature Engineering and Data Preparation for 30-Day Churn Prediction.
Transforms the gold analytics mart into cleaned, encoded ML feature matrices.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any


FEATURE_COLUMNS = [
    # Customer Contract & Profile
    "tenure_months",
    "is_month_to_month",
    "is_electronic_check",
    "is_paperless_billing",
    "has_tech_support",
    "has_online_security",
    "has_streaming_tv",
    "monthly_charges",
    "total_charges",
    
    # Billing & Payment Friction
    "total_invoices",
    "failed_invoices_count",
    "failed_invoices_ratio",
    "avg_payment_delay_days",
    "max_payment_delay_days",
    
    # Support Ticket Friction
    "total_tickets",
    "unresolved_tickets_count",
    "escalated_tickets_count",
    "critical_tickets_count",
    "avg_ticket_sentiment",
    "has_cancellation_inquiry",
    
    # Telemetry & Usage Drop Velocity
    "total_data_usage_gb",
    "avg_biweekly_usage_gb",
    "total_logins",
    "recent_usage_30d_gb",
    "prior_usage_60d_gb",
    "usage_drop_velocity",
    "recent_logins_30d",
    "total_crashes",
    "recent_crashes_30d"
]

FEATURE_LABELS_MAPPING = {
    "usage_drop_velocity": "App Usage Drop (30d vs 60d)",
    "unresolved_tickets_count": "Open / Unresolved Support Tickets",
    "escalated_tickets_count": "Escalated Support Tickets",
    "avg_ticket_sentiment": "Negative Support Sentiment",
    "failed_invoices_count": "Recent Failed Invoices",
    "avg_payment_delay_days": "Payment Delays (Days)",
    "is_month_to_month": "Month-to-Month Contract",
    "tenure_months": "Short Customer Tenure",
    "monthly_charges": "High Monthly Charges",
    "has_tech_support": "Lack of Tech Support",
    "recent_crashes_30d": "Frequent Recent App Crashes",
    "is_electronic_check": "Electronic Check Payment Method",
    "has_cancellation_inquiry": "Prior Cancellation Inquiry"
}


def prepare_features(mart_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Clean and select standard feature columns from analytics mart."""
    df = mart_df.copy()
    
    # Ensure all required feature columns exist, impute defaults if missing
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            
    # Clip any infinities in velocity calculation
    df["usage_drop_velocity"] = df["usage_drop_velocity"].replace([np.inf, -np.inf], 1.0).clip(0.0, 5.0)
    
    X = df[FEATURE_COLUMNS].copy()
    return X, FEATURE_COLUMNS
