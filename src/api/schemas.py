"""
Pydantic schemas for FastAPI serving layer.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    warehouse_connected: bool
    warehouse_type: str


class DriverExplanation(BaseModel):
    feature: str
    label: str
    shap_value: float
    impact: str


class CustomerRiskResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str
    is_churn_alert: bool
    contract_type: Optional[str] = None
    monthly_charges: Optional[float] = None
    tenure_months: Optional[int] = None
    total_tickets: Optional[int] = None
    unresolved_tickets: Optional[int] = None
    primary_churn_driver: str
    top_drivers: List[DriverExplanation]
    recommended_action: str


class CustomPredictRequest(BaseModel):
    tenure_months: int = Field(12, ge=0, le=120)
    is_month_to_month: int = Field(1, ge=0, le=1)
    is_electronic_check: int = Field(1, ge=0, le=1)
    is_paperless_billing: int = Field(1, ge=0, le=1)
    has_tech_support: int = Field(0, ge=0, le=1)
    has_online_security: int = Field(0, ge=0, le=1)
    has_streaming_tv: int = Field(1, ge=0, le=1)
    monthly_charges: float = Field(85.0, ge=10.0, le=300.0)
    total_charges: float = Field(1020.0, ge=0.0)
    total_invoices: int = Field(6, ge=1)
    failed_invoices_count: int = Field(1, ge=0)
    failed_invoices_ratio: float = Field(0.16, ge=0.0, le=1.0)
    avg_payment_delay_days: float = Field(8.0, ge=0.0)
    max_payment_delay_days: int = Field(15, ge=0)
    total_tickets: int = Field(3, ge=0)
    unresolved_tickets_count: int = Field(1, ge=0)
    escalated_tickets_count: int = Field(1, ge=0)
    critical_tickets_count: int = Field(0, ge=0)
    avg_ticket_sentiment: float = Field(-0.4, ge=-1.0, le=1.0)
    has_cancellation_inquiry: int = Field(0, ge=0, le=1)
    total_data_usage_gb: float = Field(250.0, ge=0.0)
    avg_biweekly_usage_gb: float = Field(42.0, ge=0.0)
    total_logins: int = Field(35, ge=0)
    recent_usage_30d_gb: float = Field(30.0, ge=0.0)
    prior_usage_60d_gb: float = Field(220.0, ge=0.0)
    usage_drop_velocity: float = Field(0.27, ge=0.0, le=5.0)
    recent_logins_30d: int = Field(6, ge=0)
    total_crashes: int = Field(2, ge=0)
    recent_crashes_30d: int = Field(1, ge=0)


class CustomPredictResponse(BaseModel):
    churn_probability: float
    risk_tier: str
    is_churn_alert: bool
    primary_churn_driver: str
    top_drivers: List[DriverExplanation]
    recommended_action: str


class PipelineTriggerResponse(BaseModel):
    status: str
    message: str
    execution_report: Dict[str, Any]
