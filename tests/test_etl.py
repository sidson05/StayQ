"""
Unit Tests for ETL Pipeline, Extractors, and Data Validation.
"""

import os
import pytest
import pandas as pd
from src.etl.validators import DataValidator
from src.etl.extractors import BillingExtractor, UsageLogsExtractor


@pytest.fixture
def sample_contracts_df():
    return pd.DataFrame({
        "customer_id": ["CUST-001", "CUST-002", "CUST-003"],
        "contract_type": ["Month-to-month", "One year", "Two year"],
        "monthly_charges": [45.5, 89.0, 110.2],
        "total_charges": [45.5, 1068.0, 2644.8],
        "tenure_months": [1, 12, 24],
        "signup_date": ["2026-09-01", "2025-09-01", "2024-09-01"]
    })


@pytest.fixture
def sample_invoices_df():
    return pd.DataFrame({
        "invoice_id": ["INV-01", "INV-02", "INV-03"],
        "customer_id": ["CUST-001", "CUST-002", "CUST-003"],
        "invoice_date": ["2026-09-15", "2026-09-15", "2026-09-15"],
        "amount": [45.5, 89.0, 110.2],
        "payment_status": ["PAID", "FAILED", "PAID"],
        "payment_delay_days": [0, 14, 0]
    })


def test_validator_detects_out_of_bounds(sample_contracts_df):
    validator = DataValidator()
    # Add an out-of-range monthly charge
    bad_df = sample_contracts_df.copy()
    bad_df.loc[0, "monthly_charges"] = 999.0
    
    clean_df, metrics = validator.validate_dataset(bad_df, "billing_contracts")
    assert metrics["quarantined_rows"] == 1
    assert len(clean_df) == 2


def test_validator_detects_unapproved_categories(sample_contracts_df):
    validator = DataValidator()
    bad_df = sample_contracts_df.copy()
    bad_df.loc[1, "contract_type"] = "Lifetime Free"
    
    clean_df, metrics = validator.validate_dataset(bad_df, "billing_contracts")
    assert metrics["quarantined_rows"] == 1
    assert "Lifetime Free" not in clean_df["contract_type"].values


def test_referential_integrity(sample_contracts_df, sample_invoices_df):
    validator = DataValidator()
    tickets_df = pd.DataFrame({"customer_id": ["CUST-001", "CUST-002"]})
    usage_df = pd.DataFrame({"customer_id": ["CUST-001", "CUST-002", "CUST-003"]})
    
    res = validator.validate_referential_integrity(
        sample_contracts_df, sample_invoices_df, tickets_df, usage_df
    )
    assert res["referential_integrity_intact"] is True
    assert res["orphan_invoice_customers"] == 0
