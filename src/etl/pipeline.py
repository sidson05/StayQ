"""
Automated End-to-End ETL Pipeline Orchestrator.
Ingests multi-source data, applies data quality gates, loads raw warehouse tables,
and executes SQL transformations to build the analytics churn feature mart.
"""

import os
import sys
import json
import time
from datetime import datetime
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.warehouse.db import WarehouseManager
from src.etl.extractors import BillingExtractor, SupportAPIExtractor, UsageLogsExtractor
from src.etl.validators import DataValidator
from src.etl.loaders import WarehouseLoader

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class ChurnETLPipeline:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.wh = WarehouseManager(self.config)
        self.validator = DataValidator()
        self.loader = WarehouseLoader(self.wh)

    def run(self):
        start_time = time.time()
        print("\n=======================================================")
        print(f"[START] CHURN ETL PIPELINE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=======================================================\n")

        # 1. Initialize warehouse schemas
        print("--- Step 1: Initializing Warehouse Schemas ---")
        self.wh.init_schema()

        # 2. Extract from Multi-Source Ingestion Layer
        print("\n--- Step 2: Extracting from Multi-Source Silos ---")
        src_cfg = self.config.get("sources", {})
        billing_ext = BillingExtractor(
            contracts_path=src_cfg["billing_contracts_path"],
            invoices_path=src_cfg["billing_invoices_path"]
        )
        support_ext = SupportAPIExtractor(
            api_url=src_cfg["support_api_url"],
            fallback_json_path=src_cfg["mock_tickets_db"]
        )
        usage_ext = UsageLogsExtractor(
            usage_logs_path=src_cfg["usage_logs_path"]
        )

        contracts_raw = billing_ext.extract_contracts()
        invoices_raw = billing_ext.extract_invoices()
        tickets_raw = support_ext.extract_tickets()
        usage_raw = usage_ext.extract_usage_events()

        # 3. Validate Schemas & Enforce Data Quality Rules
        print("\n--- Step 3: Executing Data Quality & Contract Validation ---")
        clean_contracts, q_contracts = self.validator.validate_dataset(contracts_raw, "billing_contracts")
        clean_invoices, q_invoices = self.validator.validate_dataset(invoices_raw, "billing_invoices")
        clean_tickets, q_tickets = self.validator.validate_dataset(tickets_raw, "support_tickets")
        clean_usage, q_usage = self.validator.validate_dataset(usage_raw, "usage_events")

        # Referential Integrity
        ref_integrity = self.validator.validate_referential_integrity(
            clean_contracts, clean_invoices, clean_tickets, clean_usage
        )

        # 4. Load Raw (Bronze) Layer
        print("\n--- Step 4: Loading Validated Data into Raw Warehouse Tables ---")
        self.loader.load_raw_tables(clean_contracts, clean_invoices, clean_tickets, clean_usage)

        # 5. Execute SQL Transformations (Staging & Analytics Marts)
        print("\n--- Step 5: Executing Silver & Gold SQL Transformations ---")
        self.wh.run_transformations()

        duration = round(time.time() - start_time, 2)
        
        # Verify analytics mart output
        mart_tbl = "analytics_mart_churn_features"
        mart_df = self.wh.read_query(f"SELECT COUNT(*) as row_count FROM {mart_tbl};")
        final_mart_count = int(mart_df["row_count"].iloc[0])

        report = {
            "status": "SUCCESS",
            "execution_timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "warehouse_type": self.wh.db_type,
            "analytics_mart_rows": final_mart_count,
            "data_quality_summary": self.validator.report
        }

        report_file = os.path.join("data", "etl_run_report.json")
        os.makedirs("data", exist_ok=True)
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print("\n=======================================================")
        print(f"[SUCCESS] ETL PIPELINE COMPLETED in {duration}s")
        print(f"[Warehouse] Analytics Mart Populated: {final_mart_count} customer feature records ready for ML.")
        print(f"[Report] Quality report saved to: {report_file}")
        print("=======================================================\n")
        return report


if __name__ == "__main__":
    pipeline = ChurnETLPipeline()
    pipeline.run()
