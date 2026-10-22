"""
Data Extraction Modules for Multi-Source Ingestion.
Extracts:
1. Billing System: CSV files (contracts and invoices)
2. Support System: REST API (Mock FastAPI) with fallback to JSON store
3. App Telemetry: JSONL event stream logs
"""

import os
import json
import requests
import pandas as pd
from typing import Dict, Any, List


class BillingExtractor:
    """Extracts billing and contract datasets from CSV sources."""

    def __init__(self, contracts_path: str, invoices_path: str):
        self.contracts_path = contracts_path
        self.invoices_path = invoices_path

    def extract_contracts(self) -> pd.DataFrame:
        if not os.path.exists(self.contracts_path):
            raise FileNotFoundError(f"Contracts file not found at: {self.contracts_path}")
        df = pd.read_csv(self.contracts_path)
        print(f"[ETL Extractor] Loaded {len(df)} contract records from {self.contracts_path}")
        return df

    def extract_invoices(self) -> pd.DataFrame:
        if not os.path.exists(self.invoices_path):
            raise FileNotFoundError(f"Invoices file not found at: {self.invoices_path}")
        df = pd.read_csv(self.invoices_path)
        print(f"[ETL Extractor] Loaded {len(df)} invoice records from {self.invoices_path}")
        return df


class SupportAPIExtractor:
    """Extracts customer support tickets from the Mock REST API with HTTP pagination and fallback."""

    def __init__(self, api_url: str, fallback_json_path: str):
        self.api_url = api_url
        self.fallback_json_path = fallback_json_path

    def extract_tickets(self, batch_size: int = 2000) -> pd.DataFrame:
        tickets = []
        try:
            # Check if API is healthy
            health_url = self.api_url.rsplit("/api/", 1)[0] + "/health"
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                print(f"[ETL Extractor] Connecting to Live Support Ticket API: {self.api_url}")
                offset = 0
                while True:
                    url = f"{self.api_url}?limit={batch_size}&offset={offset}"
                    r = requests.get(url, timeout=5)
                    r.raise_for_status()
                    batch = r.json()
                    if not batch:
                        break
                    tickets.extend(batch)
                    offset += len(batch)
                print(f"[ETL Extractor] Successfully fetched {len(tickets)} tickets via REST API.")
                return pd.DataFrame(tickets)
        except Exception as e:
            print(f"[ETL Extractor] REST API unavailable ({e}). Using local mock store: {self.fallback_json_path}")

        # Fallback to direct JSON storage
        if not os.path.exists(self.fallback_json_path):
            raise FileNotFoundError(f"Fallback ticket store not found: {self.fallback_json_path}")
        with open(self.fallback_json_path, "r") as f:
            tickets = json.load(f)
        print(f"[ETL Extractor] Loaded {len(tickets)} tickets from local JSON store.")
        return pd.DataFrame(tickets)


class UsageLogsExtractor:
    """Extracts telemetry and app usage event logs from JSONL streams."""

    def __init__(self, usage_logs_path: str):
        self.usage_logs_path = usage_logs_path

    def extract_usage_events(self) -> pd.DataFrame:
        if not os.path.exists(self.usage_logs_path):
            raise FileNotFoundError(f"Usage logs file not found: {self.usage_logs_path}")
        
        records = []
        with open(self.usage_logs_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
        print(f"[ETL Extractor] Loaded {len(df)} usage event records from JSONL stream.")
        return df
