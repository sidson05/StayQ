"""
Warehouse Loader Module.
Loads validated data into Raw (Bronze) tables using batch loading.
"""

import pandas as pd
from src.warehouse.db import WarehouseManager


class WarehouseLoader:
    """Loads validated DataFrames into warehouse tables."""

    def __init__(self, wh: WarehouseManager):
        self.wh = wh

    def load_raw_tables(
        self,
        contracts_df: pd.DataFrame,
        invoices_df: pd.DataFrame,
        tickets_df: pd.DataFrame,
        usage_df: pd.DataFrame
    ):
        """Truncate and reload Raw (Bronze) tables with newly extracted and validated data."""
        self.wh.load_df(contracts_df, "raw_contracts", if_exists="replace")
        print(f"[Warehouse Loader] Loaded {len(contracts_df)} rows into raw_contracts")

        self.wh.load_df(invoices_df, "raw_invoices", if_exists="replace")
        print(f"[Warehouse Loader] Loaded {len(invoices_df)} rows into raw_invoices")

        self.wh.load_df(tickets_df, "raw_support_tickets", if_exists="replace")
        print(f"[Warehouse Loader] Loaded {len(tickets_df)} rows into raw_support_tickets")

        self.wh.load_df(usage_df, "raw_usage_events", if_exists="replace")
        print(f"[Warehouse Loader] Loaded {len(usage_df)} rows into raw_usage_events")
