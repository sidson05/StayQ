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
        table_prefix = "" if self.wh.db_type == "sqlite" else "raw."
        
        # Load contracts
        t_contracts = f"{table_prefix}raw_contracts" if self.wh.db_type == "postgresql" else "raw_contracts"
        self.wh.load_df(contracts_df, t_contracts, if_exists="replace")
        print(f"[Warehouse Loader] Loaded {len(contracts_df)} rows into {t_contracts}")

        # Load invoices
        t_invoices = f"{table_prefix}raw_invoices" if self.wh.db_type == "postgresql" else "raw_invoices"
        self.wh.load_df(invoices_df, t_invoices, if_exists="replace")
        print(f"[Warehouse Loader] Loaded {len(invoices_df)} rows into {t_invoices}")

        # Load tickets
        t_tickets = f"{table_prefix}raw_support_tickets" if self.wh.db_type == "postgresql" else "raw_support_tickets"
        self.wh.load_df(tickets_df, t_tickets, if_exists="replace")
        print(f"[Warehouse Loader] Loaded {len(tickets_df)} rows into {t_tickets}")

        # Load usage events
        t_usage = f"{table_prefix}raw_usage_events" if self.wh.db_type == "postgresql" else "raw_usage_events"
        self.wh.load_df(usage_df, t_usage, if_exists="replace")
        print(f"[Warehouse Loader] Loaded {len(usage_df)} rows into {t_usage}")
