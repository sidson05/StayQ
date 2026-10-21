"""
Database connection factory and warehouse manager.
Supports both PostgreSQL (production) and SQLite (local dev/fallback) seamlessly.
"""

import os
import yaml
from sqlalchemy import create_engine, text, inspect
import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}


class WarehouseManager:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.db_cfg = self.config.get("database", {})
        self.engine = None
        self.db_type = None
        self._init_connection()

    def _init_connection(self):
        target_db = self.db_cfg.get("db_type", "postgresql")
        host = self.db_cfg.get("host", "localhost")
        port = self.db_cfg.get("port", 5432)
        user = self.db_cfg.get("user", "postgres")
        pwd = self.db_cfg.get("password", "password")
        dbname = self.db_cfg.get("database", "churn_warehouse")
        sqlite_file = self.db_cfg.get("sqlite_path", "data/warehouse.db")

        # Try PostgreSQL first if requested
        if target_db == "postgresql":
            try:
                pg_url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{dbname}"
                eng = create_engine(pg_url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
                with eng.connect() as conn:
                    conn.execute(text("SELECT 1;"))
                self.engine = eng
                self.db_type = "postgresql"
                print(f"[Warehouse] Connected to PostgreSQL at {host}:{port}/{dbname}")
                return
            except Exception as e:
                print(f"[Warehouse] PostgreSQL connection failed ({e}). Falling back to SQLite for seamless execution.")

        # SQLite fallback
        os.makedirs(os.path.dirname(sqlite_file) or ".", exist_ok=True)
        sqlite_url = f"sqlite:///{sqlite_file}"
        self.engine = create_engine(sqlite_url)
        self.db_type = "sqlite"
        print(f"[Warehouse] Connected to SQLite warehouse at {sqlite_file}")

    def init_schema(self):
        """Create tables for raw, staging, and analytics."""
        with self.engine.begin() as conn:
            if self.db_type == "postgresql":
                ddl_path = os.path.join(os.path.dirname(__file__), "ddl.sql")
                if os.path.exists(ddl_path):
                    with open(ddl_path, "r") as f:
                        sql = f.read()
                    # Execute statements
                    for statement in sql.split(";"):
                        stmt = statement.strip()
                        if stmt:
                            conn.execute(text(stmt))
                    print("[Warehouse] PostgreSQL schemas and tables initialized.")
            else:
                # SQLite doesn't have native multi-schemas, use prefixed table names
                sqlite_ddl = """
                CREATE TABLE IF NOT EXISTS raw_contracts (
                    customer_id TEXT, contract_type TEXT, tenure_months INTEGER,
                    internet_service TEXT, tech_support TEXT, online_security TEXT,
                    streaming_tv TEXT, payment_method TEXT, paperless_billing TEXT,
                    monthly_charges REAL, total_charges REAL, signup_date TEXT,
                    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS raw_invoices (
                    invoice_id TEXT, customer_id TEXT, invoice_date TEXT,
                    amount REAL, payment_status TEXT, payment_delay_days INTEGER,
                    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS raw_support_tickets (
                    ticket_id TEXT, customer_id TEXT, created_at TEXT,
                    issue_type TEXT, priority TEXT, status TEXT,
                    resolution_time_hrs REAL, sentiment_score REAL,
                    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS raw_usage_events (
                    event_id TEXT, customer_id TEXT, event_date TEXT,
                    period_window_days INTEGER, data_usage_gb REAL,
                    login_count INTEGER, streaming_hours REAL, app_crashes INTEGER,
                    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS staging_dim_customers (
                    customer_id TEXT PRIMARY KEY, contract_type TEXT, tenure_months INTEGER,
                    internet_service TEXT, has_tech_support INTEGER, has_online_security INTEGER,
                    has_streaming_tv INTEGER, payment_method TEXT, is_paperless_billing INTEGER,
                    monthly_charges REAL, total_charges REAL, signup_date TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS staging_fct_invoices (
                    invoice_id TEXT PRIMARY KEY, customer_id TEXT, invoice_date TEXT,
                    amount REAL, payment_status TEXT, payment_delay_days INTEGER,
                    is_failed_payment INTEGER, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS staging_fct_support_tickets (
                    ticket_id TEXT PRIMARY KEY, customer_id TEXT, created_at TEXT,
                    issue_type TEXT, priority TEXT, status TEXT,
                    resolution_time_hrs REAL, sentiment_score REAL,
                    is_unresolved INTEGER, is_escalated INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS staging_fct_usage_events (
                    event_id TEXT PRIMARY KEY, customer_id TEXT, event_date TEXT,
                    data_usage_gb REAL, login_count INTEGER, streaming_hours REAL,
                    app_crashes INTEGER, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS analytics_mart_churn_features (
                    customer_id TEXT PRIMARY KEY,
                    contract_type TEXT, is_month_to_month INTEGER, tenure_months INTEGER,
                    internet_service TEXT, has_tech_support INTEGER, has_online_security INTEGER,
                    has_streaming_tv INTEGER, monthly_charges REAL, total_charges REAL,
                    is_electronic_check INTEGER, is_paperless_billing INTEGER,
                    total_invoices INTEGER, failed_invoices_count INTEGER,
                    failed_invoices_ratio REAL, avg_payment_delay_days REAL,
                    max_payment_delay_days INTEGER, total_tickets INTEGER,
                    unresolved_tickets_count INTEGER, escalated_tickets_count INTEGER,
                    critical_tickets_count INTEGER, avg_ticket_sentiment REAL,
                    has_cancellation_inquiry INTEGER, total_data_usage_gb REAL,
                    avg_biweekly_usage_gb REAL, total_logins INTEGER,
                    recent_usage_30d_gb REAL, prior_usage_60d_gb REAL,
                    usage_drop_velocity REAL, recent_logins_30d INTEGER,
                    total_crashes INTEGER, recent_crashes_30d INTEGER,
                    feature_generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                for statement in sqlite_ddl.strip().split(";"):
                    stmt = statement.strip()
                    if stmt:
                        conn.execute(text(stmt))
                print("[Warehouse] SQLite warehouse tables initialized.")

    def run_transformations(self):
        """Execute Silver (Staging) and Gold (Mart) SQL transforms."""
        transforms_path = os.path.join(os.path.dirname(__file__), "transforms.sql")
        with open(transforms_path, "r") as f:
            sql = f.read()

        with self.engine.begin() as conn:
            # Clear staging & analytics tables before refresh
            for tbl in ["staging_dim_customers", "staging_fct_invoices", "staging_fct_support_tickets", "staging_fct_usage_events", "analytics_mart_churn_features"]:
                conn.execute(text(f"DELETE FROM {tbl};"))
                
            # Execute statement by statement
            for statement in sql.split(";"):
                stmt = statement.strip()
                if stmt:
                    conn.execute(text(stmt))

        print("[Warehouse] Transformations executed successfully. Analytics mart refreshed.")

    def load_df(self, df: pd.DataFrame, table_name: str, if_exists: str = "append"):
        """Load pandas dataframe into warehouse table."""
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)

    def read_query(self, query: str) -> pd.DataFrame:
        """Run SQL query and return DataFrame."""
        return pd.read_sql_query(query, self.engine)


if __name__ == "__main__":
    wh = WarehouseManager()
    wh.init_schema()
