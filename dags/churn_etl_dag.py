"""
Apache Airflow DAG: Automated Daily Churn ETL & Scoring Pipeline.
Orchestrates:
1. Multi-source extraction
2. Data quality checks
3. Warehouse load & SQL transforms
4. Batch Churn Scoring & Model Explainability
"""

from datetime import datetime, timedelta
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def task_extract_and_validate():
    from src.etl.extractors import BillingExtractor, SupportAPIExtractor, UsageLogsExtractor
    from src.etl.validators import DataValidator
    import yaml

    with open("config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Ingestion
    src = cfg["sources"]
    b_ext = BillingExtractor(src["billing_contracts_path"], src["billing_invoices_path"])
    s_ext = SupportAPIExtractor(src["support_api_url"], src["mock_tickets_db"])
    u_ext = UsageLogsExtractor(src["usage_logs_path"])

    c_df = b_ext.extract_contracts()
    i_df = b_ext.extract_invoices()
    t_df = s_ext.extract_tickets()
    u_df = u_ext.extract_usage_events()

    # Validation
    val = DataValidator()
    val.validate_dataset(c_df, "billing_contracts")
    val.validate_dataset(i_df, "billing_invoices")
    val.validate_dataset(t_df, "support_tickets")
    val.validate_dataset(u_df, "usage_events")
    val.validate_referential_integrity(c_df, i_df, t_df, u_df)
    print("Airflow Task: Extraction & Validation Passed.")


def task_run_warehouse_transforms():
    from src.etl.pipeline import ChurnETLPipeline
    pipeline = ChurnETLPipeline()
    report = pipeline.run()
    print(f"Airflow Task: Warehouse Transforms Complete. Status: {report['status']}")


def task_batch_score_customers():
    from src.ml.predict import batch_score_all_customers
    results = batch_score_all_customers()
    print(f"Airflow Task: Batch scoring complete for {len(results)} customers.")


default_args = {
    "owner": "data-engineering-team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

if AIRFLOW_AVAILABLE:
    dag = DAG(
        "daily_churn_etl_and_scoring_dag",
        default_args=default_args,
        description="Daily automated ETL pipeline ingesting billing, support tickets, and telemetry logs into PostgreSQL",
        schedule_interval="0 2 * * *", # Daily at 2:00 AM UTC
        catchup=False,
    )

    t1 = PythonOperator(
        task_id="extract_and_validate_sources",
        python_callable=task_extract_and_validate,
        dag=dag,
    )

    t2 = PythonOperator(
        task_id="load_and_transform_warehouse",
        python_callable=task_run_warehouse_transforms,
        dag=dag,
    )

    t3 = PythonOperator(
        task_id="batch_predict_and_score_churn",
        python_callable=task_batch_score_customers,
        dag=dag,
    )

    t1 >> t2 >> t3
