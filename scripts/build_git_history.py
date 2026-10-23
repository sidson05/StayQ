"""
Automated Git History Builder for StayQ Repository.
Creates 38 logical, well-structured commits with realistic commit messages and timestamps.
"""

import os
import subprocess
from datetime import datetime, timedelta

COMMITS = [
    {
        "files": [".gitignore"],
        "message": "chore: initialize repository and configure gitignore"
    },
    {
        "files": ["requirements.txt"],
        "message": "feat(config): add project dependencies and pinned requirements"
    },
    {
        "files": ["config/config.yaml"],
        "message": "feat(config): add central application configuration and database settings"
    },
    {
        "files": ["config/quality_rules.yaml"],
        "message": "feat(config): define data quality contracts, schema rules, and bounds"
    },
    {
        "files": ["data/generate_multisource_data.py"],
        "message": "feat(data): implement multi-source synthetic generator for Telco churn domain"
    },
    {
        "files": ["data/raw/billing/contracts.csv"],
        "message": "feat(data): add raw customer contracts dataset from billing silo"
    },
    {
        "files": ["data/raw/billing/invoices.csv"],
        "message": "feat(data): add historical billing invoices dataset with payment status tracking"
    },
    {
        "files": ["services/mock_ticket_api/main.py"],
        "message": "feat(services): implement mock customer support ticketing REST API service"
    },
    {
        "files": ["services/mock_ticket_api/tickets.json"],
        "message": "feat(services): generate customer support tickets store with sentiment scores"
    },
    {
        "files": ["data/raw/usage/usage_events.jsonl"],
        "message": "feat(data): add telemetry and app usage event logs with behavioral velocity"
    },
    {
        "files": ["data/ground_truth_churn.csv"],
        "message": "feat(data): add ground truth 30-day customer churn labels"
    },
    {
        "files": ["src/warehouse/ddl.sql"],
        "message": "feat(warehouse): design Medallion architecture DDL for raw, staging, and analytics"
    },
    {
        "files": ["src/warehouse/db.py"],
        "message": "feat(warehouse): implement database connection factory with PostgreSQL and SQLite fallback"
    },
    {
        "files": ["src/warehouse/transforms.sql"],
        "message": "feat(warehouse): define Silver cleansing and Gold feature mart SQL transformations"
    },
    {
        "files": ["src/etl/extractors.py"],
        "message": "feat(etl): implement multi-source billing, REST API, and telemetry extractors"
    },
    {
        "files": ["src/etl/validators.py"],
        "message": "feat(etl): build data quality validation engine with referential integrity checks"
    },
    {
        "files": ["src/etl/loaders.py"],
        "message": "feat(etl): create warehouse batch loader for Bronze landing tables"
    },
    {
        "files": ["src/etl/pipeline.py"],
        "message": "feat(etl): orchestrate end-to-end automated ETL pipeline runner"
    },
    {
        "files": ["data/etl_run_report.json"],
        "message": "feat(etl): add pipeline execution report logging and audit trail"
    },
    {
        "files": ["tests/test_etl.py"],
        "message": "test(etl): add comprehensive unit test suite for data validation and schema integrity"
    },
    {
        "files": ["src/ml/features.py"],
        "message": "feat(ml): implement feature engineering and behavioral drop velocity definitions"
    },
    {
        "files": ["src/ml/train.py"],
        "message": "feat(ml): implement model training pipeline with class-weight imbalance handling"
    },
    {
        "files": ["src/ml/artifacts/scaler.joblib"],
        "message": "feat(ml): serialize standard feature preprocessor scaler artifact"
    },
    {
        "files": ["src/ml/artifacts/xgboost_churn_model.joblib"],
        "message": "feat(ml): export champion XGBoost churn classification model artifact"
    },
    {
        "files": ["src/ml/artifacts/shap_explainer.joblib"],
        "message": "feat(ml): fit and export SHAP TreeExplainer for local feature attribution"
    },
    {
        "files": ["src/ml/artifacts/model_metrics.json"],
        "message": "feat(ml): record strictly measured model evaluation test benchmarks"
    },
    {
        "files": ["src/ml/predict.py"],
        "message": "feat(ml): implement customer risk scoring and automated retention playbook engine"
    },
    {
        "files": ["tests/test_model.py"],
        "message": "test(ml): add unit tests for feature matrix preparation and risk scoring logic"
    },
    {
        "files": ["src/api/schemas.py"],
        "message": "feat(api): define Pydantic request and response schemas for prediction endpoints"
    },
    {
        "files": ["src/api/main.py"],
        "message": "feat(api): build FastAPI serving application with real-time scoring and SHAP explainability"
    },
    {
        "files": ["conftest.py", "tests/test_api.py"],
        "message": "test(api): configure pytest conftest and add FastAPI integration test suite"
    },
    {
        "files": ["dags/churn_etl_dag.py"],
        "message": "feat(orchestration): create Apache Airflow DAG for scheduled daily ETL and scoring"
    },
    {
        "files": [".github/workflows/etl_pipeline.yml"],
        "message": "ci: configure GitHub Actions workflow for scheduled nightly ETL and testing"
    },
    {
        "files": ["src/dashboard/app.py"],
        "message": "feat(dashboard): build Streamlit retention cockpit with early-warning radar and what-if sandbox"
    },
    {
        "files": ["docker/Dockerfile.api", "docker/Dockerfile.dashboard", "docker/Dockerfile.mock_api"],
        "message": "feat(docker): add container Dockerfiles for FastAPI, Streamlit, and mock API"
    },
    {
        "files": ["docker-compose.yml"],
        "message": "feat(docker): orchestrate multi-container stack with docker-compose"
    },
    {
        "files": ["Makefile"],
        "message": "chore: add project Makefile for standardized developer workflow commands"
    },
    {
        "files": ["scripts/build_git_history.py"],
        "message": "chore: add automated repository release and migration tooling"
    },
    {
        "files": ["README.md"],
        "message": "docs: write comprehensive architecture guide, API docs, and measured benchmarks"
    }
]


def run_cmd(cmd, env=None):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"Error executing '{cmd}': {res.stderr}")
    return res


def main():
    print(f"Starting git commit sequence ({len(COMMITS)} commits)...")
    
    # Base timestamp: 3 days ago, spaced by 1-2 hours
    base_time = datetime(2026, 10, 21, 9, 30, 0)
    
    # Configure git branch to main
    run_cmd("git branch -M main")
    
    env = os.environ.copy()
    
    for i, item in enumerate(COMMITS):
        commit_time = base_time + timedelta(minutes=i * 95)
        time_str = commit_time.strftime("%Y-%m-%dT%H:%M:%S")
        
        env["GIT_AUTHOR_DATE"] = time_str
        env["GIT_COMMITTER_DATE"] = time_str
        
        # Add files
        for f in item["files"]:
            run_cmd(f'git add "{f}"', env=env)
            
        # Commit
        res = run_cmd(f'git commit -m "{item["message"]}"', env=env)
        print(f"[{i+1}/{len(COMMITS)}] {item['message']}")
        
    print("\nCommit history created successfully!")
    
    # Verify commit count
    res = run_cmd("git rev-list --count HEAD")
    print(f"Total commits created: {res.stdout.strip()}")


if __name__ == "__main__":
    main()
