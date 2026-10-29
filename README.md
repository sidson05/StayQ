# StayQ: Production Subscription Churn Intelligence Platform

[![CI/CD Pipeline](https://github.com/sidson05/StayQ/actions/workflows/etl_pipeline.yml/badge.svg)](https://github.com/sidson05/StayQ/actions)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=sidson05/StayQ&branch=main&mainModule=src/dashboard/app.py)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PostgreSQL 15](https://img.shields.io/badge/PostgreSQL-15-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-2.8-017CEE.svg?logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-EB5424.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **One-Line Overview**: An enterprise-grade, end-to-end data platform demonstrating the complete Data Scientist / Data Engineer lifecycle: multi-source automated ETL, Medallion data warehouse in PostgreSQL, cost-sensitive XGBoost 30-day churn modeling with SHAP explainability, FastAPI microservice serving, and an interactive Nordic Glassmorphism Streamlit retention cockpit.

---

### 🎯 Skills & Architecture Highlights for Recruiters
- **Data Engineering**: Multi-source batch ingestion (**Billing CSVs**, **Customer Support REST API**, **App Telemetry JSONL**), automated data quality contracts ([`quality_rules.yaml`](config/quality_rules.yaml)), referential integrity enforcement, and Medallion architecture (`raw` ➔ `staging` ➔ `analytics_mart_churn_features`).
- **Machine Learning**: Class imbalance calibration (`scale_pos_weight`), feature engineering (behavioral velocity ratios, ticket escalation velocity), TreeExplainer SHAP local attributions, and cost-sensitive ROI threshold tuning (**0.120** cutoff).
- **Strict Test Set Benchmarks**: **0.9869 ROC-AUC**, **0.9562 PR-AUC**, and **98.93% Churn Recall** (catching 277 / 280 churners) on held-out test data.
- **Production Serving & Orchestration**: High-throughput FastAPI endpoints, Streamlit "what-if" retention simulator, nightly scheduled Apache Airflow DAG ([`churn_etl_dag.py`](dags/churn_etl_dag.py)), and automated GitHub Actions CI/CD with PostgreSQL service containers.

---

## 1. System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │          Multi-Source Ingestion Silos        │
                       │ 1. Billing System (Contracts & Invoices CSV) │
                       │ 2. Support Ticket REST API (FastAPI Mock)    │
                       │ 3. App Telemetry & Usage Logs (JSONL)        │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │     Data Quality Validation & Cleansing      │
                       │  • Schema constraint enforcement             │
                       │  • Out-of-bounds & category checking         │
                       │  • Referential integrity cross-system check  │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │      PostgreSQL Warehouse (Medallion)        │
                       │  • raw: Bronze raw landed data               │
                       │  • staging: Silver deduplicated & typed      │
                       │  • analytics: Gold mart (churn features)     │
                       └──────────────────────┬───────────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────────┐
                       │       Machine Learning Modeling Pipeline     │
                       │  • Behavioral drop velocity features         │
                       │  • Class imbalance handling (scale_pos_wt)   │
                       │  • Baseline vs Random Forest vs XGBoost      │
                       │  • TreeExplainer SHAP attribution            │
                       │  • Cost-sensitive ROI threshold tuning       │
                       └──────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         ┌────────────────────────┐                      ┌────────────────────────┐
         │     FastAPI Server     │                      │  Streamlit Retention   │
         │  GET  /health          │                      │        Cockpit         │
         │  GET  /customers/{id}  │◄─────────────────────│  • Executive KPIs      │
         │  GET  /retention/high  │                      │  • Early-Warning Table │
         │  POST /predict         │                      │  • Local SHAP Waterfall│
         │  POST /pipeline/trigger│                      │  • "What-If" Simulator │
         └────────────────────────┘                      └────────────────────────┘
```

---

## 2. Tech Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Data Ingestion & ETL** | Python, Pandas, Requests | Multi-source batch ingestion, schema validation, streaming JSON parser |
| **Data Warehouse** | PostgreSQL (with SQLite zero-setup fallback) | Star-schema/Medallion storage (`raw`, `staging`, `analytics_mart_churn_features`) |
| **Pipeline Orchestration** | Apache Airflow & GitHub Actions | Scheduled daily ETL DAG (`dags/churn_etl_dag.py`) and CI/CD workflow |
| **Machine Learning** | Scikit-learn, XGBoost, SHAP | Feature engineering, class-weight balancing, model explainability, threshold tuning |
| **API Serving** | FastAPI, Uvicorn, Pydantic | Real-time REST endpoints for risk lookup, batch predictions, and pipeline trigger |
| **Frontend Cockpit** | Streamlit, Plotly | Interactive retention team dashboard with SHAP waterfall and "what-if" simulator |
| **DevOps & Containers** | Docker, Docker Compose | Reproducible multi-service deployment |

---

## 3. Strictly Measured Test Benchmarks

All metrics below are **strictly measured from the held-out 20% test split** ($N = 1,409$ customers; baseline churn rate: $19.89\%$):

### Model Comparison
| Model Candidate | ROC-AUC | PR-AUC (Avg Precision) |
| :--- | :---: | :---: |
| **Logistic Regression (L2 Baseline)** | `0.9840` | `0.9419` |
| **Random Forest (Challenger)** | `0.9734` | `0.9217` |
| **XGBoost Classifier (Champion)** | **`0.9869`** | **`0.9562`** |

### Champion XGBoost Test Performance
- **ROC-AUC Score**: `0.9869`
- **PR-AUC**: `0.9562`
- **Optimal Cost-Sensitive Threshold**: `0.120` (tuned to maximize retention ROI)
- **Churn Recall (Catch Rate)**: `98.93%` (caught 277 out of 280 churners)
- **Precision**: `63.53%`
- **F1-Score**: `0.7737`
- **Accuracy**: `88.50%`
- **Brier Calibration Score**: `0.0388`
- **Test Confusion Matrix**:
  ```
               Predicted Active   Predicted Churn
  Actual Active      970                159
  Actual Churn         3                277
  ```

---

## 4. Directory Layout

```
etl-pipelines/
├── .github/workflows/          # Scheduled GitHub Actions ETL workflow
├── config/
│   ├── config.yaml             # Central configuration (DB, API ports, thresholds)
│   └── quality_rules.yaml      # Data validation rules & boundary constraints
├── dags/
│   └── churn_etl_dag.py        # Apache Airflow scheduled ETL DAG
├── data/
│   ├── raw/
│   │   ├── billing/            # contracts.csv, invoices.csv
│   │   └── usage/              # usage_events.jsonl
│   ├── generate_multisource_data.py # Synthesizes & splits Telco dataset
│   └── etl_run_report.json     # Automated ETL run audit report
├── docker/
│   ├── Dockerfile.api          # FastAPI container
│   ├── Dockerfile.dashboard    # Streamlit container
│   ├── Dockerfile.mock_api     # Mock Ticket API container
│   └── docker-compose.yml      # Orchestrates Postgres, Mock API, FastAPI, Streamlit
├── services/
│   └── mock_ticket_api/        # FastAPI mock ticketing microservice
├── src/
│   ├── etl/
│   │   ├── extractors.py       # Billing, Support API, and JSONL extractors
│   │   ├── validators.py       # Data quality validation & referential integrity
│   │   ├── loaders.py          # Batch raw loader
│   │   └── pipeline.py         # End-to-end orchestrated ETL runner
│   ├── warehouse/
│   │   ├── db.py               # Database manager (Postgres + SQLite fallback)
│   │   ├── ddl.sql             # Medallion schemas DDL (Bronze, Silver, Gold)
│   │   └── transforms.sql      # Staging cleansing & feature mart aggregation
│   ├── ml/
│   │   ├── features.py         # Feature definitions and velocity formulas
│   │   ├── train.py            # Model training, benchmarking & SHAP calculation
│   │   ├── predict.py          # Customer scoring, SHAP local explainers & playbooks
│   │   └── artifacts/          # Saved model.joblib, explainer, metrics JSON
│   ├── api/
│   │   ├── main.py             # FastAPI production application
│   │   └── schemas.py          # Pydantic request/response models
│   └── dashboard/
│       └── app.py              # Streamlit Retention Cockpit
├── tests/
│   ├── test_etl.py             # ETL validation & integrity unit tests
│   ├── test_model.py           # Feature engineering & risk scoring unit tests
│   └── test_api.py             # FastAPI TestClient endpoint integration tests
├── Makefile                    # Developer CLI commands
└── requirements.txt            # Python dependencies
```

---

## 5. Quickstart Guide

### Option A: 1-Click Cloud Deployment (Streamlit Community Cloud)

Deploy the live retention dashboard with a single click (100% free, no server required):

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=sidson05/StayQ&branch=main&mainModule=src/dashboard/app.py)

- **Repository**: `sidson05/StayQ`
- **Branch**: `main`
- **Main file path**: `src/dashboard/app.py`
*(The dashboard automatically self-bootstraps its SQLite warehouse and ML scoring upon first launch).*

---

### Option B: Local Execution (No Docker Required)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate the Multi-Source Raw Datasets**:
   ```bash
   python data/generate_multisource_data.py
   ```

3. **Run the Automated ETL Pipeline**:
   ```bash
   python src/etl/pipeline.py
   ```
   *Ingests all 3 silos, validates contracts, loads warehouse, and builds the Gold analytics mart (`analytics_mart_churn_features`).*

4. **Train the ML Model & Compute Real Metrics**:
   ```bash
   python src/ml/train.py
   ```

5. **Start the FastAPI Serving Layer**:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   *Interactive Swagger docs available at `http://localhost:8000/docs`.*

6. **Start the Streamlit Retention Cockpit**:
   ```bash
   streamlit run src/dashboard/app.py
   ```
   *Opens dashboard in your browser at `http://localhost:8501`.*

---

### Option C: Docker Compose (Full-Stack Containerized Deployment)

Start PostgreSQL, Mock Ticket API, FastAPI Server, and Streamlit Dashboard simultaneously:

```bash
docker compose up --build -d
```

- **Streamlit Retention Cockpit**: `http://localhost:8501`
- **FastAPI Documentation**: `http://localhost:8000/docs`
- **Mock Ticket API**: `http://localhost:8001/docs`
- **PostgreSQL Warehouse**: `localhost:5432` (User: `postgres`, DB: `churn_warehouse`)

---

## 6. API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Healthcheck, warehouse connection type, and model status |
| `GET` | `/api/v1/customers/{customer_id}/risk` | 30-day churn probability, risk tier (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), top 3 SHAP drivers, and playbook recommendation |
| `GET` | `/api/v1/retention/high-risk` | Filterable list of top at-risk accounts sorted by expected revenue loss |
| `POST` | `/api/v1/predict` | Real-time churn scoring for arbitrary customer feature payloads |
| `GET` | `/api/v1/metrics` | Returns strictly measured model performance metrics |
| `POST` | `/api/v1/pipeline/trigger` | Trigger on-demand ETL execution to refresh warehouse tables |

---

## 7. Testing

Run the automated pytest test suite covering ETL validation, feature engineering, and API endpoints:

```bash
pytest tests/ -v
```
