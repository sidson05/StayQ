# Production Subscription Churn Intelligence Platform

An enterprise-grade, end-to-end data engineering and machine learning platform that detects and prevents customer subscription churn. 

The platform continuously extracts from heterogeneous, messy silos (**Billing CSVs**, **Customer Support REST API**, and **Telemetry JSONL event logs**), enforces automated data quality contracts, stores data in a **Medallion PostgreSQL warehouse** (`raw` ➔ `staging` ➔ `analytics`), trains a cost-sensitive **XGBoost classification model** with **SHAP explainability**, and serves real-time churn predictions through a **FastAPI microservice** and an interactive **Streamlit Retention Cockpit**.

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
| **Logistic Regression (L2 Baseline)** | `0.9839` | `0.9422` |
| **Random Forest (Challenger)** | `0.9738` | `0.9230` |
| **XGBoost Classifier (Champion)** | **`0.9866`** | **`0.9555`** |

### Champion XGBoost Test Performance
- **ROC-AUC Score**: `0.9866`
- **PR-AUC**: `0.9555`
- **Optimal Cost-Sensitive Threshold**: `0.120` (tuned to maximize retention ROI)
- **Churn Recall (Catch Rate)**: `98.93%` (caught 277 out of 280 churners)
- **Precision**: `63.68%`
- **F1-Score**: `0.7748`
- **Accuracy**: `88.57%`
- **Brier Calibration Score**: `0.0394`
- **Test Confusion Matrix**:
  ```
               Predicted Active   Predicted Churn
  Actual Active      971                158
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

### Option A: Local Execution (No Docker Required)

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

### Option B: Docker Compose (1-Click Deployment)

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
