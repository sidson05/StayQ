.PHONY: help data etl train api dashboard test docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make data       - Generate synthetic multi-source Telco churn dataset"
	@echo "  make etl        - Run end-to-end automated ETL pipeline into warehouse"
	@echo "  make train      - Train XGBoost churn model, compute SHAP and test metrics"
	@echo "  make mock-api   - Run mock customer support ticketing API service"
	@echo "  make api        - Launch FastAPI churn prediction serving endpoints"
	@echo "  make dashboard  - Launch Streamlit Retention Cockpit dashboard"
	@echo "  make test       - Run pytest test suite"
	@echo "  make docker-up  - Start full environment via Docker Compose"
	@echo "  make docker-down- Stop all Docker Compose services"

data:
	python data/generate_multisource_data.py

etl:
	python src/etl/pipeline.py

train:
	python src/ml/train.py

mock-api:
	uvicorn services.mock_ticket_api.main:app --host 0.0.0.0 --port 8001 --reload

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

test:
	pytest tests/ -v

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v
