.PHONY: help test build run-wf1 register-dev register-prod init-db clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

test: ## Run tests
	pytest tests/ -v --cov --cov-report=term-missing

build: ## Build Docker image (ARM64)
	docker buildx build --platform linux/arm64 -t quant-trading-workflows:dev .

run-wf1: ## Run WF1 data ingestion locally
	pyflyte run src/wf1_data_ingestion/workflow.py data_ingestion_workflow

register-dev: ## Register workflows to development
	pyflyte register . \
		--project quant-trading \
		--domain development \
		--image ghcr.io/biomechanoid-de/quant-trading-workflows:dev

register-prod: ## Register workflows to production
	pyflyte register . \
		--project quant-trading \
		--domain production \
		--image ghcr.io/biomechanoid-de/quant-trading-workflows:latest

init-db: ## Initialize PostgreSQL schema
	psql -h $${DB_HOST:-192.168.178.45} -U $${DB_USER:-flyte} -d $${DB_NAME:-quant_trading} -f sql/schema.sql

clean: ## Clean up
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
