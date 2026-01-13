.PHONY: help install test lint format clean run-example milvus-start milvus-stop

help:
	@echo "ADS-MRAG - Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make milvus-start  - Start Milvus with Docker"
	@echo "  make milvus-stop   - Stop Milvus"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean up generated files"
	@echo "  make run-example   - Run basic example"
	@echo "  make reset-db      - Reset vector database"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 src/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ examples/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

run-example:
	python examples/basic_usage.py

reset-db:
	python scripts/reset_vectorstore.py --confirm

ingest-sample:
	python scripts/ingest_documents.py --source data/raw/sample_document.txt --tags "sample,tutorial,rag"

query-sample:
	python scripts/query_pipeline.py --query "What is RAG?" --tags sample --show-sources

# Milvus management
milvus-start:
	docker-compose up -d
	@echo "Waiting for Milvus to be ready..."
	@sleep 10
	@echo "Milvus is ready!"

milvus-stop:
	docker-compose down

milvus-logs:
	docker-compose logs -f milvus

milvus-reset:
	docker-compose down -v
	docker-compose up -d
