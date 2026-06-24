# ===========================================================================
# Agentic Profile Matching — developer task runner
# ===========================================================================
.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install install-dev seed ingest chat ui pipeline screen test lint format typecheck clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev: ## Install runtime + dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

seed: ## Generate synthetic sample resumes + job descriptions
	$(PYTHON) -m scripts.generate_sample_data

ingest: ## Build / refresh the resume vector index
	$(PYTHON) -m scripts.ingest

chat: ## Launch the conversational CLI
	$(PYTHON) -m profile_matching.cli.chat

ui: ## Launch the Streamlit chat UI
	streamlit run src/profile_matching/app/streamlit_app.py

pipeline: ## Run the end-to-end matching pipeline on a sample JD (non-interactive)
	$(PYTHON) -m scripts.run_pipeline

screen: ## Run the multi-round screening (Part C) on a sample JD (non-interactive)
	$(PYTHON) -m scripts.run_screening

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Lint with ruff
	ruff check src tests

format: ## Auto-format with ruff
	ruff format src tests
	ruff check --fix src tests

typecheck: ## Static type-check with mypy
	mypy src

clean: ## Remove caches, vector store, checkpoints, reports
	rm -rf .pytest_cache .ruff_cache .mypy_cache chroma_db checkpoints reports logs
	find . -type d -name __pycache__ -exec rm -rf {} +
