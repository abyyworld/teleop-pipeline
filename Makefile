.DEFAULT_GOAL := help
PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Install the core package plus dev tools
	$(BIN)/pip install -e '.[dev]'

.PHONY: install-all
install-all: $(BIN)/python ## Install everything (torch, mlflow, prefect, dvc)
	$(BIN)/pip install -e '.[dev,all]'

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest

.PHONY: lint
lint: ## Lint and format-check
	$(BIN)/ruff check src tests flows
	$(BIN)/ruff format --check src tests flows

.PHONY: fmt
fmt: ## Auto-format
	$(BIN)/ruff format src tests flows
	$(BIN)/ruff check --fix src tests flows

# DVC runs each stage command in a subshell. Invoking $(BIN)/dvc does *not* put
# the virtualenv's bin directory on PATH, so a stage running `python -m ...`
# resolves against the system Python — which on macOS has python3 but no
# `python`, and the stage dies with "command not found". Export the venv onto
# PATH for every target that shells out to a stage.
VENV_PATH := PATH="$(CURDIR)/$(VENV)/bin:$$PATH"

.PHONY: repro
repro: ## Rebuild everything DVC considers stale
	$(VENV_PATH) $(BIN)/dvc repro

.PHONY: repro-force
repro-force: ## Rebuild every stage, ignoring the cache
	$(VENV_PATH) $(BIN)/dvc repro --force

.PHONY: pipeline
pipeline: ## Run every stage directly, without DVC
	$(BIN)/teleop-pipeline synth --sessions 48 --seed 0
	$(BIN)/teleop-pipeline ingest
	$(BIN)/teleop-pipeline validate
	$(BIN)/teleop-pipeline score
	$(BIN)/teleop-pipeline dataset
	$(BIN)/teleop-pipeline train
	$(BIN)/teleop-pipeline eval
	$(BIN)/teleop-pipeline report

.PHONY: metrics
metrics: ## Show tracked metrics, and how they changed against HEAD
	$(BIN)/dvc metrics show
	@$(BIN)/dvc metrics diff HEAD 2>/dev/null || true

.PHONY: dag
dag: ## Print the pipeline DAG
	$(BIN)/dvc dag

.PHONY: ingest-flow
ingest-flow: ## Run the Prefect ingestion flow once
	$(BIN)/python flows/ingest_flow.py

.PHONY: ui
ui: ## Serve the MLflow UI against the local run store
	$(BIN)/mlflow ui --backend-store-uri file:./mlruns

.PHONY: clean
clean: ## Remove generated data, artifacts and reports
	rm -rf data artifacts reports mlruns .ingest_state.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: distclean
distclean: clean ## Also remove the virtualenv
	rm -rf $(VENV) .pytest_cache .ruff_cache
