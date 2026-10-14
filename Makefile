# Use '>' as recipe prefix to avoid tab/space issues when pasting.
.RECIPEPREFIX = >
.PHONY: help sync lint format test data validate clean-data features train-baseline tune-xgb clean all

PYTHON := uv run python
PYTEST := uv run pytest
RUFF   := uv run ruff

help:
> @echo "Hospital Readmission — available targets:"
> @echo "  sync            Install dependencies (uv sync --all-groups)"
> @echo "  lint            Run ruff check + format --check"
> @echo "  format          Auto-format (writes) + auto-fix lint"
> @echo "  test            Run pytest"
> @echo "  data            Download raw dataset (UCI, fallback Kaggle)"
> @echo "  validate        Validate raw dataset schema"
> @echo "  clean-data      Build data/processed/clean.parquet"
> @echo "  features        Build data/features/{train,val,test}.parquet"
> @echo "  train-baseline  Train XGBoost + LightGBM baselines"
> @echo "  tune-xgb        Optuna tuning for XGBoost (20 trials)"
> @echo "  clean           Remove caches and build artifacts"
> @echo "  all             lint + test"

sync:
> uv sync --all-groups

lint:
> $(RUFF) check src tests
> $(RUFF) format --check src tests

format:
> $(RUFF) format src tests
> $(RUFF) check --fix src tests

test:
> $(PYTEST) -v

data:
> $(PYTHON) -m src.data.download

validate:
> $(PYTHON) -m src.data.validate

clean-data:
> $(PYTHON) -m src.data.clean

features:
> $(PYTHON) -m src.features.clinical

train-baseline:
> $(PYTHON) -m src.models.train_baseline

tune-xgb:
> $(PYTHON) -m src.models.tune_optuna --n-trials 20

clean:
> rm -rf .pytest_cache .ruff_cache .coverage htmlcov
> find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: lint test
