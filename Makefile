# Use '>' as recipe prefix to avoid tab/space issues when pasting.
.RECIPEPREFIX = >
.PHONY: help sync lint format test data validate clean-data clean all

PYTHON := uv run python
PYTEST := uv run pytest
RUFF   := uv run ruff

help:
> @echo "Hospital Readmission — available targets:"
> @echo "  sync        Install dependencies (uv sync --all-groups)"
> @echo "  lint        Run ruff check + format --check"
> @echo "  format      Auto-format (writes) + auto-fix lint"
> @echo "  test        Run pytest"
> @echo "  data        Download raw dataset (UCI, fallback Kaggle)"
> @echo "  validate    Validate raw dataset schema"
> @echo "  clean-data  Build data/processed/clean.parquet"
> @echo "  clean       Remove caches and build artifacts"
> @echo "  all         lint + test"

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

clean:
> rm -rf .pytest_cache .ruff_cache .coverage htmlcov
> find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: lint test
