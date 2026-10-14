"""Generate notebooks/01_eda.ipynb programmatically.

We avoid hand-editing ipynb JSON in nano by building it from Python.
Run: uv run python scripts/make_eda_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("notebooks/01_eda.ipynb")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md(
        "# 01 — EDA: Diabetes 130-US hospitals\n\n"
        "Lightweight exploration of the **cleaned** dataset "
        "(`data/processed/clean.parquet`).\n"
        "Goal: sanity-check distributions before feature engineering.\n"
    ),
    code(
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from src.data.clean import CLEAN_PARQUET, TARGET_BIN\n\n"
        "df = pd.read_parquet(CLEAN_PARQUET)\n"
        "df.shape\n"
    ),
    md("## Target balance\n"),
    code("df[TARGET_BIN].value_counts(normalize=True).round(4)\n"),
    md("## Numeric columns — describe\n"),
    code(
        "num_cols = df.select_dtypes(include='number').columns.tolist()\n"
        "df[num_cols].describe().T\n"
    ),
    md("## Categorical cardinalities\n"),
    code(
        "cat_cols = df.select_dtypes(include='object').columns.tolist()\n"
        "card = df[cat_cols].nunique().sort_values(ascending=False)\n"
        "card.to_frame('n_unique')\n"
    ),
    md("## Key distributions\n"),
    code(
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "df['age'].value_counts().sort_index().plot(kind='bar', ax=axes[0], title='age')\n"
        "df['gender'].value_counts().plot(kind='bar', ax=axes[1], title='gender')\n"
        "df['race'].value_counts().plot(kind='bar', ax=axes[2], title='race')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
    md("## Target vs key categoricals (base rate check)\n"),
    code(
        "for col in ['age', 'gender', 'race']:\n"
        "    print(f'--- {col} ---')\n"
        "    print(df.groupby(col)[TARGET_BIN].agg(['mean', 'size']).round(4))\n"
        "    print()\n"
    ),
    md(
        "## Notes to carry over to Phase 4 (feature engineering)\n\n"
        "- `age` is a 10-year bucket string -> ordinal encode.\n"
        "- `diag_1/2/3` are ICD-9 codes -> group into top-N + 'Other'.\n"
        "- 11 % positive class -> keep PR-AUC as primary metric.\n"
    ),
]


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(nb, indent=1))
    print(f"✅ Wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
