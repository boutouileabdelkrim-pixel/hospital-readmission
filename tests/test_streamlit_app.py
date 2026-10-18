"""Smoke tests for the Streamlit app (offline, no server launched)."""

from __future__ import annotations

import ast
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
APP_PY = APP_DIR / "streamlit_app.py"
REQ = APP_DIR / "requirements.txt"
DATA_DIR = APP_DIR / "data"


def test_app_py_exists() -> None:
    assert APP_PY.is_file(), f"Missing {APP_PY}"


def test_app_py_parses() -> None:
    ast.parse(APP_PY.read_text())


def test_app_py_has_expected_sections() -> None:
    src = APP_PY.read_text()
    for token in ("Single patient", "Cost analysis", "Model card", "shap", "xgboost"):
        assert token in src, f"Missing token: {token}"


def test_requirements_exist_with_core_deps() -> None:
    assert REQ.is_file()
    content = REQ.read_text()
    for pkg in ("streamlit", "xgboost", "shap", "pandas"):
        assert pkg in content, f"Missing dep: {pkg}"


def test_app_data_files_exist() -> None:
    assert DATA_DIR.is_dir(), f"Missing {DATA_DIR}"
    for name in ("xgb_tuned.json", "test.parquet", "fairness_report.json"):
        path = DATA_DIR / name
        assert path.is_file(), f"Missing {path}"
        assert path.stat().st_size > 0, f"Empty file: {path}"
