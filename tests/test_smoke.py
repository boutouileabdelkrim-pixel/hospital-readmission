"""Smoke tests: verify the package imports and basic paths resolve.

These tests must pass on a fresh clone with no data and no trained models.
"""

from __future__ import annotations

from src.utils.paths import (
    DATA_DIR,
    FEATURES_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    REPORTS_DIR,
    ensure_dirs,
)


def test_project_root_exists() -> None:
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_project_root_contains_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_data_subdirs_paths() -> None:
    assert DATA_DIR.name == "data"
    assert RAW_DIR.parent == DATA_DIR
    assert PROCESSED_DIR.parent == DATA_DIR
    assert FEATURES_DIR.parent == DATA_DIR


def test_artifacts_paths() -> None:
    assert MODELS_DIR.name == "models"
    assert REPORTS_DIR.name == "reports"
    assert FIGURES_DIR.parent == REPORTS_DIR


def test_ensure_dirs_creates_all() -> None:
    ensure_dirs()
    for directory in (RAW_DIR, PROCESSED_DIR, FEATURES_DIR, MODELS_DIR, FIGURES_DIR):
        assert directory.exists()
        assert directory.is_dir()
