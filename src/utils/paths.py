"""Centralized project paths.

All paths are resolved relative to the project root so that scripts can be
run from anywhere without changing directory.
"""

from __future__ import annotations

from pathlib import Path

# Project root = three levels up from this file
# src/utils/paths.py -> src/utils -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Data
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
FEATURES_DIR: Path = DATA_DIR / "features"

# Artifacts
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Source tree
SRC_DIR: Path = PROJECT_ROOT / "src"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"


def ensure_dirs() -> None:
    """Create all standard project directories if they do not exist."""
    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        FEATURES_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "DATA_DIR",
    "FEATURES_DIR",
    "FIGURES_DIR",
    "MODELS_DIR",
    "NOTEBOOKS_DIR",
    "PROCESSED_DIR",
    "PROJECT_ROOT",
    "RAW_DIR",
    "REPORTS_DIR",
    "SRC_DIR",
    "ensure_dirs",
]
