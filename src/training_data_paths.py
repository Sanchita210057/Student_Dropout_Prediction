"""Resolve local training sources with optional environment-variable overrides."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UCI_DATA_ENV = "CSE_UCI_DATASET_PATH"
BANGLADESH_DATA_ENV = "CSE_BANGLADESH_DATASET_PATH"


def configured_training_path(env_name: str, repository_default: str) -> Path:
    """Return an explicit local path or the historical in-repo fallback."""
    configured = os.environ.get(env_name, "").strip()
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT / repository_default


def require_training_source(path: str | Path, env_name: str, label: str) -> Path:
    """Fail with reproducible setup guidance when a private source is absent."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(
            f"{label} training data was not found at {source}. "
            f"Set {env_name} to an authorized local copy. Normal application "
            "predictions do not require raw training data."
        )
    return source


def uci_dataset_path() -> Path:
    return configured_training_path(UCI_DATA_ENV, "data/student_dropout.csv")


def bangladesh_dataset_path() -> Path:
    return configured_training_path(
        BANGLADESH_DATA_ENV,
        "data/Bangladesh University Research Responses.xlsx",
    )
