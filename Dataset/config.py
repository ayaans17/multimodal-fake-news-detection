from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs() -> None:
    for folder in ("data/processed", "embeddings", "models", "results"):
        project_path(folder).mkdir(parents=True, exist_ok=True)

