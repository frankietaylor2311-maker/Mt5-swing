"""YAML config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def default_config_path() -> Path:
    return Path(__file__).with_name("default.yaml")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else default_config_path()
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data
