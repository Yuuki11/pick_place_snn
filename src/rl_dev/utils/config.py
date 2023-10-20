from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from rl_dev.constants import REPO_ROOT


def repo_relative(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_launcher_config(path: str | Path) -> dict[str, Any]:
    config_path = repo_relative(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping.")
    data = copy.deepcopy(data)
    data.setdefault("args", {})
    data.setdefault("oom_fallback_ladder", [])
    data["_config_path"] = config_path
    return data

