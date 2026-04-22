from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _resolve_paths(value: Any, base_dir: Path, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        return {key: _resolve_paths(item, base_dir, key) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base_dir, parent_key) for item in value]
    if isinstance(value, str) and (parent_key.endswith("_path") or parent_key.endswith("_dir")):
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        return str(candidate)
    return value


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config = _resolve_paths(config, config_path.parent)
    config["_config_path"] = str(config_path)
    return config
