"""YAML readers used by configuration discovery."""

from pathlib import Path
from typing import Any

import yaml


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, returning an empty mapping for other YAML shapes."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}
