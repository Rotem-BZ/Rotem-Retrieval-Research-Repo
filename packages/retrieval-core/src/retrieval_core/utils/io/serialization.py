"""Conversions between framework objects and serializable values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def config_to_yaml(config: DictConfig) -> str:
    """Render a resolved OmegaConf configuration as YAML."""

    return OmegaConf.to_yaml(config, resolve=True)


def to_jsonable(value: Any) -> Any:
    """Recursively convert common framework values to JSON-compatible values."""

    if isinstance(value, (DictConfig, ListConfig)):
        return to_jsonable(OmegaConf.to_container(value, resolve=True))

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value
