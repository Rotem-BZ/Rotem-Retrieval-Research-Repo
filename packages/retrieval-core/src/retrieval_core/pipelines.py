"""Helpers for turning Hydra config into Haystack pipelines."""

from __future__ import annotations

from typing import Any

import yaml
from haystack import AsyncPipeline
from omegaconf import DictConfig, ListConfig, OmegaConf


def to_container(config: Any) -> Any:
    """Resolve OmegaConf nodes into plain Python containers."""

    if isinstance(config, (DictConfig, ListConfig)):
        return OmegaConf.to_container(config, resolve=True)
    return config


def load_async_pipeline(pipeline_config: DictConfig | dict[str, Any]) -> AsyncPipeline:
    """Load an AsyncPipeline from a Hydra field using Haystack YAML syntax."""

    pipeline_dict = to_container(pipeline_config)
    pipeline_yaml = yaml.safe_dump(pipeline_dict, sort_keys=False)
    return AsyncPipeline.loads(pipeline_yaml)


def include_outputs(value: ListConfig | list[str] | None) -> set[str] | None:
    if not value:
        return None
    return set(to_container(value))
