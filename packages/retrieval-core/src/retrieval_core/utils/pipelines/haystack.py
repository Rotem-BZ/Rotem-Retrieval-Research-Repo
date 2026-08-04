"""Helpers for turning Hydra config into Haystack pipelines."""

from __future__ import annotations

from copy import deepcopy
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


def without_component_progress_bars(
    pipeline_config: DictConfig | dict[str, Any],
) -> dict[str, Any]:
    """Copy a pipeline config and disable bars owned by individual components."""

    pipeline_dict = deepcopy(to_container(pipeline_config))
    if not isinstance(pipeline_dict, dict):
        raise TypeError("Pipeline configuration must resolve to a mapping.")
    components = pipeline_dict.get("components", {})
    if not isinstance(components, dict):
        return pipeline_dict
    for component in components.values():
        if not isinstance(component, dict):
            continue
        parameters = component.get("init_parameters")
        if not isinstance(parameters, dict):
            continue
        for parameter_name in ("progress_bar", "show_progress_bar"):
            if parameter_name in parameters:
                parameters[parameter_name] = False
    return pipeline_dict
