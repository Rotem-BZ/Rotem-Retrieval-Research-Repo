"""Hydra configuration loading for stage entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def compose_stage_config(config_name: str, overrides: Sequence[str] | None = None) -> DictConfig:
    """Compose a stage config from the repository-level configs directory."""

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(_find_config_dir()),
        job_name=f"stage-{config_name}",
    ):
        return compose(config_name=config_name, overrides=list(overrides or []))


def _find_config_dir() -> Path:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]

    for candidate in candidates:
        config_dir = candidate / "configs"
        if config_dir.is_dir() and any(config_dir.glob("*.yaml")):
            return config_dir

    searched = ", ".join(str(candidate / "configs") for candidate in candidates)
    raise FileNotFoundError(f"Could not find repository configs directory. Searched: {searched}")
