"""Hydra configuration loading for project and core config directories."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def compose_stage_config(
    config_name: str,
    overrides: Sequence[str] | None = None,
    *,
    config_dir: str | Path | None = None,
) -> DictConfig:
    """Compose with project configs taking precedence over core configs."""

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(find_config_dir(config_dir)),
        job_name=f"stage-{config_name}",
    ):
        return compose(config_name=config_name, overrides=list(overrides or []))


def core_config_dir() -> Path:
    """Return the config directory shipped by ``retrieval-core``."""

    return Path(str(files("retrieval_core.configs")))


def find_config_dir(config_dir: str | Path | None = None) -> Path:
    """Resolve the primary config directory, defaulting to the current project."""

    if config_dir is not None:
        resolved = Path(config_dir).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Config directory does not exist: {resolved}")
        return resolved

    project_configs = Path.cwd() / "configs"
    if project_configs.is_dir():
        return project_configs.resolve()

    return core_config_dir().resolve()


def config_roots(config_dir: str | Path | None = None) -> tuple[Path, ...]:
    """Return config roots in Hydra precedence order."""

    primary = find_config_dir(config_dir)
    core = core_config_dir().resolve()
    if primary == core:
        return (core,)
    return primary, core
