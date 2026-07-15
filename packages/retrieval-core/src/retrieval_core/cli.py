"""Command-line entry point for retrieval experiments."""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

from retrieval_core.config import compose_stage_config
from retrieval_core.console import print_stage_result, print_stage_start
from retrieval_core.input_mapping import validate_input_mapping_config
from retrieval_core.io import project_path
from retrieval_core.pipelines import load_async_pipeline
from retrieval_core.stages import STAGE_RUNNERS, StageResult
from retrieval_core.stages.base import prepare_stage_run_config
from retrieval_core.stages.evaluation import prepare_evaluation_config
from retrieval_core.stages.inference import prepare_inference_config, validate_inference_inputs


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help", "help"}:
        print(usage())
        return

    config_dir, args = _extract_config_dir(args)
    dry_run, validate, args = _extract_modes(args)
    if args[0] in {"-h", "--help", "help"}:
        print(usage())
        return

    config_name = args[0]
    overrides = args[1:]
    if validate:
        stage_name, cfg, result = _validate_stage_with_config(
            config_name, overrides, config_dir=config_dir
        )
        print_stage_start(stage_name, cfg, overrides=overrides, validate=True)
        print_stage_result(stage_name, result)
        return
    stage_name, cfg, result = _run_stage_with_config(
        config_name, overrides, dry_run=dry_run, config_dir=config_dir
    )
    print_stage_result(stage_name, _summarize_result(result, cfg))


def run_stage(
    config_name: str,
    overrides: Sequence[str] | None = None,
    *,
    dry_run: bool = False,
    config_dir: str | Path | None = None,
) -> StageResult:
    _, _, result = _run_stage_with_config(
        config_name, overrides, dry_run=dry_run, config_dir=config_dir
    )
    return result


def validate_stage(
    config_name: str,
    overrides: Sequence[str] | None = None,
    *,
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a composed stage configuration without executing it or writing artifacts."""

    _, _, result = _validate_stage_with_config(config_name, overrides, config_dir=config_dir)
    return result


def usage() -> str:
    stages = "|".join(sorted(STAGE_RUNNERS))
    return (
        "Usage: stage [--config-dir PATH] [--dry-run|--validate] "
        "<stage-name-or-config-name> [hydra overrides]\n"
        f"Stages: {stages}\n"
        "Materialized configs can be run by config path, for example "
        "`stage materialized/production/toy_dense_indexing_reference`.\n"
        "Use `build-command` to build and validate a command interactively.\n"
        "--dry-run executes with real inputs, redirects outputs to a temporary directory, and "
        "does not save the run.\n"
        "--validate composes and verifies the configuration and pipeline graph without execution.\n"
        "--config-dir selects a project's primary Hydra config directory; retrieval-core configs "
        "remain available as fallbacks.\n"
        "Examples:\n"
        "  stage <stage-name> dataset=toy <required-config-group>=<choice>\n"
        "  stage --dry-run <stage-name> dataset=toy <required-config-group>=<choice>\n"
        "  stage <stage-name> dataset=toy some.nested.field=value"
    )


def _run_stage_with_config(
    config_name: str,
    overrides: Sequence[str] | None = None,
    *,
    dry_run: bool = False,
    config_dir: str | Path | None = None,
) -> tuple[str, DictConfig, StageResult]:
    cfg = compose_stage_config(config_name, overrides, config_dir=config_dir)
    stage_name = _stage_name_from_config(cfg, config_name)

    try:
        runner = STAGE_RUNNERS[stage_name]
    except KeyError as exc:
        valid_stages = ", ".join(sorted(STAGE_RUNNERS))
        raise SystemExit(
            f"Config '{config_name}' declares unknown stage '{stage_name}'. "
            f"Valid stages: {valid_stages}"
        ) from exc

    prepare_stage_run_config(cfg)
    with _dry_run_artifact_context(cfg, dry_run):
        _prepare_stage_dependencies(stage_name, cfg)
        print_stage_start(stage_name, cfg, overrides=overrides, dry_run=dry_run)
        result = runner(cfg)

        if inspect.isawaitable(result):
            return stage_name, cfg, asyncio.run(result)
        return stage_name, cfg, result


def _validate_stage_with_config(
    config_name: str,
    overrides: Sequence[str] | None = None,
    *,
    config_dir: str | Path | None = None,
) -> tuple[str, DictConfig, dict[str, Any]]:
    cfg = compose_stage_config(config_name, overrides, config_dir=config_dir)
    stage_name = _stage_name_from_config(cfg, config_name)
    if stage_name not in STAGE_RUNNERS:
        valid_stages = ", ".join(sorted(STAGE_RUNNERS))
        raise SystemExit(
            f"Config '{config_name}' declares unknown stage '{stage_name}'. "
            f"Valid stages: {valid_stages}"
        )

    _prepare_stage_config(stage_name, cfg)
    _validate_dataset_paths(cfg)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if stage_name in {"indexing", "inference"}:
        load_async_pipeline(cfg.pipeline)
    if stage_name == "inference":
        validate_inference_inputs(cfg)
        validate_input_mapping_config(cfg, require_prepared=True)
    elif stage_name == "evaluation":
        predictions_path = project_path(cfg.stage.predictions_path)
        if not predictions_path.is_file():
            raise FileNotFoundError(f"Predictions file does not exist: {predictions_path}")
    elif stage_name == "prepare_mapping":
        validate_input_mapping_config(cfg, require_generated=True)

    return stage_name, cfg, {"valid": True, "stage": stage_name}


def _prepare_stage_config(stage_name: str, cfg: DictConfig) -> None:
    prepare_stage_run_config(cfg)
    _prepare_stage_dependencies(stage_name, cfg)


def _prepare_stage_dependencies(stage_name: str, cfg: DictConfig) -> None:
    if stage_name == "inference":
        prepare_inference_config(cfg)
    elif stage_name == "evaluation":
        prepare_evaluation_config(cfg)


def _validate_dataset_paths(cfg: DictConfig) -> None:
    dataset = cfg.get("dataset")
    if not dataset:
        return
    for key in ("documents_path", "queries_path", "qrels_path"):
        configured = dataset.get(key)
        if configured and not project_path(configured).is_file():
            raise FileNotFoundError(f"Dataset {key} does not exist: {project_path(configured)}")


def _stage_name_from_config(cfg: DictConfig, config_name: str) -> str:
    if "stage" not in cfg or "name" not in cfg.stage:
        raise SystemExit(f"Config '{config_name}' must define stage.name.")
    return str(cfg.stage.name)


def _summarize_result(result: StageResult, cfg: DictConfig) -> Any:
    if isinstance(result, list):
        stage = OmegaConf.to_container(cfg.stage, resolve=True)
        path_fields = {
            key: value
            for key, value in (stage or {}).items()
            if isinstance(key, str) and key.endswith("_path")
        }
        count_key = "prediction_count" if "predictions_path" in path_fields else "result_count"
        return {count_key: len(result), **path_fields}
    return result


def _extract_dry_run(args: list[str]) -> tuple[bool, list[str]]:
    dry_run, _, remaining = _extract_modes(args)
    return dry_run, remaining


def _extract_modes(args: list[str]) -> tuple[bool, bool, list[str]]:
    dry_run = False
    validate = False
    remaining: list[str] = []

    for arg in args:
        if arg == "--dry-run":
            dry_run = True
        elif arg == "--validate":
            validate = True
        else:
            remaining.append(arg)

    if dry_run and validate:
        raise SystemExit("--dry-run and --validate are mutually exclusive.")

    if not remaining:
        raise SystemExit("Missing stage name.")

    return dry_run, validate, remaining


def _extract_config_dir(args: list[str]) -> tuple[Path | None, list[str]]:
    config_dir: Path | None = None
    remaining: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--config-dir":
            if index + 1 >= len(args):
                raise SystemExit("--config-dir requires a path.")
            config_dir = Path(args[index + 1])
            index += 2
            continue
        if arg.startswith("--config-dir="):
            config_dir = Path(arg.split("=", 1)[1])
            index += 1
            continue
        remaining.append(arg)
        index += 1
    return config_dir, remaining


@contextmanager
def _dry_run_artifact_context(cfg: DictConfig, dry_run: bool):
    if not dry_run:
        yield
        return

    with TemporaryDirectory(prefix="stage-dry-run-") as temp_dir:
        with open_dict(cfg):
            cfg.stage.dry_run = True
            cfg.stage.output_dir = str(
                Path(temp_dir) / str(cfg.stage.name) / str(cfg.stage.run_id)
            )
            if "predictions_path" in cfg.stage:
                cfg.stage.predictions_path = f"{cfg.stage.output_dir}/predictions.json"
            if "metrics_path" in cfg.stage:
                cfg.stage.metrics_path = f"{cfg.stage.output_dir}/metrics.json"
        yield


if __name__ == "__main__":
    main()
