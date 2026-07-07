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

from retrieval_research.config import compose_stage_config
from retrieval_research.console import print_stage_result, print_stage_start
from retrieval_research.stages.base import prepare_stage_run_config
from retrieval_research.stages import STAGE_RUNNERS, StageResult


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help", "help"}:
        print(usage())
        return

    dry_run, args = _extract_dry_run(args)
    if args[0] in {"-h", "--help", "help"}:
        print(usage())
        return

    stage_name = args[0]
    overrides = args[1:]
    cfg, result = _run_stage_with_config(stage_name, overrides, dry_run=dry_run)
    print_stage_result(stage_name, _summarize_result(result, cfg))


def run_stage(
    stage_name: str,
    overrides: Sequence[str] | None = None,
    *,
    dry_run: bool = False,
) -> StageResult:
    _, result = _run_stage_with_config(stage_name, overrides, dry_run=dry_run)
    return result


def usage() -> str:
    stages = "|".join(sorted(STAGE_RUNNERS))
    return (
        "Usage: stage [--dry-run] <stage-name> [hydra overrides]\n"
        f"Stages: {stages}\n"
        "Use `build-command` to build and validate a command interactively.\n"
        "--dry-run redirects artifact paths to a temporary directory and skips stage output writes.\n"
        "Examples:\n"
        "  stage <stage-name> dataset=toy <required-config-group>=<choice>\n"
        "  stage --dry-run <stage-name> dataset=toy <required-config-group>=<choice>\n"
        "  stage <stage-name> dataset=toy some.nested.field=value"
    )


def _run_stage_with_config(
    stage_name: str,
    overrides: Sequence[str] | None = None,
    *,
    dry_run: bool = False,
) -> tuple[DictConfig, StageResult]:
    try:
        runner = STAGE_RUNNERS[stage_name]
    except KeyError as exc:
        valid_stages = ", ".join(sorted(STAGE_RUNNERS))
        raise SystemExit(f"Unknown stage '{stage_name}'. Valid stages: {valid_stages}") from exc

    cfg = compose_stage_config(stage_name, overrides)

    with _dry_run_artifact_context(cfg, dry_run):
        prepare_stage_run_config(cfg)
        print_stage_start(stage_name, cfg, overrides=overrides, dry_run=dry_run)
        result = runner(cfg)

        if inspect.isawaitable(result):
            return cfg, asyncio.run(result)
        return cfg, result


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
    dry_run = False
    remaining: list[str] = []

    for arg in args:
        if arg == "--dry-run":
            dry_run = True
        else:
            remaining.append(arg)

    if not remaining:
        raise SystemExit("Missing stage name after --dry-run.")

    return dry_run, remaining


@contextmanager
def _dry_run_artifact_context(cfg: DictConfig, dry_run: bool):
    if not dry_run:
        yield
        return

    with TemporaryDirectory(prefix="stage-dry-run-") as temp_dir:
        with open_dict(cfg):
            cfg.stage.dry_run = True
            cfg.stage.dry_run_artifacts_dir = str(Path(temp_dir) / "artifacts")
            if "paths" in cfg and "artifacts_dir" in cfg.paths:
                cfg.paths.artifacts_dir = cfg.stage.dry_run_artifacts_dir
        yield


if __name__ == "__main__":
    main()
