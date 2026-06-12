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

from omegaconf import DictConfig, open_dict

from retrieval_research.config import compose_stage_config
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
    print(_summarize_result(stage_name, result, cfg))


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
        "Usage: rr [--dry-run] <stage> [hydra overrides]\n"
        f"Stages: {stages}\n"
        "--dry-run redirects artifact paths to a temporary directory and skips stage output writes.\n"
        "Examples:\n"
        "  rr indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl\n"
        "  rr --dry-run inference dataset=toy pipeline/inference@pipeline=dummy_keyword\n"
        "  rr inference dataset=toy pipeline/inference@pipeline=dummy_keyword retrieval.top_k=10\n"
        "  rr evaluation dataset=toy"
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
        result = runner(cfg)

        if inspect.isawaitable(result):
            return cfg, asyncio.run(result)
        return cfg, result


def _summarize_result(stage_name: str, result: StageResult, cfg: DictConfig) -> Any:
    if stage_name == "inference" and isinstance(result, list):
        return {
            "prediction_count": len(result),
            "predictions_path": cfg.stage.predictions_path,
        }
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

    with TemporaryDirectory(prefix="rr-dry-run-") as temp_dir:
        with open_dict(cfg):
            cfg.stage.dry_run = True
            cfg.stage.dry_run_artifacts_dir = str(Path(temp_dir) / "artifacts")
            if "paths" in cfg and "artifacts_dir" in cfg.paths:
                cfg.paths.artifacts_dir = cfg.stage.dry_run_artifacts_dir
        yield


if __name__ == "__main__":
    main()
