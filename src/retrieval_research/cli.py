"""Command-line entry point for retrieval experiments."""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig

from retrieval_research.config import compose_stage_config
from retrieval_research.stages import STAGE_RUNNERS, StageResult


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help", "help"}:
        print(usage())
        return

    stage_name = args[0]
    overrides = args[1:]
    cfg, result = _run_stage_with_config(stage_name, overrides)
    print(_summarize_result(stage_name, result, cfg))


def run_stage(stage_name: str, overrides: Sequence[str] | None = None) -> StageResult:
    _, result = _run_stage_with_config(stage_name, overrides)
    return result


def usage() -> str:
    stages = "|".join(sorted(STAGE_RUNNERS))
    return (
        "Usage: rr <stage> [hydra overrides]\n"
        f"Stages: {stages}\n"
        "Examples:\n"
        "  rr indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl\n"
        "  rr inference dataset=toy pipeline/inference@pipeline=dummy_keyword retrieval.top_k=10\n"
        "  rr evaluation dataset=toy"
    )


def _run_stage_with_config(
    stage_name: str,
    overrides: Sequence[str] | None = None,
) -> tuple[DictConfig, StageResult]:
    try:
        runner = STAGE_RUNNERS[stage_name]
    except KeyError as exc:
        valid_stages = ", ".join(sorted(STAGE_RUNNERS))
        raise SystemExit(f"Unknown stage '{stage_name}'. Valid stages: {valid_stages}") from exc

    cfg = compose_stage_config(stage_name, overrides)
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


if __name__ == "__main__":
    main()
