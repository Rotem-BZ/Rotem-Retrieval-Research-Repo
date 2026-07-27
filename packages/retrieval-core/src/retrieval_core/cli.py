"""Command-line entry point for retrieval experiments."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from omegaconf import DictConfig, OmegaConf

from retrieval_core.stages import STAGE_RUNNERS, StageResult
from retrieval_core.stages.base import StageContext, prepare_stage_run_config
from retrieval_core.stages.evaluation import prepare_evaluation_config
from retrieval_core.stages.indexing import prepare_indexing_config
from retrieval_core.stages.inference import prepare_inference_config
from retrieval_core.utils.config import compose_entrypoint_config, compose_stage_config
from retrieval_core.utils.logging import (
    RUN_LOG_FILENAME,
    configure_console_logging,
    run_file_logging,
)

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    """Run a stage as a console command."""

    try:
        run_stage(argv)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception:
        raise SystemExit(1) from None


def run_stage(argv: Sequence[str] | None = None) -> StageResult:
    """Run a stage and return its result for programmatic callers."""

    configure_console_logging()
    parser = argparse.ArgumentParser(
        prog="stage",
        description="Run a retrieval experiment stage.",
        epilog=(
            "Examples:\n"
            "  stage indexing dataset=toy runtime=gpu <required-config-group>=<choice>\n"
            "  stage inference dataset=toy runtime=cpu some.nested.field=value\n"
            "  stage inference --entrypoint experiments/example/configs/runs/baseline.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage_name",
        metavar="STAGE",
        choices=sorted(STAGE_RUNNERS),
        help=f"stage to execute ({', '.join(sorted(STAGE_RUNNERS))})",
    )
    parser.add_argument(
        "--entrypoint",
        type=Path,
        help=(
            "YAML config entrypoint below a project or experiment configs/ directory; "
            "defaults to the selected core stage config"
        ),
    )
    parser.add_argument("overrides", nargs="*", metavar="OVERRIDE", help="Hydra override")
    args = parser.parse_args(argv)

    try:
        cfg = (
            compose_entrypoint_config(args.entrypoint, args.overrides)
            if args.entrypoint is not None
            else compose_stage_config(args.stage_name, args.overrides)
        )
        if "stage" not in cfg or "name" not in cfg.stage:
            source = str(args.entrypoint) if args.entrypoint is not None else args.stage_name
            parser.error(f"config entrypoint '{source}' must define stage.name")
        configured_stage = str(cfg.stage.name)
        if configured_stage != args.stage_name:
            parser.error(
                f"config entrypoint declares stage '{configured_stage}', but the command "
                f"requested '{args.stage_name}'"
            )
        prepare_stage_run_config(cfg)
        if configured_stage == "indexing":
            prepare_indexing_config(cfg)
        elif configured_stage == "inference":
            prepare_inference_config(cfg)
        elif configured_stage == "evaluation":
            prepare_evaluation_config(cfg)
        context = StageContext.create(cfg)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception:
        logger.exception("Stage setup failed: requested_stage=%s", args.stage_name)
        raise

    stage_name = args.stage_name
    runner = STAGE_RUNNERS[stage_name]
    started_at = perf_counter()
    execution_started = False

    try:
        with run_file_logging(context.output_dir / RUN_LOG_FILENAME) as log_path:
            execution_started = True
            logger.info(
                "Stage started: stage=%s run_id=%s dataset=%s output_dir=%s log_path=%s",
                stage_name,
                cfg.stage.get("run_id"),
                cfg.get("dataset", {}).get("name", "<none>"),
                context.output_dir,
                log_path,
            )
            try:
                result = runner(cfg, context=context)
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
            except KeyboardInterrupt:
                logger.warning(
                    "Stage interrupted: stage=%s run_id=%s elapsed_seconds=%.3f",
                    stage_name,
                    cfg.stage.get("run_id"),
                    perf_counter() - started_at,
                )
                raise
            except Exception:
                logger.exception(
                    "Stage failed: stage=%s run_id=%s elapsed_seconds=%.3f",
                    stage_name,
                    cfg.stage.get("run_id"),
                    perf_counter() - started_at,
                )
                raise

            logger.info(
                "Stage completed: stage=%s run_id=%s elapsed_seconds=%.3f result=%s",
                stage_name,
                cfg.stage.get("run_id"),
                perf_counter() - started_at,
                _summarize_result(cfg, result),
            )
            return result
    except (KeyboardInterrupt, Exception):
        if not execution_started:
            logger.exception(
                "Could not initialize stage logging: stage=%s run_id=%s",
                stage_name,
                cfg.stage.get("run_id"),
            )
        raise


def _summarize_result(cfg: DictConfig, result: StageResult) -> Any:
    """Return a compact result suitable for terminal and file logs."""

    if not isinstance(result, list):
        return result
    stage = OmegaConf.to_container(cfg.stage, resolve=True)
    paths = {
        key: value
        for key, value in (stage or {}).items()
        if isinstance(key, str) and key.endswith("_path")
    }
    count = "prediction_count" if "predictions_path" in paths else "result_count"
    return {count: len(result), **paths}


if __name__ == "__main__":
    main()
