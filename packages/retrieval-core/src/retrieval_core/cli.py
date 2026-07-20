"""Command-line entry point for retrieval experiments."""

from __future__ import annotations

import argparse
import asyncio
import inspect
from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

from retrieval_core.stages import STAGE_RUNNERS, StageResult
from retrieval_core.stages.base import prepare_stage_run_config
from retrieval_core.stages.evaluation import prepare_evaluation_config
from retrieval_core.stages.inference import prepare_inference_config
from retrieval_core.utils.config import compose_stage_config
from retrieval_core.utils.console import print_stage_result, print_stage_start


def main(argv: Sequence[str] | None = None) -> StageResult:
    parser = argparse.ArgumentParser(
        prog="stage",
        description="Run a retrieval experiment stage.",
        epilog=(
            "Examples:\n"
            "  stage indexing dataset=toy <required-config-group>=<choice>\n"
            "  stage inference dataset=toy some.nested.field=value\n"
            "  stage materialized/production/toy_dense_indexing_reference"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="project Hydra config directory (core configs remain available as fallbacks)",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help=(
            "experiment directory; configs resolve from its configs/ directory, then the "
            "project configs/, then retrieval-core"
        ),
    )
    parser.add_argument(
        "config_name",
        metavar="STAGE_OR_CONFIG",
        help=f"stage or config name ({', '.join(sorted(STAGE_RUNNERS))})",
    )
    parser.add_argument("overrides", nargs="*", metavar="OVERRIDE", help="Hydra override")
    args = parser.parse_args(argv)

    if args.config_dir is not None and args.experiment_dir is not None:
        parser.error("pass either --config-dir or --experiment-dir, not both")
    cfg = compose_stage_config(
        args.config_name,
        args.overrides,
        config_dir=args.config_dir,
        experiment_dir=args.experiment_dir,
    )
    if "stage" not in cfg or "name" not in cfg.stage:
        parser.error(f"config '{args.config_name}' must define stage.name")
    stage_name = str(cfg.stage.name)

    try:
        runner = STAGE_RUNNERS[stage_name]
    except KeyError:
        parser.error(
            f"config '{args.config_name}' declares unknown stage '{stage_name}'; "
            f"valid stages: {', '.join(sorted(STAGE_RUNNERS))}"
        )

    prepare_stage_run_config(cfg)
    if stage_name == "inference":
        prepare_inference_config(cfg)
    elif stage_name == "evaluation":
        prepare_evaluation_config(cfg)
    print_stage_start(stage_name, cfg, overrides=args.overrides)
    result = runner(cfg)
    if inspect.isawaitable(result):
        result = asyncio.run(result)

    printable_result = result
    if isinstance(result, list):
        stage = OmegaConf.to_container(cfg.stage, resolve=True)
        paths = {
            key: value
            for key, value in (stage or {}).items()
            if isinstance(key, str) and key.endswith("_path")
        }
        count = "prediction_count" if "predictions_path" in paths else "result_count"
        printable_result = {count: len(result), **paths}
    print_stage_result(stage_name, printable_result)
    return result


if __name__ == "__main__":
    main()
