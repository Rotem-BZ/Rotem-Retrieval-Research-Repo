"""Interactively prepare fully resolved hyperparameter sweep configurations."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf, open_dict

from retrieval_core.cli import resolve_stage_config
from retrieval_core.command_builder import BuiltCommand, run_configure
from retrieval_core.sweeps.models import (
    SWEEP_SCHEMA_VERSION,
    SweepParameter,
    SweepPlan,
    SweepRun,
    choice_name,
    save_plan,
    slugify,
    unique_choice_name,
)
from retrieval_core.utils.config import find_config_dir
from retrieval_core.utils.hashing import sha256_text
from retrieval_core.utils.io import config_to_yaml
from retrieval_core.utils.time import utc_now, utc_timestamp

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]
RESERVED_SWEEP_PATHS = {
    "paths.project_root",
    "stage.output_dir",
    "stage.run_id",
    "stage.run_name",
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interactively prepare fully resolved configurations for a sweep."
    )
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    prepare_sweep(config_dir=args.config_dir, output_root=args.output_root)


def prepare_sweep(
    *,
    config_dir: Path | None = None,
    output_root: Path | None = None,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> Path:
    project_root = Path.cwd().resolve()
    resolved_config_dir = find_config_dir(config_dir)

    output_fn("Retrieval Research sweep preparer")
    output_fn("")
    built = run_configure(
        input_fn=input_fn,
        output_fn=output_fn,
        config_dir=resolved_config_dir,
        allow_dry_run=False,
    )
    parameters = prompt_sweep_parameters(input_fn=input_fn, output_fn=output_fn)
    combination_mode = prompt_combination_mode(input_fn=input_fn, output_fn=output_fn)
    combinations = parameter_combinations(parameters, combination_mode)

    default_name = f"{built.stage_name}-sweep"
    entered_name = input_fn(f"Sweep name [{default_name}]: ").strip() or default_name
    sweep_name = slugify(entered_name, fallback=default_name)
    timestamp = utc_timestamp()
    sweep_id = f"{sweep_name}--{timestamp}"

    output_fn("")
    output_fn(f"Stage: {built.stage_name}")
    output_fn(f"Combinations: {len(combinations)} ({combination_mode})")
    output_fn(f"Sweep id: {sweep_id}")
    if not prompt_yes_no("Materialize and validate this sweep? [Y/n]: ", True, input_fn):
        raise SystemExit("Sweep preparation cancelled.")

    root = (output_root or project_root / "artifacts" / "sweeps").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / sweep_id
    if destination.exists():
        raise FileExistsError(f"Sweep directory already exists: {destination}")

    staging = root / f".{sweep_id}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        staging.mkdir(parents=True)
        plan = materialize_sweep(
            staging,
            sweep_id=sweep_id,
            sweep_name=sweep_name,
            built=built,
            parameters=parameters,
            combinations=combinations,
            combination_mode=combination_mode,
            project_root=project_root,
            config_dir=resolved_config_dir,
            output_fn=output_fn,
        )
        save_plan(staging / "sweep.yaml", plan)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    output_fn("")
    output_fn(f"Prepared {len(combinations)} runs in {destination}")
    output_fn(f"Launch them with: uv run run-sweep {destination}")
    return destination


def materialize_sweep(
    sweep_dir: Path,
    *,
    sweep_id: str,
    sweep_name: str,
    built: BuiltCommand,
    parameters: list[SweepParameter],
    combinations: list[tuple[Any, ...]],
    combination_mode: str,
    project_root: Path,
    config_dir: Path,
    output_fn: OutputFn = print,
) -> SweepPlan:
    configs_dir = sweep_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    varied_keys = {parameter.path for parameter in parameters}
    base_overrides = [
        override
        for override in built.overrides
        if override_key(override) not in varied_keys | RESERVED_SWEEP_PATHS
    ]

    runs: list[SweepRun] = []
    used_names: set[str] = set()
    for index, values in enumerate(combinations, start=1):
        generated_name = choice_name(parameters, values)
        run_name = unique_choice_name(generated_name, values, used_names)
        used_names.add(run_name)
        stage_run_id = f"{sweep_id}--{run_name}"
        parameter_values = {
            parameter.path: value for parameter, value in zip(parameters, values, strict=True)
        }
        overrides = [
            *base_overrides,
            *[
                render_override(parameter.path, value, raw=parameter.raw)
                for parameter, value in zip(parameters, values, strict=True)
            ],
            render_override("paths.project_root", str(project_root)),
            render_override("stage.run_id", stage_run_id),
            "stage.run_name=null",
        ]

        output_fn(f"[{index}/{len(combinations)}] validating {run_name}")
        stage_name, cfg = resolve_stage_config(
            built.stage_name,
            overrides,
            config_dir=config_dir,
        )
        if stage_name != built.stage_name:
            raise ValueError(
                f"Combination {run_name!r} resolved to stage {stage_name!r}, "
                f"expected {built.stage_name!r}."
            )
        with open_dict(cfg):
            cfg.stage.run_name = run_name
            cfg.stage.preserve_run_config = True
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        config_text = "# @package _global_\n\n" + config_to_yaml(cfg)
        config_file = configs_dir / f"{run_name}.yaml"
        config_file.write_text(config_text, encoding="utf-8")
        runs.append(
            SweepRun(
                index=index,
                name=run_name,
                stage_run_id=stage_run_id,
                config_file=config_file.relative_to(sweep_dir).as_posix(),
                config_sha256=sha256_text(config_text),
                parameters=parameter_values,
                output_dir=str(OmegaConf.select(cfg, "stage.output_dir")),
            )
        )

    return SweepPlan(
        schema_version=SWEEP_SCHEMA_VERSION,
        sweep_id=sweep_id,
        name=sweep_name,
        stage=built.stage_name,
        created_at=utc_now(),
        project_root=str(project_root),
        source_config_dir=str(config_dir),
        combination_mode=combination_mode,
        parameters=parameters,
        runs=runs,
    )


def prompt_sweep_parameters(
    *, input_fn: InputFn = input, output_fn: OutputFn = print
) -> list[SweepParameter]:
    output_fn("")
    output_fn("Add hyperparameters to vary. Enter a blank path when done.")
    parameters: list[SweepParameter] = []
    while True:
        path = input_fn("Hydra field or override path: ").strip()
        if not path:
            if parameters:
                return parameters
            output_fn("Add at least one hyperparameter.")
            continue
        if any(parameter.path == path for parameter in parameters):
            output_fn(f"{path!r} is already part of this sweep.")
            continue
        if path in RESERVED_SWEEP_PATHS:
            output_fn(f"{path!r} is controlled by the sweep preparer and cannot be varied.")
            continue

        default_label = default_parameter_label(path)
        label = input_fn(f"Short name used in run names [{default_label}]: ").strip()
        label = label or default_label
        raw = prompt_yes_no(
            "Treat values as raw Hydra expressions/config choices? [y/N]: ",
            False,
            input_fn,
        )
        values = prompt_values(path, input_fn=input_fn, output_fn=output_fn)
        parameters.append(SweepParameter(path=path, label=label, values=values, raw=raw))


def prompt_values(
    path: str, *, input_fn: InputFn = input, output_fn: OutputFn = print
) -> list[Any]:
    while True:
        answer = input_fn(f"Values for {path} as a YAML list (for example [0.001, 0.01]): ").strip()
        try:
            values = yaml.safe_load(answer)
        except yaml.YAMLError as exc:
            output_fn(f"Could not parse values: {exc}")
            continue
        if not isinstance(values, list) or not values:
            output_fn("Enter a non-empty YAML list.")
            continue
        return values


def prompt_combination_mode(*, input_fn: InputFn = input, output_fn: OutputFn = print) -> str:
    output_fn("")
    output_fn("Combination mode:")
    output_fn("  1. Cartesian product")
    output_fn("  2. Zip values by position")
    while True:
        answer = input_fn("Select 1-2 [1]: ").strip() or "1"
        if answer == "1":
            return "cartesian"
        if answer == "2":
            return "zip"
        output_fn("Enter 1 or 2.")


def parameter_combinations(
    parameters: list[SweepParameter], combination_mode: str
) -> list[tuple[Any, ...]]:
    if not parameters:
        raise ValueError("A sweep must contain at least one parameter.")
    if combination_mode == "cartesian":
        return list(itertools.product(*(parameter.values for parameter in parameters)))
    if combination_mode == "zip":
        lengths = {len(parameter.values) for parameter in parameters}
        if len(lengths) != 1:
            raise ValueError("Zipped sweep parameters must contain the same number of values.")
        return list(zip(*(parameter.values for parameter in parameters), strict=True))
    raise ValueError(f"Unknown combination mode: {combination_mode!r}")


def render_override(path: str, value: Any, *, raw: bool = False) -> str:
    rendered = str(value) if raw else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return f"{path}={rendered}"


def override_key(override: str) -> str:
    return override.split("=", 1)[0].lstrip("+~")


def default_parameter_label(path: str) -> str:
    normalized = path.split("@", 1)[0].rstrip("/")
    return slugify(normalized.replace("/", ".").rsplit(".", 1)[-1], fallback="parameter")


def prompt_yes_no(prompt: str, default: bool, input_fn: InputFn = input) -> bool:
    while True:
        answer = input_fn(prompt).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False


if __name__ == "__main__":
    main()
