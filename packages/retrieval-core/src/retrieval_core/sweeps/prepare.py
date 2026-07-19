"""Interactively prepare experiment run configurations."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf, open_dict

from retrieval_core.cli import resolve_stage_config
from retrieval_core.command_builder import BuiltCommand, run_configure
from retrieval_core.sweeps.models import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentParameter,
    ExperimentPlan,
    ExperimentRun,
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
RESERVED_EXPERIMENT_PATHS = {
    "paths.project_root",
    "stage.output_dir",
    "stage.run_id",
    "stage.run_name",
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interactively prepare fully resolved runs for an experiment."
    )
    parser.add_argument("experiment_dir", nargs="?", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    prepare_experiment(
        experiment_dir=args.experiment_dir,
        config_dir=args.config_dir,
        output_root=args.output_root,
    )


def prepare_experiment(
    *,
    experiment_dir: Path | None = None,
    config_dir: Path | None = None,
    output_root: Path | None = None,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> Path:
    project_root = Path.cwd().resolve()
    resolved_config_dir = find_config_dir(config_dir)

    output_fn("Retrieval Research experiment preparer")
    output_fn("")
    template_path: Path | None = None
    if experiment_dir is not None:
        destination = experiment_dir.expanduser().resolve()
        experiment_id = slugify(destination.name, fallback="experiment")
        experiment_name = experiment_id
        root = destination.parent
        candidate = destination / "configs" / "matrix.yaml"
        if candidate.is_file():
            template_path = candidate
            output_fn(f"Loading experiment template: {candidate}")

    if template_path is not None:
        built, parameters, combination_mode = load_experiment_template(template_path)
    else:
        built = run_configure(
            input_fn=input_fn,
            output_fn=output_fn,
            config_dir=resolved_config_dir,
            allow_dry_run=False,
        )
        parameters = prompt_experiment_parameters(input_fn=input_fn, output_fn=output_fn)
        combination_mode = (
            prompt_combination_mode(input_fn=input_fn, output_fn=output_fn)
            if parameters
            else "single"
        )

    combinations = parameter_combinations(parameters, combination_mode)
    if experiment_dir is None:
        default_name = f"{built.stage_name}-experiment"
        entered_name = input_fn(f"Experiment name [{default_name}]: ").strip() or default_name
        experiment_name = slugify(entered_name, fallback=default_name)
        experiment_id = f"{experiment_name}--{utc_timestamp()}"
        root = (output_root or project_root / "experiments").expanduser().resolve()
        destination = root / experiment_id

    output_fn("")
    output_fn(f"Stage: {built.stage_name}")
    output_fn(f"Combinations: {len(combinations)} ({combination_mode})")
    output_fn(f"Experiment id: {experiment_id}")
    if not prompt_yes_no("Materialize and validate this experiment? [Y/n]: ", True, input_fn):
        raise SystemExit("Experiment preparation cancelled.")

    root.mkdir(parents=True, exist_ok=True)
    if (destination / "experiment.yaml").exists() or (destination / "runs").exists():
        raise FileExistsError(f"Experiment is already materialized: {destination}")

    staging = root / f".{experiment_id}.tmp-{uuid.uuid4().hex[:8]}"
    try:
        staging.mkdir(parents=True)
        if template_path is None:
            write_experiment_template(
                staging / "configs" / "matrix.yaml",
                built=built,
                parameters=parameters,
                combination_mode=combination_mode,
            )
        plan = materialize_experiment(
            staging,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            built=built,
            parameters=parameters,
            combinations=combinations,
            combination_mode=combination_mode,
            project_root=project_root,
            config_dir=resolved_config_dir,
            output_fn=output_fn,
        )
        save_plan(staging / "experiment.yaml", plan)
        publish_experiment(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    output_fn("")
    output_fn(f"Prepared {len(combinations)} runs in {destination}")
    output_fn(f"Launch them with: uv run run-experiment {destination}")
    return destination


def materialize_experiment(
    experiment_dir: Path,
    *,
    experiment_id: str,
    experiment_name: str,
    built: BuiltCommand,
    parameters: list[ExperimentParameter],
    combinations: list[tuple[Any, ...]],
    combination_mode: str,
    project_root: Path,
    config_dir: Path,
    output_fn: OutputFn = print,
) -> ExperimentPlan:
    runs_dir = experiment_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    varied_keys = {parameter.path for parameter in parameters}
    base_overrides = [
        override
        for override in built.overrides
        if override_key(override) not in varied_keys | RESERVED_EXPERIMENT_PATHS
    ]

    runs: list[ExperimentRun] = []
    used_names: set[str] = set()
    for index, values in enumerate(combinations, start=1):
        generated_name = choice_name(parameters, values) if parameters else "run"
        run_name = unique_choice_name(generated_name, values, used_names)
        used_names.add(run_name)
        stage_run_id = f"{experiment_id}--{run_name}"
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
            # The experiment already owns the human-readable run name. Keeping the
            # stage-level prefix empty makes ``stage_run_id`` the exact artifact ID.
            cfg.stage.run_name = None
            cfg.stage.preserve_run_config = True
            cfg.experiment = {
                "id": experiment_id,
                "name": experiment_name,
                "run_name": run_name,
                "parameters": parameter_values,
            }
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        config_text = "# @package _global_\n\n" + config_to_yaml(cfg)
        config_file = runs_dir / run_name / "config.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=False)
        config_file.write_text(config_text, encoding="utf-8")
        runs.append(
            ExperimentRun(
                index=index,
                name=run_name,
                stage_run_id=stage_run_id,
                config_file=config_file.relative_to(experiment_dir).as_posix(),
                config_sha256=sha256_text(config_text),
                parameters=parameter_values,
                output_dir=str(OmegaConf.select(cfg, "stage.output_dir")),
            )
        )

    return ExperimentPlan(
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        experiment_id=experiment_id,
        name=experiment_name,
        stage=built.stage_name,
        created_at=utc_now(),
        project_root=str(project_root),
        source_config_dir=str(config_dir),
        combination_mode=combination_mode,
        parameters=parameters,
        runs=runs,
    )


def prompt_experiment_parameters(
    *, input_fn: InputFn = input, output_fn: OutputFn = print
) -> list[ExperimentParameter]:
    output_fn("")
    output_fn("Add parameters to vary. Leave the first path blank for one base run.")
    parameters: list[ExperimentParameter] = []
    while True:
        path = input_fn("Hydra field or override path: ").strip()
        if not path:
            return parameters
        if any(parameter.path == path for parameter in parameters):
            output_fn(f"{path!r} is already part of this experiment.")
            continue
        if path in RESERVED_EXPERIMENT_PATHS:
            output_fn(f"{path!r} is controlled by the experiment preparer and cannot be varied.")
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
        parameters.append(ExperimentParameter(path=path, label=label, values=values, raw=raw))


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
    parameters: list[ExperimentParameter], combination_mode: str
) -> list[tuple[Any, ...]]:
    if not parameters:
        if combination_mode == "single":
            return [()]
        raise ValueError("An experiment without varied parameters must use single mode.")
    if combination_mode == "cartesian":
        return list(itertools.product(*(parameter.values for parameter in parameters)))
    if combination_mode == "zip":
        lengths = {len(parameter.values) for parameter in parameters}
        if len(lengths) != 1:
            raise ValueError("Zipped experiment parameters must contain the same number of values.")
        return list(zip(*(parameter.values for parameter in parameters), strict=True))
    raise ValueError(f"Unknown combination mode: {combination_mode!r}")


def write_experiment_template(
    path: Path,
    *,
    built: BuiltCommand,
    parameters: list[ExperimentParameter],
    combination_mode: str,
) -> None:
    """Persist the reusable preparation choices that produced the resolved runs."""

    payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "stage": built.stage_name,
        "base_overrides": list(built.overrides),
        "combination_mode": combination_mode,
        "parameters": [
            {
                "path": parameter.path,
                "label": parameter.label,
                "values": parameter.values,
                "raw": parameter.raw,
            }
            for parameter in parameters
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_experiment_template(
    path: Path,
) -> tuple[BuiltCommand, list[ExperimentParameter], str]:
    """Load reusable experiment choices from ``configs/matrix.yaml``."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Experiment template must contain a YAML mapping: {path}")
    if payload.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported experiment template schema in {path}")

    stage = payload.get("stage")
    overrides = payload.get("base_overrides", [])
    raw_parameters = payload.get("parameters", [])
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError(f"Experiment template is missing a stage: {path}")
    if not isinstance(overrides, list) or not all(
        isinstance(override, str) for override in overrides
    ):
        raise ValueError(f"base_overrides must be a list of strings: {path}")
    if not isinstance(raw_parameters, list):
        raise ValueError(f"parameters must be a list: {path}")

    parameters: list[ExperimentParameter] = []
    for raw_parameter in raw_parameters:
        if not isinstance(raw_parameter, Mapping):
            raise ValueError(f"Each experiment parameter must be a mapping: {path}")
        values = raw_parameter.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Each experiment parameter needs non-empty values: {path}")
        parameter_path = raw_parameter.get("path")
        if not isinstance(parameter_path, str) or not parameter_path:
            raise ValueError(f"Each experiment parameter needs a path: {path}")
        label = raw_parameter.get("label") or default_parameter_label(parameter_path)
        parameters.append(
            ExperimentParameter(
                path=parameter_path,
                label=str(label),
                values=values,
                raw=bool(raw_parameter.get("raw", False)),
            )
        )

    default_mode = "cartesian" if parameters else "single"
    combination_mode = str(payload.get("combination_mode", default_mode))
    built = BuiltCommand(
        stage_name=stage,
        overrides=tuple(overrides),
        command=f"experiment template {path}",
    )
    return built, parameters, combination_mode


def publish_experiment(staging: Path, destination: Path) -> None:
    """Publish validated materialization while preserving existing research files."""

    if not destination.exists():
        staging.replace(destination)
        return

    generated_paths = {
        destination / "experiment.yaml": staging / "experiment.yaml",
        destination / "runs": staging / "runs",
    }
    staged_template = staging / "configs" / "matrix.yaml"
    if staged_template.exists():
        generated_paths[destination / "configs" / "matrix.yaml"] = staged_template
    conflicts = [path for path in generated_paths if path.exists()]
    if conflicts:
        rendered = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(f"Experiment materialization would overwrite: {rendered}")

    (destination / "configs").mkdir(parents=True, exist_ok=True)
    for target, source in generated_paths.items():
        source.replace(target)
    shutil.rmtree(staging)


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


# Compatibility API for existing integrations using the former sweep terminology.
prepare_sweep = prepare_experiment
prompt_sweep_parameters = prompt_experiment_parameters


def materialize_sweep(
    sweep_dir: Path,
    *,
    sweep_id: str,
    sweep_name: str,
    **kwargs: Any,
) -> ExperimentPlan:
    return materialize_experiment(
        sweep_dir,
        experiment_id=sweep_id,
        experiment_name=sweep_name,
        **kwargs,
    )


if __name__ == "__main__":
    main()
