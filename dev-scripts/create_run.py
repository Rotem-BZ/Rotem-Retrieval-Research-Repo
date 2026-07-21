"""Interactively create one minimal Hydra experiment run config."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from omegaconf import OmegaConf

from experiment_models import (
    load_plan,
    project_root_for_experiment,
    render_hydra_command,
    slugify,
    write_run_definition,
)
from retrieval_core.utils.config import config_roots

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interactively create a minimal runs/<name>.yaml Hydra config."
    )
    parser.add_argument("experiment_dir", nargs="?", type=Path)
    args = parser.parse_args(argv)
    create_run(experiment_dir=args.experiment_dir)


def create_run(
    *,
    experiment_dir: Path | None,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> Path:
    directory = choose_or_create_experiment(
        experiment_dir,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    project_root_for_experiment(directory)
    configs_dir = directory / "configs"
    runs_dir = directory / "runs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_configs = sorted(
        path.relative_to(configs_dir).with_suffix("").as_posix()
        for path in configs_dir.glob("*.yaml")
    )
    if not base_configs:
        raise FileNotFoundError(
            f"Create a complete experiment base config in {configs_dir} first."
        )
    output_fn("Experiment base configs:")
    for index, config_name in enumerate(base_configs, start=1):
        output_fn(f"  {index}. {config_name}")
    base_config = _prompt_choice(
        base_configs,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    entered_name = input_fn("Run name: ").strip()
    run_name = slugify(entered_name, fallback="run")
    path = runs_dir / f"{run_name}.yaml"
    output_fn(
        "Add only differences from the base as Hydra overrides. Press Enter when done."
    )
    raw_overrides: list[str] = []
    while override := input_fn("override: ").strip():
        raw_overrides.append(override)
    group_overrides, fields = split_run_overrides(
        raw_overrides,
        config_dir=configs_dir,
    )
    write_run_definition(
        path,
        base_config=base_config,
        group_overrides=group_overrides,
        fields=fields,
    )

    try:
        plan = load_plan(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    run = next(run for run in plan.runs if run.definition_file == path.resolve())
    output_fn("")
    output_fn(f"Created {path}")
    output_fn("Hydra command:")
    output_fn(render_hydra_command(run, directory))
    return path


def split_run_overrides(
    overrides: Sequence[str],
    *,
    config_dir: Path,
) -> tuple[tuple[tuple[str, str], ...], dict[str, object]]:
    """Split Hydra selections into defaults entries and values into YAML fields."""

    group_overrides: list[tuple[str, str]] = []
    dotlist: list[str] = []
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Hydra override must contain '=': {override!r}")
        key, value = override.split("=", 1)
        normalized_key = key.lstrip("+~")
        if _is_launcher_controlled(normalized_key):
            raise ValueError(
                f"Run config cannot set launcher-controlled field {key!r}."
            )
        if _is_config_group_choice(normalized_key, value, config_dir=config_dir):
            group_overrides.append((normalized_key, value))
        else:
            dotlist.append(override)

    fields = OmegaConf.to_container(OmegaConf.from_dotlist(dotlist), resolve=False)
    if not isinstance(fields, dict):
        raise ValueError("Run value overrides must compose to a YAML mapping.")
    return tuple(group_overrides), fields


def _is_config_group_choice(key: str, value: str, *, config_dir: Path) -> bool:
    group = key.split("@", 1)[0]
    choice = value.strip().strip("'\"")
    return any(
        (root / group / f"{choice}.yaml").is_file() for root in config_roots(config_dir)
    )


def _is_launcher_controlled(key: str) -> bool:
    return key in {
        "paths.project_root",
        "stage.run_id",
        "experiment",
    } or key.startswith("experiment.")


def _prompt_choice(
    choices: Sequence[str],
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> str:
    while True:
        answer = input_fn(f"Select 1-{len(choices)}: ").strip()
        try:
            selected = int(answer)
        except ValueError:
            output_fn("Enter a number from the list.")
            continue
        if 1 <= selected <= len(choices):
            return choices[selected - 1]
        output_fn("Enter a number from the list.")


def choose_or_create_experiment(
    experiment_dir: Path | None,
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> Path:
    if experiment_dir is not None:
        directory = experiment_dir.expanduser().resolve()
        if directory.exists() and not directory.is_dir():
            raise NotADirectoryError(f"Experiment path is not a directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    root = Path.cwd().resolve() / "experiments"
    choices = (
        sorted(path for path in root.iterdir() if path.is_dir())
        if root.is_dir()
        else []
    )
    if choices:
        output_fn("Experiments:")
        for index, path in enumerate(choices, start=1):
            output_fn(f"  {index}. {path.name}")
        output_fn(f"  {len(choices) + 1}. Create a new experiment")
        while True:
            answer = input_fn(f"Select 1-{len(choices) + 1}: ").strip()
            try:
                selected = int(answer)
            except ValueError:
                output_fn("Enter a number from the list.")
                continue
            if 1 <= selected <= len(choices):
                return choices[selected - 1].resolve()
            if selected == len(choices) + 1:
                break
            output_fn("Enter a number from the list.")

    name = slugify(input_fn("Experiment name: ").strip(), fallback="experiment")
    directory = root / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory.resolve()


if __name__ == "__main__":
    main()
