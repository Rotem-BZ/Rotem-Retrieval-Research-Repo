"""Interactive Hydra command builder for retrieval experiments."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from retrieval_research.config import _find_config_dir, compose_stage_config
from retrieval_research.stages import STAGE_RUNNERS

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


@dataclass(frozen=True)
class ConfigChoice:
    """One selectable Hydra config file."""

    group: str
    name: str
    path: Path
    description: str | None = None


@dataclass(frozen=True)
class RequiredDefault:
    """A required Hydra defaults-list selection."""

    group: str
    override_key: str

    @classmethod
    def from_default_key(cls, key: str) -> "RequiredDefault":
        group, package = _split_default_key(key)
        package_suffix = f"@{package}" if package else ""
        override_key = f"{group}{package_suffix}"

        if package == f"_global_.{group.replace('/', '.')}":
            override_key = group
        elif package == group.replace("/", "."):
            override_key = group

        return cls(group=group, override_key=override_key)

    def render(self, choice: ConfigChoice) -> "HydraOverride":
        return HydraOverride(compose=f"{self.override_key}={choice.name}")


@dataclass(frozen=True)
class HydraOverride:
    """A Hydra override with separate compose and shell command spellings."""

    compose: str
    command: str | None = None

    @property
    def command_text(self) -> str:
        return self.compose if self.command is None else self.command


@dataclass(frozen=True)
class BuiltCommand:
    """The generated command and the overrides used to validate it."""

    stage_name: str
    overrides: tuple[str, ...]
    command: str
    dry_run: bool = False


def run_configure(
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    config_dir: Path | None = None,
) -> BuiltCommand:
    """Run the interactive command builder and print the final command."""

    config_dir = _find_config_dir() if config_dir is None else config_dir
    output_fn("Retrieval Research command builder")
    output_fn("")

    stage_name = _prompt_stage(input_fn, output_fn)
    stage_path = config_dir / f"{stage_name}.yaml"
    override_items: list[HydraOverride] = []

    _prompt_required_defaults(
        initial_paths=[stage_path],
        overrides=override_items,
        input_fn=input_fn,
        output_fn=output_fn,
        config_dir=config_dir,
    )

    dry_run = _prompt_yes_no(
        "Use --dry-run? [y/N]: ",
        input_fn=input_fn,
        output_fn=output_fn,
        default=False,
    )

    _prompt_common_overrides(
        stage_name,
        override_items,
        input_fn=input_fn,
        output_fn=output_fn,
    )
    _prompt_free_form_overrides(
        override_items,
        input_fn=input_fn,
        output_fn=output_fn,
    )

    return _validate_and_print_command(
        stage_name,
        override_items,
        dry_run=dry_run,
        output_fn=output_fn,
    )


def main() -> None:
    """Console-script entry point for the command builder."""

    run_configure()


def discover_config_choices(group: str, config_dir: Path | None = None) -> list[ConfigChoice]:
    """Return recursive YAML choices for a Hydra config group."""

    config_dir = _find_config_dir() if config_dir is None else config_dir
    group_dir = config_dir / group
    if not group_dir.is_dir():
        return []

    choices: list[ConfigChoice] = []
    paths = group_dir.rglob("*.yaml") if "/" in group else group_dir.glob("*.yaml")
    for path in sorted(paths, key=lambda item: _choice_sort_key(group_dir, item)):
        relative = path.relative_to(group_dir).with_suffix("")
        choice_name = relative.as_posix()
        choices.append(
            ConfigChoice(
                group=group,
                name=choice_name,
                path=path,
                description=_config_description(path),
            )
        )
    return choices


def extract_required_defaults(path: Path) -> list[RequiredDefault]:
    """Read required `???` defaults from a Hydra YAML config."""

    payload = _read_yaml(path)
    defaults = payload.get("defaults", [])
    required: list[RequiredDefault] = []

    for entry in defaults:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if value == "???":
                required.append(RequiredDefault.from_default_key(str(key)))

    return required


def render_command(
    stage_name: str,
    overrides: Sequence[HydraOverride | str],
    *,
    dry_run: bool = False,
    prefix: Sequence[str] = ("uv", "run", "stage"),
) -> str:
    """Render a copyable command line."""

    tokens = list(prefix)
    if dry_run:
        tokens.append("--dry-run")
    tokens.append(stage_name)
    tokens.extend(
        override.command_text if isinstance(override, HydraOverride) else override
        for override in overrides
    )
    return " ".join(tokens)


def validate_config(stage_name: str, overrides: Sequence[str]) -> None:
    """Compose and fully resolve a command-builder result."""

    cfg = compose_stage_config(stage_name, overrides)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def _prompt_stage(input_fn: InputFn, output_fn: OutputFn) -> str:
    stages = sorted(STAGE_RUNNERS)
    return _prompt_menu(
        "Choose stage:",
        stages,
        input_fn=input_fn,
        output_fn=output_fn,
        format_item=lambda stage: stage,
    )


def _prompt_required_defaults(
    *,
    initial_paths: Iterable[Path],
    overrides: list[HydraOverride],
    input_fn: InputFn,
    output_fn: OutputFn,
    config_dir: Path,
) -> None:
    pending = list(initial_paths)
    selected_override_keys = {_override_key(item.compose) for item in overrides}

    while pending:
        path = pending.pop(0)
        for required in extract_required_defaults(path):
            if required.override_key in selected_override_keys:
                continue

            choices = discover_config_choices(required.group, config_dir=config_dir)
            if not choices:
                raise SystemExit(f"No config choices found for required group '{required.group}'.")

            choice = _prompt_menu(
                f"Choose {required.override_key}:",
                choices,
                input_fn=input_fn,
                output_fn=output_fn,
                format_item=_format_choice,
            )
            overrides.append(required.render(choice))
            selected_override_keys.add(required.override_key)
            pending.append(choice.path)


def _prompt_common_overrides(
    stage_name: str,
    overrides: list[HydraOverride],
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> None:
    if stage_name == "inference":
        current = _compose_for_prompt(stage_name, overrides).retrieval.top_k
        answer = input_fn(f"retrieval.top_k [{current}]: ").strip()
        if answer:
            top_k = _parse_positive_int(answer, "retrieval.top_k")
            overrides.append(HydraOverride(compose=f"retrieval.top_k={top_k}"))

    if stage_name == "evaluation":
        metrics = list(_compose_for_prompt(stage_name, overrides).metrics)
        default_text = ", ".join(str(metric) for metric in metrics)
        answer = input_fn(f"metrics comma-separated [{default_text}]: ").strip()
        if answer:
            selected_metrics = [metric.strip() for metric in answer.split(",") if metric.strip()]
            if not selected_metrics:
                raise SystemExit("At least one metric is required when overriding metrics.")
            metrics_json = json.dumps(selected_metrics, separators=(",", ":"))
            overrides.append(
                HydraOverride(
                    compose=f"metrics={metrics_json}",
                    command=f"metrics='{metrics_json}'",
                )
            )


def _prompt_free_form_overrides(
    overrides: list[HydraOverride],
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> None:
    output_fn("Add any extra Hydra overrides one at a time. Press Enter when done.")
    while True:
        answer = input_fn("override: ").strip()
        if not answer:
            return
        overrides.append(HydraOverride(compose=answer))


def _validate_and_print_command(
    stage_name: str,
    overrides: Sequence[HydraOverride],
    *,
    dry_run: bool,
    output_fn: OutputFn,
) -> BuiltCommand:
    compose_overrides = tuple(override.compose for override in overrides)
    try:
        validate_config(stage_name, compose_overrides)
    except Exception as exc:
        output_fn("")
        output_fn("Could not compose the final config:")
        output_fn(f"  {exc}")
        raise SystemExit(2) from exc

    command = render_command(stage_name, overrides, dry_run=dry_run)
    output_fn("")
    output_fn("Command:")
    output_fn(command)
    return BuiltCommand(
        stage_name=stage_name,
        overrides=compose_overrides,
        command=command,
        dry_run=dry_run,
    )


def _prompt_menu(
    title: str,
    items: Sequence[Any],
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
    format_item: Callable[[Any], str],
) -> Any:
    output_fn(title)
    for index, item in enumerate(items, start=1):
        output_fn(f"  {index}. {format_item(item)}")

    while True:
        answer = input_fn(f"Select 1-{len(items)}: ").strip()
        try:
            index = int(answer)
        except ValueError:
            output_fn("Enter a number from the list.")
            continue

        if 1 <= index <= len(items):
            return items[index - 1]

        output_fn("Enter a number from the list.")


def _prompt_yes_no(
    prompt: str,
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
    default: bool,
) -> bool:
    while True:
        answer = input_fn(prompt).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        output_fn("Enter y or n.")


def _format_choice(choice: ConfigChoice) -> str:
    if choice.description:
        return f"{choice.name} - {choice.description}"
    return choice.name


def _choice_sort_key(group_dir: Path, path: Path) -> tuple[int, str]:
    name = path.relative_to(group_dir).with_suffix("").as_posix()
    return (0 if name == "full" else 1, name)


def _compose_for_prompt(stage_name: str, overrides: Sequence[HydraOverride]):
    return compose_stage_config(stage_name, [override.compose for override in overrides])


def _parse_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a positive integer.") from exc

    if parsed <= 0:
        raise SystemExit(f"{name} must be a positive integer.")
    return parsed


def _config_description(path: Path) -> str | None:
    metadata = _read_yaml(path).get("metadata")
    if isinstance(metadata, dict):
        description = metadata.get("description")
        if description:
            return str(description)
    return None


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _split_default_key(key: str) -> tuple[str, str | None]:
    normalized = key.lstrip("/")
    if "@" not in normalized:
        return normalized, None
    group, package = normalized.split("@", 1)
    return group, package


def _override_key(override: str) -> str:
    return override.split("=", 1)[0]


if __name__ == "__main__":
    main()
