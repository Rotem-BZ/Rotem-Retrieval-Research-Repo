"""Interactive Hydra command builder for retrieval experiments."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from retrieval_core.stages import STAGE_RUNNERS
from retrieval_core.utils.config import compose_stage_config, config_roots, find_config_dir
from retrieval_core.utils.io import read_yaml_mapping

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
class DefaultEntry:
    """One Hydra defaults-list entry that selects a config file."""

    group: str
    override_key: str
    package: str | None
    choice_name: str | None
    required: bool = False

    @classmethod
    def from_default_item(cls, key: str, value: Any) -> "DefaultEntry":
        group, package = _split_default_key(key)
        required = value == "???"
        choice_name = None if required else str(value)
        return cls(
            group=group,
            override_key=RequiredDefault.from_default_key(key).override_key,
            package=package,
            choice_name=choice_name,
            required=required,
        )


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


@dataclass(frozen=True)
class SelectedConfig:
    """A concrete YAML config currently selected into the command graph."""

    label: str
    group: str | None
    choice_name: str
    path: Path
    field_prefix: str
    override_key: str | None = None
    status: str = "selected"


@dataclass(frozen=True)
class EditableField:
    """A scalar/list/dict leaf that can be rendered as a Hydra override."""

    path: str
    value: Any


def run_configure(
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    config_dir: Path | None = None,
    allow_dry_run: bool = True,
) -> BuiltCommand:
    """Run the interactive command builder and print the final command."""

    config_dir = find_config_dir(config_dir)
    output_fn("Retrieval Research command builder")
    output_fn("")

    stage_name = _prompt_stage(input_fn, output_fn)
    stage_path = _resolve_config_path(f"{stage_name}.yaml", config_dir)
    override_items: list[HydraOverride] = []

    _prompt_required_defaults(
        initial_paths=[stage_path],
        overrides=override_items,
        input_fn=input_fn,
        output_fn=output_fn,
        config_dir=config_dir,
    )

    _prompt_config_graph_edits(
        stage_name,
        stage_path=stage_path,
        overrides=override_items,
        input_fn=input_fn,
        output_fn=output_fn,
        config_dir=config_dir,
    )

    dry_run = False
    if allow_dry_run:
        dry_run = _prompt_yes_no(
            "Use --dry-run? [y/N]: ",
            input_fn=input_fn,
            output_fn=output_fn,
            default=False,
        )

    _prompt_configured_overrides(
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

    choices: list[ConfigChoice] = []
    seen: set[str] = set()
    for root in config_roots(config_dir):
        group_dir = root / group
        if not group_dir.is_dir():
            continue
        paths = group_dir.rglob("*.yaml") if "/" in group else group_dir.glob("*.yaml")
        for path in sorted(paths, key=lambda item: _choice_sort_key(group_dir, item)):
            relative = path.relative_to(group_dir).with_suffix("")
            choice_name = relative.as_posix()
            if choice_name in seen:
                continue
            seen.add(choice_name)
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

    payload = read_yaml_mapping(path)
    defaults = payload.get("defaults", [])
    required: list[RequiredDefault] = []

    for entry in defaults:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if value == "???":
                required.append(RequiredDefault.from_default_key(str(key)))

    return required


def extract_default_entries(path: Path) -> list[DefaultEntry]:
    """Read selectable Hydra defaults from a YAML config."""

    payload = read_yaml_mapping(path)
    defaults = payload.get("defaults", [])
    entries: list[DefaultEntry] = []

    for entry in defaults:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if str(key) == "_self_":
                continue
            entries.append(DefaultEntry.from_default_item(str(key), value))

    return entries


def collect_selected_configs(
    *,
    stage_name: str,
    stage_path: Path,
    overrides: Sequence[HydraOverride | str],
    config_dir: Path | None = None,
) -> list[SelectedConfig]:
    """Return selected config files with their mounted field prefixes."""

    config_dir = find_config_dir(config_dir)
    selected: list[SelectedConfig] = [
        SelectedConfig(
            label=f"stage {stage_name}",
            group=None,
            choice_name=stage_name,
            path=stage_path,
            field_prefix="",
            status="stage",
        )
    ]
    seen = {(stage_path.resolve(), "")}
    override_map = _override_map(overrides)

    def visit(parent_path: Path, parent_prefix: str) -> None:
        for entry in extract_default_entries(parent_path):
            override_key = _mounted_override_key(parent_prefix, entry)
            choice_name = override_map.get(override_key, entry.choice_name)
            if choice_name is None:
                continue
            try:
                path = _resolve_config_path(f"{entry.group}/{choice_name}.yaml", config_dir)
            except FileNotFoundError:
                continue

            field_prefix = _mount_prefix(
                parent_prefix,
                group=entry.group,
                package=entry.package,
            )
            status = "default"
            if override_key in override_map:
                status = "selected"
            elif entry.required:
                status = "required"
            node = SelectedConfig(
                label=f"{override_key}={choice_name}",
                group=entry.group,
                choice_name=choice_name,
                path=path,
                field_prefix=field_prefix,
                override_key=override_key,
                status=status,
            )
            key = (path.resolve(), field_prefix)
            if key in seen:
                continue
            seen.add(key)
            selected.append(node)
            visit(path, field_prefix)

    visit(stage_path, "")
    return selected


def editable_fields(config: SelectedConfig) -> list[EditableField]:
    """Return editable non-default leaf fields for one selected config."""

    payload = {
        key: value
        for key, value in read_yaml_mapping(config.path).items()
        if key not in {"defaults", "metadata"}
    }
    fields: list[EditableField] = []
    for path, value in _flatten_fields(payload):
        full_path = _join_path(config.field_prefix, path)
        if full_path:
            fields.append(EditableField(path=full_path, value=value))
    return fields


def effective_editable_fields(
    config: SelectedConfig,
    *,
    stage_name: str,
    overrides: Sequence[HydraOverride | str],
) -> list[EditableField]:
    """Return editable fields with values from the current composed config."""

    cfg = compose_stage_config(stage_name, _compose_overrides(overrides))
    fields: list[EditableField] = []
    for field in editable_fields(config):
        fields.append(
            EditableField(
                path=field.path,
                value=_get_config_value(cfg, field.path, default=field.value),
            )
        )
    return fields


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


def _prompt_config_graph_edits(
    stage_name: str,
    *,
    stage_path: Path,
    overrides: list[HydraOverride],
    input_fn: InputFn,
    output_fn: OutputFn,
    config_dir: Path,
) -> None:
    if not _prompt_yes_no(
        "Review/edit selected configs? [y/N]: ",
        input_fn=input_fn,
        output_fn=output_fn,
        default=False,
    ):
        return

    while True:
        configs = collect_selected_configs(
            stage_name=stage_name,
            stage_path=stage_path,
            overrides=overrides,
            config_dir=config_dir,
        )
        selected = _prompt_menu(
            "Selected configs:",
            [*configs, None],
            input_fn=input_fn,
            output_fn=output_fn,
            format_item=lambda item: "Done" if item is None else _format_selected_config(item),
        )
        if selected is None:
            return

        changed_choice = _prompt_selected_config_action(
            stage_name,
            selected,
            overrides=overrides,
            input_fn=input_fn,
            output_fn=output_fn,
            config_dir=config_dir,
        )
        if changed_choice:
            _prompt_required_defaults(
                initial_paths=[changed_choice.path],
                overrides=overrides,
                input_fn=input_fn,
                output_fn=output_fn,
                config_dir=config_dir,
            )


def _prompt_selected_config_action(
    stage_name: str,
    selected: SelectedConfig,
    *,
    overrides: list[HydraOverride],
    input_fn: InputFn,
    output_fn: OutputFn,
    config_dir: Path,
) -> ConfigChoice | None:
    output_fn("")
    output_fn(_format_selected_config(selected))

    if selected.group is None or selected.override_key is None:
        action = _prompt_menu(
            "Options:",
            ["Accept as-is", "Edit fields"],
            input_fn=input_fn,
            output_fn=output_fn,
            format_item=lambda item: item,
        )
        if action == "Edit fields":
            _prompt_field_edits(
                stage_name,
                selected,
                overrides,
                input_fn=input_fn,
                output_fn=output_fn,
            )
        return None

    action = _prompt_menu(
        "Options:",
        ["Accept as-is", "Switch choice", "Edit fields"],
        input_fn=input_fn,
        output_fn=output_fn,
        format_item=lambda item: item,
    )
    if action == "Accept as-is":
        return None
    if action == "Edit fields":
        _prompt_field_edits(
            stage_name,
            selected,
            overrides,
            input_fn=input_fn,
            output_fn=output_fn,
        )
        return None

    choices = discover_config_choices(selected.group, config_dir=config_dir)
    choice = _prompt_menu(
        f"Choose {selected.override_key}:",
        choices,
        input_fn=input_fn,
        output_fn=output_fn,
        format_item=_format_choice,
    )
    _upsert_override(overrides, HydraOverride(compose=f"{selected.override_key}={choice.name}"))
    return choice


def _prompt_field_edits(
    stage_name: str,
    selected: SelectedConfig,
    overrides: list[HydraOverride],
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> None:
    fields = effective_editable_fields(selected, stage_name=stage_name, overrides=overrides)
    if not fields:
        output_fn("No editable fields in this config.")
        return

    while True:
        fields = effective_editable_fields(selected, stage_name=stage_name, overrides=overrides)
        field = _prompt_menu(
            f"Editable fields in {selected.label}:",
            [*fields, None],
            input_fn=input_fn,
            output_fn=output_fn,
            format_item=lambda item: "Done" if item is None else _format_field(item),
        )
        if field is None:
            return

        answer = input_fn(f"{field.path} [{_format_value(field.value)}]: ").strip()
        if not answer:
            continue
        _upsert_override(overrides, HydraOverride(compose=f"{field.path}={answer}"))


def _prompt_configured_overrides(
    stage_name: str,
    overrides: list[HydraOverride],
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> None:
    cfg = _compose_for_prompt(stage_name, overrides)
    prompt_configs = OmegaConf.select(
        cfg,
        "metadata.command_builder.prompt_overrides",
        default=[],
    )
    prompt_items = _to_plain_value(prompt_configs)
    if not isinstance(prompt_items, list):
        return

    for prompt_config in prompt_items:
        if not isinstance(prompt_config, dict):
            continue

        path = str(prompt_config.get("path", "")).strip()
        if not path:
            continue

        prompt_type = str(prompt_config.get("type", "value"))
        prompt_text = str(prompt_config.get("prompt", f"{path}: ")).rstrip()
        current_value = _get_config_value(cfg, path, default=None)
        default_text = _format_prompt_default(current_value, prompt_type)
        answer = input_fn(f"{prompt_text} [{default_text}]: ").strip()
        if not answer:
            if prompt_config.get("require_non_empty", False):
                raise SystemExit(f"At least one value is required when overriding {path}.")
            continue

        if prompt_type == "comma_list":
            selected_values = [item.strip() for item in answer.split(",") if item.strip()]
            if prompt_config.get("require_non_empty", False) and not selected_values:
                raise SystemExit(f"At least one value is required when overriding {path}.")

            value_json = json.dumps(selected_values, separators=(",", ":"))
            command = None
            if prompt_config.get("command_quote") == "single":
                command = f"{path}='{value_json}'"
            overrides.append(HydraOverride(compose=f"{path}={value_json}", command=command))
            continue

        overrides.append(HydraOverride(compose=f"{path}={answer}"))


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


def _format_selected_config(config: SelectedConfig) -> str:
    location = f" -> {config.field_prefix}" if config.field_prefix else ""
    description = _config_description(config.path)
    suffix = f" ({config.status}{location})"
    if description:
        return f"{config.label}{suffix} - {description}"
    return f"{config.label}{suffix}"


def _format_field(field: EditableField) -> str:
    return f"{field.path} = {_format_value(field.value)}"


def _format_value(value: Any) -> str:
    value = _to_plain_value(value)
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, separators=(",", ":"))
    except TypeError:
        return str(value)


def _format_prompt_default(value: Any, prompt_type: str) -> str:
    value = _to_plain_value(value)
    if prompt_type == "comma_list" and isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return _format_value(value)


def _choice_sort_key(group_dir: Path, path: Path) -> tuple[int, str]:
    name = path.relative_to(group_dir).with_suffix("").as_posix()
    return (0 if name == "full" else 1, name)


def _compose_for_prompt(stage_name: str, overrides: Sequence[HydraOverride]):
    return compose_stage_config(stage_name, _compose_overrides(overrides))


def _parse_positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a positive integer.") from exc

    if parsed <= 0:
        raise SystemExit(f"{name} must be a positive integer.")
    return parsed


def _config_description(path: Path) -> str | None:
    metadata = read_yaml_mapping(path).get("metadata")
    if isinstance(metadata, dict):
        description = metadata.get("description")
        if description:
            return str(description)
    return None


def _resolve_config_path(relative_path: str, config_dir: Path) -> Path:
    for root in config_roots(config_dir):
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Config does not exist in the active search path: {relative_path}")


def _flatten_fields(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        fields: list[tuple[str, Any]] = []
        for key, item in value.items():
            fields.extend(_flatten_fields(item, _join_path(prefix, str(key))))
        return fields

    return [(prefix, value)]


def _mount_prefix(parent_prefix: str, *, group: str, package: str | None) -> str:
    if package:
        if package == "_global_":
            return ""
        if package.startswith("_global_."):
            return package.removeprefix("_global_.")
        return _join_path(parent_prefix, package)

    return _join_path(parent_prefix, group.replace("/", "."))


def _mounted_override_key(parent_prefix: str, entry: DefaultEntry) -> str:
    if not entry.package:
        return entry.override_key
    if entry.package.startswith("_global_"):
        return entry.override_key

    return f"{entry.group}@{_join_path(parent_prefix, entry.package)}"


def _join_path(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}.{suffix}"


def _override_map(overrides: Sequence[HydraOverride | str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for override in overrides:
        text = override.compose if isinstance(override, HydraOverride) else override
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        mapping[key] = value
    return mapping


def _compose_overrides(overrides: Sequence[HydraOverride | str]) -> list[str]:
    return [
        override.compose if isinstance(override, HydraOverride) else override
        for override in overrides
    ]


def _get_config_value(cfg: Any, path: str, *, default: Any) -> Any:
    try:
        return _to_plain_value(OmegaConf.select(cfg, path, throw_on_missing=True))
    except (KeyError, MissingMandatoryValue):
        return default


def _to_plain_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _upsert_override(overrides: list[HydraOverride], override: HydraOverride) -> None:
    key = _override_key(override.compose)
    for index, existing in enumerate(overrides):
        if _override_key(existing.compose) == key:
            overrides[index] = override
            return
    overrides.append(override)


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
