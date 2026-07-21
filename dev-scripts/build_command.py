"""Interactive Hydra command builder for retrieval experiments."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from retrieval_core.stages import STAGE_RUNNERS
from retrieval_core.utils.artifacts import discover_index_ids, validate_index_id
from retrieval_core.utils.config import (
    compose_stage_config,
    config_roots,
    find_config_dir,
)
from retrieval_core.utils.io import project_path, read_yaml_mapping

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


@dataclass(frozen=True)
class DefaultEntry:
    """One Hydra defaults-list entry that selects a config file."""

    group: str
    override_key: str
    package: str | None
    choice_name: str | None
    required: bool = False


@dataclass(frozen=True)
class HydraOverride:
    """A Hydra override with separate compose and shell command spellings."""

    compose: str
    command: str | None = None


@dataclass(frozen=True)
class BuiltCommand:
    """A generated stage command and its Hydra overrides."""

    stage_name: str
    overrides: tuple[str, ...]
    command: str


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
    indexes_dir: Path | None = None,
) -> BuiltCommand:
    """Run the interactive command builder and print the final command."""

    config_dir = find_active_config_dir(config_dir)
    output_fn("Retrieval Research command builder")
    output_fn("Config search path:")
    for root in config_roots(config_dir):
        output_fn(f"  {root}")
    output_fn("")

    stage_name = _prompt_menu(
        "Choose stage:",
        sorted(STAGE_RUNNERS),
        input_fn=input_fn,
        output_fn=output_fn,
        format_item=lambda stage: stage,
    )
    stage_path = _resolve_stage_config_path(stage_name, config_dir)
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

    _prompt_configured_overrides(
        stage_name,
        override_items,
        input_fn=input_fn,
        output_fn=output_fn,
        config_dir=config_dir,
        indexes_dir=indexes_dir,
    )
    output_fn("Add any extra Hydra overrides one at a time. Press Enter when done.")
    while answer := input_fn("override: ").strip():
        override_items.append(HydraOverride(compose=answer))

    compose_overrides = tuple(override.compose for override in override_items)
    try:
        cfg = compose_stage_config(stage_name, compose_overrides, config_dir=config_dir)
        _validate_composed_command_config(cfg)
    except Exception as exc:
        output_fn("")
        output_fn("Could not compose the final config:")
        output_fn(f"  {exc}")
        raise SystemExit(2) from exc

    command = render_command(stage_name, override_items)
    output_fn("")
    output_fn("Command:")
    output_fn(command)
    return BuiltCommand(
        stage_name=stage_name, overrides=compose_overrides, command=command
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point for the command builder."""

    parser = argparse.ArgumentParser(
        description="Interactively build a retrieval stage command."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help=(
            "Primary Hydra configs directory. By default, use the nearest configs/ "
            "directory at or above the current working directory."
        ),
    )
    args = parser.parse_args(argv)
    try:
        run_configure(config_dir=args.config_dir)
    except KeyboardInterrupt:
        print("\nCommand builder cancelled.")
        raise SystemExit(130) from None


def find_active_config_dir(
    config_dir: Path | None = None,
    *,
    working_dir: Path | None = None,
) -> Path:
    """Find the nearest local config tree, falling back to retrieval-core."""

    if config_dir is not None:
        return find_config_dir(config_dir)

    current = (working_dir or Path.cwd()).expanduser().resolve()
    for directory in (current, *current.parents):
        candidate = directory if directory.name == "configs" else directory / "configs"
        if candidate.is_dir():
            return find_config_dir(candidate)
    return find_config_dir()


def discover_config_choices(
    group: str, config_dir: Path | None = None
) -> list[ConfigChoice]:
    """Return recursive YAML choices for a Hydra config group."""

    choices: list[ConfigChoice] = []
    seen: set[str] = set()
    for root in config_roots(config_dir):
        group_dir = root / group
        if not group_dir.is_dir():
            continue
        paths = group_dir.rglob("*.yaml") if "/" in group else group_dir.glob("*.yaml")
        for path in sorted(
            paths,
            key=lambda item: (
                0
                if item.relative_to(group_dir).with_suffix("").as_posix() == "full"
                else 1,
                item.relative_to(group_dir).with_suffix("").as_posix(),
            ),
        ):
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
    override_map: dict[str, str] = {}
    for override in overrides:
        text = override.compose if isinstance(override, HydraOverride) else override
        if "=" in text:
            key, value = text.split("=", 1)
            override_map[key] = value

    def visit(parent_path: Path, parent_prefix: str) -> None:
        for raw_entry in read_yaml_mapping(parent_path).get("defaults", []):
            if not isinstance(raw_entry, dict):
                continue
            raw_key, value = next(iter(raw_entry.items()))
            key = str(raw_key)
            if key == "_self_":
                continue
            group, package = _split_default_key(key)
            required = value == "???"
            entry = DefaultEntry(
                group=group,
                override_key=RequiredDefault.from_default_key(key).override_key,
                package=package,
                choice_name=None if required else str(value),
                required=required,
            )
            override_key = (
                entry.override_key
                if not entry.package or entry.package.startswith("_global_")
                else f"{entry.group}@{_join_path(parent_prefix, entry.package)}"
            )
            choice_name = override_map.get(override_key, entry.choice_name)
            if choice_name is None:
                continue
            try:
                path = _resolve_config_path(
                    f"{entry.group}/{choice_name}.yaml", config_dir
                )
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
    config_dir: Path | None = None,
) -> list[EditableField]:
    """Return editable fields with values from the current composed config."""

    cfg = compose_stage_config(
        stage_name,
        _compose_overrides(overrides),
        config_dir=config_dir,
    )
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
    prefix: Sequence[str] = ("uv", "run", "stage"),
) -> str:
    """Render a copyable command line."""

    tokens = list(prefix)
    tokens.append(stage_name)
    tokens.extend(
        (override.compose if override.command is None else override.command)
        if isinstance(override, HydraOverride)
        else override
        for override in overrides
    )
    return " ".join(tokens)


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
                raise SystemExit(
                    f"No config choices found for required group '{required.group}'."
                )

            choice = _prompt_menu(
                f"Choose {required.override_key}:",
                choices,
                input_fn=input_fn,
                output_fn=output_fn,
                format_item=_format_choice,
            )
            overrides.append(
                HydraOverride(compose=f"{required.override_key}={choice.name}")
            )
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
    while True:
        answer = input_fn("Review/edit selected configs? [y/N]: ").strip().lower()
        if not answer or answer in {"n", "no"}:
            review_configs = False
            break
        if answer in {"y", "yes"}:
            review_configs = True
            break
        output_fn("Enter y or n.")
    if not review_configs:
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
            configs,
            input_fn=input_fn,
            output_fn=output_fn,
            format_item=_format_selected_config,
            done_option=True,
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
                config_dir=config_dir,
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
            config_dir=config_dir,
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
    _upsert_override(
        overrides, HydraOverride(compose=f"{selected.override_key}={choice.name}")
    )
    return choice


def _prompt_field_edits(
    stage_name: str,
    selected: SelectedConfig,
    overrides: list[HydraOverride],
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
    config_dir: Path,
) -> None:
    fields = effective_editable_fields(
        selected,
        stage_name=stage_name,
        overrides=overrides,
        config_dir=config_dir,
    )
    if not fields:
        output_fn("No editable fields in this config.")
        return

    while True:
        fields = effective_editable_fields(
            selected,
            stage_name=stage_name,
            overrides=overrides,
            config_dir=config_dir,
        )
        field = _prompt_menu(
            f"Editable fields in {selected.label}:",
            fields,
            input_fn=input_fn,
            output_fn=output_fn,
            format_item=lambda item: f"{item.path} = {_format_value(item.value)}",
            done_option=True,
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
    config_dir: Path,
    indexes_dir: Path | None = None,
) -> None:
    cfg = compose_stage_config(
        stage_name,
        _compose_overrides(overrides),
        config_dir=config_dir,
    )
    prompt_items = {
        "indexing": [
            {
                "path": "selections.index_id",
                "prompt": "new index id",
                "type": "new_index_id",
                "require_non_empty": True,
            },
        ],
        "inference": [
            {
                "path": "selections.index_id",
                "prompt": "index id",
                "type": "existing_index_id",
                "only_if_index_backed": True,
            },
        ],
        "evaluation": [
            {
                "path": "stage.inference_run_id",
                "prompt": "exact inference run id",
                "type": "value",
                "require_non_empty": True,
            },
            {
                "path": "metrics",
                "prompt": "metrics comma-separated",
                "type": "comma_list",
                "command_quote": "single",
                "require_non_empty": True,
            },
        ],
    }.get(stage_name, [])

    for prompt_config in prompt_items:
        if not isinstance(prompt_config, dict):
            continue

        path = str(prompt_config.get("path", "")).strip()
        if not path:
            continue
        if prompt_config.get("only_if_index_backed") and not _config_uses_index(cfg):
            continue

        prompt_type = str(prompt_config.get("type", "value"))
        prompt_text = str(prompt_config.get("prompt", f"{path}: ")).rstrip()
        configured_indexes_dir = indexes_dir or project_path(cfg.paths.indexes_dir)
        if prompt_type == "existing_index_id":
            available_index_ids = discover_index_ids(configured_indexes_dir)
            if not available_index_ids:
                raise SystemExit(
                    f"No completed indexes found under {configured_indexes_dir}. "
                    "Run the indexing stage first."
                )
            selected_index_id = _prompt_menu(
                f"Choose {prompt_text}:",
                available_index_ids,
                input_fn=input_fn,
                output_fn=output_fn,
                format_item=lambda item: str(item),
            )
            overrides.append(HydraOverride(compose=f"{path}={selected_index_id}"))
            continue

        current_value = _get_config_value(cfg, path, default=None)
        plain_value = _to_plain_value(current_value)
        has_default = _has_prompt_default(plain_value)
        if prompt_type == "new_index_id" and not has_default:
            plain_value = _suggest_index_id(cfg, configured_indexes_dir)
            has_default = True
        default_text = (
            ", ".join(str(item) for item in plain_value)
            if prompt_type == "comma_list" and isinstance(plain_value, list)
            else _format_value(plain_value)
            if has_default
            else "required"
        )
        answer = input_fn(f"{prompt_text} [{default_text}]: ").strip()
        if not answer:
            if prompt_type == "new_index_id" and has_default:
                answer = str(plain_value)
            elif has_default:
                continue
            elif prompt_config.get("require_non_empty", False):
                raise SystemExit(
                    f"At least one value is required when overriding {path}."
                )
            else:
                continue

        if prompt_type == "new_index_id":
            normalized_index_id = validate_index_id(answer)
            if normalized_index_id in discover_index_ids(configured_indexes_dir):
                raise SystemExit(
                    f"Index {normalized_index_id!r} already exists under "
                    f"{configured_indexes_dir}."
                )
            overrides.append(HydraOverride(compose=f"{path}={normalized_index_id}"))
            continue

        if prompt_type == "comma_list":
            selected_values = [
                item.strip() for item in answer.split(",") if item.strip()
            ]
            if prompt_config.get("require_non_empty", False) and not selected_values:
                raise SystemExit(
                    f"At least one value is required when overriding {path}."
                )

            value_json = json.dumps(selected_values, separators=(",", ":"))
            command = None
            if prompt_config.get("command_quote") == "single":
                command = f"{path}='{value_json}'"
            overrides.append(
                HydraOverride(compose=f"{path}={value_json}", command=command)
            )
            continue

        overrides.append(HydraOverride(compose=f"{path}={answer}"))


def _has_prompt_default(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _suggest_index_id(cfg: Any, indexes_dir: Path) -> str:
    dataset_name = _get_config_value(cfg, "dataset.name", default="")
    model_name = _get_config_value(
        cfg,
        "selections.embedding_model.artifact_name",
        default="",
    )
    if not model_name:
        model_name = _get_config_value(
            cfg,
            "pipeline.components.embedder.init_parameters.model",
            default="",
        )
        if model_name:
            model_name = str(model_name).rstrip("/").rsplit("/", 1)[-1]

    parts = [
        part
        for value in (dataset_name, model_name, "index")
        if (part := _index_id_part(value))
    ]
    base = "-".join(parts) or "index"
    existing = set(discover_index_ids(indexes_dir))
    if base not in existing:
        return base

    suffix = 2
    while f"{base}-{suffix}" in existing:
        suffix += 1
    return f"{base}-{suffix}"


def _index_id_part(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")


def _prompt_menu(
    title: str,
    items: Sequence[Any],
    *,
    input_fn: InputFn,
    output_fn: OutputFn,
    format_item: Callable[[Any], str],
    done_option: bool = False,
) -> Any:
    output_fn(title)
    if done_option:
        output_fn("  0. Done")
    for index, item in enumerate(items, start=1):
        output_fn(f"  {index}. {format_item(item)}")

    first_index = 0 if done_option else 1
    while True:
        answer = input_fn(f"Select {first_index}-{len(items)}: ").strip()
        try:
            index = int(answer)
        except ValueError:
            output_fn("Enter a number from the list.")
            continue

        if done_option and index == 0:
            return None
        if 1 <= index <= len(items):
            return items[index - 1]

        output_fn("Enter a number from the list.")


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


def _format_value(value: Any) -> str:
    value = _to_plain_value(value)
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, separators=(",", ":"))
    except TypeError:
        return str(value)


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
    raise FileNotFoundError(
        f"Config does not exist in the active search path: {relative_path}"
    )


def _resolve_stage_config_path(stage_name: str, config_dir: Path) -> Path:
    try:
        return _resolve_config_path(f"{stage_name}.yaml", config_dir)
    except FileNotFoundError:
        return _resolve_config_path(f"stages/{stage_name}.yaml", config_dir)


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


def _join_path(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}.{suffix}"


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


def _validate_composed_command_config(cfg: Any) -> None:
    """Require a fully resolved command configuration."""

    missing_keys = set(OmegaConf.missing_keys(cfg))
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise MissingMandatoryValue(f"Missing mandatory values: {missing}")
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)


def _config_uses_index(cfg: Any) -> bool:
    components = OmegaConf.select(cfg, "pipeline.components", default={})
    if not components:
        return False
    return any(
        "index_path" in component.get("init_parameters", {})
        for component in components.values()
    )


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
