"""Discover experiment choices and recommendations from repository configuration."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from retrieval_core.utils.artifacts import discover_index_ids
from retrieval_core.utils.config import config_roots
from retrieval_core.utils.io import read_yaml_mapping


GROUPS = (
    "dataset",
    "runtime",
    "pipeline/inference",
    "selections/embedding_model",
)


def _choice_files(root: Path, group: str) -> list[Path]:
    directory = root / Path(group)
    return sorted(directory.rglob("*.yaml")) if directory.is_dir() else []


def _choice_name(path: Path, root: Path, group: str) -> str:
    return path.relative_to(root / Path(group)).with_suffix("").as_posix()


def _source_name(root: Path, project_root: Path) -> str:
    if root == (project_root / "configs").resolve():
        return "project"
    return "retrieval-core"


def _description(path: Path) -> str:
    data = read_yaml_mapping(path)
    metadata = data.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("description"):
        return str(metadata["description"])
    return "No description provided."


def _available_choices(project_root: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    roots = config_roots(project_dir=project_root)
    for group in GROUPS:
        seen: set[str] = set()
        choices: list[dict[str, str]] = []
        for root in roots:
            for path in _choice_files(root, group):
                name = _choice_name(path, root, group)
                if name in seen:
                    continue
                seen.add(name)
                choices.append(
                    {
                        "name": name,
                        "source": _source_name(root, project_root),
                        "description": _description(path),
                    }
                )
        result[group] = choices
    return result


def _defaults_choices(data: dict[str, Any]) -> dict[str, str]:
    selections: dict[str, str] = {}
    defaults = data.get("defaults", [])
    if not isinstance(defaults, list):
        return selections
    for item in defaults:
        if not isinstance(item, dict):
            continue
        for raw_group, choice in item.items():
            group = str(raw_group).removeprefix("override ").split("@", maxsplit=1)[0]
            group = group.lstrip("/")
            if group in GROUPS and isinstance(choice, str):
                selections[group] = choice
    return selections


def _existing_evidence(project_root: Path) -> tuple[dict[str, Counter[str]], dict[str, list[Any]]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    observed: dict[str, list[Any]] = defaultdict(list)
    experiments = project_root / "experiments"
    if not experiments.is_dir():
        return counts, observed

    for path in sorted(experiments.glob("*/configs/base-experiment-configs/*.yaml")):
        data = read_yaml_mapping(path)
        for group, choice in _defaults_choices(data).items():
            counts[group][choice] += 1
        index_id = data.get("selections", {}).get("index_id") if isinstance(data.get("selections"), dict) else None
        if index_id and "REPLACE_WITH" not in str(index_id):
            observed["index_ids"].append(str(index_id))
        top_k = data.get("pipeline", {}).get("components", {}).get("retriever", {}).get("init_parameters", {}).get("top_k")
        if top_k is not None:
            observed["retrieval_top_k"].append(top_k)
        metrics = data.get("metrics")
        if isinstance(metrics, list):
            observed["metrics"].append(metrics)

    for path in sorted(experiments.glob("*/configs/runs/*.yaml")):
        data = read_yaml_mapping(path)
        for group, choice in _defaults_choices(data).items():
            if group == "pipeline/inference":
                counts["treatment_pipeline"][choice] += 1
    return counts, observed


def _template_defaults(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "templates" / "retrieval-experiment" / "cookiecutter.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _recommendation(
    group: str,
    available: list[dict[str, str]],
    counts: dict[str, Counter[str]],
    template: dict[str, Any],
) -> dict[str, str] | None:
    counter_key = "treatment_pipeline" if group == "treatment_pipeline" else group
    counter = counts.get(counter_key, Counter())
    names = {item["name"] for item in available}
    for name, count in counter.most_common():
        if name in names:
            return {
                "name": name,
                "reason": f"it appears in {count} existing experiment config(s)",
            }

    template_key = {
        "dataset": "dataset_config",
        "runtime": "runtime_config",
        "pipeline/inference": "baseline_pipeline",
        "treatment_pipeline": "treatment_pipeline",
        "selections/embedding_model": "embedding_model",
    }[group]
    fallback = template.get(template_key)
    if fallback in names:
        return {"name": fallback, "reason": "it is the repository template default"}
    if available:
        return {
            "name": available[0]["name"],
            "reason": "no usage evidence exists; it is the first available choice",
        }
    return None


def discover(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    repo_root = next(
        (parent for parent in (project_root, *project_root.parents) if (parent / "templates" / "retrieval-experiment").is_dir()),
        project_root,
    )
    choices = _available_choices(project_root)
    counts, observed = _existing_evidence(project_root)
    template = _template_defaults(repo_root)
    treatment_choices = choices["pipeline/inference"]
    recommendations = {
        group: _recommendation(group, available, counts, template)
        for group, available in choices.items()
    }
    recommendations["treatment_pipeline"] = _recommendation(
        "treatment_pipeline", treatment_choices, counts, template
    )

    index_ids = discover_index_ids(project_root / "artifacts" / "indexes")
    return {
        "project_root": project_root.as_posix(),
        "choices": choices,
        "recommendations": recommendations,
        "existing_index_ids": index_ids,
        "observed_values": {
            "configured_index_ids": sorted(set(observed["index_ids"])),
            "retrieval_top_k": sorted(set(observed["retrieval_top_k"])),
            "metrics": observed["metrics"],
        },
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [f"# Experiment choices for `{result['project_root']}`", ""]
    for group, choices in result["choices"].items():
        recommendation = result["recommendations"].get(group)
        lines.extend([f"## {group}", ""])
        for index, choice in enumerate(choices, start=1):
            marker = " (recommended)" if recommendation and choice["name"] == recommendation["name"] else ""
            lines.append(
                f"{index}. `{choice['name']}`{marker} — {choice['description']} [{choice['source']}]"
            )
        if recommendation:
            lines.extend(
                ["", f"Recommendation: `{recommendation['name']}` because {recommendation['reason']}."]
            )
        lines.append("")
    lines.extend(["## Existing index IDs", ""])
    lines.extend(f"- `{index_id}`" for index_id in result["existing_index_ids"])
    if not result["existing_index_ids"]:
        lines.append("- None discovered.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()
    result = discover(args.project_root)
    print(json.dumps(result, indent=2) if args.format == "json" else _markdown(result))


if __name__ == "__main__":
    main()
