"""Reject experiment values that would produce invalid Hydra choices."""

from __future__ import annotations

import json
import re
import sys


VALUES = {
    "experiment_slug": json.loads(r'''{{ cookiecutter.experiment_slug | tojson }}'''),
    "hypothesis": json.loads(r'''{{ cookiecutter.hypothesis | tojson }}'''),
    "dataset_config": json.loads(r'''{{ cookiecutter.dataset_config | tojson }}'''),
    "runtime_config": json.loads(r'''{{ cookiecutter.runtime_config | tojson }}'''),
    "embedding_model": json.loads(r'''{{ cookiecutter.embedding_model | tojson }}'''),
    "index_id": json.loads(r'''{{ cookiecutter.index_id | tojson }}'''),
    "baseline_pipeline": json.loads(r'''{{ cookiecutter.baseline_pipeline | tojson }}'''),
    "treatment_pipeline": json.loads(r'''{{ cookiecutter.treatment_pipeline | tojson }}'''),
}

RULES = {
    "experiment_slug": r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$",
    "dataset_config": r"^[a-z][a-z0-9_]*$",
    "runtime_config": r"^[a-z][a-z0-9_]*$",
    "embedding_model": r"^[A-Za-z0-9][A-Za-z0-9_/-]*$",
    "index_id": r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    "baseline_pipeline": r"^[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)+$",
    "treatment_pipeline": r"^[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)+$",
}


for field, pattern in RULES.items():
    if not re.fullmatch(pattern, VALUES[field]):
        print(f"ERROR: {field}={VALUES[field]!r} does not match {pattern!r}.", file=sys.stderr)
        sys.exit(1)

if not VALUES["hypothesis"].strip():
    print("ERROR: hypothesis must not be empty.", file=sys.stderr)
    sys.exit(1)

try:
    retrieval_top_k = int("{{ cookiecutter.retrieval_top_k }}")
except ValueError:
    print("ERROR: retrieval_top_k must be an integer.", file=sys.stderr)
    sys.exit(1)
if retrieval_top_k <= 0:
    print("ERROR: retrieval_top_k must be positive.", file=sys.stderr)
    sys.exit(1)

try:
    evaluation_metrics = json.loads(
        json.loads(r'''{{ cookiecutter.evaluation_metrics | tojson }}''')
    )
except json.JSONDecodeError as error:
    print(f"ERROR: evaluation_metrics must be a JSON list: {error}.", file=sys.stderr)
    sys.exit(1)
if not isinstance(evaluation_metrics, list) or not evaluation_metrics or not all(
    isinstance(metric, str) and metric.strip() for metric in evaluation_metrics
):
    print("ERROR: evaluation_metrics must be a non-empty JSON list of metric names.", file=sys.stderr)
    sys.exit(1)
