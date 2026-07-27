"""Reject experiment values that would produce invalid Hydra choices."""

from __future__ import annotations

import re
import sys


VALUES = {
    "experiment_slug": "{{ cookiecutter.experiment_slug }}",
    "dataset_config": "{{ cookiecutter.dataset_config }}",
    "embedding_model": "{{ cookiecutter.embedding_model }}",
    "index_id": "{{ cookiecutter.index_id }}",
    "baseline_pipeline": "{{ cookiecutter.baseline_pipeline }}",
    "treatment_pipeline": "{{ cookiecutter.treatment_pipeline }}",
}

RULES = {
    "experiment_slug": r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$",
    "dataset_config": r"^[a-z][a-z0-9_]*$",
    "embedding_model": r"^[A-Za-z0-9][A-Za-z0-9_/-]*$",
    "index_id": r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    "baseline_pipeline": r"^[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)+$",
    "treatment_pipeline": r"^[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)+$",
}


for field, pattern in RULES.items():
    if not re.fullmatch(pattern, VALUES[field]):
        print(f"ERROR: {field}={VALUES[field]!r} does not match {pattern!r}.", file=sys.stderr)
        sys.exit(1)
