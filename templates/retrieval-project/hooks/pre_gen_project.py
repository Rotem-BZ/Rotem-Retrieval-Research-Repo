"""Reject generated names that would produce invalid Python or Hydra projects."""

from __future__ import annotations

import keyword
import re
import sys


VALUES = {
    "project_slug": "{{ cookiecutter.project_slug }}",
    "package_name": "{{ cookiecutter.package_name }}",
    "pipeline_name": "{{ cookiecutter.pipeline_name }}",
    "component_class_name": "{{ cookiecutter.component_class_name }}",
    "beir_dataset": "{{ cookiecutter.beir_dataset }}",
    "dataset_config": "{{ cookiecutter.dataset_config }}",
    "embedding_model": "{{ cookiecutter.embedding_model }}",
}

RULES = {
    "project_slug": r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$",
    "package_name": r"^[a-z][a-z0-9_]*$",
    "pipeline_name": r"^[a-z][a-z0-9_]*$",
    "component_class_name": r"^[A-Z][A-Za-z0-9]*$",
    "beir_dataset": r"^[a-z][a-z0-9_-]*$",
    "dataset_config": r"^[a-z][a-z0-9_]*$",
    "embedding_model": r"^[a-zA-Z0-9][a-zA-Z0-9_/-]*$",
}


for field, pattern in RULES.items():
    if not re.fullmatch(pattern, VALUES[field]):
        print(f"ERROR: {field}={VALUES[field]!r} does not match {pattern!r}.", file=sys.stderr)
        sys.exit(1)

if keyword.iskeyword(VALUES["package_name"]):
    print("ERROR: package_name must not be a Python keyword.", file=sys.stderr)
    sys.exit(1)
