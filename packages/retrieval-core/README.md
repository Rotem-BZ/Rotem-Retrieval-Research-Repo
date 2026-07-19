# retrieval-core

Internal shared orchestration for projects in this monorepo. It owns stage execution,
artifact provenance, metrics, input mappings, and the common Hydra configuration tree.

This distribution is consumed from a local editable path and is not intended to be
published.

## Package structure

Feature modules remain directly under `retrieval_core` (`stages`, `sweeps`,
`input_mapping`, and the CLI entry points). Shared infrastructure lives under
`retrieval_core.utils` and is grouped by responsibility:

- `artifacts`: immutable run manifests and artifact resolution
- `config`: Hydra composition and config-root discovery
- `console`: stage-oriented terminal output
- `evaluation`: ranking metrics
- `io`: paths, JSON/JSONL, prediction artifacts, text, and YAML serialization
- `pipelines`: Haystack pipeline loading
- `hashing.py` and `time.py`: small cross-cutting primitives

Import shared helpers through their focused package, for example
`from retrieval_core.utils.io import read_json` or
`from retrieval_core.utils.config import compose_stage_config`.

The `prepare-experiment` and `run-experiment` commands organize resolved run configs
under a project's `experiments/<slug>/runs/` directory. The `sweeps` module name and
the older `prepare-sweep`/`run-sweep` entry points are retained for compatibility.
Stage outputs continue to use `artifacts/runs/` and carry experiment linkage in their
manifests.
