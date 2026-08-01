# retrieval-core

Internal shared orchestration for projects in this monorepo. It owns stage execution,
artifact provenance, metrics, input mappings, and the common Hydra configuration tree.

This distribution is consumed from a local editable path and is not intended to be
published.

## Package structure

Feature modules remain directly under `retrieval_core` (`stages`, `input_mapping`,
`data_schema`, and the stage CLI). `EvaluationDataSchema` is the single source of
truth for evaluation-file field names and identity validation; query and document
content is materialized when stage inputs are constructed. Shared infrastructure lives under
`retrieval_core.utils` and is grouped by responsibility:

- `artifacts`: immutable run manifests and artifact resolution
- `config`: Hydra composition and config-root discovery
- `console`: compatibility wrappers for stage lifecycle messages
- `evaluation`: ranking metrics
- `io`: paths, JSON/JSONL, prediction artifacts, text, and YAML serialization
- `logging.py`: fixed standard-library logging policy and run-file handlers
- `pipelines`: Haystack pipeline loading
- `hashing.py` and `time.py`: small cross-cutting primitives

Import shared helpers through their focused package, for example
`from retrieval_core.utils.io import read_json` or
`from retrieval_core.utils.config import compose_stage_config`.

Developer-only command building and GNU Screen experiment orchestration live in the
repository-level `awesome-dev-tools/` directory instead of this runtime package. Stage
outputs use `artifacts/runs/`, carry experiment linkage in their manifests, and include
a `run.log` containing UTC-timestamped diagnostics. Console logs are written to stderr at
`INFO`; run files include first-party `DEBUG` events. This policy is fixed in code and is
not part of Hydra experiment configuration.

The indexing stage reads dataset JSONL incrementally in
`runtime.indexing_batch_size` groups and awaits one pipeline execution per group.
It writes through a stage-owned sibling temporary artifact and publishes the
selected index atomically only after every batch succeeds. The result and manifest
record aggregate document and batch counts.
