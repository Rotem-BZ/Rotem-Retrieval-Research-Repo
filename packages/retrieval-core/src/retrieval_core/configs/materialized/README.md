# Materialized configs

Store fully materialized configuration snapshots here.

These files are meant as reference artifacts, for example production-ready
configs or reviewed snapshots copied from a stage run's `resolved_config.yaml`.
They should not rely on Hydra defaults, config-group selections, or `???`
placeholders. If a file lives here, it should be readable as the complete
configuration for that run or environment.

Suggested layout:

- `production/` for production reference configs.
- `experiments/` for named, reviewed experiment snapshots worth keeping.

Use descriptive filenames that include the stage and purpose, such as
`production/inference_bge_reranker.yaml`.

Current examples:

- `production/toy_dense_indexing_reference.yaml` is adapted from
  `artifacts/runs/indexing/20260705_231537/resolved_config.yaml`.

Run it directly with:

```bash
uv run stage indexing --entrypoint src/retrieval_core/configs/materialized/production/toy_dense_indexing_reference.yaml
```
