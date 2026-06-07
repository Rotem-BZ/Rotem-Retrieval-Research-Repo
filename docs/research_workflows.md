# Retrieval Research Workflows

This repo is organized around three ideas:

1. Reusable retrieval behavior lives in Haystack components.
2. Experiment selection and parameterization lives in Hydra configs.
3. Long-running experiment workflows are split into explicit stages.

The scaffold currently includes dummy components only. They are useful because
they exercise the intended contracts without committing the framework to a
specific retriever, document store, or evaluator too early.

## Environment

Install Python dependencies with uv:

```bash
uv sync --extra dev
```

For day-to-day execution, run the stage entry points through uv:

```bash
uv run rr indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl
```

## Configuration Layout

Hydra config entry points live at the top of `configs/`:

- `configs/indexing.yaml`
- `configs/inference.yaml`
- `configs/evaluation.yaml`

Config groups provide reusable prefills:

- `configs/dataset/` contains dataset names and file paths.
- `configs/paths/` contains artifact layout choices.
- `configs/pipeline/indexing/` contains Haystack indexing pipelines.
- `configs/pipeline/inference/` contains Haystack inference pipelines.

Dataset records live as data files, not as Hydra config payloads. The toy
dataset is in:

- `datasets/toy/documents.jsonl`
- `datasets/toy/queries.jsonl`

The indexing and inference configs each place a Haystack serialized pipeline
under the `pipeline` field. The Python runner resolves Hydra interpolation,
serializes that field to YAML, loads it with Haystack, and executes it as an
`AsyncPipeline`.

## Stage Workflow

Run indexing first:

```bash
uv run rr indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl
```

The default dummy stage writes:

```text
artifacts/indexes/toy.jsonl
```

Run inference after indexing:

```bash
uv run rr inference dataset=toy pipeline/inference@pipeline=dummy_keyword
```

The default inference stage reads the JSONL index and writes:

```text
artifacts/predictions/toy.jsonl
```

Run evaluation after inference:

```bash
uv run rr evaluation dataset=toy
```

The default evaluator writes:

```text
artifacts/metrics/toy.json
```

Each stage also writes a run folder under `artifacts/runs/<stage>/<run_id>/`
with the resolved config and a small result summary.

## Mixing Configs

Hydra config groups are intended to become the main experiment interface. The
default command should be close to the final experiment specification, with only
small overrides on the command line.

Examples:

```bash
uv run rr indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl
uv run rr inference dataset=toy pipeline/inference@pipeline=dummy_keyword retrieval.top_k=10
uv run rr evaluation dataset=toy metrics='[{name: recall_at_k, k: 10}, {name: mrr_at_k, k: 10}]'
```

When new datasets are added, place records under `datasets/<name>/` and create a
small pointer config in `configs/dataset/<name>.yaml`:

```yaml
name: my_dataset
documents_path: ${paths.project_root}/datasets/my_dataset/documents.jsonl
queries_path: ${paths.project_root}/datasets/my_dataset/queries.jsonl
```

Document JSONL records should look like:

```json
{"id":"doc-1","content":"Text to index.","meta":{}}
```

Query JSONL records should look like:

```json
{"id":"q-1","text":"Search text.","relevant_document_ids":["doc-1"]}
```

Then run:

```bash
uv run rr indexing dataset=my_dataset pipeline/indexing@pipeline=dummy_jsonl
uv run rr inference dataset=my_dataset pipeline/inference@pipeline=dummy_keyword
uv run rr evaluation dataset=my_dataset
```

## Replacing Dummy Components

Production-ready retrieval code should be added as Haystack components under
`src/retrieval_research/components/` or imported from a production package.

To add a new indexing pipeline, create a config like:

```yaml
components:
  converter:
    type: my_package.components.MyConverter
    init_parameters: {}
  writer:
    type: my_package.components.MyIndexer
    init_parameters:
      output_path: ${paths.index_dir}/${dataset.name}
connections:
  - sender: converter.documents
    receiver: writer.documents
max_runs_per_component: 100
metadata: {}
```

Save it under `configs/pipeline/indexing/my_pipeline.yaml` and select it with:

```bash
uv run rr indexing dataset=my_dataset pipeline/indexing@pipeline=my_pipeline
```

Inference pipelines follow the same pattern under
`configs/pipeline/inference/`. The inference stage only assumes that one
configured component receives the query text and one configured component output
contains retrieved `Document` objects:

```yaml
pipeline_run:
  query_input:
    component: retriever
    parameter: query
  documents_output:
    component: retriever
    field: documents
```

## Parallel Execution

The first scaffold uses Haystack `AsyncPipeline` for every pipeline invocation
and passes `pipeline_run.concurrency_limit` into `run_async`. This gives each
pipeline run an explicit concurrency budget.

For larger sweeps, use Hydra overrides and launchers. A future extension can add
Hydra launcher configs for local multiprocessing, Slurm, Kubernetes, or cloud
batch systems without changing component code.

## Design Notes

The framework keeps stage orchestration thin on purpose. Indexing, retrieval,
and later reranking/generation logic should live in Haystack components so the
same components can be imported by production services. Hydra should own
experiment assembly: datasets, pipeline variants, artifact locations, and small
runtime overrides.

Evaluation is not implemented as a Haystack pipeline yet because metrics often
need dataset-level aggregation. It is still a first-class stage and can be
swapped for a richer evaluator once prediction schemas stabilize.
