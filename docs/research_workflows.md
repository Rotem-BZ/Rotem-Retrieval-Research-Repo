# Retrieval Research Workflows

This repo is organized around three ideas:

1. Reusable retrieval behavior lives in Haystack components.
2. Experiment selection and parameterization lives in Hydra configs.
3. Long-running experiment workflows are split into explicit stages.

The scaffold currently includes dummy components only. They are useful because
they exercise the intended contracts without committing the framework to a
specific retriever, document store, or evaluator too early.

## Repository Structure

The top-level folders separate experiment configuration, small datasets,
documentation, source code, and tests:

- `configs/` contains Hydra entry points and reusable config groups.
- `datasets/` contains local research datasets and toy fixtures.
- `docs/` contains workflow and design notes.
- `src/retrieval_research/` contains the Python package.
- `tests/` contains regression tests for metrics, config composition, and stage
  behavior.

See [components.md](components.md) for the current component inventory and which
pieces are native Haystack components versus repo-specific adapters.

Inside `src/retrieval_research/`, the main modules are:

- `cli.py` dispatches `rr <stage>` commands to stage runners.
- `config.py` composes Hydra configs from the repository-level `configs/`
  directory.
- `components/` contains Haystack components that can later be shared with
  production code.
- `stages/` contains stage orchestration code for indexing, inference, and
  evaluation.
- `pipelines.py` converts Hydra pipeline config into Haystack `AsyncPipeline`
  objects.
- `io.py` and `metrics.py` contain shared IO and evaluation helpers.

## Design Philosophy

The framework separates four concerns that are often coupled in research repos:

1. **Stage orchestration** decides which workflow is running: indexing,
   inference, evaluation, or a future stage.
2. **Pipeline topology** describes the Haystack graph for a stage.
3. **Component options** describe reusable choices for individual components,
   such as a query preprocessor or bi-encoder embedder.
4. **Experiment presets** can capture favorite combinations without replacing
   the reusable pieces.

The goal is to avoid one config file per combination. For example, if there are
several query preprocessors and several bi-encoder embedders, do not create
pipeline configs for every preprocessor/embedder pair. Instead, keep one
pipeline topology and compose the selected component configs into it.

Pipeline YAML files should stay abstract whenever they represent reusable
topology. A topology config may name graph nodes and connections, but it should
not quietly choose a concrete model, checkpoint, vendor, or indexing backend
unless that choice is truly part of the topology itself. Required implementation
slots should use Hydra's `???` sentinel so configuration fails early until the
user selects the missing component config.

For example, an RRF fusion topology can require the user to select both the
bi-encoder and cross-encoder implementations at launch time:

```yaml
# configs/pipeline/inference/rrf_fusion.yaml
defaults:
  - /component/biencoder@component.biencoder: ???
  - /component/cross_encoder@component.cross_encoder: ???
  - /component/fusion@component.fusion: rrf
  - _self_

components:
  biencoder: ${component.biencoder}
  cross_encoder: ${component.cross_encoder}
  fusion: ${component.fusion}

connections:
  - sender: biencoder.documents
    receiver: cross_encoder.documents
  - sender: biencoder.documents
    receiver: fusion.biencoder_documents
  - sender: cross_encoder.documents
    receiver: fusion.cross_encoder_documents

max_runs_per_component: 100
metadata: {}
```

The command line then supplies the model choices separately from the topology:

```bash
uv run rr inference \
  dataset=toy \
  pipeline/inference@pipeline=rrf_fusion \
  component/biencoder@component.biencoder=sentence_transformers \
  component/cross_encoder@component.cross_encoder=ms_marco_minilm
```

This keeps written configs reusable: pipeline configs describe how components
are wired, component configs describe concrete implementations, and experiment
presets describe named combinations that are worth saving.

An inference pipeline topology can reference component groups:

```yaml
# configs/pipeline/inference/biencoder.yaml
components:
  query_preprocessor: ${component.query_preprocessor}
  query_embedder: ${component.query_embedder}
  retriever:
    type: my_project.components.VectorRetriever
    init_parameters:
      index_path: ${paths.index_dir}/${dataset.name}.jsonl
      top_k: ${retrieval.top_k}

connections:
  - sender: query_preprocessor.query
    receiver: query_embedder.text
  - sender: query_embedder.embedding
    receiver: retriever.query_embedding

max_runs_per_component: 100
metadata: {}
```

Then reusable component options can live under separate config groups:

```yaml
# configs/component/query_preprocessor/lowercase.yaml
type: my_project.components.LowercaseQueryPreprocessor
init_parameters:
  strip_whitespace: true
```

```yaml
# configs/component/query_embedder/e5_small.yaml
type: my_project.components.E5QueryEmbedder
init_parameters:
  model: intfloat/e5-small-v2
  prefix: "query: "
  device: cuda
```

The command line can then compose the pieces:

```bash
uv run rr inference \
  dataset=toy \
  pipeline/inference@pipeline=biencoder \
  component/query_preprocessor=lowercase \
  component/query_embedder=e5_small
```

For commonly used combinations, add experiment presets that select the dataset,
pipeline topology, and component options:

```yaml
# configs/experiment/inference/toy_e5_lowercase.yaml
defaults:
  - /dataset: toy
  - /pipeline/inference@pipeline: biencoder
  - /component/query_preprocessor: lowercase
  - /component/query_embedder: e5_small
```

Then launch the preset with:

```bash
uv run rr inference experiment/inference=toy_e5_lowercase
```

As a rule of thumb, use command-line overrides for small parameter changes, use
component config groups for reusable implementation choices, use pipeline config
groups for graph topology, and use experiment presets for named recipes.

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
- `datasets/toy/qrels.jsonl`
- `datasets/toy/input_mapping.json`

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
artifacts/predictions/toy.json
```

Prediction artifacts are JSON objects keyed first by query id and then by
document or chunk id. Each document entry contains the retrieved content, score,
and metadata.

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
qrels_path: ${paths.project_root}/datasets/my_dataset/qrels.jsonl
input_mapping_path: null
```

Document JSONL records should look like:

```json
{"id":"doc-1","content":"Text to index.","meta":{}}
```

Query JSONL records should look like:

```json
{"id":"q-1","text":"Search text."}
```

Qrels JSONL records should look like:

```json
{"query_id":"q-1","document_id":"doc-1","relevance":1}
```

`input_mapping_path` is optional and can be `null`. When present, it should point
to a JSON object that maps each query id to candidate document ids for workflows
such as reranking:

```json
{
  "q-1": ["doc-1", "doc-7", "doc-9"]
}
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
