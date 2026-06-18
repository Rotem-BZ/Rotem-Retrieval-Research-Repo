# Retrieval Research Workflows

This repo is organized around three ideas:

1. Reusable retrieval behavior lives in Haystack components.
2. Experiment selection and parameterization lives in Hydra configs.
3. Long-running experiment workflows are split into explicit stages.

The scaffold includes both dummy components and real retrieval components. The
dummy pieces are still useful because they exercise the intended contracts
without requiring a model, document store, or external service.

## Repository Structure

The top-level folders separate experiment configuration, CCDS-style data,
documentation, source code, and tests:

- `configs/` contains Hydra entry points and reusable config groups.
- `data/` contains `raw`, `interim`, and `processed` dataset files. Generated
  data is ignored except for the small checked-in toy fixture.
- `docs/` contains workflow and design notes.
- `src/retrieval_research/` contains the Python package.
- `tests/` contains regression tests for metrics, config composition, and stage
  behavior.

See [components.md](components.md) for the current component inventory and which
pieces are native Haystack components versus repo-specific adapters.

Inside `src/retrieval_research/`, the main modules are:

- `cli.py` dispatches `stage <stage-name>` commands to stage runners.
- `command_builder.py` powers `build-command`, an interactive command builder
  for Hydra selections.
- `config.py` composes Hydra configs from the repository-level `configs/`
  directory.
- `components/` contains Haystack components that can later be shared with
  production code.
- `notebooks/` contains notebook-style Python scripts for data preparation and
  exploration.
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
uv run stage inference \
  dataset=toy \
  pipeline/inference@pipeline=rrf_fusion \
  component/biencoder@component.biencoder=sentence_transformers \
  component/cross_encoder@component.cross_encoder=ms_marco_minilm
```

This keeps written configs reusable: pipeline configs describe how components
are wired, component configs describe concrete implementations, and experiment
presets describe named combinations that are worth saving.

### Semantic Selections

Some selections are not themselves Haystack components, but they still define
the meaning of a pipeline. An embedding model is a good example: it determines
the checkpoint, query prefix, document prefix, normalization behavior, and
similarity function used by several components at once.

These selections live under the root `selections` namespace:

```yaml
selections:
  embedding_model:
    name: e5_small_v2
    artifact_name: e5_small_v2
    checkpoint: intfloat/e5-small-v2
    document_prefix: "passage: "
    query_prefix: "query: "
    normalize_embeddings: true
    similarity: cosine
```

The `pipeline` field in the final resolved config should remain exact Haystack
pipeline syntax. It should contain `components`, `connections`,
`max_runs_per_component`, and `metadata`, but not helper objects such as
`embedding_model`. Pipeline topology configs can still require semantic
selections by composing them into the root `selections` namespace:

```yaml
# configs/pipeline/indexing/dense_jsonl.yaml
defaults:
  - /selections/embedding_model@_global_.selections.embedding_model: ???
  - /component/document_preprocessor@components.document_prefixer: prefix_cleanup
  - /component/document_embedder@components.embedder: sentence_transformers
  - /component/indexer@components.indexer: jsonl_embeddings
  - _self_
```

Because the pipeline config is mounted at `pipeline`, component defaults mounted
at `components.*` land inside `pipeline.components.*`. The semantic model
selection lands at root under `selections.embedding_model`, where any component
can reference it:

```yaml
# configs/component/document_embedder/sentence_transformers.yaml
type: haystack.components.embedders.sentence_transformers_document_embedder.SentenceTransformersDocumentEmbedder
init_parameters:
  model: ${selections.embedding_model.checkpoint}
  normalize_embeddings: ${selections.embedding_model.normalize_embeddings}
```

The same config group can be mounted more than once for future multi-model
topologies. Prefer role names over numbered names:

```yaml
selections:
  candidate_embedding_model: ...
  rerank_embedding_model: ...
```

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
uv run stage inference \
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
uv run stage inference experiment/inference=toy_e5_lowercase
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
uv run stage indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl
```

To build and validate a command interactively without running an experiment:

```bash
uv run build-command
```

## Configuration Layout

Hydra config entry points live at the top of `configs/`:

- `configs/indexing.yaml`
- `configs/inference.yaml`
- `configs/evaluation.yaml`

Config groups provide reusable prefills:

- `configs/dataset/` contains dataset names and file paths.
- `configs/paths/` contains artifact layout choices.
- `configs/input_mapping/` contains reusable inference candidate-set recipes.
  The default `input_mapping=full` is virtual and uses all dataset queries and
  documents without writing a giant JSON file.
- `configs/selections/` contains semantic experiment selections such as embedding
  model families and checkpoints.
- `configs/component/` contains reusable Haystack component fragments.
- `configs/pipeline/indexing/` contains Haystack indexing pipelines.
- `configs/pipeline/inference/` contains Haystack inference pipelines.

Dataset records live as data files, not as Hydra config payloads. The toy
dataset is in:

- `data/processed/toy/documents.jsonl`
- `data/processed/toy/queries.jsonl`
- `data/processed/toy/qrels.jsonl`

BEIR datasets are prepared with the Python notebook script at
`src/retrieval_research/notebooks/prepare_beir.py`. It downloads raw archives to
`data/raw`, extracts them to `data/interim`, and writes repo-native JSONL files
to `data/processed`.

The indexing and inference configs each place a Haystack serialized pipeline
under the `pipeline` field. The Python runner resolves Hydra interpolation,
serializes that field to YAML, loads it with Haystack, and executes it as an
`AsyncPipeline`.

### Input Mappings

Inference always resolves an input mapping. By default, `input_mapping=full`
uses every query and every document in the selected dataset without storing a
large file of all ids. Generated mappings are selected as reusable recipes at
the root config level, then materialized for the selected dataset when inference
first needs them:

```bash
uv run stage inference \
  dataset=toy \
  input_mapping=dev_tiny \
  pipeline/inference@pipeline=dummy_keyword
```

Materialized mappings are plain JSON objects keyed by query id:

```json
{
  "q-1": ["doc-1", "doc-7", "doc-9"]
}
```

If a mapping includes only a subset of queries, inference runs only those
queries. Candidate ids are passed into retrievers as filters, including dense
JSONL retrievers; reranking-style pipelines can additionally receive materialized
candidate `Document` objects through `pipeline_run.candidate_document_inputs`.

Useful built-in recipes live under `configs/input_mapping/`:

- `full`: virtual default; all queries against all documents, with no mapping JSON.
- `judged_only`: all queries, but only documents with qrel annotations for each query.
- `dev_tiny`: two-query development pool with easy negatives and cross-query positives.
- `random_smoke`: two-query smoke-test pool with one random extra document per query.

When inference sees `input_mapping=dev_tiny`, it writes the materialized mapping
for the selected dataset if it does not already exist:

```text
data/processed/toy/input_mappings/dev_tiny.<recipe-hash>.json
data/processed/toy/input_mappings/dev_tiny.<recipe-hash>.meta.json
```

The mapping JSON remains pure candidate data. The `.meta.json` sidecar records
the generation seed, recipe hash, source paths, subset sizes, and candidate
count summary.
Generation always includes every document with any qrel annotation for each
selected query. Gold-passage negatives are sampled from documents relevant to a
different query while excluding every document annotated for the current query.

### Abstract E5 Dense Pipelines

The concrete `pipeline/indexing@pipeline=e5_jsonl` and
`pipeline/inference@pipeline=e5_jsonl` configs remain available as simple,
fully written E5 examples. For more composable experiments, use the abstract
dense topologies and select E5 through `selections/embedding_model`:

```powershell
uv run stage indexing `
  dataset=beir_scifact `
  pipeline/indexing@pipeline=dense_jsonl `
  selections/embedding_model=e5/small_v2
```

```powershell
uv run stage inference `
  dataset=beir_scifact `
  pipeline/inference@pipeline=dense_jsonl `
  selections/embedding_model=e5/small_v2 `
  pipeline_run.query_input.component=query_preprocessor `
  pipeline_run.query_input.parameter=text `
  retrieval.top_k=100
```

```powershell
uv run stage evaluation dataset=beir_scifact
```

To run the same model through the chunked topology, switch both pipeline
selections:

```powershell
uv run stage indexing `
  dataset=beir_scifact `
  pipeline/indexing@pipeline=dense_chunked_jsonl `
  selections/embedding_model=e5/small_v2
```

```powershell
uv run stage inference `
  dataset=beir_scifact `
  pipeline/inference@pipeline=dense_chunked_jsonl `
  selections/embedding_model=e5/small_v2 `
  pipeline_run.query_input.component=query_preprocessor `
  pipeline_run.query_input.parameter=text `
  retrieval.top_k=100
```

## Stage Workflow

Run indexing first:

```bash
uv run stage indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl
```

The default dummy stage writes:

```text
artifacts/indexes/toy.jsonl
```

Run inference after indexing:

```bash
uv run stage inference dataset=toy pipeline/inference@pipeline=dummy_keyword
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
uv run stage evaluation dataset=toy
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
uv run stage indexing dataset=toy pipeline/indexing@pipeline=dummy_jsonl
uv run stage inference dataset=toy pipeline/inference@pipeline=dummy_keyword retrieval.top_k=10
uv run stage evaluation dataset=toy metrics='["Recall@10","MRR@10","NDCG@10","Precision@10","HitRate@10"]'
```

When new datasets are added, place processed records under
`data/processed/<name>/` and create a small pointer config in
`configs/dataset/<name>.yaml`:

```yaml
name: my_dataset
documents_path: ${paths.processed_data_dir}/my_dataset/documents.jsonl
queries_path: ${paths.processed_data_dir}/my_dataset/queries.jsonl
qrels_path: ${paths.processed_data_dir}/my_dataset/qrels.jsonl
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

Then run:

```bash
uv run stage indexing dataset=my_dataset pipeline/indexing@pipeline=dummy_jsonl
uv run stage inference dataset=my_dataset pipeline/inference@pipeline=dummy_keyword
uv run stage evaluation dataset=my_dataset
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
uv run stage indexing dataset=my_dataset pipeline/indexing@pipeline=my_pipeline
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

