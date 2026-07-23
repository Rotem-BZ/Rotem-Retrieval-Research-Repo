# Retrieval Research Workflows

This repo is organized around three ideas:

1. Reusable retrieval behavior lives in Haystack components.
2. Experiment selection and parameterization lives in Hydra configs.
3. Long-running experiment workflows are split into explicit stages.

The scaffold includes both minimal scaffold components and real retrieval components. The
scaffold pieces are still useful because they exercise the intended contracts
without requiring a model, document store, or external service.

## Research Lifecycle

Use the repository as an evidence-producing workflow rather than as a collection
of isolated commands:

1. Create `projects/<project>/experiments/<experiment-slug>/` and write a falsifiable
   hypothesis in its `experiment.md` card.
2. Identify one controlled treatment change, its baseline, the primary metric,
   and the settings that must remain fixed.
3. Select or implement the required component, pipeline topology, dataset,
   semantic selections, and input mapping.
4. Exercise configuration and pipeline changes with focused tests before spending compute.
5. Reuse exact upstream artifacts. A treatment that changes only inference should
   normally share the baseline's index, mapping, dataset, and qrels.
6. Put complete shared stage configurations in
   `projects/<project>/experiments/<experiment-slug>/configs/base-experiment-configs/`,
   then record each invocation as a minimal Hydra entrypoint in that experiment's
   `configs/runs/` directory.
7. Run inference and evaluation with immutable, exact run ids. Large stage artifacts
   remain below `artifacts/runs/`; their manifests link back to the experiment.
8. Inspect aggregate and query-level behavior in the experiment's `analysis.ipynb`,
   then write `report.md` beside the card from manifests, resolved configs, results,
   predictions, and metrics—not from remembered commands.

Repository-local agent skills support the same lifecycle:

- `create-experiment-card` records a concise experiment description and a
  user-supplied hypothesis.
- `implement-new-component` adds reusable Haystack behavior.
- `implement-new-stage` adds a new artifact-producing workflow phase.
- `generate-experiment-report` checks provenance and reports completed results.

## Repository Structure

The monorepo separates reusable components, shared orchestration, and isolated
research projects:

- `packages/retrieval-core/src/retrieval_core/configs/` contains shared Hydra entry
  points and reusable config groups.
- `data/processed/toy/` at the repository root is the checked-in core test fixture.
  Each research project normally owns its own `data/` and `artifacts/` directories.
- `docs/` contains workflow and design notes.
- `packages/retrieval-core/` contains stage orchestration and its regression tests.
- `packages/retrieval-components/` contains reusable Haystack components and tests.
- `projects/` contains independently locked research projects and their config
  overlays. Each project groups cards, run definitions, analysis, and reports below
  `experiments/<experiment-slug>/`.

See the [retrieval-components README](../packages/retrieval-components/README.md#available-components)
for the current component inventory and which pieces are native Haystack components
versus repo-specific adapters.

The main runtime modules are:

- `cli.py` dispatches `stage <stage-name>` commands to stage runners.
- `awesome-dev-tools/` contains the interactive command builder and GNU Screen experiment tools.
- reusable components live separately under `packages/retrieval-components/`.
- `input_mapping/` owns candidate-set recipes and materialized mappings.
- `notebooks/` contains cell-marked data-preparation scripts such as `prepare_beir.py`.
- `stages/` contains orchestration for `prepare_mapping`, indexing, inference,
  and evaluation.
- `utils/` groups shared helpers by responsibility: artifacts, config, console,
  evaluation, IO, pipelines, hashing, and time.

The separate top-level `packages/retrieval-core/src/hydra_plugins/` namespace is
intentional. Hydra auto-discovers its search-path plugin, which supplies the fallback
roots configured for each composition. An experiment entrypoint composes in this
order: experiment `configs/`, project `configs/`, then
`pkg://retrieval_core.configs`. That ordering lets an experiment or project override
only the config groups it owns while using broader defaults as fallbacks.

## Design Philosophy

The framework separates four concerns that are often coupled in research repos:

1. **Stage orchestration** decides which workflow is running: mapping
   preparation, indexing, inference, evaluation, or a future registered stage.
2. **Pipeline topology** describes the Haystack graph for a stage.
3. **Component options** describe reusable choices for individual components,
   such as a query preprocessor or bi-encoder embedder.
4. **Research protocol** records hypotheses, controlled differences, metrics,
   and decision rules independently of the executable configuration.

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

For example, the shared dense inference topology fixes the graph while requiring
the semantic embedding-model selection:

```yaml
# configs/pipeline/inference/retrieve/dense_jsonl.yaml
defaults:
  - /selections/embedding_model@_global_.selections.embedding_model: ???
  - /component/query_preprocessor@components.query_preprocessor: prefix_cleanup
  - /component/query_embedder@components.query_embedder: sentence_transformers
  - /component/retriever@components.retriever: jsonl_embeddings
  - _self_

components:
  input:
    type: retrieval_components.interfaces.stage_io.InferenceInput
  output:
    type: retrieval_components.interfaces.stage_io.InferenceOutput

connections:
  - sender: input.query
    receiver: query_preprocessor.text
  - sender: query_preprocessor.text
    receiver: query_embedder.text
  - sender: query_embedder.embedding
    receiver: retriever.query_embedding
  - sender: retriever.documents
    receiver: output.documents

max_runs_per_component: 100
metadata: {}
```

The command line supplies the model selection separately from the topology:

```bash
uv run stage inference \
  dataset=beir_scifact \
  runtime=gpu \
  pipeline/inference@pipeline=retrieve/dense_jsonl \
  selections/embedding_model=e5/small_v2 \
  selections.index_id=YOUR_INDEX_ID
```

This keeps written configs reusable: pipeline configs describe how components
are wired, component configs describe concrete implementations, and semantic
selections keep settings shared by several components consistent.

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
# configs/pipeline/indexing/dense/documents_jsonl.yaml
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

Project-specific pipeline configs can reuse the shared component groups while
inserting one local treatment component. The query-repetition project follows
this pattern:

```yaml
# projects/query-repetition-e5/configs/pipeline/inference/query_repetition_e5/dense_query_repetition.yaml
defaults:
  - /selections/embedding_model@_global_.selections.embedding_model: ???
  - /component/query_preprocessor@components.query_preprocessor: prefix_cleanup
  - /component/query_embedder@components.query_embedder: sentence_transformers
  - /component/retriever@components.retriever: jsonl_embeddings
  - _self_

components:
  input:
    type: retrieval_components.interfaces.stage_io.InferenceInput
  query_repeater:
    type: query_repetition_e5.components.QueryRepeater
    init_parameters:
      separator: " "
  output:
    type: retrieval_components.interfaces.stage_io.InferenceOutput
```

Its graph routes `input.query` through `query_repeater` before the shared query
preprocessor and embedder. The baseline uses the shared `retrieve/dense_jsonl` topology,
so the local component is the intended difference.

As a rule of thumb, use command-line overrides for small scalar changes,
component config groups for reusable implementations, pipeline config groups for
graph topology, and experiment cards for the research claim and decision rule.
If a fully resolved configuration must be preserved as a runnable reference,
store it under the appropriate `configs/materialized/` tree and pass the YAML file
to `stage <stage> --entrypoint`.

## Environment

Each research project owns its environment and lockfile. Run project commands
from that project's directory so `paths.project_root: .`, `data/`, `artifacts/`,
and the project-local `configs/` tree resolve consistently:

```powershell
Set-Location projects/query-repetition-e5
uv sync --extra dev
```

For day-to-day execution, run the stage entry points through that environment:

```bash
uv run stage --help
```

The repository-root `data/processed/toy/` fixture is used by retrieval-core smoke
tests. It is not automatically copied into every generated research project.

To build a command interactively without running an experiment:

```bash
uv run python ../../awesome-dev-tools/interactive_build_command.py
```

For `prepare_mapping`, the builder suggests a unique run id from the selected
dataset and recipe, such as `toy-dev-tiny`, and prompts before adding it to the
command. Press Enter to accept the suggestion or enter another folder-safe id.
For evaluation, it scans completed inference manifests and presents only runs for
the selected dataset; older runs are matched through their saved resolved config.

After required Hydra choices are selected, the builder can review the selected
config graph. From there you can switch leaf fields such as `selections.input_mapping`,
enter nested YAML configs such as selected embedding models, and render edited
leaf fields as command-line overrides.

## Configuration Layout

Composition can involve three distinct config roots:

1. `packages/retrieval-core/src/retrieval_core/configs/` is the packaged core root.
2. `projects/<project>/configs/` is the project overlay.
3. `projects/<project>/experiments/<experiment-slug>/configs/` is the experiment
   overlay and the home of checked-in run entrypoints.

For an experiment entrypoint, Hydra searches those roots from most specific to least
specific: experiment, project, then core. The CLI infers the experiment and project
roots from the path passed to `--entrypoint`; YAML files do not import those roots.

Shared Hydra stage entry points live under the core `configs/stages/` group:

- `configs/stages/indexing.yaml`
- `configs/stages/inference.yaml`
- `configs/stages/evaluation.yaml`
- `configs/stages/prepare_mapping.yaml`

Without `--entrypoint`, `stage` composes the selected bare stage name, such as
`inference`, against the current project's `configs/` tree when the working directory
has one, then the core tree. The name resolves to `stages/inference` when no explicit
top-level config with that name exists. The CLI never infers an experiment from the
working directory; select an experiment run by passing its YAML path explicitly.

Config groups provide reusable prefills:

- `configs/dataset/` contains dataset names and file paths.
- `configs/paths/` contains artifact layout choices.
- `configs/input_mapping_recipe/` contains reusable candidate-set recipes for the
  `prepare_mapping` stage. Inference uses all dataset queries and documents when
  its `selections.input_mapping` value is null.
- `configs/runtime/` contains the required `gpu` and `cpu` execution profiles.
- `configs/selections/` contains semantic experiment selections such as embedding
  model families and checkpoints.
- `configs/component/` contains reusable Haystack component fragments.
- `configs/pipeline/indexing/` contains Haystack indexing pipelines.
- `configs/pipeline/inference/` contains Haystack inference pipelines.

Pipeline choices are grouped by intent below each stage. See
`configs/pipeline/README.md` for the concise choice catalog.

### Ownership namespaces for overlay choices

Reusable choices in a project overlay must keep the existing Hydra group and add
the owning project's snake-case Python package name as the first choice path
segment:

```text
projects/query-repetition-e5/configs/
└── pipeline/inference/
    └── query_repetition_e5/
        └── dense_query_repetition.yaml
```

Select that project-owned topology as:

```yaml
- override /pipeline/inference@pipeline: query_repetition_e5/dense_query_repetition
```

The same rule applies to project-owned `component` and `selections` choices. For
example, an experimental-components topology selects its local embedder with:

```yaml
- /component/query_embedder@components.query_embedder: experimental_components/fastembed_dense
```

Core-owned choices remain unqualified, such as `retrieve/dense_jsonl`,
`sentence_transformers`, and `e5/small_v2`. This makes ownership visible at every
selection site without changing the Hydra group being overridden. Do not wrap the
group itself in a directory such as `project-configs/pipeline/inference`; that
creates a different Hydra group and will not satisfy the core stage's
`pipeline/inference` override slot.

If a reusable choice truly belongs to only one experiment, use the normalized
experiment slug as the first choice segment under that experiment's config
overlay. Base experiment configs and concrete files under `configs/runs/` are
entrypoints rather than reusable choices, so they remain directly in their
established directories.

Dataset records live as data files, not as Hydra config payloads. The core test
fixture is in:

- `data/processed/toy/documents.jsonl`
- `data/processed/toy/queries.jsonl`
- `data/processed/toy/qrels.jsonl`

The canonical field names live in the `EvaluationDataSchema` dataclass in
`retrieval_core.data_schema`. `IN` is the join key shared by query and qrel
records; `query_id` remains the query's external identifier. Records may include
any additional fields.

BEIR datasets are prepared with the cell-marked Python script at
`packages/retrieval-core/src/retrieval_core/notebooks/prepare_beir.py`, exposed as
the `prepare-beir` command. When invoked from a research project, it downloads
raw archives to that project's `data/raw`, extracts them to `data/interim`, and
writes repo-native JSONL files to `data/processed`:

```bash
uv run prepare-beir --data-dir data --dataset scifact
```

Project analysis uses a real Jupyter notebook stored with its experiment. Add exact
inference run ids to `experiments/<experiment-slug>/analysis.ipynb`; it resolves
predictions through each manifest and builds qrel-enriched `predictions_df` and
`query_summary_df` tables ready for project-specific plots.

The indexing and inference configs each place a Haystack serialized pipeline
under the `pipeline` field. The Python runner resolves Hydra interpolation,
serializes that field to YAML, loads it with Haystack, and executes it as an
`AsyncPipeline`.

Index-backed pipeline templates derive their component paths from
`paths.indexes_dir` and the global `selections.index_id`. Indexing writes
`<indexes-dir>/<index-id>/index.jsonl`, and inference validates that exact artifact
before loading Haystack. The pipeline defaults list mounts the index selection into
the global `selections` package, just like an embedding-model selection. Candidate-only
pipelines omit that default, have no `index_path` parameter, and therefore compose
without an `index_id` field.

### Input Mappings

Inference uses every query and every document in the selected dataset when
`selections.input_mapping` is null. Smaller candidate sets are selected as reusable recipes
for `prepare_mapping` and written under the stage run id:

```bash
uv run stage prepare_mapping \
  dataset=beir_scifact \
  input_mapping_recipe=dev_tiny \
  stage.run_id=scifact_dev_tiny
```

Inference selects the prepared folder by name under `paths.input_mappings_dir`,
so the chosen artifact is explicit and can be reused by any number of runs:

```bash
uv run stage inference \
  dataset=beir_scifact \
  runtime=cpu \
  selections.input_mapping=scifact_dev_tiny \
  pipeline/inference@pipeline=rerank/bi_encoder \
  selections/embedding_model=e5/small_v2
```

The interactive command builder scans that directory after composing an inference
command. It offers all completed mappings whose `meta.json` names the selected
dataset, plus an all-documents option that leaves `selections.input_mapping` null.
For index-backed pipelines, the input-mapping menu appears immediately before the
required index-id menu.

Materialized mappings are plain JSON objects keyed by query input (`IN`):

```json
{
  "q-1": ["doc-1", "doc-7", "doc-9"]
}
```

If a mapping includes only a subset of queries, inference runs only those
queries. Query metadata, candidate ids, and materialized candidate `Document`
objects are passed to each inference pipeline through the fixed `input` interface
component; each pipeline decides which internal components consume them.

Useful built-in recipes live under `configs/input_mapping_recipe/`:

- `judged_only`: all queries, but only documents with qrel annotations for each query.
- `dev_tiny`: two-query development pool with easy negatives and cross-query positives.
- `random_smoke`: two-query smoke-test pool with one random extra document per query.

Prepared mappings are stored outside the dataset tree:

```text
artifacts/input_mappings/<run-id>/input_mapping.json
artifacts/input_mappings/<run-id>/meta.json
```

The mapping JSON remains pure candidate data. `meta.json` records the run id,
dataset, recipe parameters, subset sizes, and candidate-count summary. Run ids
are unique: preparation refuses to overwrite an existing mapping directory.
Generation always includes every document with any qrel annotation for each
selected query. Gold-passage negatives are sampled from documents relevant to a
different query while excluding every document annotated for the current query.

A complete reranking example over the toy dataset is:

```powershell
uv run stage prepare_mapping `
  dataset=toy `
  input_mapping_recipe=dev_tiny `
  stage.run_id=toy_dev_tiny

uv run stage inference `
  dataset=toy `
  runtime=cpu `
  selections.input_mapping=toy_dev_tiny `
  pipeline/inference@pipeline=rerank/bi_encoder `
  selections/embedding_model=e5/small_v2 `
  stage.run_id=toy_e5_rerank
```

### Dense Pipelines

Dense topologies remain model-agnostic. Select the embedding model separately
through `selections/embedding_model`:

```powershell
uv run stage indexing `
  dataset=beir_scifact `
  runtime=gpu `
  pipeline/indexing@pipeline=dense/documents_jsonl `
  selections/embedding_model=e5/small_v2 `
  selections.index_id=YOUR_NEW_INDEX_ID
```

```powershell
uv run stage inference `
  dataset=beir_scifact `
  runtime=gpu `
  pipeline/inference@pipeline=retrieve/dense_jsonl `
  selections.index_id=YOUR_INDEX_ID `
  selections/embedding_model=e5/small_v2 `
  pipeline.components.retriever.init_parameters.top_k=100
```

```powershell
uv run stage evaluation dataset=beir_scifact stage.inference_run_id=YOUR_EXACT_INFERENCE_RUN_ID
```

To create a chunked index, switch the indexing choice. The same dense retrieval
topology consumes both document- and chunk-level indexes:

```powershell
uv run stage indexing `
  dataset=beir_scifact `
  runtime=gpu `
  pipeline/indexing@pipeline=dense/chunks_jsonl `
  selections/embedding_model=e5/small_v2 `
  selections.index_id=YOUR_NEW_CHUNKED_INDEX_ID
```

```powershell
uv run stage inference `
  dataset=beir_scifact `
  runtime=gpu `
  pipeline/inference@pipeline=retrieve/dense_jsonl `
  selections.index_id=YOUR_CHUNKED_INDEX_ID `
  selections/embedding_model=e5/small_v2 `
  pipeline.components.retriever.init_parameters.top_k=100
```

### Hybrid RRF Retrieval

Use `retrieve/hybrid_rrf_jsonl` to combine keyword and dense retrieval from the
same JSONL index. The shared RRF component fragment defines `lexical` and
`dense` inputs with equal weights, producing classic reciprocal rank fusion:

```powershell
uv run stage inference `
  dataset=beir_scifact `
  runtime=gpu `
  pipeline/inference@pipeline=retrieve/hybrid_rrf_jsonl `
  selections.index_id=YOUR_INDEX_ID `
  selections/embedding_model=e5/small_v2
```

Complete project topologies can select `/component/fusion@components.fusion:
rrf` and override its source-weight mapping when they use different producer
names or weighted RRF.

### Reranking Pipelines

The inference stage always sends legacy query text, complete query metadata,
candidate ids, and materialized candidate documents through the `input` component.
That lets query parsers and reranking pipelines reuse the same stage contract.

To rerank a candidate pool with a bi-encoder, use the candidate reranker
topology. This embeds `input.candidate_documents`, embeds the query, scores by
embedding similarity, and writes ranked documents through `output`:

```powershell
uv run stage prepare_mapping dataset=beir_scifact input_mapping_recipe=judged_only stage.run_id=scifact_judged_only

uv run stage inference `
  dataset=beir_scifact `
  runtime=gpu `
  selections.input_mapping=scifact_judged_only `
  pipeline/inference@pipeline=rerank/bi_encoder `
  selections/embedding_model=e5/small_v2 `
  pipeline.components.ranker.init_parameters.top_k=10
```

For larger candidate pools, prepare an input mapping that limits the documents
per query before reranking. With `selections.input_mapping=null`, every dataset document is
passed as a candidate.

To rerank the same candidate pool with a cross-encoder, select a reranker model
such as BGE reranker v2 M3:

```powershell
uv run stage inference `
  dataset=beir_scifact `
  runtime=gpu `
  selections.input_mapping=scifact_judged_only `
  pipeline/inference@pipeline=rerank/cross_encoder `
  selections/reranker_model=bge/v2_m3 `
  stage.run_id=bge_v2_m3 `
  pipeline.components.ranker.init_parameters.top_k=10
```

Evaluate a completed inference run by passing its exact run id:

```powershell
uv run stage evaluation `
  dataset=beir_scifact `
  stage.inference_run_id=YOUR_EXACT_INFERENCE_RUN_ID
```

Prefixes are intentionally not resolved: exact ids keep lineage unambiguous.

## Stage Workflow

Every stage has a narrow contract:

| Stage | Required inputs | Durable artifact names |
| --- | --- | --- |
| `prepare_mapping` | Dataset plus an `input_mapping_recipe` and run id | `input_mapping`, `input_mapping_metadata` |
| `indexing` | Dataset plus an indexing pipeline | `index` |
| `inference` | Dataset, inference pipeline, mapping, and any required exact index | `predictions` |
| `evaluation` | Qrels plus an exact inference run or explicit predictions path | `metrics` |

Generated input mappings are stored under the prepare-mapping run id. Leaving
inference `selections.input_mapping` null needs no preparation:

```bash
uv run stage prepare_mapping dataset=beir_scifact input_mapping_recipe=dev_tiny stage.run_id=scifact_dev_tiny
```

The following single-line commands are shell-neutral examples for the
query-repetition project. Prepare SciFact first and choose unique ids:

```text
uv run prepare-beir --data-dir data --dataset scifact
uv run stage indexing dataset=beir_scifact runtime=cpu pipeline/indexing@pipeline=dense/documents_jsonl selections/embedding_model=e5/small_v2 selections.index_id=YOUR_UNIQUE_INDEX_ID stage.run_id=YOUR_UNIQUE_INDEXING_RUN_ID
uv run stage inference dataset=beir_scifact runtime=cpu pipeline/inference@pipeline=retrieve/dense_jsonl selections/embedding_model=e5/small_v2 selections.index_id=YOUR_UNIQUE_INDEX_ID stage.run_id=YOUR_UNIQUE_INFERENCE_RUN_ID
uv run stage evaluation dataset=beir_scifact stage.inference_run_id=YOUR_UNIQUE_INFERENCE_RUN_ID stage.run_id=YOUR_UNIQUE_EVALUATION_RUN_ID
```

The artifact locations are:

```text
artifacts/indexes/<index-id>/index.jsonl
artifacts/runs/indexing/<indexing-run-id>/{resolved_config.yaml,result.json,manifest.json}
artifacts/runs/inference/<inference-run-id>/predictions.json
artifacts/runs/evaluation/<evaluation-run-id>/metrics.json
```

Each saved run contains its outputs, `resolved_config.yaml`, `result.json`, and a
`manifest.json` with exact input references, artifact paths, the resolved-config
hash, package/Python versions, and Git commit when available.

`stage.run_id` is the single identifier for a stage run. It defaults to a unique,
Hydra-override-safe timestamp such as `20260723-011220-179277` and may be set
explicitly when a descriptive or stable id is useful:

```bash
uv run stage inference \
  dataset=beir_scifact \
  runtime=gpu \
  pipeline/inference@pipeline=retrieve/dense_jsonl \
  selections/embedding_model=e5/small_v2 \
  selections.index_id=YOUR_INDEX_ID \
  stage.run_id=keyword_smoke
```

Artifact paths and downstream dependency references use this exact id. Run ids
are never modified after configuration preparation.

## Evaluation, Analysis, and Reporting

Prediction artifacts are JSON objects keyed first by query `IN` and then by
document or chunk id. Each document entry contains retrieved content, score,
and metadata. Evaluation uses `meta.source_document_id` when present, so several
retrieved chunks from one source document collapse to one evaluated document.

Qrels with labels less than or equal to zero are excluded. NDCG uses graded
labels; Recall, Precision, HitRate, MAP, and MRR use binary relevance. Record
the exact metric list in versioned experiment configuration before inspecting
results, and use the same list for every run in a comparison.

For query-level analysis, open the experiment's `analysis.ipynb`, configure readable
labels and exact inference run ids, and run the cells. The notebook resolves
predictions through each run manifest, joins qrels, and creates:

- `predictions_df`: one row per retrieved result, including run, query, rank,
  score, content, metadata, source document id, and relevance;
- `query_summary_df`: one row per run and query, including retrieval depth,
  relevant counts, first relevant rank, reciprocal rank, recall, and query lengths.

Add plots below the preparation cells so plot selection remains specific to the
research question while loading and joining stay reproducible. Do not save large
cell outputs in Git; the repository's `nbstripout` configuration removes them
from commits.

Before interpreting a baseline-versus-treatment delta, verify:

- exact run ids and manifest-declared artifact paths;
- the same dataset, split, qrels, and input mapping;
- the same index when indexing is not the treatment;
- the same metric configuration, runtime settings, device, and relevant seeds;
- the intended resolved-config difference after ignoring dynamic run/output fields;
- Git commit, Python version, and installed package versions from each manifest.

Reports belong beside their experiment card as
`projects/<project>/experiments/<experiment-slug>/report.md`. Link exact stage run
directories, calculate deltas as treatment minus baseline, distinguish observations
from interpretations, and report provenance mismatches or missing artifacts. A
single run on one dataset supports an exploratory result, not a claim of statistical
significance or broad generalization.

## Mixing Configs

Hydra config groups are intended to become the main experiment interface. The
default command should be close to the final experiment specification, with only
small overrides on the command line.

Examples:

```bash
uv run stage indexing dataset=beir_scifact runtime=cpu pipeline/indexing@pipeline=dense/documents_jsonl selections/embedding_model=e5/small_v2 selections.index_id=YOUR_NEW_INDEX_ID
uv run stage inference dataset=beir_scifact runtime=cpu pipeline/inference@pipeline=retrieve/dense_jsonl selections/embedding_model=e5/small_v2 selections.index_id=YOUR_INDEX_ID pipeline.components.retriever.init_parameters.top_k=10
uv run stage evaluation dataset=beir_scifact stage.inference_run_id=YOUR_EXACT_INFERENCE_RUN_ID metrics='["Recall@10","MRR@10","NDCG@10","Precision@10","HitRate@10"]'
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
{"doc_id":"doc-1","text":"Text to index.","title":"An optional extra field"}
```

Only `doc_id` is required by the dataset schema. All other JSON-serializable fields,
including `text`, are preserved as document metadata. Built-in pipelines use
`DocumentContentFieldParser(content_field="text")`; select or implement another
document parser for datasets whose embedding text comes from different fields.

Query JSONL records should look like:

```json
{"query_id":"external-q-1","IN":"q-1","query_content":"Search text.","language":"en"}
```

Only `query_id` and `IN` are required by the dataset schema. Other fields are exposed
through `input.query_meta`. Built-in pipelines use
`QueryContentFieldParser(content_field="query_content")`; alternate pipelines can
select another field or render several metadata fields before embedding.

Qrels JSONL records should look like:

```json
{"IN":"q-1","doc_id":"doc-1","label":1,"annotator":"optional"}
```

Then select those paths from a project-local dataset config and run the relevant
pipeline. The following scaffold pipelines are useful for contract tests and do not
require a model:

```bash
uv run stage indexing dataset=my_dataset runtime=cpu pipeline/indexing@pipeline=scaffold/documents_jsonl selections.index_id=YOUR_NEW_INDEX_ID
uv run stage inference dataset=my_dataset runtime=cpu pipeline/inference@pipeline=scaffold/keyword_jsonl selections.index_id=YOUR_INDEX_ID
uv run stage evaluation dataset=my_dataset stage.inference_run_id=YOUR_EXACT_INFERENCE_RUN_ID
```

## Implementing Components and Pipelines

Production-ready retrieval code should be added as Haystack components under
`packages/retrieval-components/src/retrieval_components/<category>/` or imported
from another production package.

To add a new indexing pipeline, create a config like:

```yaml
defaults:
  - /selections@_global_.selections: index
  - _self_

components:
  output:
    type: retrieval_components.interfaces.stage_io.IndexingOutput
  converter:
    type: my_package.components.MyConverter
    init_parameters: {}
  writer:
    type: my_package.components.MyIndexer
    init_parameters:
      output_path: ${paths.indexes_dir}/${selections.index_id}/index.jsonl
connections:
  - sender: converter.documents
    receiver: writer.documents
  - sender: writer.index_path
    receiver: output.index_path
  - sender: writer.indexed_count
    receiver: output.indexed_count
max_runs_per_component: 100
metadata: {}
```

Save it under `configs/pipeline/indexing/my_pipeline.yaml` and select it with:

```bash
uv run stage indexing dataset=my_dataset runtime=gpu pipeline/indexing@pipeline=my_pipeline selections.index_id=YOUR_NEW_INDEX_ID
```

Inference pipelines follow the same pattern under
`configs/pipeline/inference/`. The inference stage always sends query and
candidate data to an `input` component and reads ranked `Document` objects from
an `output` component. The pipeline graph owns all internal routing:

```yaml
defaults:
  - /selections@_global_.selections: index
  - _self_

components:
  input:
    type: retrieval_components.interfaces.stage_io.InferenceInput
  output:
    type: retrieval_components.interfaces.stage_io.InferenceOutput
  retriever:
    type: my_package.components.MyRetriever
    init_parameters:
      index_path: ${paths.indexes_dir}/${selections.index_id}/index.jsonl
connections:
  - sender: input.query
    receiver: retriever.query
  - sender: input.candidate_document_ids
    receiver: retriever.candidate_document_ids
  - sender: retriever.documents
    receiver: output.documents
```

## Parallel Execution

The inference stage runs independent queries concurrently. Configure the number
of simultaneous query runs with `runtime.query_concurrency_limit`; predictions
are still written in dataset order. Each invocation uses Haystack
`AsyncPipeline` and receives `runtime.concurrency_limit`, which controls the
concurrency budget within that individual pipeline run.

For larger sweeps, use Hydra overrides and launchers. A future extension can add
Hydra launcher configs for local multiprocessing, Slurm, Kubernetes, or cloud
batch systems without changing component code.

### Explicit experiment runs in GNU Screen

The repository includes a workflow for a controlled set of local runs. An
experiment is the durable research unit; it may be a baseline/treatment pair, a
hyperparameter sweep, or a single run. Its layout is:

```text
projects/<project>/experiments/<experiment-slug>/
├── experiment.md
├── analysis.ipynb
├── report.md                    # added after results exist
└── configs/
    ├── base-experiment-configs/
    │   └── inference.yaml       # complete shared experiment configuration
    ├── pipeline/...             # optional experiment-only config groups
    └── runs/
        ├── baseline.yaml        # inherits the base unchanged
        └── treatment.yaml       # contains only treatment differences
```

The base config uses Hydra's defaults list to select the stage, dataset, pipeline,
models, and other shared choices, and holds all shared field values:

```yaml
# @package _global_
defaults:
  - /stages/inference
  - override /dataset: beir_scifact
  - override /pipeline/inference@pipeline: retrieve/dense_jsonl
  - override /selections/embedding_model@selections.embedding_model: e5/small_v2
  - override /runtime: gpu
  - _self_

selections:
  index_id: EXACT_INDEX_ID
```

The baseline run is only an entry layer:

```yaml
# @package _global_
defaults:
  - /base-experiment-configs/inference
  - _self_
```

The treatment adds only its differing Hydra selection:

```yaml
# @package _global_
defaults:
  - /base-experiment-configs/inference
  - override /pipeline/inference@pipeline: my_project/treatment_pipeline
  - _self_
```

Create these files directly or use the interactive command builder:

```bash
uv run python ../../awesome-dev-tools/interactive_create_run.py experiments/<experiment-slug>
```

The filename becomes the run name. Run definitions must be direct YAML children of
`configs/runs/`. Their defaults are composed with this search order:
`<experiment>/configs/`, `<project>/configs/`, then the configs packaged by
`retrieval-core`. Launch a selected run by passing its YAML file as the entrypoint:

```bash
uv run stage inference --entrypoint experiments/<experiment-slug>/configs/runs/baseline.yaml
```

The runtime derives the project root, stable run ID, and experiment manifest metadata
from the entrypoint path and run filename; run files may not override those fields.
Resolved configs and heavy outputs remain under `artifacts/runs/<stage>/<run-id>/`.
Launcher status and Screen logs live under
`artifacts/experiments/<experiment>/<run>/`, keeping the experiment workspace
declarative and suitable for version control.

On Linux, install GNU Screen and launch a subset interactively:

```bash
uv run python ../../awesome-dev-tools/interactive_run_in_parallel_screens.py
```

The launcher lists experiments containing `configs/runs/*.yaml`, prints the generated Hydra
command for every run, asks which run files to use, and accepts selections such as
`1,3,4-7`, `ready`, and `all`. It asks for a maximum number of executing runs, assigns
selected runs to that many persistent lanes, launches the Screen sessions, and exits.
The first run in each lane starts immediately; later workers wait for their
predecessor's terminal status using a polling sleep. A failed, cancelled, or lost
predecessor releases its lane because lane dependencies represent execution capacity
rather than experimental data dependencies.

The cap applies to executing runs, not to Screen processes: waiting sessions remain
visible and can be attached to while they consume negligible compute. Lane tails are
kept below `artifacts/experiments/.launcher/`, allowing later launcher invocations to
append work without exceeding the existing cap. The cap cannot be changed until all
current lanes are terminal.

## Troubleshooting

- **Hydra cannot find an entrypoint or config group:** pass an existing YAML file
  below a project or experiment `configs/` directory to `--entrypoint`. Confirm that
  `retrieval-core` is installed in the active environment so its `hydra_plugins`
  search-path plugin can expose the shared config package.
- **A required value is `???`:** select the missing config group, usually a
  dataset, pipeline, input mapping, embedding model, or reranker model. The
  interactive `uv run python ../../awesome-dev-tools/interactive_build_command.py` flow can
  discover choices.
- **Inference cannot find an index:** select a completed directory under
  `paths.indexes_dir` with `selections.index_id`. The command builder lists valid
  directories that contain `index.jsonl`. Prefix matching and implicit "latest"
  resolution are intentionally unsupported. Candidate-only reranking pipelines
  do not use the selection.
- **A prepared mapping is missing:** run `stage prepare_mapping` with a unique
  `stage.run_id`, then pass that folder name as inference `selections.input_mapping`.
- **The run directory already exists:** choose a new `stage.run_id`. Runs are
  immutable and are never overwritten.
- **A command works from one directory but not another:** remember that the
  default project root is `.`, so relative data and artifact paths follow the
  current project directory.
- **A dense pipeline fails on a machine without CUDA:** select `runtime=cpu`
  instead of `runtime=gpu`.

## Design Notes

The framework keeps stage orchestration thin on purpose. Indexing, retrieval,
and later reranking/generation logic should live in Haystack components so the
same components can be imported by production services. Hydra should own
experiment assembly: datasets, pipeline variants, artifact locations, and small
runtime overrides.

Evaluation is deliberately not a Haystack pipeline because metrics need
dataset-level aggregation. It remains a first-class immutable stage with the
same resolved-config, result, and manifest provenance as pipeline-backed stages.

