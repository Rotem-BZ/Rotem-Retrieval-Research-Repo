# Retrieval Research Workflows

This repo is organized around three ideas:

1. Reusable retrieval behavior lives in Haystack components.
2. Experiment selection and parameterization lives in Hydra configs.
3. Long-running experiment workflows are split into explicit stages.

The scaffold includes both dummy components and real retrieval components. The
dummy pieces are still useful because they exercise the intended contracts
without requiring a model, document store, or external service.

## Research Lifecycle

Use the repository as an evidence-producing workflow rather than as a collection
of isolated commands:

1. Write a falsifiable hypothesis and preregister the comparison in
   `projects/<project>/experiments/<experiment-slug>.md`.
2. Identify one controlled treatment change, its baseline, the primary metric,
   and the settings that must remain fixed.
3. Select or implement the required component, pipeline topology, dataset,
   semantic selections, and input mapping.
4. Run `stage --validate` before spending compute. Use `--dry-run` when executing
   real components against temporary outputs is useful.
5. Reuse exact upstream artifacts. A treatment that changes only inference should
   normally share the baseline's index, mapping, dataset, and qrels.
6. Run inference and evaluation with immutable, exact run ids.
7. Inspect aggregate metrics and query-level behavior using the project's
   `notebooks/analyze_predictions.ipynb` notebook.
8. Write `projects/<project>/reports/<experiment-slug>.md` from manifests,
   resolved configs, results, predictions, and metrics—not from remembered commands.

Repository-local agent skills support the same lifecycle:

- `create-experiment-card` plans a baseline-versus-treatment experiment.
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
- `projects/` contains independently locked experiments and their config overlays.

See [components.md](components.md) for the current component inventory and which
pieces are native Haystack components versus repo-specific adapters.

Inside `packages/retrieval-core/src/retrieval_core/`, the main modules are:

- `cli.py` dispatches `stage <stage-name>` commands to stage runners.
- `command_builder.py` powers `build-command`, an interactive command builder
  for Hydra selections.
- reusable components live separately under `packages/retrieval-components/`.
- `input_mapping.py` owns candidate-set recipes and materialized mappings.
- `notebooks/` contains cell-marked data-preparation scripts such as `prepare_beir.py`.
- `stages/` contains orchestration for `prepare_mapping`, indexing, inference,
  and evaluation.
- `utils/` groups shared helpers by responsibility: artifacts, config, console,
  evaluation, IO, pipelines, hashing, and time.

The separate top-level `packages/retrieval-core/src/hydra_plugins/` namespace is
intentional. Hydra auto-discovers its search-path plugin, which appends
`pkg://retrieval_core.configs` after the consuming project's config directory.
That ordering lets projects override only the config groups they own while using
the shared core entry points and defaults as fallbacks.

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
# configs/pipeline/inference/dense_jsonl.yaml
defaults:
  - /selections/embedding_model@_global_.selections.embedding_model: ???
  - /component/query_preprocessor@components.query_preprocessor: prefix_cleanup
  - /component/query_embedder@components.query_embedder: sentence_transformers
  - /component/retriever@components.retriever: jsonl_embeddings
  - _self_

components:
  input:
    type: retrieval_components.components.interfaces.stage_io.InferenceInput
  output:
    type: retrieval_components.components.interfaces.stage_io.InferenceOutput

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
  pipeline/inference@pipeline=dense_jsonl \
  selections/embedding_model=e5/small_v2 \
  stage.indexing_run_id=YOUR_EXACT_INDEXING_RUN_ID
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

Project-specific pipeline configs can reuse the shared component groups while
inserting one local treatment component. The query-repetition project follows
this pattern:

```yaml
# projects/query-repetition-e5/configs/pipeline/inference/dense_query_repetition.yaml
defaults:
  - /selections/embedding_model@_global_.selections.embedding_model: ???
  - /component/query_preprocessor@components.query_preprocessor: prefix_cleanup
  - /component/query_embedder@components.query_embedder: sentence_transformers
  - /component/retriever@components.retriever: jsonl_embeddings
  - _self_

components:
  input:
    type: retrieval_components.components.interfaces.stage_io.InferenceInput
  query_repeater:
    type: query_repetition_e5.components.QueryRepeater
    init_parameters:
      separator: " "
  output:
    type: retrieval_components.components.interfaces.stage_io.InferenceOutput
```

Its graph routes `input.query` through `query_repeater` before the shared query
preprocessor and embedder. The baseline uses the shared `dense_jsonl` topology,
so the local component is the intended difference.

As a rule of thumb, use command-line overrides for small scalar changes,
component config groups for reusable implementations, pipeline config groups for
graph topology, and experiment cards for the research claim and decision rule.
If a fully resolved configuration must be preserved as a runnable reference,
store it under `configs/materialized/` and run it by config path.

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

To build and validate a command interactively without running an experiment:

```bash
uv run build-command
```

After required Hydra choices are selected, the builder can review the selected
config graph. From there you can switch default choices such as `input_mapping`,
enter nested YAML configs such as selected embedding models, and render edited
leaf fields as command-line overrides.

## Configuration Layout

Hydra config entry points live at the top of `configs/`:

- `configs/indexing.yaml`
- `configs/inference.yaml`
- `configs/evaluation.yaml`
- `configs/prepare_mapping.yaml`

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

Dataset records live as data files, not as Hydra config payloads. The core test
fixture is in:

- `data/processed/toy/documents.jsonl`
- `data/processed/toy/queries.jsonl`
- `data/processed/toy/qrels.jsonl`

BEIR datasets are prepared with the cell-marked Python script at
`packages/retrieval-core/src/retrieval_core/notebooks/prepare_beir.py`, exposed as
the `prepare-beir` command. When invoked from a research project, it downloads
raw archives to that project's `data/raw`, extracts them to `data/interim`, and
writes repo-native JSONL files to `data/processed`:

```bash
uv run prepare-beir --data-dir data --dataset scifact
```

Project analysis uses a real Jupyter notebook. Add exact inference run ids to
`notebooks/analyze_predictions.ipynb`; it resolves predictions through each
manifest and builds qrel-enriched `predictions_df` and `query_summary_df` tables
ready for project-specific plots.

The indexing and inference configs each place a Haystack serialized pipeline
under the `pipeline` field. The Python runner resolves Hydra interpolation,
serializes that field to YAML, loads it with Haystack, and executes it as an
`AsyncPipeline`.

### Input Mappings

Inference always resolves an input mapping. By default, `input_mapping=full`
uses every query and every document in the selected dataset without storing a
large file of all ids. Generated mappings are selected as reusable recipes at
the root config level and prepared explicitly before inference:

```bash
uv run stage prepare_mapping \
  dataset=beir_scifact \
  input_mapping=dev_tiny
```

The prepared mapping can then be reused by any number of inference runs that
select `input_mapping=dev_tiny`. Its content-addressed cache key includes both
the recipe and SHA-256 fingerprints of the documents, queries, and qrels files.

Materialized mappings are plain JSON objects keyed by query id:

```json
{
  "q-1": ["doc-1", "doc-7", "doc-9"]
}
```

If a mapping includes only a subset of queries, inference runs only those
queries. Candidate ids and materialized candidate `Document` objects are passed
to each inference pipeline through the fixed `input` interface component; each
pipeline decides which internal components consume them.

Useful built-in recipes live under `configs/input_mapping/`:

- `full`: virtual default; all queries against all documents, with no mapping JSON.
- `judged_only`: all queries, but only documents with qrel annotations for each query.
- `dev_tiny`: two-query development pool with easy negatives and cross-query positives.
- `random_smoke`: two-query smoke-test pool with one random extra document per query.

Prepared mappings are stored outside the dataset tree:

```text
artifacts/input_mappings/toy/dev_tiny.<cache-key>.json
artifacts/input_mappings/toy/dev_tiny.<cache-key>.meta.json
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
  stage.indexing_run_id=YOUR_EXACT_INDEXING_RUN_ID `
  selections/embedding_model=e5/small_v2 `
  pipeline.components.retriever.init_parameters.top_k=100
```

```powershell
uv run stage evaluation dataset=beir_scifact stage.inference_run_id=YOUR_EXACT_INFERENCE_RUN_ID
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
  stage.indexing_run_id=YOUR_EXACT_INDEXING_RUN_ID `
  selections/embedding_model=e5/small_v2 `
  pipeline.components.retriever.init_parameters.top_k=100
```

### Reranking Pipelines

The inference stage always sends the raw query, candidate ids, and materialized
candidate documents through the `input` component. That lets reranking pipelines
reuse the same stage contract.

To rerank a candidate pool with a bi-encoder, use the candidate reranker
topology. This embeds `input.candidate_documents`, embeds the query, scores by
embedding similarity, and writes ranked documents through `output`:

```powershell
uv run stage prepare_mapping dataset=beir_scifact input_mapping=judged_only

uv run stage inference `
  dataset=beir_scifact `
  input_mapping=judged_only `
  pipeline/inference@pipeline=dense_candidate_reranker `
  selections/embedding_model=e5/small_v2 `
  pipeline.components.ranker.init_parameters.top_k=10
```

For larger candidate pools, create or select an `input_mapping` that limits the
documents per query before reranking. Without a mapping, `input_mapping=full`
passes every dataset document as a candidate.

To rerank the same candidate pool with a cross-encoder, select a reranker model
such as BGE reranker v2 M3:

```powershell
uv run stage inference `
  dataset=beir_scifact `
  input_mapping=judged_only `
  pipeline/inference@pipeline=cross_encoder_candidate_reranker `
  selections/reranker_model=bge/v2_m3 `
  stage.run_name=bge_v2_m3 `
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
| `prepare_mapping` | Dataset plus a generated `input_mapping` recipe | `input_mapping`, `input_mapping_metadata` |
| `indexing` | Dataset plus an indexing pipeline | `index` |
| `inference` | Dataset, inference pipeline, mapping, and any required exact index | `predictions` |
| `evaluation` | Qrels plus an exact inference run or explicit predictions path | `metrics` |

Generated input mappings are prepared once and cached by recipe and source-file
fingerprints. `input_mapping=full` is virtual and needs no preparation:

```bash
uv run stage prepare_mapping dataset=beir_scifact input_mapping=dev_tiny
```

The following single-line commands are shell-neutral examples for the
query-repetition project. Prepare SciFact first, choose unique ids, and validate
with the same scientific overrides that will be used for execution:

```text
uv run prepare-beir --data-dir data --dataset scifact
uv run stage --validate indexing dataset=beir_scifact pipeline/indexing@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 runtime.device.device=cpu stage.run_id=YOUR_UNIQUE_INDEX_RUN_ID
uv run stage indexing dataset=beir_scifact pipeline/indexing@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 runtime.device.device=cpu stage.run_id=YOUR_UNIQUE_INDEX_RUN_ID
uv run stage --validate inference dataset=beir_scifact pipeline/inference@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 runtime.device.device=cpu stage.indexing_run_id=YOUR_UNIQUE_INDEX_RUN_ID stage.run_id=YOUR_UNIQUE_INFERENCE_RUN_ID
uv run stage inference dataset=beir_scifact pipeline/inference@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 runtime.device.device=cpu stage.indexing_run_id=YOUR_UNIQUE_INDEX_RUN_ID stage.run_id=YOUR_UNIQUE_INFERENCE_RUN_ID
uv run stage evaluation dataset=beir_scifact stage.inference_run_id=YOUR_UNIQUE_INFERENCE_RUN_ID stage.run_id=YOUR_UNIQUE_EVALUATION_RUN_ID
```

The artifact locations are:

```text
artifacts/runs/indexing/<indexing-run-id>/index.jsonl
artifacts/runs/inference/<inference-run-id>/predictions.json
artifacts/runs/evaluation/<evaluation-run-id>/metrics.json
```

Each saved run contains its outputs, `resolved_config.yaml`, `result.json`, and a
`manifest.json` with exact input references, artifact paths, the resolved-config
hash, package/Python versions, and Git commit when available.

All stages accept an optional `stage.run_name` label. It is prepended to the
resolved `stage.run_id`, whether that id was generated or explicitly supplied:

```bash
uv run stage inference \
  dataset=beir_scifact \
  pipeline/inference@pipeline=dense_jsonl \
  selections/embedding_model=e5/small_v2 \
  stage.indexing_run_id=YOUR_EXACT_INDEXING_RUN_ID \
  stage.run_name=keyword_smoke
```

With the generated timestamp id, this creates a run id like
`keyword_smoke_20260623_153000_123456`. If scripts set a descriptive
`stage.run_id` directly, they should normally omit `stage.run_name`; otherwise
the exact id becomes `<run-name>_<supplied-run-id>`. Upstream references must
always use the complete stored id.

Use `--validate` to compose the config, resolve exact upstream references, check
input files, and load the Haystack graph without executing components or writing
a run:

```bash
uv run stage --validate inference \
  dataset=beir_scifact \
  pipeline/inference@pipeline=dense_jsonl \
  selections/embedding_model=e5/small_v2 \
  stage.indexing_run_id=YOUR_EXACT_INDEXING_RUN_ID
```

Use `--dry-run` to execute against real inputs while redirecting all outputs to
a temporary directory. No run record or durable output is saved. Unlike
`--validate`, a dry run executes components and may still consume substantial
compute.

## Evaluation, Analysis, and Reporting

Prediction artifacts are JSON objects keyed first by query id and then by
document or chunk id. Each document entry contains retrieved content, score,
and metadata. Evaluation uses `meta.source_document_id` when present, so several
retrieved chunks from one source document collapse to one evaluated document.

Qrels with relevance less than or equal to zero are excluded. NDCG uses graded
relevance; Recall, Precision, HitRate, MAP, and MRR use binary relevance. Record
the exact metric list in the experiment card before inspecting results, and use
the same list for every run in a comparison.

For query-level analysis, open the project's
`notebooks/analyze_predictions.ipynb`, configure readable labels and exact
inference run ids, and run the cells. The notebook resolves predictions through
each run manifest, joins qrels, and creates:

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

Reports belong in `projects/<project>/reports/`. Link the experiment card and
exact run directories, calculate deltas as treatment minus baseline, distinguish
observations from interpretations, and report provenance mismatches or missing
artifacts. A single run on one dataset supports an exploratory result, not a
claim of statistical significance or broad generalization.

## Mixing Configs

Hydra config groups are intended to become the main experiment interface. The
default command should be close to the final experiment specification, with only
small overrides on the command line.

Examples:

```bash
uv run stage indexing dataset=beir_scifact pipeline/indexing@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 runtime.device.device=cpu
uv run stage inference dataset=beir_scifact pipeline/inference@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 stage.indexing_run_id=YOUR_EXACT_INDEXING_RUN_ID pipeline.components.retriever.init_parameters.top_k=10 runtime.device.device=cpu
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

Then select those paths from a project-local dataset config and run the relevant
pipeline. The following dummy pipelines are useful for contract tests and do not
require a model:

```bash
uv run stage indexing dataset=my_dataset pipeline/indexing@pipeline=dummy_jsonl
uv run stage inference dataset=my_dataset pipeline/inference@pipeline=dummy_keyword stage.indexing_run_id=YOUR_EXACT_INDEXING_RUN_ID
uv run stage evaluation dataset=my_dataset stage.inference_run_id=YOUR_EXACT_INFERENCE_RUN_ID
```

## Implementing Components and Pipelines

Production-ready retrieval code should be added as Haystack components under
`packages/retrieval-components/src/retrieval_components/components/` or imported
from another production package.

To add a new indexing pipeline, create a config like:

```yaml
components:
  output:
    type: retrieval_components.components.interfaces.stage_io.IndexingOutput
  converter:
    type: my_package.components.MyConverter
    init_parameters: {}
  writer:
    type: my_package.components.MyIndexer
    init_parameters:
      output_path: ${stage.output_dir}/index
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
uv run stage indexing dataset=my_dataset pipeline/indexing@pipeline=my_pipeline
```

Inference pipelines follow the same pattern under
`configs/pipeline/inference/`. The inference stage always sends query and
candidate data to an `input` component and reads ranked `Document` objects from
an `output` component. The pipeline graph owns all internal routing:

```yaml
components:
  input:
    type: retrieval_components.components.interfaces.stage_io.InferenceInput
  output:
    type: retrieval_components.components.interfaces.stage_io.InferenceOutput
  retriever:
    type: my_package.components.MyRetriever
    init_parameters: {}
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

### Prepared Screen sweeps

The repository includes a two-phase workflow for local hyperparameter sweeps. Run
the preparer from the project whose environment and config tree should be used:

```bash
uv run prepare-sweep
```

The interactive preparer first uses the normal command builder to choose a valid
base stage configuration. It then prompts for one or more Hydra fields or override
paths, short labels, YAML value lists, and Cartesian or zipped combination mode.
Every combination is composed, validated, assigned a descriptive name such as
`lr-0.01--chunksize-14--model-E5-base`, and written as a fully resolved config under
`artifacts/sweeps/<sweep-id>/configs/`. The folder is published only after every
configuration validates successfully.

On Linux, install GNU Screen and launch a prepared subset interactively:

```bash
uv run run-sweep
```

The launcher highlights existing run states and accepts selections such as
`1,3,4-7`, as well as `ready` and `all`. It asks for a maximum number of
executing experiments, assigns selected runs to that many persistent lanes, launches
all selected Screen sessions, and exits. The first run in each lane starts immediately;
later workers wait for their predecessor's terminal status using a polling sleep.
A failed, cancelled, or lost predecessor releases its lane because lane dependencies
represent execution capacity rather than experimental data dependencies.

The cap applies to executing experiments, not to Screen processes: waiting sessions
remain visible and can be attached to while they consume negligible compute. Lane
tails are kept below `artifacts/sweeps/.launcher/`, allowing later launcher invocations
to append work without exceeding the existing cap. The cap cannot be changed until
all current lanes are terminal.

## Troubleshooting

- **Hydra cannot find a stage or config group:** run from the intended project
  directory, or pass `--config-dir PATH`. Confirm that `retrieval-core` is
  installed in the active environment so its `hydra_plugins` search-path plugin
  can expose the shared config package.
- **A required value is `???`:** select the missing config group, usually a
  dataset, pipeline, input mapping, embedding model, or reranker model. The
  interactive `uv run build-command` flow can discover and validate choices.
- **Inference cannot find an index:** pass the complete immutable indexing run id
  in `stage.indexing_run_id`, or explicitly set `stage.index_path` for a legacy
  artifact. Prefix matching and implicit "latest" resolution are intentionally
  unsupported.
- **A generated mapping is missing:** run `stage prepare_mapping` with the same
  dataset and recipe before validating or executing inference.
- **The run directory already exists:** choose a new `stage.run_id`. Runs are
  immutable and are never overwritten.
- **A command works from one directory but not another:** remember that the
  default project root is `.`, so relative data and artifact paths follow the
  current project directory.
- **A dense pipeline fails on a machine without CUDA:** override
  `runtime.device.device=cpu` or install and configure the intended CUDA runtime.

## Design Notes

The framework keeps stage orchestration thin on purpose. Indexing, retrieval,
and later reranking/generation logic should live in Haystack components so the
same components can be imported by production services. Hydra should own
experiment assembly: datasets, pipeline variants, artifact locations, and small
runtime overrides.

Evaluation is deliberately not a Haystack pipeline because metrics need
dataset-level aggregation. It remains a first-class immutable stage with the
same resolved-config, result, and manifest provenance as pipeline-backed stages.

