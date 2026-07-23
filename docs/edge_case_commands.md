# Edge-Case Test Commands

Run these PowerShell commands from the repository root. The stage commands use
isolated artifact directories and unique run IDs so they do not collide with
normal experiment runs.

Use `python -m retrieval_core.cli` instead of the `stage.exe` wrapper when exit
codes matter. The current wrapper returns the stage result object and therefore
exits with code 1 even after a successful stage.

## Setup

```powershell
$edgeTag = Get-Date -Format 'yyyyMMdd-HHmmss'
$edgeRoot = "./artifacts/edge-cases-$edgeTag"
$edgeRuns = "$edgeRoot/runs"
$edgeIndexes = "$edgeRoot/indexes"
$edgeMappings = "$edgeRoot/input-mappings"

$indexId = "edge-index-$edgeTag"
$indexingRun = "edge-indexing-run-$edgeTag"
$fullInferenceRun = "edge-full-inference-$edgeTag"
$evaluationRun = "edge-evaluation-$edgeTag"
$mappingRun1 = "edge-mapping-1-$edgeTag"
$mappingRun2 = "edge-mapping-2-$edgeTag"
$mappedInferenceRun = "edge-mapped-inference-$edgeTag"

function Invoke-EdgeStage {
    uv run --project packages/retrieval-core python -m retrieval_core.cli @args
}
```

## Successful Workflow Cases

Basic indexing:

```powershell
Invoke-EdgeStage indexing `
  dataset=toy `
  runtime=cpu `
  'pipeline/indexing@pipeline=scaffold/documents_jsonl' `
  "paths.runs_dir=$edgeRuns" `
  "paths.indexes_dir=$edgeIndexes" `
  "selections.index_id=$indexId" `
  "stage.run_id=$indexingRun" `
  runtime.progress_bar=false
```

Full-dataset inference with one-query concurrency and `top_k` larger than the
corpus:

```powershell
Invoke-EdgeStage inference `
  dataset=toy `
  runtime=cpu `
  'pipeline/inference@pipeline=scaffold/keyword_jsonl' `
  "paths.runs_dir=$edgeRuns" `
  "paths.indexes_dir=$edgeIndexes" `
  "selections.index_id=$indexId" `
  "stage.run_id=$fullInferenceRun" `
  runtime.query_concurrency_limit=1 `
  runtime.progress_bar=false `
  pipeline.components.retriever.init_parameters.top_k=999
```

Evaluation with cutoffs larger than the corpus:

```powershell
Invoke-EdgeStage evaluation `
  dataset=toy `
  "paths.runs_dir=$edgeRuns" `
  "stage.inference_run_id=$fullInferenceRun" `
  "stage.run_id=$evaluationRun" `
  'metrics=[Recall@1,Recall@999,MRR@999,NDCG@999,Precision@999]'
```

Prepare a judged-only mapping:

```powershell
Invoke-EdgeStage prepare_mapping `
  dataset=toy `
  input_mapping_recipe=judged_only `
  "paths.runs_dir=$edgeRuns" `
  "paths.input_mappings_dir=$edgeMappings" `
  "stage.run_id=$mappingRun1"
```

Inference using that prepared mapping:

```powershell
Invoke-EdgeStage inference `
  dataset=toy `
  runtime=cpu `
  "selections.input_mapping=$mappingRun1" `
  'pipeline/inference@pipeline=scaffold/keyword_jsonl' `
  "paths.runs_dir=$edgeRuns" `
  "paths.input_mappings_dir=$edgeMappings" `
  "paths.indexes_dir=$edgeIndexes" `
  "selections.index_id=$indexId" `
  "stage.run_id=$mappedInferenceRun" `
  runtime.progress_bar=false
```

## Empty-Input Cases

Prepare a valid mapping containing zero queries:

```powershell
$emptyMappingRun = "edge-empty-mapping-$edgeTag"

Invoke-EdgeStage prepare_mapping `
  dataset=toy `
  input_mapping_recipe=random_smoke `
  input_mapping_recipe.query_subset_size=0 `
  input_mapping_recipe.random_docs_per_query=0 `
  "paths.runs_dir=$edgeRuns" `
  "paths.input_mappings_dir=$edgeMappings" `
  "stage.run_id=$emptyMappingRun"
```

Run inference over that empty mapping:

```powershell
$emptyInferenceRun = "edge-empty-inference-$edgeTag"

Invoke-EdgeStage inference `
  dataset=toy `
  runtime=cpu `
  "selections.input_mapping=$emptyMappingRun" `
  'pipeline/inference@pipeline=scaffold/keyword_jsonl' `
  "paths.runs_dir=$edgeRuns" `
  "paths.input_mappings_dir=$edgeMappings" `
  "paths.indexes_dir=$edgeIndexes" `
  "selections.index_id=$indexId" `
  "stage.run_id=$emptyInferenceRun" `
  runtime.progress_bar=false

Get-Content -Raw "$edgeRuns/inference/$emptyInferenceRun/predictions.json"
```

The prediction artifact should be `{}`.

Evaluate empty predictions against non-empty qrels:

```powershell
Invoke-EdgeStage evaluation `
  dataset=toy `
  "paths.runs_dir=$edgeRuns" `
  "stage.inference_run_id=$emptyInferenceRun" `
  "stage.run_id=edge-empty-evaluation-$edgeTag" `
  'metrics=[Recall@1,MRR@1,HitRate@1]'
```

## Expected Configuration Failures

Run these commands individually. Each should exit nonzero.

Runtime selection missing:

```powershell
Invoke-EdgeStage indexing dataset=toy 'pipeline/indexing@pipeline=scaffold/documents_jsonl'
```

Missing dataset and indexing pipeline:

```powershell
Invoke-EdgeStage indexing runtime=cpu
```

Dataset supplied but indexing pipeline missing:

```powershell
Invoke-EdgeStage indexing dataset=toy runtime=cpu
```

Inference pipeline supplied but dataset missing:

```powershell
Invoke-EdgeStage inference runtime=cpu 'pipeline/inference@pipeline=scaffold/keyword_jsonl'
```

Evaluation dataset missing:

```powershell
Invoke-EdgeStage evaluation
```

Unknown dataset:

```powershell
Invoke-EdgeStage evaluation dataset=does_not_exist
```

Invalid prepare-mapping run-id path characters:

```powershell
Invoke-EdgeStage prepare_mapping `
  dataset=toy `
  input_mapping_recipe=random_smoke `
  'stage.run_id=bad/name'
```

## Expected Dependency Failures

Unknown index id:

```powershell
Invoke-EdgeStage inference `
  dataset=toy `
  runtime=cpu `
  'pipeline/inference@pipeline=scaffold/keyword_jsonl' `
  "paths.runs_dir=$edgeRuns" `
  "paths.indexes_dir=$edgeIndexes" `
  selections.index_id=does-not-exist `
  "stage.run_id=edge-missing-index-$edgeTag"
```

Non-canonical explicit index path:

```powershell
Invoke-EdgeStage inference `
  dataset=toy `
  runtime=cpu `
  'pipeline/inference@pipeline=scaffold/keyword_jsonl' `
  "paths.runs_dir=$edgeRuns" `
  "paths.indexes_dir=$edgeIndexes" `
  "selections.index_id=$indexId" `
  pipeline.components.retriever.init_parameters.index_path=./data/processed/toy/documents.jsonl `
  "stage.run_id=edge-conflicting-index-$edgeTag"
```

Unknown exact inference run:

```powershell
Invoke-EdgeStage evaluation `
  dataset=toy `
  "paths.runs_dir=$edgeRuns" `
  stage.inference_run_id=does-not-exist `
  "stage.run_id=edge-missing-inference-$edgeTag"
```

Use a prepared-mapping folder name that does not exist:

```powershell
Invoke-EdgeStage inference `
  dataset=toy `
  runtime=cpu `
  selections.input_mapping=missing `
  'pipeline/inference@pipeline=scaffold/keyword_jsonl' `
  "paths.runs_dir=$edgeRuns" `
  "paths.input_mappings_dir=$edgeRoot/unprepared-mappings" `
  "paths.indexes_dir=$edgeIndexes" `
  "selections.index_id=$indexId" `
  "stage.run_id=edge-unprepared-mapping-$edgeTag"
```

## Expected Mapping-Generation Failures

Query subset larger than the dataset:

```powershell
Invoke-EdgeStage prepare_mapping `
  dataset=toy `
  input_mapping_recipe=dev_tiny `
  input_mapping_recipe.query_subset_size=999 `
  "paths.runs_dir=$edgeRuns" `
  "paths.input_mappings_dir=$edgeMappings" `
  "stage.run_id=edge-too-many-queries-$edgeTag"
```

Negative document subset:

```powershell
Invoke-EdgeStage prepare_mapping `
  dataset=toy `
  input_mapping_recipe=random_smoke `
  input_mapping_recipe.document_subset_size=-1 `
  "paths.runs_dir=$edgeRuns" `
  "paths.input_mappings_dir=$edgeMappings" `
  "stage.run_id=edge-negative-docs-$edgeTag"
```

Request too many random documents:

```powershell
Invoke-EdgeStage prepare_mapping `
  dataset=toy `
  input_mapping_recipe=random_smoke `
  input_mapping_recipe.random_docs_per_query=999 `
  "paths.runs_dir=$edgeRuns" `
  "paths.input_mappings_dir=$edgeMappings" `
  "stage.run_id=edge-too-many-random-$edgeTag"
```

Zero query concurrency:

```powershell
Invoke-EdgeStage inference `
  dataset=toy `
  runtime=cpu `
  'pipeline/inference@pipeline=scaffold/keyword_jsonl' `
  "paths.runs_dir=$edgeRuns" `
  "paths.indexes_dir=$edgeIndexes" `
  "selections.index_id=$indexId" `
  "stage.run_id=edge-zero-concurrency-$edgeTag" `
  runtime.query_concurrency_limit=0
```

Invalid metric syntax:

```powershell
Invoke-EdgeStage evaluation `
  dataset=toy `
  "paths.runs_dir=$edgeRuns" `
  "stage.inference_run_id=$fullInferenceRun" `
  "stage.run_id=edge-invalid-metric-$edgeTag" `
  'metrics=[recall_at_k]'
```

## Data-Schema Probes

Valid records with extra fields:

```powershell
uv run --project packages/retrieval-core python -c 'from retrieval_core import EVALUATION_DATA_SCHEMA as s; s.validate_query({s.query_id:"external",s.IN:"join-key",s.query_content:"query","language":"en"}); s.validate_document({s.doc_id:"d1",s.text:"text","title":"extra"}); s.validate_qrel({s.IN:"join-key",s.doc_id:"d1",s.label:2,"annotator":"extra"}); print("schema accepted optional fields")'
```

Missing query content, expected failure:

```powershell
uv run --project packages/retrieval-core python -c 'from retrieval_core import EVALUATION_DATA_SCHEMA as s; s.validate_query({s.query_id:"q1",s.IN:"input-1"})'
```

Missing document text, expected failure:

```powershell
uv run --project packages/retrieval-core python -c 'from retrieval_core import EVALUATION_DATA_SCHEMA as s; s.validate_document({s.doc_id:"d1"})'
```

Missing qrel label, expected failure:

```powershell
uv run --project packages/retrieval-core python -c 'from retrieval_core import EVALUATION_DATA_SCHEMA as s; s.validate_qrel({s.IN:"input-1",s.doc_id:"d1"})'
```

## Component Edge Cases

Chunk cascade must select by score regardless of input order:

```powershell
uv run --project packages/retrieval-components python -c 'from haystack import Document; from retrieval_components.cascade import ChunkCascade; docs=[Document(id="first",score=.1,meta={"source_document_id":"d1"}),Document(id="second",score=.9,meta={"source_document_id":"d1"})]; print([d.id for d in ChunkCascade(top_k=1).run(docs)["documents"]])'
```

Expected output: `['second']`.

Reject `top_k=0`:

```powershell
uv run --project packages/retrieval-components python -c 'from retrieval_components.cascade import ChunkCascade; ChunkCascade(top_k=0)'
```

Z-normalization with constant scores:

```powershell
uv run --project packages/retrieval-components python -c 'from haystack import Document; from retrieval_components import ZScoreFusion; result=ZScoreFusion(weights={"only":2}).run(only=[Document(id="d1",score=3),Document(id="d2",score=3)]); print([(d.id,d.score) for d in result["documents"]])'
```

Both scores should be `0.0`.

Linear normalization with one document:

```powershell
uv run --project packages/retrieval-components python -c 'from haystack import Document; from retrieval_components import LinearScoreFusion; result=LinearScoreFusion(weights={"only":1}).run(only=[Document(id="d1",score=42)]); print([(d.id,d.score) for d in result["documents"]])'
```

The only document should receive `1.0`.

Empty fusion sources:

```powershell
uv run --project packages/retrieval-components python -c 'from retrieval_components import LinearScoreFusion,ZScoreFusion; print(LinearScoreFusion(weights={"a":1}).run(a=[])); print(ZScoreFusion(weights={"a":1}).run(a=[]))'
```

Both should return an empty document list.

## Full Regression and Lint Commands

```powershell
uv run --project packages/retrieval-components pytest packages/retrieval-components/tests -q
uv run --project packages/retrieval-core pytest packages/retrieval-core/tests -q
uv run --project projects/query-repetition-e5 pytest projects/query-repetition-e5/tests -q

uv run --project packages/retrieval-components ruff check packages/retrieval-components
uv run --project packages/retrieval-components ruff format --check packages/retrieval-components
uv run --project packages/retrieval-core ruff check packages/retrieval-core
```

All generated stage artifacts remain under `$edgeRoot`, isolated from normal
experiment runs.
