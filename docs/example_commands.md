# Toy dataset command examples

These PowerShell examples use only the checked-in `toy` dataset. Stage outputs,
indexes, and input mappings are immutable, so change the example IDs before rerunning
a command whose artifact already exists.

## Core indexing, inference, and evaluation

Run this section from the repository root. Prepare the core environment once:

```powershell
uv sync --project packages/retrieval-core --extra dev
```

Create a keyword-search index and use that exact index for inference:

```powershell
uv run --project packages/retrieval-core stage indexing `
  dataset=toy `
  runtime=cpu `
  pipeline/indexing@pipeline=dummy_jsonl `
  selections.index_id=toy-keyword-index `
  stage.run_id=toy-keyword-indexing

uv run --project packages/retrieval-core stage inference `
  dataset=toy `
  runtime=cpu `
  pipeline/inference@pipeline=dummy_keyword `
  selections.index_id=toy-keyword-index `
  stage.run_id=toy-keyword-inference
```

Evaluate the completed inference run by its exact run ID:

```powershell
uv run --project packages/retrieval-core stage evaluation `
  dataset=toy `
  stage.inference_run_id=toy-keyword-inference `
  stage.run_id=toy-keyword-evaluation
```

The resulting metrics are written to
`artifacts/runs/evaluation/toy-keyword-evaluation/metrics.json`.

## Reranking a prepared input mapping without an index

First materialize a small candidate mapping from the toy documents and qrels:

```powershell
uv run --project packages/retrieval-core stage prepare_mapping `
  dataset=toy `
  input_mapping_recipe=dev_tiny `
  stage.run_id=toy-dev-tiny
```

Then rerank those materialized candidate documents with E5-small:

```powershell
uv run --project packages/retrieval-core stage inference `
  dataset=toy `
  runtime=cpu `
  input_mapping=toy-dev-tiny `
  pipeline/inference@pipeline=dense_candidate_reranker `
  selections/embedding_model=e5/small_v2 `
  pipeline.components.ranker.init_parameters.top_k=5 `
  stage.run_id=toy-e5-reranker
```

This inference command intentionally has no `selections.index_id`. The reranker
receives candidate documents from `artifacts/input_mappings/toy-dev-tiny/` and does
not read a retrieval index.

Evaluate the reranker output with:

```powershell
uv run --project packages/retrieval-core stage evaluation `
  dataset=toy `
  stage.inference_run_id=toy-e5-reranker `
  stage.run_id=toy-e5-reranker-evaluation
```

## A specific query-repetition project component

Run the remaining commands from the query-repetition project:

```powershell
Set-Location projects/query-repetition-e5
uv sync --extra dev
```

The project examples use E5-small, so first create one dense toy index. The
`paths.processed_data_dir` override points the project to the repository-level toy
fixture:

```powershell
uv run stage indexing `
  dataset=toy `
  paths.processed_data_dir=../../data/processed `
  runtime=cpu `
  pipeline/indexing@pipeline=dense_jsonl `
  selections/embedding_model=e5/small_v2 `
  selections.index_id=toy-e5-small-index `
  stage.run_id=toy-e5-small-indexing
```

Select the project-owned `dense_query_repetition` pipeline to execute
`query_repetition_e5.components.QueryRepeater` directly:

```powershell
uv run stage inference `
  dataset=toy `
  paths.processed_data_dir=../../data/processed `
  runtime=cpu `
  pipeline/inference@pipeline=dense_query_repetition `
  selections/embedding_model=e5/small_v2 `
  selections.index_id=toy-e5-small-index `
  stage.run_id=toy-query-repetition-component
```

## A specific project experiment run

The project includes the toy experiment
[`query-repetition-e5-small-toy`](../projects/query-repetition-e5/experiments/query-repetition-e5-small-toy/experiment.md).
After creating `toy-e5-small-index` above, launch its checked-in `repeated` run by
passing the YAML file as the stage entrypoint:

```powershell
uv run stage inference `
  --entrypoint experiments/query-repetition-e5-small-toy/configs/runs/repeated.yaml
```

The entrypoint infers the project and experiment context and produces the stable run
ID `query-repetition-e5-small-toy--repeated`. Evaluate it with:

```powershell
uv run stage evaluation `
  dataset=toy `
  paths.processed_data_dir=../../data/processed `
  stage.inference_run_id=query-repetition-e5-small-toy--repeated `
  stage.run_id=toy-query-repetition-experiment-evaluation
```
