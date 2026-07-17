---
experiment_id: query-repetition-e5-small-scifact
status: draft
project: query-repetition-e5
created: 2026-07-17
owner: TBD
---

# Repeating E5-small queries on BEIR SciFact

## Research question

Does repeating each raw query twice before adding the standard E5 query prefix improve dense retrieval effectiveness on the BEIR SciFact test split compared with the unchanged query pipeline?

## Hypothesis

Repeating the raw query twice with one separating space before E5 preprocessing will produce a higher `NDCG@10` than the unchanged `intfloat/e5-small-v2` baseline on the SciFact test split.

## Rationale

The treatment tests whether the gains associated with input repetition transfer to retrieval embeddings. It is motivated by [Repetition Improves Language Model Embeddings](https://arxiv.org/abs/2402.15449) and [Prompt Repetition Improves Non-Reasoning LLMs](https://arxiv.org/abs/2512.14982), as summarized in the [project README](../README.md). This is an exploratory transfer test rather than a reproduction because E5-small is a bidirectional encoder and the proposed explanations may not transfer from causal language models.

## Comparison

| Dimension | Baseline | Treatment | Held fixed? |
| --- | --- | --- | --- |
| Inference pipeline | Core [`dense_jsonl`](../../../packages/retrieval-core/src/retrieval_core/configs/pipeline/inference/dense_jsonl.yaml) | Project [`dense_query_repetition`](../configs/pipeline/inference/dense_query_repetition.yaml) | No; this is the treatment boundary |
| Query transformation | Raw query goes directly to E5 preprocessing | [`QueryRepeater`](../src/query_repetition_e5/components.py) emits `"<query> <query>"` before E5 preprocessing | No; primary treatment |
| Embedding model | `intfloat/e5-small-v2` | `intfloat/e5-small-v2` | Yes |
| Query prefix | `"query: "` after normalizing whitespace | `"query: "` after normalizing whitespace | Yes |
| Document index | Shared document-level E5-small index | Exact same index run | Yes |
| Dataset and split | BEIR SciFact test | BEIR SciFact test | Yes |
| Candidate mapping | `input_mapping=full` | `input_mapping=full` | Yes |
| Retrieved depth | `top_k=100` | `top_k=100` | Yes |
| Device and concurrency | CPU; query concurrency 4; pipeline concurrency 4 | Same | Yes |
| Evaluation metrics | Same explicit metric list and qrels | Same | Yes |

Primary treatment change: insert a `QueryRepeater(separator=" ")` between the inference input and the standard E5 query preprocessor so the raw query appears exactly twice.

## Data and sampling

- Dataset/split: full BEIR SciFact corpus and the standard `test.tsv` qrels, converted by `prepare-beir --dataset scifact` into the repository JSONL format.
- Query/document inclusion: all converted SciFact queries and documents; no query or document subsampling.
- Input mapping: `full_dataset`, selected explicitly as `input_mapping=full`; every query is retrieved against the full indexed corpus.
- Retrieval depth: 100 documents per query in both arms so `Recall@10`, `Recall@50`, and `Recall@100` are all observable rather than capped by the repository's default `top_k=5`.
- Repetitions or seeds: one paired deterministic execution per arm. No random sampling or seed is exposed by this pipeline. Device and package environment will remain fixed.
- Exclusions: none beyond records excluded by the repository's standard BEIR conversion. Any conversion warnings or count mismatches will abort the comparison.

## Metrics and decision rule

- Primary metric: `NDCG@10`; higher is better.
- Secondary metrics: `Recall@10` and `MRR@50` for early retrieval quality, plus `Recall@50`, `Recall@100`, and `HitRate@10` for diagnostic coverage.
- Diagnostics: per-query rank changes, the fraction of improved/degraded/unchanged queries, query token lengths before and after repetition, and the count of queries affected by the tokenizer's 512-token maximum.
- Success: treatment `NDCG@10 - baseline NDCG@10 > 0`, with matching provenance and complete outputs. A small positive delta will count only as directional support on SciFact, not as evidence of general improvement.
- Failure: treatment `NDCG@10 - baseline NDCG@10 < 0`, with matching provenance and complete outputs.
- Inconclusive: the primary delta is exactly zero at stored precision, any run is incomplete, or dataset, index, mapping, metric, code, package, device, or resolved non-treatment configuration differs between arms.

No minimum effect-size threshold is preregistered because the project defines this as an exploratory sign-and-size comparison and provides no prior variance estimate. The report will show the absolute delta at full stored precision.

## Execution plan

1. Install the project environment and run its component and pipeline tests.
2. Download and convert the complete SciFact test dataset.
3. Validate and create one shared E5-small document index with retrieval depth fixed independently at inference.
4. Validate and run the baseline and repeated-query pipelines against the exact shared index.
5. Evaluate both inference runs using the same explicit metric list and SciFact qrels.
6. Compare metrics and verify the two resolved configs differ only at the intended query-repetition node and dynamic run/output fields.

Run these commands from `projects/query-repetition-e5`:

```powershell
$ErrorActionPreference = "Stop"
$suffix = Get-Date -Format "yyyyMMdd-HHmmss"
$indexRun = "query-repeat-e5-index-$suffix"
$baselineRun = "query-repeat-e5-baseline-$suffix"
$treatmentRun = "query-repeat-e5-treatment-$suffix"
$baselineEval = "query-repeat-e5-baseline-eval-$suffix"
$treatmentEval = "query-repeat-e5-treatment-eval-$suffix"
$metrics = '["Recall@10","Recall@50","Recall@100","MRR@50","NDCG@10","HitRate@10"]'

uv sync --extra dev
uv run pytest
uv run prepare-beir --data-dir data --dataset scifact

uv run stage --validate indexing dataset=beir_scifact pipeline/indexing@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 runtime.device.device=cpu runtime.concurrency_limit=4 stage.run_id=$indexRun
uv run stage indexing dataset=beir_scifact pipeline/indexing@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 runtime.device.device=cpu runtime.concurrency_limit=4 stage.run_id=$indexRun

uv run stage --validate inference dataset=beir_scifact input_mapping=full pipeline/inference@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 stage.indexing_run_id=$indexRun pipeline.components.retriever.init_parameters.top_k=100 runtime.device.device=cpu runtime.query_concurrency_limit=4 runtime.concurrency_limit=4 stage.run_id=$baselineRun
uv run stage inference dataset=beir_scifact input_mapping=full pipeline/inference@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 stage.indexing_run_id=$indexRun pipeline.components.retriever.init_parameters.top_k=100 runtime.device.device=cpu runtime.query_concurrency_limit=4 runtime.concurrency_limit=4 stage.run_id=$baselineRun

uv run stage --validate inference dataset=beir_scifact input_mapping=full pipeline/inference@pipeline=dense_query_repetition selections/embedding_model=e5/small_v2 stage.indexing_run_id=$indexRun pipeline.components.retriever.init_parameters.top_k=100 runtime.device.device=cpu runtime.query_concurrency_limit=4 runtime.concurrency_limit=4 stage.run_id=$treatmentRun
uv run stage inference dataset=beir_scifact input_mapping=full pipeline/inference@pipeline=dense_query_repetition selections/embedding_model=e5/small_v2 stage.indexing_run_id=$indexRun pipeline.components.retriever.init_parameters.top_k=100 runtime.device.device=cpu runtime.query_concurrency_limit=4 runtime.concurrency_limit=4 stage.run_id=$treatmentRun

uv run stage --validate evaluation dataset=beir_scifact stage.inference_run_id=$baselineRun metrics=$metrics stage.run_id=$baselineEval
uv run stage --validate evaluation dataset=beir_scifact stage.inference_run_id=$treatmentRun metrics=$metrics stage.run_id=$treatmentEval
uv run stage evaluation dataset=beir_scifact stage.inference_run_id=$baselineRun metrics=$metrics stage.run_id=$baselineEval
uv run stage evaluation dataset=beir_scifact stage.inference_run_id=$treatmentRun metrics=$metrics stage.run_id=$treatmentEval

uv run python scripts/compare_metrics.py "artifacts/runs/evaluation/$baselineEval/metrics.json" "artifacts/runs/evaluation/$treatmentEval/metrics.json"
```

Run naming scheme: `query-repeat-e5-<role>-<YYYYMMDD-HHMMSS>`. Commands set exact `stage.run_id` values and intentionally omit `stage.run_name`; the latter is prepended to the run id by stage preparation and would require downstream references to use the resulting combined id.

Expected artifacts:

- `artifacts/runs/indexing/$indexRun/{index.jsonl,resolved_config.yaml,result.json,manifest.json}`
- `artifacts/runs/inference/$baselineRun/{predictions.json,resolved_config.yaml,result.json,manifest.json}`
- `artifacts/runs/inference/$treatmentRun/{predictions.json,resolved_config.yaml,result.json,manifest.json}`
- `artifacts/runs/evaluation/$baselineEval/{metrics.json,resolved_config.yaml,result.json,manifest.json}`
- `artifacts/runs/evaluation/$treatmentEval/{metrics.json,resolved_config.yaml,result.json,manifest.json}`

## Validity and risk checks

- Confounders: the only substantive resolved pipeline difference must be the treatment's `query_repeater` component and its two adjacent connections. Dynamic run ids, output paths, descriptions, and config hashes will differ and will be classified separately.
- Index fairness: both inference manifests must name the same exact `indexing_run_id` and resolved index artifact.
- Data fairness: both inference and evaluation configs must point to identical SciFact documents, queries, qrels, and `input_mapping=full`.
- Truncation: repetition may push long queries past the E5 tokenizer's 512-token limit. This is part of the treatment's behavior but will be measured as a diagnostic because it may explain degraded queries.
- Leakage risk: do not tune separator, repetition count, retrieval depth, or the primary decision rule after inspecting SciFact results. Any follow-up must receive a new card or be labeled exploratory.
- External dependencies: BEIR dataset hosting and the Hugging Face `intfloat/e5-small-v2` checkpoint must be available or cached. Run manifests must record non-`unknown` package versions and a Git commit when possible.
- Existing script risk: [`run_experiment.ps1`](../scripts/run_experiment.ps1) currently combines explicit `stage.run_id` and `stage.run_name`, while downstream commands reuse only the unprefixed id; it also inherits `top_k=5`. Use the commands in this card unless that script is aligned with this preregistered design.
- Resource/time constraints: TBD after the first validated environment check; use CPU for both arms unless the card is amended before execution to use the same CUDA environment throughout.
- Abort conditions: failed unit/config validation, incomplete dataset conversion, an existing immutable run directory, a missing manifest artifact, unequal prediction query counts, non-finite metrics, or any unplanned resolved-config difference.

## Analysis plan

Create a report table with baseline, treatment, and `treatment - baseline` for every preregistered metric, highlighting `NDCG@10` as primary. Verify provenance and resolved-config comparability before interpreting values. Report per-query improved/degraded/unchanged counts and summarize rank changes without choosing examples based solely on favorable outcomes. Stratify diagnostics by whether repetition reaches the 512-token limit if any query does.

Because this is one model, one dataset, and one paired execution, do not claim statistical significance or generalization. A positive primary delta will motivate a new multi-dataset replication card; a negative delta will motivate error analysis before changing the treatment.

## Results

Pending. Link the completed report here without changing the preregistered hypothesis or decision rule.
