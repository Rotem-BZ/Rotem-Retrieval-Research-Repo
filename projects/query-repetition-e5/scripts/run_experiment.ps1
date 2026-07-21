param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$Runtime = if ($Device -eq "cuda") { "gpu" } else { "cpu" }

$suffix = Get-Date -Format "yyyyMMdd-HHmmss"
$indexId = "e5-small-index-$suffix"
$indexingRun = "e5-small-indexing-$suffix"
$baselineRun = "e5-small-baseline-$suffix"
$repeatedRun = "e5-small-repeated-$suffix"
$baselineEvaluation = "e5-small-baseline-eval-$suffix"
$repeatedEvaluation = "e5-small-repeated-eval-$suffix"

uv run prepare-beir --data-dir data --dataset scifact
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run stage indexing dataset=beir_scifact pipeline/indexing@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 selections.index_id=$indexId runtime=$Runtime stage.run_id=$indexingRun
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run stage inference dataset=beir_scifact pipeline/inference@pipeline=dense_jsonl selections/embedding_model=e5/small_v2 selections.index_id=$indexId runtime=$Runtime stage.run_id=$baselineRun
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run stage inference dataset=beir_scifact pipeline/inference@pipeline=dense_query_repetition selections/embedding_model=e5/small_v2 selections.index_id=$indexId runtime=$Runtime stage.run_id=$repeatedRun
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run stage evaluation dataset=beir_scifact stage.inference_run_id=$baselineRun stage.run_id=$baselineEvaluation
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run stage evaluation dataset=beir_scifact stage.inference_run_id=$repeatedRun stage.run_id=$repeatedEvaluation
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run python scripts/compare_metrics.py "artifacts/runs/evaluation/$baselineEvaluation/metrics.json" "artifacts/runs/evaluation/$repeatedEvaluation/metrics.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "Shared index: $indexId"
Write-Output "Baseline inference: $baselineRun"
Write-Output "Repeated inference: $repeatedRun"
