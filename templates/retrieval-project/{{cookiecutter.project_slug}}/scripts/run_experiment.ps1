param(
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$Runtime = if ($Device -eq "cuda") { "gpu" } else { "cpu" }

$suffix = Get-Date -Format "yyyyMMdd-HHmmss"
$indexId = "{{ cookiecutter.project_slug }}-index-$suffix"
$indexingRun = "{{ cookiecutter.project_slug }}-indexing-$suffix"
$baselineRun = "{{ cookiecutter.project_slug }}-baseline-$suffix"
$treatmentRun = "{{ cookiecutter.project_slug }}-treatment-$suffix"
$baselineEvaluation = "{{ cookiecutter.project_slug }}-baseline-eval-$suffix"
$treatmentEvaluation = "{{ cookiecutter.project_slug }}-treatment-eval-$suffix"

uv run prepare-beir --data-dir data --dataset {{ cookiecutter.beir_dataset }}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run stage indexing dataset={{ cookiecutter.dataset_config }} pipeline/indexing@pipeline=dense/documents_jsonl selections/embedding_model={{ cookiecutter.embedding_model }} selections.index_id=$indexId runtime=$Runtime stage.run_id=$indexingRun
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run stage inference dataset={{ cookiecutter.dataset_config }} pipeline/inference@pipeline=retrieve/dense_jsonl selections/embedding_model={{ cookiecutter.embedding_model }} selections.index_id=$indexId runtime=$Runtime stage.run_id=$baselineRun
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run stage inference dataset={{ cookiecutter.dataset_config }} pipeline/inference@pipeline={{ cookiecutter.package_name }}/{{ cookiecutter.pipeline_name }} selections/embedding_model={{ cookiecutter.embedding_model }} selections.index_id=$indexId runtime=$Runtime stage.run_id=$treatmentRun
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run stage evaluation dataset={{ cookiecutter.dataset_config }} stage.inference_run_id=$baselineRun stage.run_id=$baselineEvaluation
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run stage evaluation dataset={{ cookiecutter.dataset_config }} stage.inference_run_id=$treatmentRun stage.run_id=$treatmentEvaluation
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run python scripts/compare_metrics.py "artifacts/runs/evaluation/$baselineEvaluation/metrics.json" "artifacts/runs/evaluation/$treatmentEvaluation/metrics.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "Shared index: $indexId"
Write-Output "Baseline inference: $baselineRun"
Write-Output "Treatment inference: $treatmentRun"
