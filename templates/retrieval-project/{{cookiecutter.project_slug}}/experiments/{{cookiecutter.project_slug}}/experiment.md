# {{ cookiecutter.project_name }} experiment

## Research question

State the concrete retrieval question this experiment answers.

## Hypothesis

Describe the expected treatment effect and name the primary metric before running it.

## Comparison

The baseline uses `dense_jsonl`; the treatment uses
`{{ cookiecutter.pipeline_name }}`. Keep the dataset, embedding model, shared index,
retrieval depth, runtime, and evaluation procedure fixed.

## Execution

1. Create the shared index.
2. Replace `REPLACE_WITH_EXACT_INDEX_ID` in
   `configs/base-experiment-configs/inference.yaml`.
3. On Linux, run `uv run python ../../dev-scripts/run_in_parallel_screens.py --experiment {{ cookiecutter.project_slug }}`.
4. Record evaluation commands, acceptance criteria, and results here.

The complete shared configuration lives in
`configs/base-experiment-configs/inference.yaml`. Each `configs/runs/*.yaml`
entrypoint extends it through Hydra's defaults list and contains only fields or
config selections that differ. Experiment-local configs take precedence over
project configs, which take precedence over core configs.
Resolved stage artifacts remain below `artifacts/runs/` and are linked back to this
experiment through their manifests; launcher status and logs live below
`artifacts/experiments/`.

## Results

Pending.
