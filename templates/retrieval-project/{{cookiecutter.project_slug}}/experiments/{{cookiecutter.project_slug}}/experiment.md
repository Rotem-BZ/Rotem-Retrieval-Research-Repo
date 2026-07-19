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
2. Replace `REPLACE_WITH_EXACT_INDEXING_RUN_ID` in `configs/matrix.yaml`.
3. Run `uv run prepare-experiment experiments/{{ cookiecutter.project_slug }}`.
4. On Linux, run `uv run run-experiment --experiment {{ cookiecutter.project_slug }}`.
5. Record evaluation commands, acceptance criteria, and results here.

The preparer writes `experiment.yaml` and one immutable resolved config per run below
`runs/`. Stage artifacts remain below `artifacts/runs/` and are linked back to this
experiment through their manifests.

## Results

Pending.
