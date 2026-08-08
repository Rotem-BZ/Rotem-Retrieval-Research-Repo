# {{ cookiecutter.experiment_name }}

## Experiment

{{ cookiecutter.experiment_short_description }} The baseline uses
`{{ cookiecutter.baseline_pipeline }}` and the treatment uses
`{{ cookiecutter.treatment_pipeline }}`.

## Hypothesis

{{ cookiecutter.hypothesis }}

## Runs

| Run | Stage | Difference |
| --- | --- | --- |
| `baseline` | Inference | Uses the shared inference configuration unchanged. |
| `treatment` | Inference | Replaces the baseline pipeline with `{{ cookiecutter.treatment_pipeline }}`. |
| `baseline-evaluation` | Evaluation | Evaluates the exact `baseline` inference run. |
| `treatment-evaluation` | Evaluation | Evaluates the exact `treatment` inference run; otherwise uses the same evaluation configuration as `baseline-evaluation`. |
