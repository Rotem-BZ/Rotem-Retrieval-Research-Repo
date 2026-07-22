# {{ cookiecutter.project_name }}

{{ cookiecutter.project_short_description }}

This project compares one project-local query treatment with the unchanged dense
pipeline while holding the dataset, embedding model, document index, and evaluation
procedure constant.

## Define the treatment

The generated `{{ cookiecutter.component_class_name }}` is an identity
transformation. Replace its `run` implementation with the experimental behavior,
then update its unit test. Constructor parameters belong in
`experiments/{{ cookiecutter.project_slug }}/configs/pipeline/inference/{{ cookiecutter.pipeline_name }}.yaml`.

Keeping the identity implementation is useful for a parity check: baseline and
treatment metrics should be identical when both arms consume the same index.

## Set up the project

From this directory:

```powershell
uv sync --extra dev
uv run nbstripout --install --attributes ../../.gitattributes
uv run pre-commit install --install-hooks
uv run pytest
```

The project owns its environment and lockfile. It declares
`retrieval-components=={{ cookiecutter.retrieval_components_version }}` while
resolving both monorepo packages locally and editably during development.

## Run the comparison

```powershell
./scripts/run_experiment.ps1
```

The runner defaults to CPU. Use a CUDA 12.6 PyTorch environment with:

```powershell
uv sync --extra dev --extra torch-cu126
./scripts/run_experiment.ps1 -Device cuda
```

The experiment prepares BEIR `{{ cookiecutter.beir_dataset }}`, validates both
pipeline graphs, creates one shared index, runs both inference arms against it,
evaluates both runs, and prints per-metric deltas. Every invocation uses timestamped,
immutable run IDs.

The generated experiment workspace is
[`experiments/{{ cookiecutter.project_slug }}`](experiments/{{ cookiecutter.project_slug }}).
After creating a shared index, replace the placeholder index ID in
`experiments/{{ cookiecutter.project_slug }}/configs/base-experiment-configs/inference.yaml`.
On Linux, use
`uv run python ../../awesome-dev-tools/interactive_run_in_parallel_screens.py` to choose the experiment
and its run subset. Use
`uv run python ../../awesome-dev-tools/interactive_create_run.py experiments/{{ cookiecutter.project_slug }}`
to add another explicit run interactively.

The default selections are:

- dataset: `{{ cookiecutter.dataset_config }}`;
- embedding model: `{{ cookiecutter.embedding_model }}`;
- baseline pipeline: `dense_jsonl`; and
- treatment pipeline: `{{ cookiecutter.pipeline_name }}`.

Small changes on one dataset are exploratory evidence. Broaden the dataset and seed
coverage before drawing a general conclusion.

## Analyze predictions

Open the experiment's
[`analysis.ipynb`](experiments/{{ cookiecutter.project_slug }}/analysis.ipynb).
Add readable labels and exact inference run IDs to `RUNS`, then run the cells. The
notebook resolves each run's prediction artifact through its manifest and builds:

- `predictions_df`, one row per retrieved result with rank, score, content, metadata,
  and matched qrel relevance; and
- `query_summary_df`, one row per run and query with retrieval depth, relevant counts,
  first relevant rank, reciprocal rank, recall, and query-length fields.

Use these DataFrames directly for project-specific plots below the final notebook cell.
