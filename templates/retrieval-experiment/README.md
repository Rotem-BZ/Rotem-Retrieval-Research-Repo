# Retrieval experiment Cookiecutter

This template creates one version-controlled experiment workspace containing:

- a concise experiment card with its hypothesis and run differences;
- complete inference and evaluation base configurations;
- minimal baseline, treatment, and evaluation run entrypoints; and
- a dataset-agnostic prediction-analysis notebook.

Generate it from a retrieval project's root:

```powershell
uvx cookiecutter ../../templates/retrieval-experiment --output-dir experiments
```

The experiment is written to `experiments/<experiment-slug>/`. Its config search
order is experiment, project, then `retrieval-core`, so both pipeline selections must
name choices available from those roots.

## Important prompts

- `dataset_config` selects a Hydra dataset configuration.
- `runtime_config` selects the execution profile.
- `embedding_model` selects the shared embedding-model configuration.
- `index_id` must identify an existing immutable index used by both arms.
- `baseline_pipeline` and `treatment_pipeline` are full Hydra choices such as
  `retrieve/dense_in_memory` and `my_project/dense_experiment`.
- `retrieval_top_k` sets the shared retrieval depth.
- `evaluation_metrics` is a JSON list recorded in the shared evaluation base.
- `hypothesis` is copied verbatim into the experiment card and analysis notebook.

Every generated run entrypoint declares its immutable `stage.run_id` explicitly.
Ordinary stage commands outside an experiment retain their timestamp-based default.

The analysis notebook discovers qrels from the first selected inference run's
`resolved_config.yaml`; it does not assume a dataset family or storage layout.
After adding exact inference and evaluation run IDs, export the completed notebook
from the project root with:

```powershell
uv run python ../../awesome-dev-tools/export_experiment_report.py experiments/<experiment-slug>
```
