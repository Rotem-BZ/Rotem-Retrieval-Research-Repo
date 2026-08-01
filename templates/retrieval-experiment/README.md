# Retrieval experiment Cookiecutter

This template creates one version-controlled experiment workspace containing:

- a concise experiment card;
- one complete inference base configuration;
- minimal baseline and treatment run entrypoints; and
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
- `embedding_model` selects the shared embedding-model configuration.
- `index_id` must identify an existing immutable index used by both arms.
- `baseline_pipeline` and `treatment_pipeline` are full Hydra choices such as
  `retrieve/dense_in_memory` and `my_project/dense_experiment`.

The analysis notebook discovers qrels from the first selected inference run's
`resolved_config.yaml`; it does not assume a dataset family or storage layout.
