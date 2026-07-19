# Retrieval research project Cookiecutter

This template creates an isolated research project that follows the repository's
baseline-versus-treatment pattern. Generated projects contain:

- an independently locked Python package;
- editable links to `retrieval-core` and `retrieval-components`;
- one project-local Haystack query component and Hydra pipeline;
- an experiment workspace containing a card, reusable run matrix, and Jupyter
  analysis notebook;
- focused component and pipeline-composition tests; and
- a PowerShell runner that builds one shared index, evaluates both arms, and prints
  metric deltas.

## Generate a project

From the repository root:

```powershell
uvx cookiecutter templates/retrieval-project --output-dir projects
Set-Location projects/<project-slug>
uv sync --extra dev
uv run nbstripout --install --attributes ../../.gitattributes
uv run pre-commit install --install-hooks
uv run pytest
```

The generated treatment is deliberately an identity transformation. Edit
`src/<package_name>/components.py` and its initialization parameters in
`configs/pipeline/inference/<pipeline_name>.yaml` before treating the comparison as
a research run. Leaving it unchanged is useful as an end-to-end parity smoke test.

`uv sync` creates the generated project's own `uv.lock`; the template does not copy
another experiment's resolved dependency graph.

## Important prompts

- `project_slug` becomes the directory, distribution, and run-ID prefix.
- `package_name` must be a valid Python package name.
- `pipeline_name` names the project-owned Hydra inference configuration.
- `component_class_name` names the starter Haystack component.
- `beir_dataset` is the name accepted by `prepare-beir`, such as `scifact`.
- `dataset_config` is the corresponding Hydra dataset selection, such as
  `beir_scifact`.
- `embedding_model` is a shared config selection, such as `e5/small_v2`.

The generated relative dependency paths assume the project is created directly
under `projects/`.

Before materializing the generated experiment, create its shared index and replace
`REPLACE_WITH_EXACT_INDEXING_RUN_ID` in
`experiments/<project-slug>/configs/matrix.yaml`. Then run
`uv run prepare-experiment experiments/<project-slug>`; on Linux,
`uv run run-experiment` chooses the experiment and any subset of its runs.
