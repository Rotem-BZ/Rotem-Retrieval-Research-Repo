# Retrieval research project Cookiecutter

This template creates an isolated research project containing:

- an independently locked Python package;
- editable links to `retrieval-core` and `retrieval-components`;
- one project-local Haystack query component and Hydra pipeline;
- focused component and pipeline-composition tests; and
- an empty `experiments/` directory ready for separately scaffolded experiments.

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
`configs/pipeline/inference/<package_name>/<pipeline_name>.yaml` before treating it
as research code. Leaving it unchanged is useful as an end-to-end parity smoke test.

`uv sync` creates the generated project's own `uv.lock`; the template does not copy
another experiment's resolved dependency graph.

## Important prompts

- `project_slug` becomes the directory and distribution name.
- `package_name` must be a valid Python package name.
- `pipeline_name` names the project-owned Hydra inference configuration selected as
  `<package_name>/<pipeline_name>`.
- `component_class_name` names the starter Haystack component.

Dataset, model, index, baseline, and treatment selections belong to an experiment,
not the project scaffold. Generate one from inside the project:

```powershell
uvx cookiecutter ../../templates/retrieval-experiment --output-dir experiments
```

The generated relative dependency paths assume the project is created directly
under `projects/`.
