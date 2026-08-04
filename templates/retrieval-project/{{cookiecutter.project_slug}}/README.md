# {{ cookiecutter.project_name }}

{{ cookiecutter.project_short_description }}

This project owns reusable treatment code and pipeline configuration. Individual
research comparisons live in independently generated workspaces below
`experiments/`.

## Define the treatment

The generated `{{ cookiecutter.component_class_name }}` is an identity
transformation. Replace its `run` implementation with the project behavior, then
update its unit test. Constructor parameters belong in
`configs/pipeline/inference/{{ cookiecutter.package_name }}/{{ cookiecutter.pipeline_name }}.yaml`.

Keeping the identity implementation is useful for a parity check: baseline and
treatment metrics should be identical when both arms consume the same inputs.

## Set up the project

From this directory:

```powershell
uv sync --extra dev
uv run nbstripout --install --attributes ../../.gitattributes
uv run pre-commit install --install-hooks
uv run pytest
```

For a CUDA 12.6 PyTorch environment, sync with
`uv sync --extra dev --extra torch-cu126`. On an older NVIDIA driver that requires
PyTorch 2.5.1 with CUDA 12.4, use
`uv sync --extra dev --extra torch-cu124-legacy` instead. The two PyTorch extras are
mutually exclusive. Use the legacy extra with a Python version supported by PyTorch
2.5.1, such as Python 3.12; PyTorch 2.5.1 does not provide Python 3.14 wheels.

The project owns its environment and lockfile. It declares
`retrieval-components=={{ cookiecutter.retrieval_components_version }}` while
resolving both monorepo packages locally and editably during development.

## Create an experiment

The generated `experiments/` directory starts empty. Scaffold each comparison
separately:

```powershell
uvx cookiecutter ../../templates/retrieval-experiment --output-dir experiments
```

Select `{{ cookiecutter.package_name }}/{{ cookiecutter.pipeline_name }}` as the
treatment pipeline when the experiment should exercise this project's generated
component. The experiment template owns the dataset, embedding model, index, baseline,
and treatment selections.

Run an explicit experiment entrypoint with:

```powershell
uv run stage inference --entrypoint experiments/<experiment-slug>/configs/runs/baseline.yaml
```

On Linux, the interactive launcher can run any subset of an experiment's entrypoints:

```bash
uv run python ../../awesome-dev-tools/interactive_run_in_parallel_screens.py
```
