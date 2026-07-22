# Retrieval Research Monorepo

Hydra-managed information-retrieval experiments built around Haystack
`AsyncPipeline` execution.

## Quick start

Install [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/), then run
the following commands from a terminal on Windows or Linux:

```shell
git clone https://github.com/Rotem-BZ/Rotem-Retrieval-Research-Repo.git
cd Rotem-Retrieval-Research-Repo/projects/query-repetition-e5
uv sync --extra dev
uv run nbstripout --install --attributes ../../.gitattributes
uv run pre-commit install --install-hooks
uv run stage --help
```

Each project owns its environment and resolves the shared monorepo packages through
editable local dependencies. Run stage commands through `uv run` from the project
directory so project-specific components and configuration are available.

The setup also installs two repository-local Git protections. `nbstripout` keeps
notebook outputs in the working copy but strips them from commits, while the
pre-commit hook rejects newly added files larger than 10 MiB unless they are tracked
with Git LFS. Check the setup at any time with:

```shell
uv run nbstripout --status
uv run pre-commit run --all-files
```

For example, after creating an index, inference can be launched with:

```shell
uv run stage inference dataset=beir_scifact runtime=cpu pipeline/inference@pipeline=dense_query_repetition selections/embedding_model=e5/small_v2 selections.index_id=YOUR_INDEX_ID runtime.query_concurrency_limit=8
```

## Experiments and parallel runs

Each project keeps durable research workspaces below `experiments/`. An experiment
can contain its card, complete shared stage configs under
`configs/base-experiment-configs/`, minimal Hydra entrypoints under `configs/runs/`,
an analysis notebook, and a report. Create a run interactively with:

```shell
uv run python ../../awesome-dev-tools/interactive_create_run.py experiments/<experiment-slug>
```

With no path, the helper lets you select or create an experiment. On Linux with GNU
Screen installed, the launcher chooses an experiment and then a subset of its run
files:

```shell
uv run python ../../awesome-dev-tools/interactive_run_in_parallel_screens.py
```

The launcher displays ready, waiting, running, succeeded, and failed runs; accepts
selections such as `1,3,4-7`; asks for the maximum number of executing runs;
launches the selected Screen sessions; and exits. Waiting sessions form persistent
dependency lanes, so the concurrency cap remains effective after the launcher exits.
See the [awesome dev tools guide](awesome-dev-tools/README.md) for every
exposed script and its purpose.

The repository is split into independently installable units:

- `packages/retrieval-components/` is the reusable, publishable Haystack component
  library. Its distribution name and version are `retrieval-components==0.1.0`.
- `packages/retrieval-core/` is internal orchestration shared through an editable
  path. It owns stages, metrics, artifacts, mappings, and the common Hydra config tree.
- `projects/` contains research projects. Every project owns a `pyproject.toml` and
  lockfile, so it can declare its own component-library version.
- `data/processed/toy/` is the checked-in fixture used by core smoke tests.

Passing `--entrypoint experiments/<experiment>/configs/runs/<run>.yaml` makes the
stage CLI infer the experiment and project from that path. Hydra then searches the
experiment's `configs/`, the project's `configs/`, and finally the config package
shipped by `retrieval-core`.

## Research workflow

See [the research workflow guide](docs/research_workflows.md) for the complete
experiment lifecycle: preregistering an experiment card, composing Hydra configs,
testing and running immutable stages, reusing exact artifacts, analyzing
predictions in a notebook, generating an evidence-led report, and launching
explicit experiment runs.

For copy-ready stage invocations that use only the checked-in toy dataset, see
[the example command guide](docs/example_commands.md).

## Query-repetition example

[`projects/query-repetition-e5`](projects/query-repetition-e5/README.md) demonstrates
the complete pattern. It installs both monorepo dependencies editably, explicitly
declares `retrieval-components==0.1.0`, adds a project-local `QueryRepeater`
component, and includes a reproducible E5-small/SciFact baseline comparison.

```powershell
Set-Location projects/query-repetition-e5
uv sync --extra dev
./scripts/run_experiment.ps1
```

Every run writes immutable outputs, `resolved_config.yaml`, `result.json`, and
`manifest.json` below the active project's `artifacts/runs/`. Manifests record the
installed `retrieval-core` and `retrieval-components` distribution versions.

## Create a research project

Generate another isolated baseline-versus-treatment project from the repository
Cookiecutter:

```powershell
uvx cookiecutter templates/retrieval-project --output-dir projects
Set-Location projects/<project-slug>
uv sync --extra dev
uv run nbstripout --install --attributes ../../.gitattributes
uv run pre-commit install --install-hooks
uv run pytest
```

The generated treatment starts as an identity transform so its first run can verify
baseline parity. Implement the project-specific behavior in the generated component
and update its focused test before running the research comparison. See the
[`retrieval-project` template](templates/retrieval-project/README.md) for its prompts
and assumptions.

## Development

Each unit is checked independently:

```powershell
uv sync --project packages/retrieval-components --extra dev
uv run --project packages/retrieval-components pytest packages/retrieval-components/tests

uv sync --project packages/retrieval-core --extra dev
uv run --project packages/retrieval-core pytest packages/retrieval-core/tests awesome-dev-tools/tests

uv sync --project projects/query-repetition-e5 --extra dev
uv run --project projects/query-repetition-e5 pytest projects/query-repetition-e5/tests
```

See [the package plan](docs/multiple_projects_package_plan.md) for the architectural
rationale behind the monorepo split.
