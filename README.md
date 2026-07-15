# Retrieval Research Monorepo

Hydra-managed information-retrieval experiments built around Haystack
`AsyncPipeline` execution.

The repository is split into independently installable units:

- `packages/retrieval-components/` is the reusable, publishable Haystack component
  library. Its distribution name and version are `retrieval-components==0.1.0`.
- `packages/retrieval-core/` is internal orchestration shared through an editable
  path. It owns stages, metrics, artifacts, mappings, and the common Hydra config tree.
- `projects/` contains research projects. Every project owns a `pyproject.toml` and
  lockfile, so it can declare its own component-library version.
- `data/processed/toy/` is the checked-in fixture used by core smoke tests.

Hydra uses the consuming project's `configs/` directory as the primary source and
the config package shipped by `retrieval-core` as the fallback. A project can add or
override only the groups it owns.

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

## Development

Each unit is checked independently:

```powershell
uv sync --project packages/retrieval-components --extra dev
uv run --project packages/retrieval-components pytest packages/retrieval-components/tests

uv sync --project packages/retrieval-core --extra dev
uv run --project packages/retrieval-core pytest packages/retrieval-core/tests

uv sync --project projects/query-repetition-e5 --extra dev
uv run --project projects/query-repetition-e5 pytest projects/query-repetition-e5/tests
```

See [the research workflow guide](docs/research_workflows.md) for stage semantics and
[the package plan](docs/multiple_projects_package_plan.md) for the architectural
rationale.
