# Multi-project package architecture

The repository now uses three layers with independent Python project metadata.

```text
packages/retrieval-components/     publishable reusable components
packages/retrieval-core/           internal shared orchestration and Hydra configs
projects/<project>/                isolated research environment and local extensions
```

Each layer has its own `pyproject.toml` and `uv.lock`. There is intentionally no root
workspace lock: independently locking projects is what allows one project to stay on
an older component release while another upgrades.

## Version contract per project

A project expresses the component version it expects in normal PEP 621 dependencies:

```toml
[project]
dependencies = [
  "retrieval-core",
  "retrieval-components==0.1.0",
]
```

During monorepo development it can resolve those dependencies from editable paths:

```toml
[tool.uv.sources]
retrieval-core = { path = "../../packages/retrieval-core", editable = true }
retrieval-components = { path = "../../packages/retrieval-components", editable = true }
```

The version requirement remains meaningful: the editable component distribution must
still report `0.1.0`. To test the published artifact instead, remove only the
`retrieval-components` source override and regenerate that project's lockfile.

Run manifests also record the installed versions of `retrieval-core` and
`retrieval-components`, so results retain the resolved runtime versions in addition
to the project's declared and locked versions.

## Config ownership

`retrieval-core` ships reusable stage entry points and config groups as package data.
Its Hydra search-path plugin exposes `pkg://retrieval_core.configs` as the final
fallback. A project entrypoint composes project configs before core configs; an
experiment entrypoint composes experiment configs, then project configs, then core
configs. The CLI derives these roots from the YAML path passed to `--entrypoint`.

Project components are referenced directly by import path from project-owned YAML.
Reusable project YAML choices stay in the same Hydra groups as core choices but
are stored below the project's Python package namespace, for example
`configs/pipeline/inference/query_repetition_e5/dense_query_repetition.yaml` and
selected as `query_repetition_e5/dense_query_repetition`. Core choices remain
unqualified.
They should be promoted to `retrieval-components` only when they become reusable.

## Implemented example

[`projects/query-repetition-e5`](../projects/query-repetition-e5/README.md) exercises
the whole design: editable local dependencies, an exact component version, a custom
component, a project-only Hydra pipeline, independent tests and lockfile, and
declarative baseline-versus-treatment run entrypoints.

## New-project scaffold

[`templates/retrieval-project`](../templates/retrieval-project/README.md) turns that
pattern into a Cookiecutter. It creates an independently lockable package with
editable monorepo dependencies, a project-local component and Hydra pipeline,
component and composition tests, and experiment configs with explicit baseline and
treatment run entrypoints. The generated identity treatment doubles as an end-to-end
baseline-parity check before the project implements its actual treatment.

## Deferred work

- Split heavy integrations into optional dependency profiles.
- Redesign dataset ownership and preparation across projects.
- Establish publication automation and a compatibility policy for config schemas.
