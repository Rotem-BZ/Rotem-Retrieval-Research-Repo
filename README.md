# Retrieval Research

![R4 logo](docs/assets/r4-logo.png)

Hydra-managed information retrieval experiments built around Haystack
`AsyncPipeline` execution.

The repo provides reusable stage runners for indexing, inference, and
evaluation; composable Hydra configs for datasets, pipelines, components, and
model selections; and a checked-in toy dataset for smoke tests.

## Quickstart

Install dependencies with uv:

```bash
uv sync --extra dev
```

Run the toy keyword workflow:

```bash
uv run stage indexing \
  dataset=toy \
  pipeline/indexing@pipeline=dummy_jsonl

uv run stage inference \
  dataset=toy \
  pipeline/inference@pipeline=dummy_keyword \
  stage.run_name=toy_keyword

uv run stage evaluation \
  dataset=toy \
  stage.inference_run_name=toy_keyword
```

Use the interactive command builder when you want help assembling Hydra choices:

```bash
uv run build-command
```

## Repository Layout

- `configs/` contains Hydra entry points and reusable config groups.
- `data/processed/toy/` contains the small checked-in toy fixture.
- `docs/` contains workflow and component notes.
- `src/retrieval_research/` contains the package code.
- `tests/` contains regression tests for config composition, metrics, IO, and components.

See [docs/research_workflows.md](docs/research_workflows.md) for the full workflow.

