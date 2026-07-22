# Retrieval Component Contract

## Placement

| Scope | Location | Test location |
| --- | --- | --- |
| Reusable retrieval behavior | `packages/retrieval-components/src/retrieval_components/<category>/` | `packages/retrieval-components/tests/` |
| Private component helper | The Python module that uses it; minimal duplication is acceptable | Tests for the owning package |
| One research hypothesis | `projects/<project>/src/<package>/components.py` or the project's established component layout | `projects/<project>/tests/` |
| Workflow/artifact orchestration | `packages/retrieval-core/src/retrieval_core/stages/` | `packages/retrieval-core/tests/` |

Use snake_case module names and PascalCase classes. In the shared package, follow the closest existing category rather than creating a near-duplicate category. In a project, follow its existing layout and prefer `components.py` for a small set of project-owned components. Do not introduce an inner `components/` package solely for one component or add a package-level `utils/` directory.

## Haystack Interface

- Decorate static components with `@component`.
- Declare static outputs with `@component.output_types(...)`.
- Use `component.set_input_types` and `component.set_output_types` only for genuinely dynamic named sockets, such as weighted fusion inputs.
- Return a dictionary whose keys exactly match output socket names.
- Keep constructor arguments serializable through Haystack pipeline YAML.
- Validate invalid constructor settings early with specific messages.
- Preserve document `id`, `content`, `meta`, `score`, and `embedding` when the transformation does not intentionally replace them.
- Keep constructor values compatible with Haystack pipeline serialization. Isolate external systems behind module helpers, importable factories, or explicitly serialized client configuration so tests stay local and deterministic.

## Configuration Boundaries

- Put cross-project component choices under `packages/retrieval-core/src/retrieval_core/configs/component/`, project-wide choices under `projects/<project>/configs/component/`, and choices used only by one experiment under that experiment's `configs/component/` overlay.
- Put experiment base entrypoints under `experiments/<experiment>/configs/base-experiment-configs/` and concrete run entrypoints directly under `experiments/<experiment>/configs/runs/`; do not put reusable component fragments in either directory.
- Put semantic choices shared by multiple components, such as embedding checkpoint and prefixes, under root `selections`.
- Keep the resolved `pipeline` subtree valid Haystack serialization: `components`, `connections`, `max_runs_per_component`, and `metadata`.
- Use Hydra `???` for required choices so composition fails early.

## Verification

From the repository root, use the narrowest relevant commands:

```powershell
uv run --extra dev --project packages/retrieval-components pytest packages/retrieval-components/tests/<test-file>.py
uv run --extra dev --project packages/retrieval-components pytest packages/retrieval-components/tests
uv run --project packages/retrieval-core pytest packages/retrieval-core/tests/test_pipeline_configs.py
uv run --project projects/<project> pytest projects/<project>/tests
```

If the component has a Hydra fragment, exercise at least one consuming pipeline through focused config and pipeline tests. Do not trigger model downloads or external services merely to claim unit-level verification.
