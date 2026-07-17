# Retrieval Component Contract

## Placement

| Scope | Location | Test location |
| --- | --- | --- |
| Reusable retrieval behavior | `packages/retrieval-components/src/retrieval_components/components/<category>/` | `packages/retrieval-components/tests/` |
| Shared non-component helper | `packages/retrieval-components/src/retrieval_components/utils/` | `packages/retrieval-components/tests/` |
| One research hypothesis | `projects/<project>/src/<package>/components/` | `projects/<project>/tests/` |
| Workflow/artifact orchestration | `packages/retrieval-core/src/retrieval_core/stages/` | `packages/retrieval-core/tests/` |

Use snake_case module names and PascalCase classes. Follow the closest existing category rather than creating a near-duplicate category.

## Haystack Interface

- Decorate static components with `@component`.
- Declare static outputs with `@component.output_types(...)`.
- Use `component.set_input_types` and `component.set_output_types` only for genuinely dynamic named sockets, such as weighted fusion inputs.
- Return a dictionary whose keys exactly match output socket names.
- Keep constructor arguments serializable through Haystack pipeline YAML.
- Validate invalid constructor settings early with specific messages.
- Preserve document `id`, `content`, `meta`, `score`, and `embedding` when the transformation does not intentionally replace them.
- Prefer injected clients or callables for external systems so tests stay local and deterministic.

## Configuration Boundaries

- Put reusable component choices under `packages/retrieval-core/src/retrieval_core/configs/component/` or the consuming project's `configs/component/` overlay.
- Put semantic choices shared by multiple components, such as embedding checkpoint and prefixes, under root `selections`.
- Keep the resolved `pipeline` subtree valid Haystack serialization: `components`, `connections`, `max_runs_per_component`, and `metadata`.
- Use Hydra `???` for required choices so composition fails early.

## Verification

From the repository root, use the narrowest relevant commands:

```powershell
uv run --project packages/retrieval-components pytest packages/retrieval-components/tests/<test-file>.py
uv run --project packages/retrieval-components pytest packages/retrieval-components/tests
uv run --project packages/retrieval-core pytest packages/retrieval-core/tests/test_pipeline_configs.py
uv run --project projects/<project> pytest projects/<project>/tests
```

If the component has a Hydra fragment, validate at least one consuming pipeline with `uv run stage --validate ...` from the owning project directory. Do not trigger model downloads or external services merely to claim unit-level verification.
