---
name: implement-new-component
description: Implement or extend a Haystack retrieval component in this monorepo, including placement, typed input/output sockets, exports, Hydra configuration, focused tests, and component inventory documentation. Use when adding reusable retrieval behavior to packages/retrieval-components, adding research-specific behavior to a project package, adapting an external integration, or changing an existing component contract.
---

# Implement New Component

Add the smallest component that satisfies a clear pipeline contract and fits the repository's shared-versus-project ownership model.

## Workflow

1. Read [references/repository-contract.md](references/repository-contract.md).
2. Inspect `packages/retrieval-components/README.md`, neighboring components, their `__init__.py` exports, tests, and the pipeline/config that will consume the component.
3. Search Haystack and the repository before writing code. Prefer a native Haystack component when it already satisfies the contract.
4. Choose ownership:
   - Put generally reusable retrieval behavior in `packages/retrieval-components`.
   - Put hypothesis-specific behavior in the consuming project's Python package.
   - Keep orchestration and artifact lifecycle behavior out of components; it belongs in `retrieval-core` stages.
5. State the contract before implementation: constructor settings, input socket names and types, output socket names and types, sync versus async behavior, mutation policy, failure behavior, and optional dependencies.
6. Implement the component in the closest category directly under `src/retrieval_components/`. Use Haystack's `@component` API and explicit output types. Use dynamic sockets only when topology truly requires them. Keep private helpers in the component module; do not create a package-level `utils` directory.
7. Preserve semantically relevant `Document` fields when transforming documents. Avoid hidden filesystem or network work unless that behavior is the component's declared purpose and can be injected or mocked.
8. Export the public class from the category `__init__.py`; export it from `retrieval_components` when it is part of the package's convenient public surface.
9. Add or update a Hydra component fragment and pipeline topology only when the component needs to be selectable or wired into a runnable experiment. Keep concrete model choices in component or `selections` config, not reusable topology.
10. Add focused tests for the socket contract, normal behavior, edge cases, configuration, and integration boundaries. Mock HTTP, Elasticsearch, model downloads, and other external systems.
11. Add the component to the complete inventory in `packages/retrieval-components/README.md` and update the native-versus-repository explanation when that decision changes. Run the inventory validation test.
12. Run the narrow tests first, then the owning package's full tests. Report exactly what was and was not verified.

## Guardrails

- Do not duplicate a native Haystack component for naming or import convenience alone; re-export or configure it when appropriate.
- Do not place a one-experiment hypothesis in the shared component package without evidence that it is reusable.
- Do not add an optional runtime dependency to the base import path. Import it lazily and provide an actionable error when the component is used.
- Do not modify lockfiles unless the implementation actually changes declared dependencies and the user asked for a complete dependency update.
- Preserve existing public imports unless the requested change explicitly includes a breaking change.

## Completion Criteria

Complete the task only when the implementation, exports, tests, and any required config/docs agree on the same socket and constructor contract. Include the affected package, verification commands, and remaining external-runtime checks in the handoff.
