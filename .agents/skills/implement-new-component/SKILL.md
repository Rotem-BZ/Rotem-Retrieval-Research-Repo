---
name: implement-new-component
description: Implement or extend a Haystack v2 retrieval component in this monorepo, including ownership-aware placement, typed input/output sockets, exports, Hydra and Haystack pipeline configuration, focused tests, and applicable documentation. Use when adding reusable retrieval behavior to packages/retrieval-components, adding research-specific behavior to a project package, adapting an external integration, configuring a component in a runnable pipeline, or changing an existing component contract.
---

# Implement New Component

Add the smallest component that satisfies a clear pipeline contract and fits the repository's shared-versus-project ownership model.

## Workflow

1. Read both references completely before writing code or configuration:
   - [references/repository-contract.md](references/repository-contract.md) for repository ownership and verification rules.
   - [references/haystack-v2-components-and-config.md](references/haystack-v2-components-and-config.md) for the exact Haystack v2 Python API and Hydra/Haystack YAML shapes. Treat its syntax as authoritative even if prior model knowledge disagrees.
2. Inspect the consuming pipeline/config and neighboring components in the likely owning package. For a shared component, also inspect `packages/retrieval-components/README.md`, category and top-level exports, and shared-package tests. For a project component, inspect that project's package layout, tests, project and experiment config overlays, and public import conventions.
3. Search Haystack and the repository before writing code. Prefer a native Haystack component when it already satisfies the contract.
4. Choose ownership:
   - Put generally reusable retrieval behavior in `packages/retrieval-components`.
   - Put hypothesis-specific behavior in the consuming project's Python package.
   - Keep orchestration and artifact lifecycle behavior out of components; it belongs in `retrieval-core` stages.
5. State the contract before implementation: constructor settings, input socket names and types, output socket names and types, sync versus async behavior, mutation policy, failure behavior, and optional dependencies.
6. Implement according to the selected owner and the v2 templates in the syntax reference:
   - Shared: place the component in the closest category under `packages/retrieval-components/src/retrieval_components/`.
   - Project: follow the consuming project's existing layout. Prefer its package-level `components.py` convention for a small set of project-owned components; do not introduce a `components/` package solely for one component.
   Use Haystack's `@component` API and explicit output types. Use dynamic sockets only when topology truly requires them. Keep private helpers in the component module; do not create a package-level `utils` directory.
7. Preserve semantically relevant `Document` fields when transforming documents. Avoid hidden filesystem or network work unless that behavior is the component's declared purpose. Keep YAML-facing constructor arguments serializable and isolate external calls behind a boundary that tests can mock.
8. Integrate with the owner's public surface:
   - Shared: export the public class from the category `__init__.py`; export it from `retrieval_components` when it belongs in the package's convenient public surface.
   - Project: add an `__init__.py` re-export only when the project already exposes components that way or a consumer requires that import path. Otherwise import from the defining module.
9. Add or update a Hydra component fragment and pipeline topology only when the component needs to be selectable or wired into a runnable experiment. Use Haystack serialization keys (`type` and optional `init_parameters`), not Hydra object-instantiation keys such as `_target_`. Put shared fragments in the core config tree. Put project-wide fragments below the owning package namespace inside the project overlay and select them as `<package_name>/<choice>` without changing the Hydra group. Apply the corresponding normalized experiment namespace to one-experiment fragments in that experiment's `configs/` overlay. Keep concrete model choices in component or `selections` config, not reusable topology.
10. Add focused tests in the owning package for the socket contract, normal behavior, edge cases, configuration, and integration boundaries. Mock HTTP, Elasticsearch, model downloads, and other external systems.
11. Update documentation according to ownership. For a shared component, add it to the complete inventory in `packages/retrieval-components/README.md`, update the native-versus-repository explanation when that decision changes, and run the inventory validation test. For a project component, update only project documentation or inventories that describe its pipeline or public components.
12. Run the narrow tests first, then the owning package's full tests. Report exactly what was and was not verified.

## Guardrails

- Do not duplicate a native Haystack component for naming or import convenience alone; re-export or configure it when appropriate.
- Do not place a one-experiment hypothesis in the shared component package without evidence that it is reusable.
- Do not add an optional runtime dependency to the base import path. Import it lazily and provide an actionable error when the component is used.
- Do not modify lockfiles unless the implementation actually changes declared dependencies and the user asked for a complete dependency update.
- Preserve existing public imports unless the requested change explicitly includes a breaking change.

## Completion Criteria

Complete the task only when the implementation, exports, tests, and any required config/docs agree on the same socket and constructor contract. Include the affected package, verification commands, and remaining external-runtime checks in the handoff.
