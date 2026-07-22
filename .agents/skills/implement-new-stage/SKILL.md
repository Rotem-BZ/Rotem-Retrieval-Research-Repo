---
name: implement-new-stage
description: Implement a new retrieval-core workflow stage end to end, including its runner, Hydra entry configuration, stage registry, dependency preparation and validation, immutable run artifacts, CLI behavior, tests, and workflow documentation. Use when adding a new long-running experiment phase, a new artifact-producing or artifact-consuming stage, or changing a stage's lifecycle contract.
---

# Implement New Stage

Add a stage as a complete workflow boundary, not only as a runner function.

## Workflow

1. Read [references/stage-contract.md](references/stage-contract.md).
2. Inspect `retrieval_core.stages.base`, the closest existing runner, its entry YAML, `retrieval_core.cli`, registry, artifact helpers, and relevant tests.
3. Define the stage contract before editing:
   - required config groups and leaf fields;
   - upstream run references or explicit input paths;
   - primary output artifacts and result payload;
   - sync or async execution;
   - input and configuration checks that should happen before expensive work.
4. Add `packages/retrieval-core/src/retrieval_core/stages/<stage_name>.py`. Accept one resolved `DictConfig`. Use `StageContext.from_config` before material writes so immutable output directories cannot be overwritten.
5. Resolve exact upstream runs through `artifact_for_run`. Permit explicit artifact paths only when the stage contract intentionally supports legacy or external inputs. Reject conflicting run-id and path selections.
6. Write the resolved config and compact `result.json`; write `manifest.json` with named output artifacts and provenance-bearing inputs. Use shared IO helpers and `project_path` for repository-relative paths.
7. Add the shared Hydra entry config under `packages/retrieval-core/src/retrieval_core/configs/stages/<stage_name>.yaml`. Include `stage.name`, run naming/id/output layout, relevant output paths, runtime settings, command-builder prompts, and Hydra run directory. Bare CLI stage names resolve through the current project's config overlay when present, then the core group. Use project or experiment YAML only as a more specific entrypoint/overlay for a concrete use; invoke a specific file with `stage <stage_name> --entrypoint <path>.yaml`.
8. Register the runner in `retrieval_core.stages.STAGE_RUNNERS`. Extend `StageResult` only if the existing result union cannot represent the new result.
9. Update CLI dependency preparation when the stage has upstream inputs. Keep generic CLI behavior generic; avoid adding a special command path when registry/config dispatch is sufficient.
10. Add tests for config composition, registry visibility, input resolution, failure handling, immutable output behavior, and written result/manifest content. Mock the expensive workload.
11. Update `docs/research_workflows.md` and CLI examples when users need to understand or invoke the new stage.
12. Run focused tests, the full `retrieval-core` suite, and a lightweight end-to-end stage command using real fixture paths. If the change adds a project or experiment entrypoint, exercise that exact YAML through `--entrypoint`.

## Guardrails

- Do not write into an existing run directory.
- Do not infer an upstream run from "latest"; require an exact run id or explicit path.
- Keep reusable retrieval algorithms in components. A stage coordinates data, pipelines, metrics, and artifacts.
- Keep project-only stage config in the project overlay unless the workflow is broadly reusable.

## Completion Criteria

Complete the stage only when it appears in CLI help, composes from Hydra, executes through the registry, records immutable outputs and provenance, and has regression coverage for its failure paths.
