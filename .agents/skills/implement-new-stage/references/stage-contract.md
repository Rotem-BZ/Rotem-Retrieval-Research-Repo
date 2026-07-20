# Retrieval Stage Contract

## Integration Surface

| Concern | Repository location |
| --- | --- |
| Runner | `packages/retrieval-core/src/retrieval_core/stages/<name>.py` |
| Shared lifecycle | `packages/retrieval-core/src/retrieval_core/stages/base.py` |
| Registry and result type | `packages/retrieval-core/src/retrieval_core/stages/__init__.py` |
| CLI preparation | `packages/retrieval-core/src/retrieval_core/cli.py` |
| Entry config | `packages/retrieval-core/src/retrieval_core/configs/stages/<name>.yaml` |
| Artifact provenance | `packages/retrieval-core/src/retrieval_core/artifacts.py` |
| IO helpers | `packages/retrieval-core/src/retrieval_core/io.py` |
| Regression tests | `packages/retrieval-core/tests/` |

## Required Lifecycle

1. Compose the config and determine `stage.name`.
2. Freeze the run id with `prepare_stage_run_config`.
3. Prepare exact upstream dependencies without creating output artifacts.
4. Check missing config and file-backed inputs before expensive work.
5. Create the output directory once through `StageContext.from_config`.
6. Execute the stage workload.
7. Write `resolved_config.yaml` and a compact, machine-readable `result.json`.
8. Write `manifest.json` when the stage exposes reusable artifacts.

Manifests use named artifacts and explicit inputs. Downstream stages must resolve these names through `artifact_for_run`, which verifies the exact run directory, manifest entry, and artifact existence.

## Configuration Shape

Follow existing entry configs. A typical stage includes:

```yaml
defaults:
  - paths: local
  - dataset: ???
  - _self_

stage:
  name: <stage-name>
  run_name: null
  run_id: ${now:%Y%m%d_%H%M%S_%f}
  output_dir: ${paths.runs_dir}/${stage.name}/${stage.run_id}

hydra:
  job:
    chdir: false
  run:
    dir: ${paths.runs_dir}/hydra/${stage.name}/${stage.run_id}
```

Add only groups the stage needs. Use `???` for required selections.

## Test Matrix

- Registry: the name maps to the intended runner and appears in usage.
- Composition: required choices fail and a fixture configuration resolves.
- Preparation: exact upstream run ids resolve to manifest-declared artifacts.
- Conflict: a run id plus a different explicit path fails.
- Inputs: missing files fail with a specific error.
- Execution: runner writes the expected result and manifest keys.
- Immutability: an existing run directory is rejected.

Run:

```powershell
uv run --project packages/retrieval-core pytest packages/retrieval-core/tests/<focused-test>.py
uv run --project packages/retrieval-core pytest packages/retrieval-core/tests
```

From a project directory, also run a representative lightweight `uv run stage <stage-name> ...` command.
