---
name: create-experiment
description: Interactively design and scaffold a complete retrieval experiment workspace, including a concise experiment card, complete shared base configs, minimal run entrypoints, paired evaluation configs, and an analysis notebook. Use when creating a new experiment, formalizing ad hoc retrieval commands as a versioned experiment, or replacing card-only experiment setup.
---

# Create Experiment

Interview the user before writing files. Create versioned configuration and reporting artifacts; do not run stages or pipelines.

## Discover available choices

Run this from the selected project root, resolving this skill directory first:

```powershell
uv run python <skill-dir>/scripts/discover_choices.py . --format json
```

Treat its output as the source of truth for choices and evidence-based recommendations. Read `references/experiment-layout.md` completely before scaffolding.

## Interview the user

Ask one decision at a time and wait for the answer. For every configuration choice:

1. Show the full numbered list of available choices with each choice's source and description.
2. Mark one recommendation and explain whether it comes from existing experiment usage or the repository template fallback.
3. Accept only an available choice unless the user explicitly asks to implement a new one.

Collect, in order:

1. Experiment name, short description, and hypothesis. The hypothesis must come directly from the user; never infer it.
2. Run shape: paired baseline/treatment inference plus paired evaluation (recommended), a single run, or a user-defined shape.
3. Dataset, runtime, baseline pipeline, treatment pipeline, and embedding-model choices.
4. The exact existing index ID when inference consumes an index. Show all discovered IDs. If indexing belongs to the new experiment, interview for its mandatory choices and create its base and run configs too.
5. Retrieval depth, target metrics, and every run-specific override. Show observed values and recommend the closest existing experiment pattern.

Continue until every mandatory Hydra field is resolved. Summarize the complete design and get confirmation before writing files.

## Scaffold the experiment

Render `templates/retrieval-experiment` with the confirmed values. Prefer Cookiecutter; if it is unavailable, create the same layout described in the reference.

- Make base configs complete and directly composable.
- Keep run entrypoints minimal: they contain only their base include and true differences.
- Do not put launcher-controlled `stage.run_id`, `paths.project_root`, or `experiment.*` values in run files.
- Make evaluation entries reference the exact upstream inference run IDs.
- Limit `experiment.md` to the short description, user-supplied hypothesis, and a run-difference table derived from the files created.
- Copy the analysis notebook and align its run labels and target metrics with the confirmed design.
- Never overwrite an existing experiment without explicit approval.

## Validate and report

Compose every run entrypoint with `retrieval_core.utils.config.compose_entrypoint_config`. Verify that no mandatory value is missing, evaluation inputs use exact run IDs, and no placeholder such as `REPLACE_WITH_EXACT_INDEX_ID` remains.

Report the created paths and the selected choices. Do not launch the experiment.
