---
name: create-experiment-card
description: Create a concise, reproducible Markdown experiment card for this retrieval research monorepo. Use when planning or preregistering a baseline-versus-treatment experiment, turning a research idea into runnable stage/config choices, defining metrics and acceptance criteria before execution, or documenting a proposed sweep without fabricating results.
---

# Create Experiment Card

Turn a research question into an auditable execution and analysis plan. Keep proposed decisions separate from observed results.

## Workflow

1. Read [references/card-template.md](references/card-template.md).
2. Inspect the target project's README, configs, components, scripts, and prior cards. Read `docs/research_workflows.md` for stage and configuration semantics.
3. Establish the smallest testable claim:
   - phrase one falsifiable hypothesis;
   - define exactly one primary treatment change;
   - name the baseline;
   - list invariants that must remain identical.
4. Resolve as much of the plan from repository evidence as possible: project, dataset and split, pipeline topology, component/selections overrides, shared upstream artifacts, metrics, runtime/device, seeds, and expected commands.
5. Mark unknown values as `TBD` with a concrete decision needed. Never invent a dataset, checkpoint, run id, result, budget, owner, or date.
6. Define one primary metric and its direction before looking at outcomes. Add secondary and diagnostic metrics only when they answer distinct failure questions.
7. Define success, failure, and inconclusive criteria. Avoid arbitrary thresholds unless the user or existing protocol supplies them; explain the rationale for every threshold.
8. Plan provenance and fairness: exact git/package versions captured by manifests, immutable run ids, one shared index where appropriate, identical candidate mappings, identical qrels, and controlled runtime settings.
9. Include executable stage commands or a sweep plan. Use placeholders only for values that cannot exist until execution, such as exact timestamped run ids.
10. Create or reuse `projects/<project>/experiments/<experiment-slug>/` and save the card as `experiment.md`. Keep its executable matrix in `configs/matrix.yaml`, its analysis notebook as `analysis.ipynb`, and its eventual report as `report.md` in the same experiment workspace.
11. Review the card against the checklist in the template. The card must be understandable before any results exist.

## Writing Rules

- Use future tense for planned actions and expected outcomes.
- Label rationale, assumptions, and risks explicitly.
- Distinguish "must remain fixed" from "will be varied."
- Prefer exact config group paths and command overrides over prose descriptions.
- Link repository files with relative Markdown links from the card location.
- Keep raw results and interpretation out of the card. Link the eventual report under `Results` when it exists.

## Completion Criteria

Complete the card only when another researcher can identify the controlled difference, reproduce the planned runs, determine the primary decision rule, and recognize confounders without asking what the experiment is actually comparing.
