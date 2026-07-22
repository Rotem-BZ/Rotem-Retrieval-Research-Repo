---
name: generate-experiment-report
description: Generate an evidence-led Markdown report from completed retrieval experiment artifacts, including manifests, resolved configs, results, predictions, and evaluation metrics. Use when comparing baseline and treatment runs, summarizing a sweep, documenting experiment outcomes, checking provenance and fairness, calculating metric deltas, or turning immutable run directories into a reproducible research conclusion.
---

# Generate Experiment Report

Build conclusions from run artifacts, not from remembered commands or intended configuration.

## Workflow

1. Read [references/report-template.md](references/report-template.md).
2. Locate the experiment card when one exists. Treat any hypothesis, primary metric, and decision rule actually recorded there as preregistered; do not add or rewrite them after seeing results.
3. Inspect checked-in run entrypoints under `projects/<project>/experiments/<experiment-slug>/configs/runs/` when available, then identify the exact baseline and treatment run ids for every relevant stage. Do not select "latest" implicitly. If run identity is ambiguous, stop and request the exact runs.
4. Read each run's `manifest.json`, `resolved_config.yaml`, and `result.json`. Read manifest-declared artifacts such as `metrics.json` and inspect predictions only when needed for diagnostics.
5. Verify comparability before calculating conclusions:
   - same dataset and qrels;
   - same input mapping or sampling;
   - same shared index when the treatment does not target indexing;
   - same metric configuration;
   - intended config difference isolated after excluding dynamic run/output fields;
   - disclosed git commit, package versions, Python version, device, and seeds.
6. Record every provenance mismatch. Classify it as expected, harmless dynamic metadata, a limitation, or a comparison-invalidating confounder. Do not silently normalize substantive differences.
7. Calculate absolute deltas as `treatment - baseline`. Calculate relative deltas only when the baseline is nonzero and the percentage aids interpretation. Preserve enough precision to reproduce the calculation; format display values consistently.
8. For repeated seeds or datasets, report per-run values plus the aggregation unit, count, mean, and dispersion. Do not claim statistical significance without an appropriate test and independent replication structure.
9. Compare the primary result with the card's decision rule. Separate:
   - observations directly supported by artifacts;
   - interpretations or plausible mechanisms;
   - limitations and untested alternatives.
10. Add concise diagnostic analysis only when it helps explain the aggregate result. Avoid presenting cherry-picked queries as representative.
11. Save the report as `projects/<project>/experiments/<experiment-slug>/report.md`, beside the source `experiment.md`, declarative `configs/`, and analysis notebook. Resolved configs remain in exact stage artifact directories under `artifacts/runs/`; link those directories rather than copying resolved configs into the experiment tree.
12. Re-read every numeric and provenance claim against its source artifact before handing off the report.

## Guardrails

- Do not rerun experiments, mutate artifacts, or fabricate missing metrics unless the user explicitly asks for execution.
- Do not infer success from secondary metrics when the preregistered primary decision rule failed.
- Do not call a small single-run delta meaningful, significant, or generalizable without supporting uncertainty evidence.
- Do not hide null package versions, different commits, different candidate sets, or failed runs.
- Use "inconclusive" when missing artifacts or confounders prevent a defensible comparison.

## Completion Criteria

Complete the report only when every reported value traces to an exact artifact, delta arithmetic has been checked, provenance differences are disclosed, the conclusion follows the predeclared decision rule, and limitations distinguish evidence from inference.
