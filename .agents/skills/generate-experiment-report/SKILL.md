---
name: generate-experiment-report
description: Generate a concise HTML report from a retrieval experiment's saved run artifacts. Use when documenting completed experiment results, selecting exact inference and evaluation runs, or presenting the hypothesis and target metrics in the experiment analysis notebook.
---

# Generate Experiment Report

Use saved artifacts only; do not run stages or pipelines.

1. Read `experiment.md` for the brief description, hypothesis, and target metrics.
2. Identify the exact completed inference and evaluation run IDs from `configs/runs/` and their manifests. Never select the latest run implicitly.
3. Update `analysis.ipynb` with those run IDs and keep the report to:
   - brief experiment description;
   - exact run list;
   - hypothesis;
   - target metric values and treatment-minus-baseline deltas.
4. Export from the project root:

```shell
uv run python ../../awesome-dev-tools/export_experiment_report.py experiments/<experiment-slug>
```

If exact runs or target metrics are ambiguous, ask for them instead of guessing.
