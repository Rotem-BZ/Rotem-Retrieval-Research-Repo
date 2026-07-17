# Experiment Card Template

Use this structure unless the project already has a stronger convention. Omit irrelevant optional sections; do not omit the hypothesis, comparison, metrics, execution plan, or validity checks.

```markdown
---
experiment_id: <kebab-case-id>
status: draft
project: <project-slug>
created: <YYYY-MM-DD-or-TBD>
owner: <name-or-TBD>
---

# <Experiment title>

## Research question

<One answerable question.>

## Hypothesis

<One falsifiable statement, including the expected direction of the primary metric.>

## Rationale

<Why the treatment could affect retrieval; link the motivating code, issue, or paper when available.>

## Comparison

| Dimension | Baseline | Treatment | Held fixed? |
| --- | --- | --- | --- |
| Pipeline | ... | ... | yes/no |
| Component/config change | ... | ... | yes/no |
| Dataset and split | ... | ... | yes |
| Index | ... | ... | yes/no/not applicable |
| Candidate mapping | ... | ... | yes |
| Runtime/device | ... | ... | yes |

Primary treatment change: `<exactly one sentence>`

## Data and sampling

- Dataset/split: ...
- Query/document inclusion: ...
- Input mapping: ...
- Repetitions or seeds: ...
- Exclusions: ...

## Metrics and decision rule

- Primary metric: `<name>`, direction `<higher/lower is better>`
- Secondary metrics: ...
- Diagnostics: ...
- Success: ...
- Failure: ...
- Inconclusive: ...

## Execution plan

1. Prepare prerequisites: ...
2. Validate configurations: ...
3. Run shared upstream stages: ...
4. Run baseline and treatment: ...
5. Evaluate with the same qrels and metric config: ...

```powershell
<exact or parameterized commands>
```

Run naming scheme: ...

Expected artifacts:

- `artifacts/runs/<stage>/<run-id>/resolved_config.yaml`
- `artifacts/runs/<stage>/<run-id>/result.json`
- `artifacts/runs/<stage>/<run-id>/manifest.json`
- evaluation `metrics.json`

## Validity and risk checks

- Confounders: ...
- Leakage risks: ...
- External dependencies: ...
- Resource/time constraints: ...
- Abort conditions: ...

## Analysis plan

<State the comparison table, aggregation across seeds/datasets, and any qualitative error analysis planned before seeing results.>

## Results

Pending. Link the completed experiment report here without rewriting the preregistered plan.
```

## Review Checklist

- The hypothesis is falsifiable and names a metric direction.
- Baseline and treatment differ in the intended variable only, or every additional difference is disclosed.
- Dataset, split, mapping, index reuse, metrics, seeds, and runtime controls are explicit.
- Commands reference real config groups or clearly marked placeholders.
- The decision rule was written before results.
- Expected immutable artifacts and report linkage are explicit.
- Unknowns are `TBD`, not guessed.
