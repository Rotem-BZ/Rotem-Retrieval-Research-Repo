# Experiment Report Template

## Evidence Collection

For each run, resolve these files from the exact run directory:

| File | Use |
| --- | --- |
| `manifest.json` | Stage/run identity, named artifacts, upstream inputs, config hash, git/package/Python provenance |
| `resolved_config.yaml` | Dataset, mapping, pipeline, selections, metrics, runtime, seeds, output paths |
| `result.json` | Compact execution outcome and artifact summary |
| manifest-declared `metrics.json` | Evaluation values |
| manifest-declared predictions | Optional query-level diagnostics |

Treat `stage.run_id`, `stage.run_name`, output paths, Hydra run directories, timestamps, and config hashes as dynamic fields only after confirming they are the sole reason for a diff. Never discard dataset, pipeline, mapping, metric, model, or runtime differences as dynamic.

## Report Structure

```markdown
# <Experiment title>: results

## Executive summary

<Two to four sentences: comparison, primary outcome, decision, and largest caveat.>

## Experiment identity

- Experiment card: [link](...)
- Baseline inference/evaluation runs: ...
- Treatment inference/evaluation runs: ...
- Git commit(s): ...
- Package versions: ...

## Hypothesis and decision rule

<Copy or faithfully summarize the preregistered hypothesis, primary metric, and success/failure/inconclusive rule. If no card existed, say the rule was retrospective.>

## Provenance and comparability

| Check | Baseline | Treatment | Assessment |
| --- | --- | --- | --- |
| Dataset/qrels | ... | ... | matched/mismatch |
| Input mapping | ... | ... | matched/mismatch |
| Upstream index | ... | ... | shared/different/N/A |
| Metric config | ... | ... | matched/mismatch |
| Intended treatment | ... | ... | isolated/not isolated |
| Runtime/seeds | ... | ... | matched/mismatch |
| Code/packages | ... | ... | matched/mismatch |

## Results

| Metric | Baseline | Treatment | Absolute delta | Relative delta | Direction |
| --- | ---: | ---: | ---: | ---: | --- |
| ... | ... | ... | ... | ... | higher/lower is better |

Primary metric outcome: ...

## Diagnostics

<Optional aggregate slices or representative error categories. State selection method.>

## Interpretation

### Observations

- <Artifact-supported statements only.>

### Possible explanations

- <Clearly labeled inference.>

## Limitations

- <Single run, dataset scope, confounders, missing artifacts, variance, external services, etc.>

## Conclusion and next step

<Succeeded, failed, or inconclusive under the stated rule; name the smallest justified follow-up.>

## Reproduction

<Exact project directory, run ids, commands/configs, and linked artifact paths.>
```

## Arithmetic and Aggregation Checks

- Compute `absolute delta = treatment - baseline` for every metric.
- Compute `relative delta = (treatment - baseline) / abs(baseline)` only when baseline is nonzero; label it as a percentage.
- Preserve raw machine values while rounding display cells consistently.
- For multiple seeds, pair baseline and treatment by seed when the design is paired.
- For multiple datasets, show per-dataset results before any macro average and state weighting.
- Recalculate at least one table row manually or with an independent expression to catch column/order mistakes.

## Conclusion Vocabulary

- **Succeeded:** the preregistered success rule is met and no comparison-invalidating confounder exists.
- **Failed:** the preregistered failure rule is met with a valid comparison.
- **Inconclusive:** the rule is not met decisively, uncertainty is too large, artifacts are missing, or comparability is compromised.

Avoid "significant" unless a named statistical test, assumptions, sample unit, and result support it.
