---
experiment_id: query-repetition-e5-small-toy
status: example
project: query-repetition-e5
created: 2026-07-21
---

# Query repetition on the toy dataset

This experiment is a small executable example for the stage-entrypoint workflow. It
uses the repository's checked-in toy dataset, the project-owned `QueryRepeater`, and
the shared `intfloat/e5-small-v2` dense index configuration.

The experiment is intended for configuration and command validation, not for drawing
research conclusions from the toy metrics. Before launching its run, create the
`toy-e5-small-index` index with the command documented in
[`docs/example_commands.md`](../../../../docs/example_commands.md).

The `repeated` run inherits the complete toy inference configuration from
`configs/base-experiment-configs/inference.yaml` and changes only the inference
pipeline to the project-owned `dense_query_repetition` topology.
