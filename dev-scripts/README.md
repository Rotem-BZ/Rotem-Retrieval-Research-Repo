# Development scripts

These repository-only tools are intentionally kept outside the `retrieval-core`
runtime package. Run them through the active research project's `uv` environment:

```shell
uv run python ../../dev-scripts/build_command.py
uv run python ../../dev-scripts/prepare_experiment.py experiments/<experiment-slug>
uv run python ../../dev-scripts/run_experiment.py
```

The experiment launcher requires Linux and GNU Screen. Its worker, state models,
and Screen adapter are implementation details used by the three scripts above.
