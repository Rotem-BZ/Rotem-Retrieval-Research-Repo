# Development scripts

These repository-only tools are intentionally kept outside the `retrieval-core`
runtime package. Run them through the active research project's `uv` environment:

```shell
uv run python ../../dev-scripts/build_command.py
uv run python ../../dev-scripts/create_run.py experiments/<experiment-slug>
uv run python ../../dev-scripts/run_in_parallel_screens.py
```

`build_command.py` uses the nearest `configs/` directory at or above the current
working directory. For an experiment, choices are resolved from experiment configs,
then project configs, then the configs packaged by `retrieval-core`. Pass
`--config-dir <path>` to select a config tree explicitly.

`create_run.py` writes one minimal Hydra `configs/runs/<name>.yaml` entrypoint that
inherits a complete config below `configs/base-experiment-configs/`. The experiment
launcher
requires Linux and GNU Screen; its worker, state models, and Screen adapter are
implementation details used by the scripts above. The old `prepare_experiment.py`
and `run_experiment.py` names remain compatibility aliases.
