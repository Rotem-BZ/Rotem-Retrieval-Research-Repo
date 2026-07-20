# Development scripts

These repository-only tools are intentionally kept outside the `retrieval-core`
runtime package. Run them through the active research project's `uv` environment:

```shell
uv run python ../../dev-scripts/build_command.py
uv run python ../../dev-scripts/create_run.py experiments/<experiment-slug>
uv run python ../../dev-scripts/run_in_parallel_screens.py
```

`create_run.py` writes one minimal Hydra `runs/<name>.yaml` file that inherits a
complete experiment base config. The experiment launcher
requires Linux and GNU Screen; its worker, state models, and Screen adapter are
implementation details used by the scripts above. The old `prepare_experiment.py`
and `run_experiment.py` names remain compatibility aliases.
