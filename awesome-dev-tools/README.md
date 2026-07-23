# Awesome dev tools

These repository-only tools are intentionally kept outside the `retrieval-core`
runtime package. Run them through the active research project's `uv` environment:

```shell
uv run python ../../awesome-dev-tools/interactive_build_command.py
uv run python ../../awesome-dev-tools/interactive_create_run.py experiments/<experiment-slug>
uv run python ../../awesome-dev-tools/visualize_pipeline.py artifacts/runs/inference/<run-id>/resolved_config.yaml
uv run python ../../awesome-dev-tools/run_in_screen.py --name toy-index -- uv run stage indexing dataset=toy
uv run python ../../awesome-dev-tools/interactive_run_in_parallel_screens.py
```

When `bash_aliases.sh` has been sourced from the repository root, the command builder
and Screen cleanup utility have short Bash commands:

```shell
build-command
kill-screens
```

The exposed scripts are:

- `interactive_build_command.py`: interactively build a `retrieval-core` stage command. It uses
  the nearest `configs/` directory at or above the current working directory. For
  an experiment, choices are resolved from experiment configs, then project
  configs, then the configs packaged by `retrieval-core`. Pass `--config-dir
  <path>` to select a config tree explicitly. The final command is copied to the
  clipboard. On Linux, install an `xclip` or `xsel` system package to provide a
  clipboard backend when the desktop environment does not already provide one.
- `interactive_create_run.py`: write one minimal Hydra `configs/runs/<name>.yaml` entrypoint
  that inherits a complete config below `configs/base-experiment-configs/`.
- `visualize_pipeline.py`: render the Haystack pipeline in a stage run's
  `resolved_config.yaml`. SVG output defaults to
  `artifacts/visualizations/pipelines/<stage>/<run-id>.svg`, outside the immutable
  run directory. Use `--output` to select another location, or `--format` to render
  PNG, JPEG, WebP, or PDF. Rendering uses `https://mermaid.ink` by default; point
  `--server-url` at a private Mermaid server when required. Stage boundary
  components named `input` and `output` are labeled `stage_input` and
  `stage_output` in the diagram to avoid names reserved by Haystack's renderer.
- `interactive_run_in_parallel_screens.py`: choose run definitions and launch them through GNU
  Screen on Linux.
- `run_in_screen.py`: launch one arbitrary command in a detached GNU Screen session
  on Linux without creating an experiment. Use `--cwd <project>` when invoking it
  outside the command's project directory; logs default to
  `<cwd>/artifacts/screens/<session>.log`.
- `kill_screens.sh`: close every GNU Screen session owned by the current user.
- `interactive_prepare_experiment.py`: compatibility alias for
  `interactive_create_run.py`.
- `interactive_run_experiment.py`: compatibility alias for
  `interactive_run_in_parallel_screens.py`.

Private implementation modules live in `_internal/`, and tests live in `tests/`.
