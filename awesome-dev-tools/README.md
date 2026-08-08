# Awesome dev tools

These repository-only tools are intentionally kept outside the `retrieval-core`
runtime package. Run them through the active research project's `uv` environment:

```shell
uv run python ../../awesome-dev-tools/interactive_build_command.py
uv run python ../../awesome-dev-tools/interactive_create_run.py experiments/<experiment-slug>
uv run python ../../awesome-dev-tools/export_experiment_report.py experiments/<experiment-slug>
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
  clipboard through `pyperclip`. On Linux, pyperclip needs either a Wayland session
  with `wl-clipboard`, or an X11 session with `xclip`/`xsel`. A plain SSH terminal
  has no graphical clipboard even when `xclip` is installed. To use the local
  clipboard over SSH, run an X server locally, install `xauth` and `xclip` on the
  server, and connect with `ssh -X`; verify that `$DISPLAY` is set before running
  the builder. Clipboard failures print diagnostics for the active session.
- `interactive_create_run.py`: write one minimal Hydra `configs/runs/<name>.yaml` entrypoint
  that inherits a complete config below `configs/base-experiment-configs/`. It suggests
  `<experiment>--<run-name>` and records the confirmed `stage.run_id` directly in YAML.
- `export_experiment_report.py`: execute an experiment's `analysis.ipynb` against saved
  artifacts and atomically export `report.html` without adding outputs to the source
  notebook. Execution uses the project root as its working directory.
- `visualize_pipeline.py`: render the Haystack pipeline in a stage run's
  `resolved_config.yaml`. SVG output defaults to
  `artifacts/runs/<stage>/<run-id>/pipeline.svg`, alongside the immutable run's
  config and outputs. Indexing and inference generate that SVG automatically; this
  tool remains useful for alternate formats, themes, and explicit destinations. Its
  default renderer is shared with `retrieval-core`, reads resolved YAML directly, and
  uses headless Matplotlib, so it needs no network service and does not initialize
  pipeline components. Use `--output` to select another location, or
  `--format` to render PNG, JPEG, WebP, or PDF. The optional `--renderer mermaid`
  mode uses `https://mermaid.ink` by default; point `--server-url` at a private
  Mermaid server when required. In Mermaid diagrams, stage boundary components
  named `input` and `output` are labeled `stage_input` and `stage_output` to avoid
  names reserved by Haystack's renderer.
- `interactive_run_in_parallel_screens.py`: choose run definitions and launch them through GNU
  Screen on Linux.
- `run_in_screen.py`: launch one arbitrary command in a detached GNU Screen session
  on Linux without creating an experiment. Use `--cwd <project>` when invoking it
  outside the command's project directory; logs default to
  `<cwd>/artifacts/screens/<session>.log`. This Screen log is the complete session
  transcript; a stage also writes its own diagnostic `run.log` inside its immutable
  run directory.
- `kill_screens.sh`: close every GNU Screen session owned by the current user.
- `interactive_prepare_experiment.py`: compatibility alias for
  `interactive_create_run.py`.
- `interactive_run_experiment.py`: compatibility alias for
  `interactive_run_in_parallel_screens.py`.

Private implementation modules live in `_internal/`, and tests live in `tests/`.
