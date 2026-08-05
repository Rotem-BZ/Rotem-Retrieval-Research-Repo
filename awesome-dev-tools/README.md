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
  clipboard. It first uses the operating system clipboard through `pyperclip`. In
  a headless SSH session, where tools such as `xclip` cannot work without an X
  display, it falls back to the OSC 52 terminal protocol so the clipboard belongs
  to the local terminal. The terminal emulator must permit OSC 52 clipboard access.
- `interactive_create_run.py`: write one minimal Hydra `configs/runs/<name>.yaml` entrypoint
  that inherits a complete config below `configs/base-experiment-configs/`.
- `visualize_pipeline.py`: render the Haystack pipeline in a stage run's
  `resolved_config.yaml`. SVG output defaults to
  `artifacts/visualizations/pipelines/<stage>/<run-id>.svg`, outside the immutable
  run directory. The default NetworkX renderer reads the resolved YAML directly
  and uses headless Matplotlib, so it needs no network service and does not
  initialize pipeline components. Use `--output` to select another location, or
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
