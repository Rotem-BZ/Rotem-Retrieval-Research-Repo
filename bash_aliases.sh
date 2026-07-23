#!/usr/bin/env bash

# Repository-local shortcuts. Source this file from Bash; do not execute it.
RETRIEVAL_RESEARCH_REPO_ROOT="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"

alias build-command='uv run python "$RETRIEVAL_RESEARCH_REPO_ROOT/awesome-dev-tools/interactive_build_command.py"'
alias kill-screens='bash "$RETRIEVAL_RESEARCH_REPO_ROOT/awesome-dev-tools/kill_screens.sh"'
