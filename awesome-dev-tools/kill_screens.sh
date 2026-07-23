#!/usr/bin/env bash

set -uo pipefail

if ! command -v screen >/dev/null 2>&1; then
  echo "GNU Screen is not installed or is not available on PATH." >&2
  exit 1
fi

screen_output="$(screen -ls 2>&1)"
screen_status=$?

sessions=()
while IFS= read -r session; do
  sessions+=("$session")
done < <(
  printf '%s\n' "$screen_output" |
    awk '$1 ~ /^[0-9]+\./ { print $1 }'
)

if ((${#sessions[@]} == 0)); then
  if ((screen_status != 0)) && [[ "$screen_output" != *"No Sockets found"* ]]; then
    printf '%s\n' "$screen_output" >&2
    exit "$screen_status"
  fi
  echo "No GNU Screen sessions are open."
  exit 0
fi

failed=0
for session in "${sessions[@]}"; do
  if screen -S "$session" -X quit; then
    echo "Closed GNU Screen session: $session"
  else
    echo "Failed to close GNU Screen session: $session" >&2
    failed=1
  fi
done

if ((failed != 0)); then
  exit 1
fi

echo "Closed ${#sessions[@]} GNU Screen session(s)."
