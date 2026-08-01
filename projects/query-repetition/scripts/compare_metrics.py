"""Compare repeated-query metrics against the baseline from two JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("repeated", type=Path)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    repeated = json.loads(args.repeated.read_text(encoding="utf-8"))
    names = sorted(set(baseline) | set(repeated))

    print(f"{'metric':<16} {'baseline':>12} {'repeated':>12} {'delta':>12}")
    for name in names:
        baseline_value = float(baseline[name])
        repeated_value = float(repeated[name])
        print(
            f"{name:<16} {baseline_value:>12.6f} {repeated_value:>12.6f} "
            f"{repeated_value - baseline_value:>+12.6f}"
        )


if __name__ == "__main__":
    main()
