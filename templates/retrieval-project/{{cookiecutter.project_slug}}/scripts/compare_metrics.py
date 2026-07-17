"""Compare treatment metrics against a baseline from two JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("treatment", type=Path)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    treatment = json.loads(args.treatment.read_text(encoding="utf-8"))
    names = sorted(set(baseline) | set(treatment))

    print(f"{'metric':<16} {'baseline':>12} {'treatment':>12} {'delta':>12}")
    for name in names:
        baseline_value = float(baseline[name])
        treatment_value = float(treatment[name])
        print(
            f"{name:<16} {baseline_value:>12.6f} {treatment_value:>12.6f} "
            f"{treatment_value - baseline_value:>+12.6f}"
        )


if __name__ == "__main__":
    main()
