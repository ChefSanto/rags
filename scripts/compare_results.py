#!/usr/bin/env python
"""Compare all experiments in experiments/results/ as a markdown table.

    python scripts/compare_results.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.experiments.registry import comparison_rows, load_results, to_markdown  # noqa: E402


def main() -> None:
    results_dir = ROOT / "experiments" / "results"
    rows = comparison_rows(load_results(results_dir))
    print(to_markdown(rows))
    print(f"\n({len(rows)} experiment result(s) in {results_dir})")


if __name__ == "__main__":
    main()
