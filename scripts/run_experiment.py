#!/usr/bin/env python
"""Run an experiment defined by a YAML config.

Example:
    python scripts/run_experiment.py --config experiments/configs/smoke_bm25.yaml
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.experiments.runner import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument(
        "--results-dir",
        default=str(ROOT / "experiments" / "results"),
        help="Where to write the result JSON",
    )
    args = parser.parse_args()

    result = run_experiment(args.config, results_dir=args.results_dir, repo_root=ROOT)

    print(f"experiment : {result['experiment_id']}")
    print(f"dataset    : {result['dataset']}@{result['dataset_version']} "
          f"({result['num_documents']} docs, {result['num_queries']} queries)")
    print(f"retriever  : {result['retriever']}")
    print("metrics    :")
    print(json.dumps(result["metrics"], indent=2))
    commit = result["metadata"]["git_commit"]
    print(f"git commit : {commit or '(not a git repo)'}")
    print(f"saved to   : {result['result_path']}")


if __name__ == "__main__":
    main()
