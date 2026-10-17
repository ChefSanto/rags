#!/usr/bin/env python
"""Idempotent baseline guard: run only the baseline experiments whose
run-artifacts are missing. Safe to run every Kaggle session — skips
everything when artifacts already exist (e.g. committed to git).

Usage: python scripts/ensure_baselines.py
"""

import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

BASELINES = [
    "fiqa_bm25", "fiqa_dense", "fiqa_hybrid",
    "nfcorpus_bm25", "nfcorpus_dense", "nfcorpus_hybrid",
]


def artifact_ok(cfg: str) -> bool:
    # config files use underscores (fiqa_bm25) but experiment/result names use
    # hyphens (fiqa-bm25-v1__*.json)
    results = sorted(glob.glob(f"experiments/results/{cfg.replace('_', '-')}-v1__*.json"))
    if not results:
        return False
    try:
        art = json.load(open(results[-1])).get("run_artifact")
    except Exception:
        return False
    return bool(art) and os.path.exists(art)


def main() -> None:
    missing = [c for c in BASELINES if not artifact_ok(c)]
    print("baseline artifacts missing:", missing or "none")
    for cfg in missing:
        print(f"--- running {cfg} ---")
        subprocess.run([sys.executable, "scripts/run_experiment.py",
                        "--config", f"experiments/configs/{cfg}.yaml"], check=True)


if __name__ == "__main__":
    main()
