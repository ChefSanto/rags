#!/usr/bin/env python
"""Oracle routing analysis — the upper bound of per-query strategy selection.

H2 in papers/property-routing/proposal.md: if an oracle that picks the best
strategy per query barely beats the best fixed strategy, routing has no
headroom and the research question is dead on this dataset.

Reads result JSONs for one dataset, loads their saved ranked-run artifacts
(written by the runner), and reports:
  - mean metric per strategy (sanity check against the result files)
  - oracle: mean over queries of the per-query best strategy
  - oracle gap vs the best fixed strategy (absolute and relative)
  - oracle-best share per strategy (does the winner actually vary?)
  - head-to-head win rates between strategies

Example:
    python scripts/oracle_analysis.py --dataset fiqa --metric ndcg_at_10
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.datasets.beir import load_beir  # noqa: E402
from core.evaluation.retrieval import (  # noqa: E402
    per_query_ndcg_at_k,
    per_query_recall_at_k,
    parse_metric,
)
from core.experiments.registry import load_results  # noqa: E402

PER_QUERY = {"ndcg": per_query_ndcg_at_k, "recall": per_query_recall_at_k}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset name, e.g. fiqa")
    parser.add_argument("--metric", default="ndcg_at_10",
                        help="per-query metric for oracle comparison (ndcg_at_K or recall_at_K)")
    parser.add_argument("--results-dir", default=str(ROOT / "experiments" / "results"))
    parser.add_argument("--out", default=None,
                        help="path for the markdown report (default: experiments/results/)")
    args = parser.parse_args()

    base, k = parse_metric(args.metric)
    if base not in PER_QUERY:
        raise SystemExit(f"oracle analysis supports ndcg_at_K / recall_at_K, got {args.metric}")

    results = [r for r in load_results(args.results_dir)
               if r.get("dataset") == args.dataset and r.get("run_artifact")]
    if len(results) < 2:
        raise SystemExit(
            f"need >= 2 result files for dataset '{args.dataset}' with run artifacts; "
            f"found {len(results)}. Note: results produced before the run-artifact "
            f"runner change have no 'run_artifact' field — re-run those experiments."
        )

    # keep the newest result per retriever (same dataset, same config lineage)
    by_strategy: dict[str, dict] = {}
    for r in sorted(results, key=lambda r: r["experiment_id"]):
        by_strategy[r["retriever"]] = r

    cfg = next(iter(by_strategy.values()))["config"]
    dataset = load_beir(cfg["dataset"]["path"],
                        name=cfg["dataset"]["name"],
                        version=cfg["dataset"]["version"])

    per_strategy: dict[str, dict[str, float]] = {}
    for strategy, r in sorted(by_strategy.items()):
        run = json.loads(Path(r["run_artifact"]).read_text(encoding="utf-8"))
        per_strategy[strategy] = PER_QUERY[base](run, dataset.qrels, k)

    common = sorted(set.intersection(*[set(v) for v in per_strategy.values()]))
    if not common:
        raise SystemExit("no common judged queries across strategies")

    # sanity: macro means must match the metrics stored in the result files
    means = {s: sum(v[q] for q in common) / len(common) for s, v in per_strategy.items()}
    lines = [f"# Oracle analysis — {args.dataset} ({args.metric})", "",
             f"- judged queries compared: {len(common)}",
             f"- strategies: {', '.join(sorted(per_strategy))}", ""]
    for s, r in sorted(by_strategy.items()):
        stored = r["metrics"].get(args.metric)
        computed = round(means[s], 4)
        flag = "OK" if stored is None or abs(stored - computed) < 1e-3 else "MISMATCH"
        lines.append(f"- sanity {s}: computed {computed:.4f} vs result file {stored} [{flag}]")
    lines.append("")

    oracle_per_q = {q: max(per_strategy[s][q] for s in per_strategy) for q in common}
    oracle = sum(oracle_per_q.values()) / len(common)
    best_fixed_s = max(means, key=means.get)
    best_fixed = means[best_fixed_s]
    gap = oracle - best_fixed
    rel = gap / best_fixed * 100 if best_fixed else 0.0

    lines += ["## Oracle gap", "",
              "| quantity | value |", "|---|---|",
              f"| oracle (per-query best) | {oracle:.4f} |",
              f"| best fixed strategy | {best_fixed_s} @ {best_fixed:.4f} |",
              f"| **oracle gap** | **+{gap:.4f} (+{rel:.1f}%)** |", ""]

    shares = {s: sum(1 for q in common if per_strategy[s][q] >= max(per_strategy[t][q] for t in per_strategy) - 1e-12) / len(common)
              for s in per_strategy}
    lines += ["## Oracle-best share (ties counted for each)", "",
              "| strategy | share |", "|---|---|"]
    lines += [f"| {s} | {shares[s]:.1%} |" for s in sorted(shares, key=shares.get, reverse=True)]
    lines.append("")

    strategies = sorted(per_strategy)
    lines += ["## Head-to-head win rates", "", "| vs \\ wins against | " + " | ".join(strategies) + " |",
              "|" + "---|" * (len(strategies) + 1)]
    for a in strategies:
        row = [a]
        for b in strategies:
            if a == b:
                row.append("—")
            else:
                w = sum(1 for q in common if per_strategy[a][q] > per_strategy[b][q] + 1e-12) / len(common)
                row.append(f"{w:.1%}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    out = Path(args.out) if args.out else Path(args.results_dir) / (
        f"oracle-{args.dataset}-{args.metric}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nreport saved: {out}")


if __name__ == "__main__":
    main()
