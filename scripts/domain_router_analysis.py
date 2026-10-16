#!/usr/bin/env python
"""Domain-routing baseline (H4 in papers/property-routing/proposal.md).

Quantifies what a router achieves using ONLY domain labels: route each query
to the best fixed strategy for its domain. The property-based router (H3)
must beat this number — and the gap between domain routing and the oracle is
what document properties must explain beyond domain membership.

For each dataset, per-query scores per strategy are recomputed from the run
artifacts (same machinery as oracle_analysis.py), then combined:

  - per-dataset strategy table (which strategy wins where)
  - domain router: weighted mean over queries using each dataset's best
  - best uniform: the single best strategy applied to every dataset
  - combined oracle: weighted mean of per-dataset oracles (any router's ceiling)
  - headline: domain-routing gain over best uniform (what H3 must beat)

Example:
    python scripts/domain_router_analysis.py --datasets fiqa nfcorpus --metric ndcg_at_10
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


def collect_scores(dataset_name: str, metric_base: str, k: int, results_dir: str) -> dict:
    """Per-dataset per-strategy means + oracle, computed over common judged queries."""
    results = [r for r in load_results(results_dir)
               if r.get("dataset") == dataset_name and r.get("run_artifact")]
    by_strategy: dict[str, dict] = {}
    for r in sorted(results, key=lambda r: r["experiment_id"]):
        by_strategy[r["retriever"]] = r
    if len(by_strategy) < 2:
        raise SystemExit(
            f"need >= 2 strategies with run artifacts for '{dataset_name}', "
            f"found {len(by_strategy)}"
        )

    cfg = next(iter(by_strategy.values()))["config"]
    dataset = load_beir(cfg["dataset"]["path"],
                        name=cfg["dataset"]["name"],
                        version=cfg["dataset"]["version"])

    per_strategy: dict[str, dict[str, float]] = {}
    for strategy, r in sorted(by_strategy.items()):
        run = json.loads(Path(r["run_artifact"]).read_text(encoding="utf-8"))
        per_strategy[strategy] = PER_QUERY[metric_base](run, dataset.qrels, k)

    common = sorted(set.intersection(*[set(v) for v in per_strategy.values()]))
    if not common:
        raise SystemExit(f"no common judged queries for '{dataset_name}'")
    means = {s: sum(per_strategy[s][q] for q in common) / len(common)
             for s in per_strategy}
    oracle = sum(max(per_strategy[s][q] for s in per_strategy) for q in common) / len(common)
    return {"n": len(common), "means": means, "oracle": oracle}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="dataset names, e.g. --datasets fiqa nfcorpus")
    parser.add_argument("--metric", default="ndcg_at_10")
    parser.add_argument("--results-dir", default=str(ROOT / "experiments" / "results"))
    args = parser.parse_args()

    base, k = parse_metric(args.metric)
    if base not in PER_QUERY:
        raise SystemExit(f"supports ndcg_at_K / recall_at_K, got {args.metric}")

    per_dataset = {name: collect_scores(name, base, k, args.results_dir)
                   for name in args.datasets}
    total_n = sum(d["n"] for d in per_dataset.values())

    def weighted(values: dict[str, float]) -> float:
        return sum(per_dataset[name]["n"] * v for name, v in values.items()) / total_n

    routing = {name: max(d["means"], key=d["means"].get) for name, d in per_dataset.items()}
    domain_router = weighted({name: max(d["means"].values()) for name, d in per_dataset.items()})

    all_strategies = sorted({s for d in per_dataset.values() for s in d["means"]})
    uniform = {}
    for s in all_strategies:
        vals = {name: d["means"][s] for name, d in per_dataset.items() if s in d["means"]}
        w = sum(per_dataset[name]["n"] for name in vals)
        uniform[s] = sum(per_dataset[name]["n"] * v for name, v in vals.items()) / w
    best_uniform_s = max(uniform, key=uniform.get)

    combined_oracle = weighted({name: d["oracle"] for name, d in per_dataset.items()})
    gain = domain_router - uniform[best_uniform_s]
    rel = gain / uniform[best_uniform_s] * 100 if uniform[best_uniform_s] else 0.0

    lines = [f"# Domain-routing baseline — {', '.join(args.datasets)} ({args.metric})",
             "", f"- combined judged queries: {total_n}", "",
             "## Per-domain strategy means", "",
             "| domain | judged | " + " | ".join(all_strategies) + " | best |", "|---|---|" + "---|" * (len(all_strategies) + 1)]
    for name, d in per_dataset.items():
        row = [name, str(d["n"])] + [f"{d['means'].get(s, float('nan')):.4f}" for s in all_strategies]
        row.append(f"**{routing[name]}**")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines += ["## Routing comparison", "",
              "| router | combined score |", "|---|---|",
              f"| domain router (per-domain best) | {domain_router:.4f} |",
              f"| best uniform: {best_uniform_s} everywhere | {uniform[best_uniform_s]:.4f} |",
              f"| combined oracle (ceiling) | {combined_oracle:.4f} |", "",
              f"**Domain routing beats best uniform by +{gain:.4f} (+{rel:.1f}%).** "
              f"The property-based router (H3) must beat the domain router; the "
              f"remaining gap to the oracle (+{combined_oracle - domain_router:.4f}) "
              f"is what document properties must explain beyond domain membership.", ""]

    report = "\n".join(lines)
    print(report)

    out = Path(args.results_dir) / (
        f"domain-router-{'-'.join(args.datasets)}-{args.metric}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"report saved: {out}")


if __name__ == "__main__":
    main()
