#!/usr/bin/env python
"""When does routing help? Characterizing the oracle's winning queries.

Converts the flat negative result ("routing doesn't beat uniform on average")
into an understanding: on which queries does the oracle actually win, by how
much, and what would a router's accuracy need to be to profit?

For each test query: per-strategy nDCG (from committed run artifacts),
oracle = per-query max, uniform = best single-strategy mean. "Win queries"
are those where oracle > uniform. We then compare query properties
(length, term rarity, coverage, strategy margin) between win and non-win
queries, and compute the break-even pick accuracy:

    p* = loss / (gain + loss)
    gain = mean(oracle - uniform) on win queries
    loss = mean(uniform - mean(other strategies)) on non-win queries

Output: markdown report to experiments/results/.

    python scripts/routing_win_analysis.py --datasets fiqa nfcorpus
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.evaluation.retrieval import per_query_ndcg_at_k  # noqa: E402
from core.experiments.registry import load_results  # noqa: E402
import router_v0 as rv  # noqa: E402

STRATEGIES = ("bm25", "dense", "hybrid")
FEATS = ["n_terms", "n_unique", "coverage", "avg_idf", "max_idf", "stopword_hits"]



def _artifact_path(path_str):
    """Run artifacts may carry absolute paths from the machine that produced
    them (e.g. /kaggle/working/...); fall back to the local committed file."""
    p = Path(path_str)
    return p if p.exists() else ROOT / "artifacts" / "runs" / p.name


def _local_run(path_str):
    return json.loads(_artifact_path(path_str).read_text())

def load_runs3(name, results_dir):
    results = [r for r in load_results(results_dir)
               if r.get("dataset") == name and r.get("run_artifact")]
    by = {}
    for r in sorted(results, key=lambda r: r["experiment_id"]):
        if r["retriever"] in STRATEGIES and _artifact_path(r["run_artifact"]).exists():
            by[r["retriever"]] = r
    if set(by) != set(STRATEGIES):
        raise SystemExit(f"{name}: need bm25+dense+hybrid run artifacts")
    return {s: _local_run(r["run_artifact"]) for s, r in by.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--results-dir", default=str(ROOT / "experiments" / "results"))
    args = ap.parse_args()

    results = load_results(args.results_dir)
    sections, summary = [], []

    for name in args.datasets:
        runs = load_runs3(name, args.results_dir)
        ds_dir = next(r for r in results if r.get("dataset") == name)["config"]["dataset"]["path"]
        from core.datasets.beir import load_beir
        ds = load_beir(ds_dir, name=name, version="1.0")
        qrels, texts = ds.qrels, {q.id: q.text for q in ds.queries}
        n_docs, df = rv.corpus_stats(ds_dir, name)
        test_q = [q for q in rv.load_split_qrels(ds_dir, args.test_split)
                  if all(q in runs[s] for s in STRATEGIES) and q in qrels and q in texts]

        per_strat = {s: per_query_ndcg_at_k(runs[s], qrels, 10) for s in STRATEGIES}
        uniform = {s: sum(per_strat[s].get(q, 0.0) for q in test_q) / len(test_q) for s in STRATEGIES}
        best_u = max(uniform, key=uniform.get)   # strategy name
        best_u_s = uniform[best_u]               # its mean score

        wins, losses, margins = [], [], []
        X = {q: rv.query_features(texts[q], df, n_docs) for q in test_q}
        winner_dist = {s: 0 for s in STRATEGIES}
        for q in test_q:
            scores = {s: per_strat[s].get(q, 0.0) for s in STRATEGIES}
            orc = max(scores.values())
            ranked = sorted(scores.values(), reverse=True)
            margins.append(ranked[0] - ranked[1])
            winner = max(scores, key=scores.get)
            if orc > best_u_s + 1e-9:
                wins.append((q, orc - best_u_s))
                winner_dist[winner] += 1
            else:
                others = [v for s, v in scores.items() if s != winner]
                losses.append(best_u_s - sum(others) / len(others))

        gain = sum(g for _, g in wins) / max(len(wins), 1)
        loss = sum(losses) / max(len(losses), 1)
        p_star = loss / (gain + loss) if wins else 1.0

        def mean_feat(qs, i):
            return sum(X[q][i] for q in qs) / max(len(qs), 1)

        win_q = [q for q, _ in wins]
        win_set = set(win_q)
        rest_q = [q for q in test_q if q not in win_set]
        feat_rows = "\n".join(
            f"| {FEATS[i]} | {mean_feat(win_q, i):.3f} | {mean_feat(rest_q, i):.3f} |"
            for i in range(len(FEATS)))
        margin_win = sum(m for m, q in zip(margins, test_q) if q in win_set) / max(len(wins), 1)

        sections.append(
            f"### {name}\n\n"
            f"- test queries: {len(test_q)}; best uniform: {best_u} @ {best_u_s:.4f}\n"
            f"- **oracle wins on {len(wins)} queries ({len(wins)/len(test_q):.1%})**, "
            f"mean gain on those: +{gain:.4f} nDCG\n"
            f"- oracle-best winner distribution on win queries: " +
            ", ".join(f"{s} {winner_dist[s]}" for s in STRATEGIES) + "\n"
            f"- mean strategy margin (top1-top2) on win queries: {margin_win:.3f} "
            f"vs all: {sum(margins)/len(margins):.3f}\n\n"
            f"| query property | win queries | others |\n|---|---|---|\n{feat_rows}\n\n"
            f"- break-even pick accuracy: **p* = {p_star:.1%}** — a router must correctly "
            f"identify winning queries (and pick their best strategy) {p_star:.0%} of the time, "
            f"given the losses on the rest, to merely tie the uniform baseline.\n"
        )
        summary.append((name, len(test_q), len(wins), gain, loss, p_star))

    total = sum(n for _, n, *_ in summary)
    lines = [f"# When does routing help? — {', '.join(args.datasets)}", "",
             "| domain | test q | oracle wins | mean gain (win) | mean loss (rest) | break-even p* |",
             "|---|---|---|---|---|---|"]
    for n_, q, w, g, l, p in summary:
        lines.append(f"| {n_} | {q} | {w} ({w/q:.1%}) | +{g:.4f} | -{l:.4f} | {p:.1%} |")
    lines += ["", "## Per-domain detail", ""] + sections
    report = "\n".join(lines)
    print(report)
    out = Path(args.results_dir) / (
        f"routing-wins-{'-'.join(args.datasets)}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"report saved: {out}")


if __name__ == "__main__":
    main()
