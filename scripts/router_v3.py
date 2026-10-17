#!/usr/bin/env python
"""Router v3 (H3): per-corpus training on pseudo-labeled queries.

Why this exists — the established findings (experiments/results/):
  - in-sample recovery is high (v1 lexical 53.6%, v2a embedding 72.5%): the
    routing signal is real;
  - cross-split transfer is ~5% and cross-domain transfer is negative: the
    signal is LOCAL. Routers must be trained per-corpus on test-distributed
    data.

v3 provides exactly that: pseudo-queries generated from the corpus's own
documents (scripts/make_pseudo_queries.py), scored by all three strategies
through the standard runner, used to train per-corpus regressors, then
evaluated on the corpus's REAL test split.

Honest risk, reported with the results: pseudo-queries are document-derived,
so they are lexically easy and may over-favor BM25. A routed score below the
best uniform is a finding about pseudo-query distribution, not just routing.

Example:
    python scripts/router_v3.py --datasets fiqa nfcorpus --metric ndcg_at_10
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.evaluation.retrieval import parse_metric  # noqa: E402
from core.experiments.registry import load_results  # noqa: E402
import router_v0 as rv  # noqa: E402
import router_v2 as r2  # noqa: E402

PER_QUERY = rv.PER_QUERY


def load_qrels_tsv(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["doc-id"]] = int(row["relevance"])
    return qrels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--metric", default="ndcg_at_10")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-cap", type=int, default=15000)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default=str(ROOT / "experiments" / "results"))
    args = parser.parse_args()

    base, k = parse_metric(args.metric)
    if base not in PER_QUERY:
        raise SystemExit(f"supports ndcg_at_K / recall_at_K, got {args.metric}")

    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingRegressor

    results = load_results(args.results_dir)
    model_name = next(r["config"]["models"]["embedding"] for r in results
                      if r.get("dataset") in args.datasets
                      and r.get("config", {}).get("models", {}).get("embedding")
                      and r["config"]["models"]["embedding"] not in (None, "hashing"))

    def dataset_dir_of(name):
        return next(r for r in results
                    if r.get("dataset") == name)["config"]["dataset"]["path"]

    import numpy as np

    sections = []
    summary = []
    for name in args.datasets:
        real = rv.collect_dataset(name, base, k, args.results_dir)
        pseudo = rv.collect_dataset(f"pseudo-{name}", base, k, args.results_dir)
        strategies = real["strategies"]
        assert pseudo["strategies"] == strategies, f"strategy mismatch for {name}"

        r2.embed_queries(pseudo["texts"], model_name)
        r2.embed_queries(real["texts"], model_name)
        cache = json.loads((ROOT / "artifacts" / "query-embeddings" /
                            f"{model_name.replace('/', '-')}.json").read_text())

        # ---- train on pseudo-labeled queries ----
        pseudo_qrels = load_qrels_tsv(Path(dataset_dir_of(f"pseudo-{name}")) / "qrels.tsv")
        train_qids = r2.row_qids(pseudo, pseudo_qrels)[: args.train_cap]
        X_train = np.array([cache[q] for q in train_qids])
        Y_train = np.array([[pseudo["scores"][s][q] for s in strategies] for q in train_qids])

        # ---- test on the real test split ----
        test_qrels = rv.load_split_qrels(dataset_dir_of(name), args.test_split)
        test_qids = r2.row_qids(real, test_qrels)
        X_test = np.array([cache[q] for q in test_qids])
        Y_test = np.array([[real["scores"][s][q] for s in strategies] for q in test_qids])

        if len(X_train) == 0 or len(X_test) == 0:
            raise SystemExit(f"{name}: empty train ({len(X_train)}) or test ({len(X_test)})")

        pca = PCA(n_components=min(args.components, len(X_train)), random_state=args.seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        models = []
        for i in range(len(strategies)):
            m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                          learning_rate=0.05, subsample=0.8,
                                          random_state=args.seed)
            m.fit(X_train, Y_train[:, i])
            models.append(m)
        preds = np.array([[m.predict([x])[0] for m in models] for x in X_test])
        pred_idx = preds.argmax(axis=1)

        routed = float(np.mean([Y_test[j, i] for j, i in enumerate(pred_idx)]))
        uniform = {s: float(np.mean(Y_test[:, i])) for i, s in enumerate(strategies)}
        pseudo_means = {s: float(np.mean(Y_train[:, i])) for i, s in enumerate(strategies)}
        domain_router = max(pseudo_means.values())
        oracle = float(np.mean(Y_test.max(axis=1)))
        best_uniform_s = max(uniform, key=uniform.get)
        recovery = ((routed - uniform[best_uniform_s]) / (oracle - uniform[best_uniform_s])
                    if oracle > uniform[best_uniform_s] else 0.0)

        from collections import Counter
        pred_dist = Counter(strategies[i] for i in pred_idx)
        sections.append(
            f"### {name}\n\n"
            f"- train: {len(X_train)} pseudo-queries (from pseudo-{name})\n"
            f"- test: {len(X_test)} real {args.test_split}-split queries\n"
            f"- pseudo-label means: " +
            ", ".join(f"{s} {pseudo_means[s]:.4f}" for s in strategies) + "\n\n"
            f"| router | score |\n|---|---|\n"
            f"| **routed (v3, per-corpus pseudo-trained)** | **{routed:.4f}** |\n" +
            "".join(f"| uniform {s} | {v:.4f} |\n" for s, v in
                    sorted(uniform.items(), key=lambda kv: -kv[1])) +
            f"| pseudo-mean domain router | {domain_router:.4f} |\n"
            f"| oracle (ceiling) | {oracle:.4f} |\n\n"
            f"- gain over best uniform ({best_uniform_s}): **{routed - uniform[best_uniform_s]:+.4f}**\n"
            f"- oracle-gap recovery: {recovery:.1%}\n"
            f"- predicted distribution: " +
            ", ".join(f"{s} {pred_dist.get(s, 0)/len(test_qids):.1%}" for s in strategies) + "\n"
        )
        summary.append((name, len(X_test), routed, uniform[best_uniform_s], oracle, recovery))

    total = sum(n for _, n, *_ in summary)
    agg_routed = sum(n * r for _, n, r, *_ in summary) / total
    agg_uniform = sum(n * u for _, n, _, u, *_ in summary) / total
    agg_oracle = sum(n * o for _, n, *_, o, _ in summary) / total

    lines = [f"# Router v3 (per-corpus pseudo-labeled) — {', '.join(args.datasets)} ({args.metric})",
             "", f"- embedding: {model_name}, PCA-{args.components}",
             "- model: GradientBoostingRegressor per strategy, trained per-corpus",
             "", "## Summary", "",
             "| domain | test queries | routed | best uniform | oracle | recovery |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {q} | {r:.4f} | {u:.4f} | {o:.4f} | {rec:.1%} |"
              for n, q, r, u, o, rec in summary]
    lines += [f"| **weighted** | {total} | **{agg_routed:.4f}** | {agg_uniform:.4f} "
              f"| {agg_oracle:.4f} | "
              f"{(agg_routed - agg_uniform) / (agg_oracle - agg_uniform):.1%} |", ""]
    lines += ["## Per-domain detail", ""] + sections

    report = "\n".join(lines)
    print(report)

    out = Path(args.results_dir) / (
        f"router-v3-{'-'.join(args.datasets)}-{args.metric}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"report saved: {out}")


if __name__ == "__main__":
    main()
