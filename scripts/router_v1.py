#!/usr/bin/env python
"""Router v1 (H3): per-strategy score regression routing.

v0 (query features -> argmax classification) failed with a collapsed tree
(see experiments/results/router-v0-*.md): hard labels on tie-biased train
splits, distribution shift, and margin blindness. v1 replaces labels with
regression targets:

  - train one regressor per strategy to predict that strategy's per-query
    metric from the same query features
  - route each test query to the strategy with the highest PREDICTED score

This removes hard labels and ties entirely, lets margins shape the decision,
and keeps the identical train/test protocol as v0 for a controlled comparison.

Example:
    python scripts/router_v1.py --datasets fiqa nfcorpus --metric ndcg_at_10
"""

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.evaluation.retrieval import parse_metric  # noqa: E402
from core.experiments.registry import load_results  # noqa: E402
import router_v0 as rv  # noqa: E402  (shared data pipeline)

PER_QUERY = rv.PER_QUERY


def assemble_regression(data: dict, split_qrels: dict[str, dict[str, int]]):
    """Feature matrix + per-strategy score targets for one split."""
    strategies = data["strategies"]
    X, Y, qs = [], [], []
    for qid, rels in sorted(split_qrels.items()):
        if not any(r >= 1 for r in rels.values()):
            continue
        if not all(qid in data["scores"][s] for s in strategies):
            continue
        if qid not in data["texts"]:
            continue
        X.append(rv.query_features(data["texts"][qid], data["df"], data["n_docs"]))
        Y.append([data["scores"][s][qid] for s in strategies])
        qs.append(qid)
    return X, Y, qs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="datasets for BOTH train and test (or use --train-datasets/--test-datasets)")
    parser.add_argument("--train-datasets", nargs="+", default=None)
    parser.add_argument("--test-datasets", nargs="+", default=None)
    parser.add_argument("--metric", default="ndcg_at_10")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default=str(ROOT / "experiments" / "results"))
    args = parser.parse_args()

    train_names = args.train_datasets or args.datasets
    test_names = args.test_datasets or args.datasets
    if not train_names or not test_names:
        raise SystemExit("provide --datasets, or both --train-datasets and --test-datasets")
    all_names = list(dict.fromkeys(train_names + test_names))

    base, k = parse_metric(args.metric)
    if base not in PER_QUERY:
        raise SystemExit(f"supports ndcg_at_K / recall_at_K, got {args.metric}")

    from sklearn.ensemble import GradientBoostingRegressor

    per_dataset = {name: rv.collect_dataset(name, base, k, args.results_dir)
                   for name in all_names}
    strategies = per_dataset[args.datasets[0]]["strategies"]
    for name in args.datasets:
        if per_dataset[name]["strategies"] != strategies:
            raise SystemExit(f"strategy sets differ across datasets: {name} has "
                             f"{per_dataset[name]['strategies']} vs {strategies}")

    def dataset_dir_of(name):
        return next(r for r in load_results(args.results_dir)
                    if r.get("dataset") == name)["config"]["dataset"]["path"]

    X_train, Y_train, q_train = [], [], []
    X_test, Y_test, q_test = [], [], []
    test_domains: dict[str, str] = {}
    for name in train_names:
        data = per_dataset[name]
        X, Y, q = assemble_regression(data, rv.load_split_qrels(dataset_dir_of(name), args.train_split))
        X_train += X; Y_train += Y; q_train += [f"{name}:{x}" for x in q]
    for name in test_names:
        data = per_dataset[name]
        X, Y, q = assemble_regression(data, rv.load_split_qrels(dataset_dir_of(name), args.test_split))
        X_test += X; Y_test += Y; q_test += q
        for x in q:
            test_domains[x] = name

    if not X_train or not X_test:
        raise SystemExit(f"empty train ({len(X_train)}) or test ({len(X_test)}) split")

    # one regressor per strategy; target = that strategy's per-query score
    models = []
    for i, s in enumerate(strategies):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=args.seed)
        m.fit(X_train, [y[i] for y in Y_train])
        models.append(m)

    import numpy as np
    preds = np.array([[m.predict([x])[0] for m in models] for x in X_test])
    pred_idx = preds.argmax(axis=1)

    def score_of(qid, strategy):
        return per_dataset[test_domains[qid]]["scores"][strategy][qid]

    Y_test_arr = np.array(Y_test)
    routed = sum(score_of(q, strategies[i]) for q, i in zip(q_test, pred_idx)) / len(q_test)
    uniform = {s: sum(score_of(q, s) for q in q_test) / len(q_test) for s in strategies}
    oracle = sum(max(per_dataset[test_domains[q]]["scores"][s][q] for s in strategies)
                 for q in q_test) / len(q_test)

    # domain router: per-domain best by TRAIN means, applied to test
    train_means: dict[str, dict[str, float]] = {name: {} for name in train_names}
    for name in args.datasets:
        for i, s in enumerate(strategies):
            vals = [y[i] for x, y in zip(q_train, Y_train)
                    if x.startswith(name + ":")]
            if vals:
                train_means[name][s] = sum(vals) / len(vals)
    dom_route = {name: max(train_means[name], key=train_means[name].get)
                 for name in train_names if train_means[name]}
    if all(test_domains[q] in dom_route for q in q_test):
        domain_router = sum(score_of(q, dom_route[test_domains[q]]) for q in q_test) / len(q_test)
        domain_router_s = f"{domain_router:.4f}"
        vs_domain = f"{routed - domain_router:+.4f}"
    else:
        domain_router_s = "n/a (test domains unseen in train)"
        vs_domain = "n/a"

    best_uniform_s = max(uniform, key=uniform.get)
    recovery = ((routed - uniform[best_uniform_s]) / (oracle - uniform[best_uniform_s])
                if oracle > uniform[best_uniform_s] else 0.0)

    # diagnostics: predicted vs actual mean per strategy; train/test shift table
    pred_means = preds.mean(axis=0)
    actual_means = Y_test_arr.mean(axis=0)
    train_means_all = [sum(y[i] for y in Y_train) / len(Y_train) for i in range(len(strategies))]

    lines = [f"# Router v1 (score regression) — {', '.join(args.datasets)} ({args.metric})",
             "", f"- train split: {args.train_split} ({len(X_train)} queries)",
             f"- test split: {args.test_split} ({len(q_test)} queries)",
             f"- strategies: {', '.join(strategies)}",
             "- model: GradientBoostingRegressor per strategy "
               "(n=200, depth=3, lr=0.05, subsample=0.8)", "",
             "## Test-split scores", "",
             "| router | score |", "|---|---|",
             f"| **routed (v1 regression)** | **{routed:.4f}** |"]
    lines += [f"| uniform {s} | {v:.4f} |" for s, v in sorted(uniform.items(), key=lambda kv: -kv[1])]
    lines += [f"| domain router {dom_route} | {domain_router_s} |",
              f"| oracle (ceiling) | {oracle:.4f} |", "",
              f"- routing gain over best uniform ({best_uniform_s}): "
              f"**{routed - uniform[best_uniform_s]:+.4f}**",
              f"- vs domain router: {vs_domain}",
              f"- oracle-gap recovery vs best uniform: {recovery:.1%}",
              f"- gap remaining to oracle: {oracle - routed:+.4f}", "",
              "## Prediction diagnostics (test)", "",
              "| strategy | predicted mean | actual mean | train-split mean |",
              "|---|---|---|---|"]
    for i, s in enumerate(strategies):
        lines.append(f"| {s} | {pred_means[i]:.4f} | {actual_means[i]:.4f} "
                     f"| {train_means_all[i]:.4f} |")
    lines += ["", "## Predicted routing distribution (test)", "",
              "| strategy | share |", "|---|---|"]
    pred_dist = Counter(strategies[i] for i in pred_idx)
    lines += [f"| {s} | {pred_dist.get(s, 0)/len(q_test):.1%} |" for s in strategies]
    lines += ["", "## Mean feature importances (across strategy regressors)", "",
              "| feature | importance |", "|---|---|"]
    importances = [m.feature_importances_ for m in models]
    mean_imp = [sum(col) / len(col) for col in zip(*importances)]
    lines += [f"| {n} | {imp:.3f} |" for n, imp in
              sorted(zip(rv.FEATURE_NAMES, mean_imp), key=lambda kv: -kv[1])]
    lines.append("")

    report = "\n".join(lines)
    print(report)

    out = Path(args.results_dir) / (
        f"router-v1-{'-'.join(args.datasets)}-{args.metric}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"report saved: {out}")


if __name__ == "__main__":
    main()
