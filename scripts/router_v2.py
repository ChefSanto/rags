#!/usr/bin/env python
"""Router v2a (H3): routing on query-embedding features.

v1 (lexical-statistical query features) routed slightly above baselines but
recovered only ~5% of the oracle gap: it predicted strategy LEVELS well and
per-query deviations not at all. v2a tests whether semantic query features
carry what lexical stats miss:

  - embed every query with the same model the dense baseline uses
    (read from the result configs, e.g. all-MiniLM-L6-v2)
  - PCA (fit on TRAIN only) to --components dims
  - per-strategy GradientBoostingRegressor on those components (v1 unchanged)
  - optional --with-lexical concatenates the six v1 features

Embeddings are cached under artifacts/query-embeddings/ per dataset+model.

Example:
    python scripts/router_v2.py --datasets fiqa nfcorpus --metric ndcg_at_10
    python scripts/router_v2.py --datasets fiqa nfcorpus --metric ndcg_at_10 --with-lexical
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.evaluation.retrieval import parse_metric  # noqa: E402
from core.experiments.registry import load_results  # noqa: E402
import router_v0 as rv  # noqa: E402

PER_QUERY = rv.PER_QUERY


def row_qids(data: dict, split_qrels: dict[str, dict[str, int]]) -> list[str]:
    """Queries of a split usable for routing (judged, scored, has text)."""
    strategies = data["strategies"]
    out = []
    for qid, rels in sorted(split_qrels.items()):
        if not any(r >= 1 for r in rels.values()):
            continue
        if not all(qid in data["scores"][s] for s in strategies):
            continue
        if qid not in data["texts"]:
            continue
        out.append(qid)
    return out


def embed_queries(texts: dict[str, str], model_name: str) -> dict[str, list[float]]:
    """qid -> normalized embedding; cached under artifacts/query-embeddings/."""
    slug = model_name.replace("/", "-")
    cache_dir = ROOT / "artifacts" / "query-embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{slug}.json"
    stored = {}
    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))

    missing = {q: t for q, t in texts.items() if q not in stored}
    if missing:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        qids = list(missing)
        vecs = model.encode([missing[q] for q in qids],
                            normalize_embeddings=True, show_progress_bar=False)
        for q, v in zip(qids, vecs):
            stored[q] = [float(x) for x in v]
        cache.write_text(json.dumps(stored), encoding="utf-8")
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--metric", default="ndcg_at_10")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--with-lexical", action="store_true",
                        help="concatenate the six lexical v1 features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default=str(ROOT / "experiments" / "results"))
    args = parser.parse_args()

    base, k = parse_metric(args.metric)
    if base not in PER_QUERY:
        raise SystemExit(f"supports ndcg_at_K / recall_at_K, got {args.metric}")

    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingRegressor

    results = load_results(args.results_dir)
    per_dataset = {name: rv.collect_dataset(name, base, k, args.results_dir)
                   for name in args.datasets}
    strategies = per_dataset[args.datasets[0]]["strategies"]
    for name in args.datasets:
        if per_dataset[name]["strategies"] != strategies:
            raise SystemExit(f"strategy sets differ across datasets: {name}")

    def dataset_dir_of(name):
        return next(r for r in results
                    if r.get("dataset") == name)["config"]["dataset"]["path"]

    model_name = next(r for r in results
                      if r.get("dataset") in args.datasets
                      and r.get("models", {}).get("embedding")
                      and r["models"]["embedding"] not in (None, "hashing")
                      )["models"]["embedding"]

    for name in args.datasets:
        embed_queries(per_dataset[name]["texts"], model_name)

    cache = json.loads((ROOT / "artifacts" / "query-embeddings" /
                        f"{model_name.replace('/', '-')}.json").read_text())

    def features(data, qid):
        emb = cache[qid]
        if args.with_lexical:
            return emb + rv.query_features(data["texts"][qid], data["df"], data["n_docs"])
        return list(emb)

    X_train, Y_train, q_train = [], [], []
    X_test, Y_test, q_test, test_domains = [], [], [], {}
    for name in args.datasets:
        data = per_dataset[name]
        for qid in row_qids(data, rv.load_split_qrels(dataset_dir_of(name), args.train_split)):
            X_train.append(features(data, qid))
            Y_train.append([data["scores"][s][qid] for s in strategies])
            q_train.append(f"{name}:{qid}")
        for qid in row_qids(data, rv.load_split_qrels(dataset_dir_of(name), args.test_split)):
            X_test.append(features(data, qid))
            Y_test.append([data["scores"][s][qid] for s in strategies])
            q_test.append(qid)
            test_domains[qid] = name

    if not X_train or not X_test:
        raise SystemExit(f"empty train ({len(X_train)}) or test ({len(X_test)}) split")

    pca = PCA(n_components=min(args.components, len(X_train)), random_state=args.seed)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

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

    routed = sum(score_of(q, strategies[i]) for q, i in zip(q_test, pred_idx)) / len(q_test)
    uniform = {s: sum(score_of(q, s) for q in q_test) / len(q_test) for s in strategies}
    oracle = sum(max(per_dataset[test_domains[q]]["scores"][s][q] for s in strategies)
                 for q in q_test) / len(q_test)

    train_means: dict[str, dict[str, float]] = {name: {} for name in args.datasets}
    for name in args.datasets:
        for i, s in enumerate(strategies):
            vals = [y[i] for x, y in zip(q_train, Y_train) if x.startswith(name + ":")]
            if vals:
                train_means[name][s] = sum(vals) / len(vals)
    dom_route = {name: max(train_means[name], key=train_means[name].get)
                 for name in args.datasets if train_means[name]}
    domain_router = sum(score_of(q, dom_route[test_domains[q]]) for q in q_test
                        if test_domains[q] in dom_route) / len(q_test)

    best_uniform_s = max(uniform, key=uniform.get)
    recovery = ((routed - uniform[best_uniform_s]) / (oracle - uniform[best_uniform_s])
                if oracle > uniform[best_uniform_s] else 0.0)

    pred_means = preds.mean(axis=0)
    actual_means = np.array(Y_test).mean(axis=0)
    train_means_all = [sum(y[i] for y in Y_train) / len(Y_train) for i in range(len(strategies))]

    tag = "emb+lex" if args.with_lexical else "emb"
    lines = [f"# Router v2a ({tag}, PCA-{args.components}) — {', '.join(args.datasets)} ({args.metric})",
             "", f"- train split: {args.train_split} ({len(X_train)} queries)",
             f"- test split: {args.test_split} ({len(q_test)} queries)",
             f"- strategies: {', '.join(strategies)}",
             f"- embedding model: {model_name} (same as dense baseline)",
             "- model: PCA + GradientBoostingRegressor per strategy", "",
             "## Test-split scores", "",
             "| router | score |", "|---|---|",
             f"| **routed (v2a)** | **{routed:.4f}** |"]
    lines += [f"| uniform {s} | {v:.4f} |" for s, v in sorted(uniform.items(), key=lambda kv: -kv[1])]
    lines += [f"| domain router {dom_route} | {domain_router:.4f} |",
              f"| oracle (ceiling) | {oracle:.4f} |", "",
              f"- routing gain over best uniform ({best_uniform_s}): "
              f"**{routed - uniform[best_uniform_s]:+.4f}**",
              f"- vs domain router: {routed - domain_router:+.4f}",
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
    lines += ["", f"- PCA explained variance (first 5): "
              f"{', '.join(f'{v:.3f}' for v in pca.explained_variance_ratio_[:5])}", ""]
    importances = [m.feature_importances_ for m in models]
    mean_imp = [sum(col) / len(col) for col in zip(*importances)]
    top = sorted(zip(range(len(mean_imp)), mean_imp), key=lambda kv: -kv[1])[:10]
    lines += ["## Top feature importances (PCA components)", "",
              "| component | importance |", "|---|---|"]
    lines += [f"| PC{c+1} | {imp:.3f} |" for c, imp in top]
    lines.append("")

    report = "\n".join(lines)
    print(report)

    out = Path(args.results_dir) / (
        f"router-v2a-{tag}-{'-'.join(args.datasets)}-{args.metric}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"report saved: {out}")


if __name__ == "__main__":
    main()
