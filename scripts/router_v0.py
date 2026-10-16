#!/usr/bin/env python
"""Router v0 (H3): can the oracle's per-query strategy choices be learned?

Trains an interpretable decision tree mapping query features to the
oracle-best retrieval strategy. Trained on one qrels split, evaluated on
another. Reads existing run artifacts — no retrieval re-runs.

Features (query text + corpus statistics only; no relevance labels):
    n_terms, n_unique, coverage (fraction of query terms present in corpus),
    avg_idf, max_idf, stopword_hits (query terms in >50% of documents)

Label: argmax over strategies of the per-query metric (ties -> alphabetical).

Evaluation on the test split compares:
    - routed score: per-query score of the predicted strategy
    - each uniform strategy
    - domain router (per-domain best, computed on TRAIN — honest)
    - oracle (ceiling)

Example:
    python scripts/router_v0.py --datasets fiqa nfcorpus \
        --train-split train --test-split test --metric ndcg_at_10
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.evaluation.retrieval import (  # noqa: E402
    per_query_ndcg_at_k,
    per_query_recall_at_k,
    parse_metric,
)
from core.experiments.registry import load_results  # noqa: E402
from core.retrieval.bm25 import tokenize  # noqa: E402

PER_QUERY = {"ndcg": per_query_ndcg_at_k, "recall": per_query_recall_at_k}
FEATURE_NAMES = ["n_terms", "n_unique", "coverage", "avg_idf", "max_idf", "stopword_hits"]


def corpus_stats(dataset_dir: str, dataset_name: str) -> tuple[int, Counter]:
    """Document frequencies over the corpus; cached under artifacts/corpus-stats/."""
    cache_dir = ROOT / "artifacts" / "corpus-stats"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{dataset_name}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return data["n_docs"], Counter(data["df"])

    df: Counter = Counter()
    n_docs = 0
    with open(Path(dataset_dir) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            df.update(set(tokenize(doc["text"])))
            n_docs += 1
    cache.write_text(json.dumps({"n_docs": n_docs, "df": dict(df)}), encoding="utf-8")
    return n_docs, df


def query_features(text: str, df: Counter, n_docs: int) -> list[float]:
    toks = tokenize(text)
    uniq = set(toks)
    freqs = [df.get(t, 0) for t in uniq]

    def idf(f: int) -> float:
        return math.log((n_docs - f + 0.5) / (f + 0.5) + 1.0) if f else 0.0

    idfs = [idf(f) for f in freqs]
    return [
        float(len(toks)),
        float(len(uniq)),
        sum(1 for f in freqs if f > 0) / max(len(uniq), 1),
        sum(idfs) / max(len(idfs), 1),
        max(idfs) if idfs else 0.0,
        float(sum(1 for f in freqs if f > 0.5 * n_docs)),
    ]


def load_split_qrels(dataset_dir: str, split: str) -> dict[str, dict[str, int]]:
    p = Path(dataset_dir) / f"qrels-{split}.tsv"
    if not p.exists():
        raise SystemExit(
            f"{p} not found. Run scripts/build_qrels.py on this dataset first "
            f"(it writes qrels-<split>.tsv alongside the merged qrels.tsv)."
        )
    qrels: dict[str, dict[str, int]] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(r["query-id"], {})[r["doc-id"]] = int(r["relevance"])
    return qrels


def collect_dataset(name: str, metric_base: str, k: int, results_dir: str) -> dict:
    """Per-dataset: strategy artifacts, per-query scores, query texts, features."""
    results = [r for r in load_results(results_dir)
               if r.get("dataset") == name and r.get("run_artifact")]
    by_strategy: dict[str, dict] = {}
    for r in sorted(results, key=lambda r: r["experiment_id"]):
        by_strategy[r["retriever"]] = r
    if len(by_strategy) < 2:
        raise SystemExit(f"need >= 2 strategies with run artifacts for '{name}'")

    cfg = next(iter(by_strategy.values()))["config"]
    dataset_dir = cfg["dataset"]["path"]

    from core.datasets.beir import load_beir
    dataset = load_beir(dataset_dir, name=name, version=cfg["dataset"]["version"])

    per_strategy: dict[str, dict[str, float]] = {}
    for strategy, r in sorted(by_strategy.items()):
        run = json.loads(Path(r["run_artifact"]).read_text(encoding="utf-8"))
        per_strategy[strategy] = PER_QUERY[metric_base](run, dataset.qrels, k)

    n_docs, df = corpus_stats(dataset_dir, name)
    texts = {q.id: q.text for q in dataset.queries}
    return {"strategies": sorted(per_strategy), "scores": per_strategy,
            "texts": texts, "n_docs": n_docs, "df": df}


def assemble(data: dict, split_qrels: dict[str, dict[str, int]]):
    """Feature matrix + oracle labels for one split. Rows: judged queries with
    at least one relevant doc and scores available for every strategy."""
    strategies = data["strategies"]
    rows_X, rows_y, rows_q = [], [], []
    for qid, rels in sorted(split_qrels.items()):
        if not any(r >= 1 for r in rels.values()):
            continue
        if not all(qid in data["scores"][s] for s in strategies):
            continue
        if qid not in data["texts"]:
            continue
        scores = [data["scores"][s][qid] for s in strategies]
        rows_X.append(query_features(data["texts"][qid], data["df"], data["n_docs"]))
        rows_y.append(strategies[max(range(len(scores)), key=lambda i: scores[i])])
        rows_q.append(qid)
    return rows_X, rows_y, rows_q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--metric", default="ndcg_at_10")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default=str(ROOT / "experiments" / "results"))
    args = parser.parse_args()

    base, k = parse_metric(args.metric)
    if base not in PER_QUERY:
        raise SystemExit(f"supports ndcg_at_K / recall_at_K, got {args.metric}")

    from sklearn.tree import DecisionTreeClassifier, export_text

    per_dataset = {name: collect_dataset(name, base, k, args.results_dir)
                   for name in args.datasets}
    strategies = per_dataset[args.datasets[0]]["strategies"]
    for name in args.datasets:
        if per_dataset[name]["strategies"] != strategies:
            raise SystemExit(f"strategy sets differ across datasets: {name} has "
                             f"{per_dataset[name]['strategies']} vs {strategies}")

    def dataset_dir_of(name):
        return next(r for r in load_results(args.results_dir)
                    if r.get("dataset") == name)["config"]["dataset"]["path"]

    X_train, y_train, q_train = [], [], []
    X_test, y_test, q_test = [], [], []
    test_domains: dict[str, str] = {}
    for name in args.datasets:
        data = per_dataset[name]
        X, y, q = assemble(data, load_split_qrels(dataset_dir_of(name), args.train_split))
        X_train += X; y_train += y; q_train += [f"{name}:{x}" for x in q]
        X, y, q = assemble(data, load_split_qrels(dataset_dir_of(name), args.test_split))
        X_test += X; y_test += y; q_test += q
        for x in q:
            test_domains[x] = name

    if not X_train or not X_test:
        raise SystemExit(f"empty train ({len(X_train)}) or test ({len(X_test)}) split")

    clf = DecisionTreeClassifier(max_depth=args.max_depth,
                                 min_samples_leaf=args.min_samples_leaf,
                                 random_state=args.seed)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    def score_of(qid, strategy):
        return per_dataset[test_domains[qid]]["scores"][strategy][qid]

    routed = sum(score_of(q, p) for q, p in zip(q_test, pred)) / len(q_test)
    uniform = {s: sum(score_of(q, s) for q in q_test) / len(q_test) for s in strategies}
    oracle = sum(max(per_dataset[test_domains[q]]["scores"][s][q] for s in strategies)
                 for q in q_test) / len(q_test)

    # domain router: per-domain best by TRAIN means, applied to test
    train_means: dict[str, dict[str, float]] = {name: {} for name in args.datasets}
    for name in args.datasets:
        for s in strategies:
            vals = [per_dataset[name]["scores"][s][x.split(":", 1)[1]]
                    for x in q_train if x.startswith(name + ":")
                    and x.split(":", 1)[1] in per_dataset[name]["scores"][s]]
            if vals:
                train_means[name][s] = sum(vals) / len(vals)
    dom_route = {name: max(train_means[name], key=train_means[name].get)
                 for name in args.datasets if train_means[name]}
    domain_router = sum(score_of(q, dom_route[test_domains[q]]) for q in q_test
                        if test_domains[q] in dom_route) / len(q_test)

    acc = sum(1 for a, b in zip(pred, y_test) if a == b) / len(y_test)
    pred_dist = Counter(pred)
    best_uniform_s = max(uniform, key=uniform.get)

    lines = [f"# Router v0 — {', '.join(args.datasets)} ({args.metric})",
             "", f"- train split: {args.train_split} ({len(X_train)} queries)",
             f"- test split: {args.test_split} ({len(X_test)} queries)",
             f"- strategies: {', '.join(strategies)}",
             f"- model: decision tree (max_depth={args.max_depth}, "
             f"min_samples_leaf={args.min_samples_leaf})", "",
             "## Test-split scores", "",
             "| router | score |", "|---|---|",
             f"| **routed (decision tree)** | **{routed:.4f}** |"]
    lines += [f"| uniform {s} | {v:.4f} |" for s, v in sorted(uniform.items(), key=lambda kv: -kv[1])]
    lines += [f"| domain router {dom_route} | {domain_router:.4f} |",
              f"| oracle (ceiling) | {oracle:.4f} |", "",
              f"- routing gain over best uniform ({best_uniform_s}): "
              f"**{routed - uniform[best_uniform_s]:+.4f}**",
              f"- gap remaining to oracle: {oracle - routed:+.4f}",
              f"- label accuracy (predicted == oracle-best): {acc:.1%}", "",
              "## Predicted strategy distribution (test)", "",
              "| strategy | share |", "|---|---|"]
    lines += [f"| {s} | {pred_dist.get(s, 0)/len(q_test):.1%} |" for s in strategies]
    lines += ["", "## Feature importances", "",
              "| feature | importance |", "|---|---|"]
    lines += [f"| {n} | {imp:.3f} |" for n, imp in
              sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda kv: -kv[1])]
    lines += ["", "## Decision tree", "", "```",
              export_text(clf, feature_names=FEATURE_NAMES).rstrip(), "```", ""]

    report = "\n".join(lines)
    print(report)

    out = Path(args.results_dir) / (
        f"router-v0-{'-'.join(args.datasets)}-{args.metric}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"report saved: {out}")


if __name__ == "__main__":
    main()
