#!/usr/bin/env python
"""Router v5: SOFT routing — per-query fusion weights (the missing experiment).

Context: RCT-MARS (J. Intell. Inf. Syst., 2026) showed hard per-query routing
fails ("routing paradox") while SOFT routing — predicting per-query fusion
weights — recovers part of the oracle gap. Our v1-v4 matrix tested only HARD
selection. v5 closes the loop on our FiQA/NFCorpus stack:

  oracle-soft : per-query dense_weight w* in a 0..1 grid maximizing nDCG@10 of
                the RRF-fused bm25+dense ranking (soft paradigm's ceiling)
  learned-soft: regress w* from query features (train split) -> predict on test
  heuristic   : w = pred_dense_nDCG / (pred_dense + pred_bm25) from v1-style
                per-strategy score regressors
  reference   : best UNIFORM weight on test (the sweep result), hard-oracle
                (from earlier reports), hard-routed v1 (from earlier reports)

Only the ranked top-10 lists (committed run artifacts) are needed — RRF fusion
works from orderings alone.

Example:
    python scripts/router_v5_soft.py --datasets fiqa nfcorpus --metric ndcg_at_10
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.evaluation.retrieval import per_query_ndcg_at_k, parse_metric  # noqa: E402
from core.experiments.registry import load_results  # noqa: E402
import router_v0 as rv  # noqa: E402
import router_v2 as r2  # noqa: E402

RRF_K = 60
GRID = [i / 10 for i in range(11)]



def _local_run(path_str):
    """Run artifacts may carry absolute paths from the machine that produced
    them (e.g. /kaggle/working/...); fall back to the local committed file."""
    p = Path(path_str)
    if not p.exists():
        p = ROOT / "artifacts" / "runs" / p.name
    return json.loads(p.read_text())

def load_runs(dataset_name: str, results_dir: str) -> dict:
    results = [r for r in load_results(results_dir)
               if r.get("dataset") == dataset_name and r.get("run_artifact")]
    by = {}
    for r in sorted(results, key=lambda r: r["experiment_id"]):
        by[r["retriever"]] = r
    if "bm25" not in by or "dense" not in by:
        raise SystemExit(f"{dataset_name}: need bm25+dense run artifacts")
    return {s: _local_run(r["run_artifact"]) for s, r in by.items() if s in ("bm25", "dense")}


def fuse(bm25_ids: list, dense_ids: list, w: float) -> list:
    """RRF-style weighted rank fusion. Deterministic ties (doc id asc)."""
    score: dict[str, float] = {}
    for rank, d in enumerate(bm25_ids):
        score[d] = score.get(d, 0.0) + (1.0 - w) / (rank + RRF_K)
    for rank, d in enumerate(dense_ids):
        score[d] = score.get(d, 0.0) + w / (rank + RRF_K)
    return [d for d, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))][:10]


def ndcg1(ranked: list, rels: dict) -> float:
    return per_query_ndcg_at_k({"q": ranked}, {"q": rels}, 10).get("q", 0.0)


def best_weight(bm25_ids: list, dense_ids: list, rels: dict) -> tuple[float, float]:
    best_w, best_s = 0.0, -1.0
    for w in GRID:
        s = ndcg1(fuse(bm25_ids, dense_ids, w), rels)
        if s > best_s + 1e-12:
            best_s, best_w = s, w
    return best_w, best_s


def features_for(qids, data, cache, args):
    if args.features == "emb":
        return [cache[q] for q in qids]
    return [rv.query_features(data["texts"][q], data["df"], data["n_docs"]) for q in qids]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--metric", default="ndcg_at_10")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--features", choices=["emb", "lexical"], default="emb")
    ap.add_argument("--components", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", default=str(ROOT / "experiments" / "results"))
    args = ap.parse_args()

    base, k = parse_metric(args.metric)
    if base != "ndcg":
        raise SystemExit("v5 uses ndcg_at_K (fusion ranking quality)")

    from sklearn.decomposition import PCA
    from sklearn.ensemble import GradientBoostingRegressor

    results = load_results(args.results_dir)

    def dir_of(name):
        return next(r for r in results if r.get("dataset") == name)["config"]["dataset"]["path"]

    sections, summary = [], []
    for name in args.datasets:
        runs = load_runs(name, args.results_dir)
        from core.datasets.beir import load_beir
        ds = load_beir(dir_of(name), name=name, version="1.0")
        qrels = ds.qrels

        model_name = next(r["config"]["models"]["embedding"] for r in results
                          if r.get("dataset") == name
                          and r.get("config", {}).get("models", {}).get("embedding")
                          and r["config"]["models"]["embedding"] not in (None, "hashing"))
        r2.embed_queries({q.id: q.text for q in ds.queries}, model_name)
        cache = json.loads((ROOT / "artifacts" / "query-embeddings" /
                            f"{model_name.replace('/', '-')}.json").read_text())
        data = {"texts": {q.id: q.text for q in ds.queries}, "df": None, "n_docs": 0}

        train_q = [q for q in rv.load_split_qrels(dir_of(name), args.train_split)
                   if q in runs["bm25"] and q in runs["dense"] and q in qrels and q in data["texts"]]
        test_q = [q for q in rv.load_split_qrels(dir_of(name), args.test_split)
                  if q in runs["bm25"] and q in runs["dense"] and q in qrels and q in data["texts"]]

        w_train = []
        for q in train_q:
            w_star, _ = best_weight(runs["bm25"][q], runs["dense"][q], qrels[q])
            w_train.append(w_star)
        oracle_scores = []
        for q in test_q:
            _, s = best_weight(runs["bm25"][q], runs["dense"][q], qrels[q])
            oracle_scores.append(s)
        oracle_soft = sum(oracle_scores) / len(oracle_scores)

        X_train = features_for(train_q, data, cache, args)
        X_test = features_for(test_q, data, cache, args)
        if args.features == "emb":
            pca = PCA(n_components=min(args.components, len(X_train)), random_state=args.seed)
            X_train, X_test = pca.fit_transform(X_train), pca.transform(X_test)
        reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, random_state=args.seed)
        reg.fit(X_train, w_train)
        w_pred = [min(1.0, max(0.0, w)) for w in reg.predict(X_test)]
        learned_scores = [ndcg1(fuse(runs["bm25"][q], runs["dense"][q], w), qrels[q])
                          for q, w in zip(test_q, w_pred)]
        learned_soft = sum(learned_scores) / len(learned_scores)

        per_strat = {s: per_query_ndcg_at_k(runs[s], qrels, k) for s in ("bm25", "dense")}
        Xtr, Ytr = [], []
        for q in train_q:
            Xtr.append(features_for([q], data, cache, args)[0])
            Ytr.append([per_strat["bm25"].get(q, 0.0), per_strat["dense"].get(q, 0.0)])
        if args.features == "emb":
            Xtr = pca.transform(Xtr)
        sregs = []
        for i in range(2):
            m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, random_state=args.seed)
            m.fit(Xtr, [y[i] for y in Ytr])
            sregs.append(m)
        Xte = features_for(test_q, data, cache, args)
        if args.features == "emb":
            Xte = pca.transform(Xte)
        heur_scores = []
        for q, x in zip(test_q, Xte):
            pd_ = max(0.0, sregs[1].predict([x])[0])
            pb_ = max(0.0, sregs[0].predict([x])[0])
            w = pd_ / (pd_ + pb_ + 1e-9)
            heur_scores.append(ndcg1(fuse(runs["bm25"][q], runs["dense"][q], w), qrels[q]))
        heuristic_soft = sum(heur_scores) / len(heur_scores)

        uniform = {}
        for w in GRID:
            uniform[w] = sum(ndcg1(fuse(runs["bm25"][q], runs["dense"][q], w), qrels[q])
                             for q in test_q) / len(test_q)
        best_w = max(uniform, key=uniform.get)

        sections.append(
            f"### {name}\n\n"
            f"- train queries: {len(train_q)}, test queries: {len(test_q)}\n\n"
            f"| router | test nDCG@10 |\n|---|---|\n"
            f"| uniform bm25 (w=0) | {uniform[0.0]:.4f} |\n"
            f"| uniform dense (w=1) | {uniform[1.0]:.4f} |\n"
            f"| **best uniform (w={best_w})** | **{uniform[best_w]:.4f}** |\n"
            f"| heuristic soft (predicted scores) | {heuristic_soft:.4f} |\n"
            f"| **learned soft (w* regression)** | **{learned_soft:.4f}** |\n"
            f"| oracle soft (per-query w*) | {oracle_soft:.4f} |\n\n"
            f"- uniform sweep: " + ", ".join(f"w={w}: {uniform[w]:.4f}" for w in GRID) + "\n"
            f"- predicted mean weight (test): {sum(w_pred)/len(w_pred):.3f}\n"
        )
        summary.append((name, len(test_q), uniform[best_w], learned_soft, heuristic_soft, oracle_soft))

    total = sum(n for _, n, *_ in summary)
    lines = [f"# Router v5 (soft routing) — {', '.join(args.datasets)} ({args.metric}, {args.features})",
             "", "## Weighted summary", "",
             "| domain | test q | best uniform | learned soft | heuristic soft | oracle soft |", "|---|---|---|---|---|---|"]
    for n_, q, u, l, h, o in summary:
        lines.append(f"| {n_} | {q} | {u:.4f} | {l:.4f} | {h:.4f} | {o:.4f} |")
    bu = sum(n * r[2] for r, n in [(r, r[1]) for r in summary]) / total
    ls = sum(n * r[3] for r, n in [(r, r[1]) for r in summary]) / total
    hs = sum(n * r[4] for r, n in [(r, r[1]) for r in summary]) / total
    os_ = sum(n * r[5] for r, n in [(r, r[1]) for r in summary]) / total
    lines += [f"| **weighted** | {total} | **{bu:.4f}** | **{ls:.4f}** | {hs:.4f} | {os_:.4f} |", "",
              "Hard-oracle ceiling reference (earlier reports): 0.4177.", "",
              "## Per-domain detail", ""] + sections
    report = "\n".join(lines)
    print(report)
    out = Path(args.results_dir) / (
        f"router-v5-soft-{'-'.join(args.datasets)}-{args.metric}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"report saved: {out}")


if __name__ == "__main__":
    main()
