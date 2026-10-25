#!/usr/bin/env python
"""Reviewer-requested statistics (local, minutes).

1. Unified baseline: every router number in the matrix recomputed against ONE
   canonical test-query set (intersection across strategies), removing the
   per-run subset wobble noted by reviewers.
2. Pick accuracy: v0-style decision tree (the paper's best classifier) rerun
   locally; bootstrap CI on per-query pick accuracy vs the oracle-best label.
3. Paired bootstrap test: P(pick_accuracy > p*) over joint query resamples --
   the formal version of "predictors operate at the break-even threshold".
4. Hedging quantification: per-query, soft-oracle (weight grid) vs hard-oracle
   (strategy argmax) vs the tuned constant weight -- how much of the hedging
   advantage does the constant already capture?

Run:  python3 -m pip install scikit-learn   (once)
      python3 scripts/reviewer_stats.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import router_v0 as rv  # noqa: E402
import router_v5_soft as v5  # noqa: E402
from core.evaluation.retrieval import per_query_ndcg_at_k  # noqa: E402
from core.datasets.beir import load_beir  # noqa: E402
from core.experiments.registry import load_results  # noqa: E402

DATASETS = ["fiqa", "nfcorpus"]
TUNED_W = {"fiqa": 0.7, "nfcorpus": 0.5}  # from the committed weight sweep
GRID = v5.GRID
B = 10_000
SEED = 42


def per_query_scores(runs, qrels, qids, k=10):
    return {s: per_query_ndcg_at_k(runs[s], qrels, k) for s in runs}


def main() -> None:
    rng = np.random.default_rng(SEED)
    results_dir = str(ROOT / "experiments" / "results")
    results = load_results(results_dir)

    all_print = []
    pooled = {}

    for name in DATASETS:
        runs = v5.load_runs(name, results_dir)  # bm25 + dense
        runs3 = dict(runs)
        hyb = [r for r in results if r.get("dataset") == name
               and r.get("run_artifact") and r.get("retriever") == "hybrid"
               and v5._artifact_path(r["run_artifact"]).exists()]
        if not hyb:
            raise SystemExit(f"{name}: no local hybrid artifact")
        runs3["hybrid"] = json.loads(v5._artifact_path(hyb[-1]["run_artifact"]).read_text())

        ds_dir = next(r for r in results if r.get("dataset") == name)["config"]["dataset"]["path"]
        ds = load_beir(ds_dir, name=name, version="1.0")
        qrels = ds.qrels
        texts = {q.id: q.text for q in ds.queries}

        test_q = [q for q in rv.load_split_qrels(ds_dir, "test")
                  if all(q in runs3[s] for s in ("bm25", "dense", "hybrid"))
                  and q in qrels and q in texts]

        pstrat = per_query_scores(runs3, qrels, test_q)
        S = np.array([[pstrat[s].get(q, 0.0) for s in ("bm25", "dense", "hybrid")]
                      for q in test_q])
        uniform_means = S.mean(axis=0)
        best_u_idx = int(uniform_means.argmax())
        best_u = float(uniform_means[best_u_idx])
        oracle_hard = S.max(axis=1)

        n = len(test_q)
        F = np.zeros((n, len(GRID)))
        for i, q in enumerate(test_q):
            for j, w in enumerate(GRID):
                ranked = v5.fuse(runs3["bm25"][q], runs3["dense"][q], w)
                F[i, j] = per_query_ndcg_at_k({q: ranked}, {q: qrels[q]}, 10).get(q, 0.0)
        oracle_soft = F.max(axis=1)
        const_scores = F[:, GRID.index(TUNED_W[name])]

        soft_beats_hard = oracle_soft > oracle_hard + 1e-12
        hedge_gain_total = float((oracle_soft - oracle_hard).clip(min=0).sum())
        const_captured = float((const_scores - oracle_hard).clip(min=0).sum())
        captured_frac = const_captured / hedge_gain_total if hedge_gain_total > 0 else 0.0

        wins = oracle_hard > best_u + 1e-9
        gains = np.where(wins, oracle_hard - best_u, 0.0)
        losses = np.where(wins, 0.0, best_u - (S.sum(axis=1) - S.max(axis=1)) / 2.0)

        n_docs, df = rv.corpus_stats(ds_dir, name)
        train_q = [q for q in rv.load_split_qrels(ds_dir, "train")
                   if q in texts and q in qrels and all(q in runs3[s] for s in ("bm25", "dense", "hybrid"))]
        strat_train = per_query_scores(runs3, qrels, train_q)
        from sklearn.tree import DecisionTreeClassifier
        X_train = [rv.query_features(texts[q], df, n_docs) for q in train_q]
        y_train = [int(np.argmax([strat_train[s].get(q, 0.0) for s in ("bm25", "dense", "hybrid")]))
                   for q in train_q]
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
        tree.fit(X_train, y_train)
        X_test = [rv.query_features(texts[q], df, n_docs) for q in test_q]
        pred = tree.predict(X_test)
        oracle_label = S.argmax(axis=1)
        correct = (pred == oracle_label).astype(float)

        idx = rng.integers(0, n, size=(B, n))
        acc_b = correct[idx].mean(axis=1)
        gain_b = gains[idx].sum(axis=1)
        loss_b = losses[idx].sum(axis=1)
        pstar_b = loss_b / (gain_b + loss_b)
        diff_b = acc_b - pstar_b
        acc_ci = np.percentile(acc_b, [2.5, 97.5])
        pstar_ci = np.percentile(pstar_b, [2.5, 97.5])
        p_win = float((diff_b > 0).mean())

        sec = f"""### {name}
- canonical test queries: {n}
- unified best uniform: {['bm25','dense','hybrid'][best_u_idx]} @ {best_u:.4f}
  (strategy means: bm25 {uniform_means[0]:.4f}, dense {uniform_means[1]:.4f}, hybrid {uniform_means[2]:.4f})
- hard oracle {oracle_hard.mean():.4f} | soft oracle {oracle_soft.mean():.4f} | tuned constant {const_scores.mean():.4f}
- hedging: soft beats hard on {soft_beats_hard.mean():.1%} of queries;
  tuned constant captures {captured_frac:.1%} of the positive hedge mass
- pick accuracy {correct.mean():.1%} (95% CI [{acc_ci[0]:.1%}, {acc_ci[1]:.1%}])
- break-even p* {float(pstar_b.mean()):.1%} (95% CI [{pstar_ci[0]:.1%}, {pstar_ci[1]:.1%}])
- paired bootstrap P(pick accuracy > p*) = {p_win:.3f}
"""
        all_print.append(sec)
        print(sec)
        pooled[name] = dict(n=n, correct=correct, gains=gains, losses=losses,
                            hedge=dict(soft_beats=soft_beats_hard.mean(),
                                       captured=captured_frac))

    correct = np.concatenate([pooled[d]["correct"] for d in DATASETS])
    gains = np.concatenate([pooled[d]["gains"] for d in DATASETS])
    losses = np.concatenate([pooled[d]["losses"] for d in DATASETS])
    n = len(correct)
    idx = rng.integers(0, n, size=(B, n))
    acc_b = correct[idx].mean(axis=1)
    pstar_b = losses[idx].sum(axis=1) / (gains[idx].sum(axis=1) + losses[idx].sum(axis=1))
    diff_b = acc_b - pstar_b
    acc_ci = np.percentile(acc_b, [2.5, 97.5])
    pstar_ci = np.percentile(pstar_b, [2.5, 97.5])
    p_win = float((diff_b > 0).mean())
    cap = np.average([pooled[d]["hedge"]["captured"] for d in DATASETS],
                     weights=[pooled[d]["n"] for d in DATASETS])
    soft_beats = np.average([pooled[d]["hedge"]["soft_beats"] for d in DATASETS],
                            weights=[pooled[d]["n"] for d in DATASETS])

    pooled_sec = f"""### pooled ({n} queries)
- pick accuracy {correct.mean():.1%} (95% CI [{acc_ci[0]:.1%}, {acc_ci[1]:.1%}])
- break-even p* {float(pstar_b.mean()):.1%} (95% CI [{pstar_ci[0]:.1%}, {pstar_ci[1]:.1%}])
- paired bootstrap P(pick accuracy > p*) = {p_win:.3f}
- soft oracle beats hard on {soft_beats:.1%} of queries (weighted);
  tuned constant captures {cap:.1%} of positive hedge mass
"""
    print(pooled_sec)

    verdict = (
        "Reviewer-stats verdict for the paper:\n"
        f"  pick accuracy {correct.mean():.1%} vs break-even {float(pstar_b.mean()):.1%}; "
        f"P(acc > p*) = {p_win:.2f} over {B:,} paired resamples.\n"
        + ("  -> cannot reject break-even operation: the CI overlaps p*.\n"
           if p_win < 0.95 else
           "  -> pick accuracy significantly exceeds p*; revisit the claim.\n")
        + f"  The tuned constant weight already captures {cap:.0%} of the hedging\n"
          "  mass that makes the soft oracle exceed the hard oracle."
    )
    print(verdict)
    out = ROOT / "experiments" / "results" / "reviewer-stats.md"
    out.write_text("# Reviewer-requested statistics\n\n" + "\n".join(all_print)
                   + pooled_sec + "\n```\n" + verdict + "\n```\n")
    print("saved:", out)


if __name__ == "__main__":
    main()
