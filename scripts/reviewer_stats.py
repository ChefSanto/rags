#!/usr/bin/env python
"""Reviewer-requested statistics, v2 (local, minutes).

1. Unified baseline on ONE canonical test-query set (all three strategies).
2. Pick accuracy AND routed gain of the v1 router (per-strategy score
   regressors on lexical features, argmax routing) -- the model whose pooled
   gain (+0.003) the paper cites. Bootstrap CI on pick accuracy.
3. Paired bootstrap: P(pick_accuracy > p*) and the sign prediction:
   where acc > p* routing should profit, where acc < p* it should lose.
4. Hedging quantification: soft-oracle vs hard-oracle vs tuned constant.

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
TUNED_W = {"fiqa": 0.7, "nfcorpus": 0.5}  # committed weight sweep
GRID = v5.GRID
B = 10_000
SEED = 42
STRATS = ("bm25", "dense", "hybrid")


def per_query_scores(runs, qrels, k=10):
    return {s: per_query_ndcg_at_k(runs[s], qrels, k) for s in runs}


def main() -> None:
    rng = np.random.default_rng(SEED)
    results_dir = str(ROOT / "experiments" / "results")
    results = load_results(results_dir)
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
                  if all(q in runs3[s] for s in STRATS) and q in qrels and q in texts]
        n = len(test_q)

        pstrat = per_query_scores(runs3, qrels)
        S = np.array([[pstrat[s].get(q, 0.0) for s in STRATS] for q in test_q])
        uniform_means = S.mean(axis=0)
        best_u_idx = int(uniform_means.argmax())
        best_u = float(uniform_means[best_u_idx])
        oracle_hard = S.max(axis=1)

        F = np.zeros((n, len(GRID)))
        for i, q in enumerate(test_q):
            for j, w in enumerate(GRID):
                ranked = v5.fuse(runs3["bm25"][q], runs3["dense"][q], w)
                F[i, j] = per_query_ndcg_at_k({q: ranked}, {q: qrels[q]}, 10).get(q, 0.0)
        oracle_soft = F.max(axis=1)
        const_scores = F[:, GRID.index(TUNED_W[name])]

        soft_beats_hard = oracle_soft > oracle_hard + 1e-12
        hedge_mass = float((oracle_soft - oracle_hard).clip(min=0).sum())
        const_captured = float((const_scores - oracle_hard).clip(min=0).sum())
        captured_frac = const_captured / hedge_mass if hedge_mass > 0 else 0.0

        wins = oracle_hard > best_u + 1e-9
        gains = np.where(wins, oracle_hard - best_u, 0.0)
        losses = np.where(wins, 0.0, best_u - (S.sum(axis=1) - S.max(axis=1)) / 2.0)

        # ---- v1 router: per-strategy regressors on lexical features, argmax route
        n_docs, df = rv.corpus_stats(ds_dir, name)
        train_q = [q for q in rv.load_split_qrels(ds_dir, "train")
                   if q in texts and q in qrels and all(q in runs3[s] for s in STRATS)]
        strat_train = per_query_scores(runs3, qrels)
        from sklearn.ensemble import GradientBoostingRegressor
        X_train = np.array([rv.query_features(texts[q], df, n_docs) for q in train_q])
        Y_train = np.array([[strat_train[s].get(q, 0.0) for s in STRATS] for q in train_q])
        regs = []
        for j in range(3):
            m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, random_state=42)
            m.fit(X_train, Y_train[:, j])
            regs.append(m)
        X_test = np.array([rv.query_features(texts[q], df, n_docs) for q in test_q])
        preds = np.column_stack([m.predict(X_test) for m in regs])
        pred_label = preds.argmax(axis=1)
        oracle_label = S.argmax(axis=1)
        correct = (pred_label == oracle_label).astype(float)
        routed_scores = S[np.arange(n), pred_label]
        routed_gain = float(routed_scores.mean() - best_u)
        delta = routed_scores - S[:, best_u_idx]
        gain_ci = np.percentile(
            delta[rng.integers(0, n, size=(B, n))].mean(axis=1), [2.5, 97.5])
        margin = np.sort(S, axis=1)[:, -1] - np.sort(S, axis=1)[:, -2]
        decisive = margin > 0.05
        nd = int(decisive.sum())
        acc_dec = float(correct[decisive].mean())
        idd = rng.integers(0, max(nd, 1), size=(B, max(nd, 1)))
        acc_dec_ci = np.percentile(correct[decisive][idd].mean(axis=1), [2.5, 97.5])
        # reviewer v5: decisive-subset break-even + stratum gain decomposition
        gains_d, losses_d = gains[decisive], losses[decisive]
        nd2 = max(int(decisive.sum()), 1)
        idd2 = rng.integers(0, nd2, size=(B, nd2))
        pstar_dec = losses_d[idd2].sum(1) / (gains_d[idd2].sum(1) + losses_d[idd2].sum(1))
        pstar_dec_ci = np.percentile(pstar_dec, [2.5, 97.5])
        win_rate = float(wins.mean())
        G = float(gains[wins].mean()) if wins.any() else 0.0
        L = float(losses[~wins].mean()) if (~wins).any() else 0.0
        p_formula = (1 - win_rate) * L / (win_rate * G + (1 - win_rate) * L) if (G + L) > 0 else 1.0
        delta_q = routed_scores - S[:, best_u_idx]
        gain_dec = float(delta_q[decisive].mean())
        gain_non = float(delta_q[~decisive].mean())
        # weight-selection robustness: best w on TRAIN split only, scored on test
        train_judged = [q for q in train_q if q in qrels]
        sweep_train = {}
        for w in GRID:
            sc = [per_query_ndcg_at_k({q: v5.fuse(runs3['bm25'][q], runs3['dense'][q], w)}, {q: qrels[q]}, 10).get(q, 0.0) for q in train_judged]
            sweep_train[w] = sum(sc) / max(len(sc), 1)
        w_train = max(sweep_train, key=sweep_train.get)
        flat = float(F[:, GRID.index(0.5):GRID.index(0.8) + 1].mean(axis=0).max()
                     - F[:, GRID.index(0.5):GRID.index(0.8) + 1].mean(axis=0).min())
        w_lines = (f'    weight selection: train-split-best w={w_train}, test score at w_train '
                   f'{F[:, GRID.index(w_train)].mean():.4f} vs at 0.7/0.5 {const_scores.mean():.4f}; '
                   f'test sweep flatness in [0.5,0.8] = {flat:.4f}')
        # reviewer v5: margin-gated routing (route only when the model is confident)
        pmargin = np.sort(preds, axis=1)[:, -1] - np.sort(preds, axis=1)[:, -2]
        gate_lines = []
        for t in (0.0, 0.02, 0.05, 0.1, 0.2):
            gate = pmargin > t
            if int(gate.sum()) < 10:
                continue
            rg = S[np.arange(n), pred_label].copy()
            rg[~gate] = S[~gate, best_u_idx]
            d_g = rg - S[:, best_u_idx]
            ci_g = np.percentile(d_g[rng.integers(0, n, size=(B, n))].mean(axis=1), [2.5, 97.5])
            gate_lines.append(f'    gate>{t}: routed {gate.mean():.1%}, gain {d_g.mean():+.4f} CI [{ci_g[0]:+.4f}, {ci_g[1]:+.4f}]')
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_dec = roc_auc_score(decisive.astype(int), pmargin)
        ap_dec = average_precision_score(decisive.astype(int), pmargin)
        gate_lines.append(f'    gate predictive value: AUC {auc_dec:.3f}, PR-AUC {ap_dec:.3f} (decisive base rate {decisive.mean():.1%})')
        gate_block = '\n'.join(gate_lines)
        # gate predictive value: does predicted margin identify decisiveness?

        # ---- paired bootstrap: pick accuracy vs p*
        idx = rng.integers(0, n, size=(B, n))
        acc_b = correct[idx].mean(axis=1)
        pstar_b = losses[idx].sum(axis=1) / (gains[idx].sum(axis=1) + losses[idx].sum(axis=1))
        diff_b = acc_b - pstar_b
        acc_ci = np.percentile(acc_b, [2.5, 97.5])
        pstar_ci = np.percentile(pstar_b, [2.5, 97.5])
        p_win = float((diff_b > 0).mean())

        sec = f"""### {name}
- canonical test queries: {n}
- unified best uniform: {STRATS[best_u_idx]} @ {best_u:.4f}
- v1 router: pick accuracy {correct.mean():.1%} (95% CI [{acc_ci[0]:.1%}, {acc_ci[1]:.1%}]);
  break-even p* {float(pstar_b.mean()):.1%} (95% CI [{pstar_ci[0]:.1%}, {pstar_ci[1]:.1%}]);
  P(acc > p*) = {p_win:.3f}; ROUTED GAIN {routed_gain:+.4f} (95% CI [{gain_ci[0]:+.4f}, {gain_ci[1]:+.4f}])
- decisive queries (oracle margin > 0.05): {nd}/{n} ({nd/n:.1%}); pick accuracy {acc_dec:.1%} (95% CI [{acc_dec_ci[0]:.1%}, {acc_dec_ci[1]:.1%}])
- decisive-subset break-even p* {float(pstar_dec.mean()):.1%} (95% CI [{pstar_dec_ci[0]:.1%}, {pstar_dec_ci[1]:.1%}]); win-rate-corrected formula p*={p_formula:.3f} (w={win_rate:.1%}, G={G:.3f}, L={L:.3f})
- stratum routed gain: decisive {gain_dec:+.4f}, non-decisive {gain_non:+.4f}
- margin-gated routing (predicted-margin threshold):
{w_lines}
{gate_block}
- hedging: soft beats hard on {soft_beats_hard.mean():.1%} of queries;
  tuned constant captures {captured_frac:.1%} of positive hedge mass
"""
        print(sec)
        pooled[name] = dict(n=n, correct=correct, gains=gains, losses=losses,
                            routed_gain=routed_gain, gain_ci=gain_ci, p_win=p_win,
                            acc_dec=acc_dec, nd=nd,
                            acc=float(correct.mean()), pstar=float(pstar_b.mean()),
                            hedge=dict(soft_beats=soft_beats_hard.mean(), captured=captured_frac))

    # ---- pooled
    correct = np.concatenate([pooled[d]["correct"] for d in DATASETS])
    gains = np.concatenate([pooled[d]["gains"] for d in DATASETS])
    losses = np.concatenate([pooled[d]["losses"] for d in DATASETS])
    n = len(correct)
    idx = rng.integers(0, n, size=(B, n))
    acc_b = correct[idx].mean(axis=1)
    pstar_b = losses[idx].sum(axis=1) / (gains[idx].sum(axis=1) + losses[idx].sum(axis=1))
    p_win = float((acc_b - pstar_b > 0).mean())
    cap = np.average([pooled[d]["hedge"]["captured"] for d in DATASETS],
                     weights=[pooled[d]["n"] for d in DATASETS])
    soft_beats = np.average([pooled[d]["hedge"]["soft_beats"] for d in DATASETS],
                            weights=[pooled[d]["n"] for d in DATASETS])
    rgain_w = np.average([pooled[d]["routed_gain"] for d in DATASETS],
                         weights=[pooled[d]["n"] for d in DATASETS])

    lines = [f"### pooled ({n} queries)",
             f"- pick accuracy {correct.mean():.1%}; p* {float(pstar_b.mean()):.1%}; "
             f"P(acc > p*) = {p_win:.3f}",
             f"- routed gain (weighted): {rgain_w:+.4f}",
             f"- soft beats hard on {soft_beats:.1%} of queries (weighted); "
             f"constant captures {cap:.1%} of hedge mass", ""]
    sign_ok = all((pooled[d]["routed_gain"] > 0) == (pooled[d]["acc"] > pooled[d]["pstar"])
                  for d in DATASETS)
    lines.append("per-domain sign check (routed gain sign == sign(acc - p*)): "
                 + ("CONFIRMED on both domains" if sign_ok else "NOT clean — inspect"))
    for d in DATASETS:
        lines.append(f"  {d}: acc {pooled[d]['acc']:.1%} vs p* {pooled[d]['pstar']:.1%} "
                     f"(P={pooled[d]['p_win']:.2f}) -> gain {pooled[d]['routed_gain']:+.4f} CI [{pooled[d]['gain_ci'][0]:+.4f}, {pooled[d]['gain_ci'][1]:+.4f}]; decisive pick acc {pooled[d]['acc_dec']:.1%} on {pooled[d]['nd']} q")
    print("\n".join(lines))

    out = ROOT / "experiments" / "results" / "reviewer-stats.md"
    out.write_text("# Reviewer-requested statistics (v2)\n\n```\n" + "\n".join(lines) + "\n```\n")
    print("saved:", out)


if __name__ == "__main__":
    main()
