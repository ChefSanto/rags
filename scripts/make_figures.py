#!/usr/bin/env python
"""Generate the paper's figures from committed results (reproducible).

Outputs (papers/property-routing/figures/):
  fig_ladder.png    the routing ladder: fixed strategies, routers, oracle
  fig_valley.png    per-query fusion-weight sweep with IQR bands (both domains),
                    recomputed from committed run artifacts
  fig_breakeven.png break-even pick accuracy vs achieved predictor accuracy

Run: pip install matplotlib && python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "papers" / "property-routing" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GRAY, BLUE, GREEN, RED = "#9aa3b2", "#5b9dff", "#4ade80", "#ff7b72"
ORANGE = "#f5a623"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- fig_ladder
# Weighted test nDCG@10, honest protocol.
# Sources: router-v1/v2a/v3/v4/v5/v6 reports, oracle reports (experiments/results/)
ladder = [
    ("bm25 uniform",            0.2603, GRAY),
    ("v0 classify (lexical)",   0.2651, BLUE),
    ("v3 pseudo-sentences",     0.2603, BLUE),
    ("v5a regress w*",          0.2907, BLUE),
    ("v6f LLM forced",          0.3386, BLUE),
    ("v4 pseudo-questions",     0.3469, BLUE),
    ("dense uniform",           0.3523, GRAY),
    ("v6h LLM hedge",           0.3599, BLUE),
    ("best uniform (tuned w)",  0.3597, RED),
    ("v5b strength hedge",      0.3607, BLUE),
    ("v1 regress (lexical)",    0.3626, BLUE),
    ("v2a regress (emb)",       0.3605, BLUE),
    ("oracle (cheat)",          0.4173, GREEN),
]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
names = [x[0] for x in ladder]
vals = [x[1] for x in ladder]
cols = [x[2] for x in ladder]
bars = ax.barh(names, vals, color=cols, edgecolor="none")
ax.axvline(0.3597, color=RED, lw=1.2, ls="--", alpha=0.8)
ax.text(0.3597 + 0.002, -0.4, "tuned uniform\n0.3597", color=RED, fontsize=8, va="top")
for b, v in zip(bars, vals):
    ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.4f}",
            va="center", fontsize=8, color="#333")
ax.set_xlim(0.2, 0.45)
ax.set_xlabel("weighted test nDCG@10 (honest train/test protocol)")
ax.set_title("The routing rainbow: every paradigm, one bar - nobody climbs to the oracle")
fig.tight_layout()
fig.savefig(FIG / "fig_ladder.png", dpi=180)
plt.close(fig)

# ---------------------------------------------------------------- fig_valley
# Per-query nDCG across the full fusion-weight sweep, recomputed from committed
# run artifacts (bm25 + dense ranked lists, RRF fusion). Solid = mean,
# band = interquartile range across test queries. Validated to reproduce the
# FiQA means reported in router-v5-soft report.
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import router_v0 as rv  # noqa: E402
import router_v5_soft as v5  # noqa: E402
from core.evaluation.retrieval import per_query_ndcg_at_k  # noqa: E402
from core.datasets.beir import load_beir  # noqa: E402
from core.experiments.registry import load_results  # noqa: E402

GRID = [i / 10 for i in range(11)]
RESULTS_DIR = str(ROOT / "experiments" / "results")
PRED_W = {"fiqa": 0.323, "nfcorpus": 0.272}  # v5-direct predicted mean weights

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=True)
for ax, (name, pretty, color) in zip(axes, [
        ("fiqa", "FiQA", BLUE), ("nfcorpus", "NFCorpus", ORANGE)]):
    try:
        runs = v5.load_runs(name, RESULTS_DIR)
        ds_dir = next(r for r in load_results(RESULTS_DIR)
                      if r.get("dataset") == name)["config"]["dataset"]["path"]
        ds = load_beir(ds_dir, name=name, version="1.0")
        test_q = [q for q in rv.load_split_qrels(ds_dir, "test")
                  if q in runs["bm25"] and q in runs["dense"] and q in ds.qrels]
        per_w = {w: [] for w in GRID}
        for q in test_q:
            for w in GRID:
                ranked = v5.fuse(runs["bm25"][q], runs["dense"][q], w)
                s = per_query_ndcg_at_k({q: ranked}, {q: ds.qrels[q]}, 10).get(q, 0.0)
                per_w[w].append(s)
        means = [sum(per_w[w]) / len(per_w[w]) for w in GRID]
        p25 = [sorted(per_w[w])[len(per_w[w]) // 4] for w in GRID]
        p75 = [sorted(per_w[w])[(3 * len(per_w[w])) // 4] for w in GRID]
        ax.fill_between(GRID, p25, p75, color=color, alpha=0.15, label="IQR across queries")
        ax.plot(GRID, means, marker="o", color=color, lw=2, label="mean nDCG@10")
        ax.axvline(PRED_W[name], color=RED, ls="--", lw=1.1)
        ax.text(PRED_W[name] + 0.02, min(means) + 0.02,
                f"predicted mean w\n{PRED_W[name]:.2f}", color=RED, fontsize=8)
        if name == "fiqa":
            ax.axvspan(0.28, 0.42, color=RED, alpha=0.10)
            ax.text(0.35, max(means) - 0.05, "the valley", color=RED, fontsize=9, ha="center")
            ax.annotate("w=0.4: 0.2714", xy=(0.4, 0.2714), xytext=(0.55, 0.24),
                        arrowprops=dict(arrowstyle="->", color="#555"), fontsize=8)
            ax.annotate("w=0.5: 0.3574", xy=(0.5, 0.3574), xytext=(0.62, 0.33),
                        arrowprops=dict(arrowstyle="->", color="#555"), fontsize=8)
        ax.set_title(pretty)
        ax.set_xlabel("fusion weight w (dense share)")
        print(f"{name}: sweep means {[round(m, 4) for m in means]}")
    except Exception as e:  # missing local artifacts/qrels: skip domain
        ax.set_title(f"{pretty} (skipped: {e})")
        ax.text(0.5, 0.5, "run on machine with\ncommitted artifacts", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color=GRAY)
axes[0].set_ylabel("test nDCG@10")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Per-query fusion-weight surfaces: the non-convex valley (v5 mechanism)")
fig.tight_layout()
fig.savefig(FIG / "fig_valley.png", dpi=180)
plt.close(fig)

# ------------------------------------------------------------ fig_breakeven
domains = ["FiQA", "NFCorpus"]
p_star = [44.7, 46.0]        # reviewer-stats.md bootstrap means
achieved = [44.1, 57.6]      # v0 tree pick accuracy, retrained on canonical set

fig, ax = plt.subplots(figsize=(6.0, 3.4))
x = range(len(domains))
b1 = ax.bar([i - 0.18 for i in x], p_star, width=0.36, color=RED, label="break-even pick accuracy $p^*$")
b2 = ax.bar([i + 0.18 for i in x], achieved, width=0.36, color=BLUE,
            label="achieved pick accuracy (v0 tree, retrained)")
for b, v in zip(b1, p_star):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
for b, v in zip(b2, achieved):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
ax.axhline(50, color=GRAY, ls=":", lw=1)
ax.set_xticks(list(x), domains)
ax.set_ylabel("%")
ax.set_ylim(0, 70)
ax.set_title("Predictors sit exactly at the profitability threshold")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig_breakeven.png", dpi=180)
plt.close(fig)

print("wrote", ", ".join(p.name for p in sorted(FIG.glob("*.png"))))
