#!/usr/bin/env python
"""Generate the paper's figures from committed results (reproducible).

Outputs (papers/property-routing/figures/):
  fig_ladder.png    the routing ladder: fixed strategies, routers, oracle
  fig_valley.png    the non-convex fusion-weight surface (v5 mechanism)
  fig_breakeven.png break-even pick accuracy vs achieved predictor accuracy

All numbers are from the committed reports in experiments/results/ (cited per
figure). Run: pip install matplotlib && python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "papers" / "property-routing" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GRAY, BLUE, GREEN, RED = "#9aa3b2", "#5b9dff", "#4ade80", "#ff7b72"
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
# FiQA uniform weight sweep (router-v5 report) + v5-direct predicted mean weight.
w = [i / 10 for i in range(11)]
ndcg = [0.2312, 0.2619, 0.2642, 0.2673, 0.2714, 0.3574,
        0.3686, 0.3695, 0.3689, 0.3694, 0.3687]

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(w, ndcg, marker="o", color=BLUE, lw=2)
ax.axvspan(0.28, 0.42, color=RED, alpha=0.12)
ax.text(0.35, 0.30, "the valley:\nmean predictions\nland here", ha="center", color=RED, fontsize=9)
ax.axvline(0.323, color=RED, ls="--", lw=1.2)
ax.text(0.323, 0.375, " v5-direct predicted\n mean weight 0.323", color=RED, fontsize=8, va="bottom")
ax.annotate("w=0.4: 0.2714", xy=(0.4, 0.2714), xytext=(0.52, 0.25),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=8)
ax.annotate("w=0.5: 0.3574", xy=(0.5, 0.3574), xytext=(0.62, 0.33),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=8)
ax.set_xlabel("fusion weight w (dense share)")
ax.set_ylabel("FiQA test nDCG@10")
ax.set_title("The non-convex weight valley (v5-direct collapse mechanism)")
fig.tight_layout()
fig.savefig(FIG / "fig_valley.png", dpi=180)
plt.close(fig)

# ------------------------------------------------------------ fig_breakeven
# routing-wins report: p* per domain; achieved predictor label accuracy ~51%.
domains = ["FiQA", "NFCorpus"]
p_star = [49.6, 45.5]
achieved = [51, 51]

fig, ax = plt.subplots(figsize=(6.0, 3.4))
x = range(len(domains))
b1 = ax.bar([i - 0.18 for i in x], p_star, width=0.36, color=RED, label="break-even pick accuracy $p^*$")
b2 = ax.bar([i + 0.18 for i in x], achieved, width=0.36, color=BLUE,
            label="achieved predictor accuracy (approx. chance)")
for b, v in zip(b1, p_star):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
for b, v in zip(b2, achieved):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
ax.axhline(50, color=GRAY, ls=":", lw=1)
ax.set_xticks(list(x), domains)
ax.set_ylabel("%")
ax.set_ylim(0, 62)
ax.set_title("Predictors sit exactly at the profitability threshold")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig_breakeven.png", dpi=180)
plt.close(fig)

print("wrote", ", ".join(p.name for p in sorted(FIG.glob("*.png"))))
