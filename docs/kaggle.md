# Kaggle Workflow (Phase 6)

The same code runs locally and on Kaggle. Manual execution is fine until the
workflow stabilizes — no orchestration automation yet.

## One-time setup

1. Create a Kaggle notebook in your account.
2. Add the GitHub repo as a **Notebook input** (Add Data → search your repo,
   or use `git clone` in a cell: `!git clone https://github.com/<you>/ai-lab.git`).
3. Enable GPU (Settings → Accelerator → GPU T4/P100) for embedding models.

## Running an experiment

```python
# Cell 1 — install
!cd ai-lab && pip install -q -e ".[ml]"

# Cell 2 — dataset (small BEIR subset for iteration; full corpus for final runs)
#          upload the converted BEIR files as a Kaggle Dataset input, or
#          regenerate in-notebook per datasets/README.md

# Cell 3 — run
!cd ai-lab && python scripts/run_experiment.py --config experiments/configs/smoke_bm25.yaml
```

For real embedding models, copy a config and set:

```yaml
models:
  embedding: sentence-transformers/all-MiniLM-L6-v2
```

## Getting results back

Download `experiments/results/*.json` from the notebook output and commit them
to GitHub. Result files contain full reproducibility metadata (git commit of the
code that produced them — commit the Kaggle-side code changes first, then the
results reference the right commit).

## Rules

- Keep configs in git; never hardcode Kaggle paths in configs — use the
  manifest + relative paths.
- Record dataset provenance in `datasets/manifest.yaml` notes when you subset.
- Prefer multiple small runs (per-domain, per-fold) over one long run: free
  GPU sessions are time-limited.
