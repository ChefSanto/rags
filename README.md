# AI Lab

A reproducible experimentation platform for RAG (Retrieval-Augmented Generation) research.

> Research first. Infrastructure second. UI third.

AI Lab is the laboratory: the research questions and discoveries produced inside it
become papers, workshops, benchmarks, and potentially public tools.

## What works today (V0.1)

- [x] Define an experiment using YAML
- [x] Run it locally: `python scripts/run_experiment.py --config ...`
- [x] BM25 baseline
- [x] Dense retrieval baseline (with an offline hashing embedder for smoke tests)
- [x] Hybrid retrieval (score fusion)
- [x] Recall@K, Precision@K, MRR, nDCG
- [x] Reproducible results: Git commit, seed, Python/package versions, hardware, timestamp
- [x] File-based result registry + experiment comparison
- [x] Document profiler v0 (PDF properties for routing research)

## Quickstart

```bash
# 1. Create a virtual environment and install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core deps: pyyaml, numpy
pip install -e ".[ml]"           # optional: sentence-transformers, torch (GPU work / Kaggle)
pip install -e ".[pdf]"          # optional: pymupdf (document profiler)

# 2. Generate the tiny smoke-test dataset (deterministic, seed 42)
python scripts/make_synthetic_dataset.py

# 3. Run the three baselines
python scripts/run_experiment.py --config experiments/configs/smoke_bm25.yaml
python scripts/run_experiment.py --config experiments/configs/smoke_dense.yaml
python scripts/run_experiment.py --config experiments/configs/smoke_hybrid.yaml

# 4. Compare results
python scripts/compare_results.py
```

Everything runs on CPU. Real research runs (bigger embeddings, GPU) execute on Kaggle
using the same code and configs — see `docs/kaggle.md`.

## Project layout

```text
ai-lab/
├── core/                  # research code (no web code here)
│   ├── experiments/       # schema, runner, reproducibility metadata, registry
│   ├── datasets/          # dataset loaders (BEIR format + synthetic smoke)
│   ├── retrieval/         # bm25, dense, hybrid retrievers
│   ├── evaluation/        # recall@k, precision@k, mrr, ndcg, comparison report
│   ├── models/            # embedding model wrappers (with offline fallback)
│   └── profiling/         # PDF document profiler v0 (routing properties)
├── experiments/
│   ├── configs/           # YAML experiment definitions (version-controlled)
│   └── results/           # machine-readable results (committed for reproducibility)
├── datasets/              # datasets + manifest.yaml (large downloads are gitignored)
├── scripts/               # CLI entry points
├── notebooks/             # exploration (not part of the runner)
├── dashboard/             # future Cloudflare Pages app
├── worker/                # future Cloudflare Worker API
├── docs/                  # schema, datasets, architecture decisions (ADRs)
└── papers/                # research proposals and reports
    └── property-routing/
```

## Key design decisions

1. **Local-first results.** Experiment results are JSON files in `experiments/results/`.
   Every comparison, report, and the future dashboard reads those files. The D1 database
   (Phase 7–8) becomes a thin sync layer, not a dependency. See
   `docs/decisions/0001-local-first-results.md`.
2. **Declarative datasets.** `datasets/manifest.yaml` maps dataset name/version → location
   and format. Experiments reference datasets by name, never by hardcoded paths.
3. **Versioned profiler.** Document properties come from an explicit
   `profiler_version`, so routing experiments remain reproducible when extraction improves.
4. **No premature infrastructure.** No auth, no billing, no microservices. The scientific
   loop works before anything touches Cloudflare.

## Research program

The flagship hypothesis (see `papers/property-routing/proposal.md`):

> Can measurable document properties be used to select the retrieval strategy
> that maximizes RAG performance?

Baselines: fixed BM25 · fixed dense · fixed hybrid · domain router · property router.
Treated as a hypothesis to test — not an assumed conclusion — across document types
with non-cherry-picked datasets (BEIR subsets first: FiQA, NFCorpus).

## Status

V0.1 checklist per `AI_Lab_Project_Roadmap.md` §24. Kaggle workflow documented in
`docs/kaggle.md`. Dashboard intentionally not started.
