# Research Checkpoint — 2 November 2026

## Research project

**Title:** Why Per-Query Retrieval Routing Fails to Beat Tuned Fusion: A Mechanism-Level Study.

## Completed: FiQA weight-0.7 hybrid reproduction

Experiment:
- ID: `fiqa-hybrid-w07-v1__20261102T064555Z`
- Result: `experiments/results/fiqa-hybrid-w07-v1__20261102T064555Z.json`
- Ranking: `artifacts/runs/fiqa-hybrid-w07-v1__20261102T064555Z.json`
- Dataset: FiQA
- Documents: 57,638
- Queries: 6,648
- Dense weight: 0.7
- BM25: k1=1.5, b=0.75
- Dense model: sentence-transformers/all-MiniLM-L6-v2
- Top-k: 10
- Seed: 42

Runner merged-qrels nDCG@10: 0.3827

Independent official test evaluation:
- Test queries: 648
- Covered: 648
- Missing: 0
- Extra non-test queries: 6,000
- Duplicate document IDs: 0
- Test nDCG@10: **0.387637**

This exactly reproduces the previously reported FiQA hybrid test score.

## Current FiQA test results

| Method | Test nDCG@10 |
|---|---:|
| BM25 | 0.231182 |
| Dense | 0.368671 |
| Hybrid, weight 0.7 | 0.387637 |

Hybrid improves over dense by 0.018966.

## Reproducibility and provenance

- Historical commit `fca5e9594683eddbd016c528241ce0174b26e6c8` exists locally.
- The historical versions of hybrid.py, bm25.py, dense.py, embeddings.py, and fiqa_hybrid_w07.yaml match current HEAD.
- New run commit: `772d7e35aea5b22d400427ecce3e4fae1142ab7d`.
- New run metadata records `git_dirty=true`.
- `scripts/router_v0.py` contains the title-plus-body corpus-statistics correction.
- Official FiQA train/dev/test qrels remain preserved.
- A derived combined `datasets/downloads/fiqa/qrels.tsv` was created for compatibility with the experiment runner.

## Next task

Continue FiQA weight selection using the official DEV split only.

Available ranking artifacts:
- Weight 0.5: `fiqa-hybrid-v1__20261018T155728Z.json`
- Weight 0.7: `fiqa-hybrid-w07-v1__20261102T064555Z.json`
- Weight 0.8: `fiqa-hybrid-w08-v1__20261019T062234Z.json` is missing.

The weight-0.8 metadata exists, but its ranking artifact is unavailable.

Do not select weights based on test results. The test split has already been inspected; document this limitation and use dev for further selection.

## Research integrity notes

- The runner's nDCG uses combined qrels; do not report it as test-only nDCG.
- Use standard exponential-gain nDCG@10 for evaluation.
- Do not modify the manuscript or previous checkpoint without explicit approval.
- Do not commit or push without explicit approval.
