# FiQA Controlled Hybrid Fusion Weight Sweep

**Date:** 2026-11-02
**Project:** AI Lab / rags
**Repository commit:** `4195c3986e4ee5efe7061d4c007126d65d546eb2`

## Research question

How does the dense retrieval weight affect BM25+dense hybrid retrieval performance on FiQA?

## Experimental setup

- Dataset: official BEIR FiQA
- Corpus: 57,638 documents
- Queries: 6,648
- Official dev split: 500 queries
- Evaluation: nDCG@10
- BM25: k1=1.5, b=0.75
- Dense encoder: sentence-transformers/all-MiniLM-L6-v2
- Hybrid top-k: 10
- Random seed: 42
- Hardware: NVIDIA Tesla T4
- PyTorch: 2.11.0+cu128
- Python: 3.13.15
- Official archive MD5: `17918ed23cd04fb15047f73e6c3bd9d9`

All five new runs used the same recorded Git commit.

## Dev results

| Dense weight | Dev nDCG@10 |
|---:|---:|
| 0.3 | 0.312774 |
| 0.5 | 0.380627 |
| 0.6 | 0.402663 |
| 0.7 | 0.402643 |
| 0.8 | 0.400993 |

All five runs evaluated all 500 dev queries. There were zero missing queries and zero duplicate retrieved documents.

The highest observed dev score was at weight 0.6 (0.402663). Weight 0.7 was nearly identical (0.402643).

## Paired bootstrap analysis

Percentile paired bootstrap with 10,000 resamples and seed 42, using the same 500 dev queries.

### Weight 0.6 minus weight 0.7

- Mean difference: +0.00001965
- 95% bootstrap CI: [-0.00592870, +0.00614296]
- Improved queries: 61
- Worsened queries: 79
- Tied queries: 360

The interval includes zero. There is no evidence here that 0.6 meaningfully outperforms 0.7.

### Weight 0.6 minus weight 0.5

- Mean difference: +0.02203561
- 95% bootstrap CI: [+0.01438614, +0.02997678]
- Improved queries: 140
- Worsened queries: 42
- Tied queries: 318

The interval is above zero. Weight 0.6 outperforms 0.5 on this dev sample.

## Interpretation

The observed best setting is 0.6, but weights 0.6 and 0.7 form an effectively tied performance plateau in this experiment. The results support tuning constant fusion rather than assuming equal weighting is optimal. They do not establish that 0.6 is the uniquely optimal weight.

## Important limitations

- The test split was inspected in earlier work. It is not an untouched confirmatory holdout and was not used to select the weights in this sweep.
- The sweep covers only five discrete weights; it does not establish a global optimum.
- Bootstrap intervals quantify query-level sampling uncertainty on this dev set, not variation across random seeds, datasets, or model training.
- The earlier historical runs had `git_dirty: true`. This new sweep uses one recorded commit, but the historical runs should not be treated as equally controlled.

## Experiment records

- Weight 0.3: `fiqa-hybrid-sweep-w03-v1__20261102T134658Z`
  - Result: `experiments/results/fiqa-hybrid-sweep-w03-v1__20261102T134658Z.json`
  - Ranking: `artifacts/runs/fiqa-hybrid-sweep-w03-v1__20261102T134658Z.json`
- Weight 0.5: `fiqa-hybrid-sweep-w05-v1__20261102T140203Z`
  - Result: `experiments/results/fiqa-hybrid-sweep-w05-v1__20261102T140203Z.json`
  - Ranking: `artifacts/runs/fiqa-hybrid-sweep-w05-v1__20261102T140203Z.json`
- Weight 0.6: `fiqa-hybrid-sweep-w06-v1__20261102T141710Z`
  - Result: `experiments/results/fiqa-hybrid-sweep-w06-v1__20261102T141710Z.json`
  - Ranking: `artifacts/runs/fiqa-hybrid-sweep-w06-v1__20261102T141710Z.json`
- Weight 0.7: `fiqa-hybrid-sweep-w07-v1__20261102T143153Z`
  - Result: `experiments/results/fiqa-hybrid-sweep-w07-v1__20261102T143153Z.json`
  - Ranking: `artifacts/runs/fiqa-hybrid-sweep-w07-v1__20261102T143153Z.json`
- Weight 0.8: `fiqa-hybrid-sweep-w08-v1__20261102T144633Z`
  - Result: `experiments/results/fiqa-hybrid-sweep-w08-v1__20261102T144633Z.json`
  - Ranking: `artifacts/runs/fiqa-hybrid-sweep-w08-v1__20261102T144633Z.json`

## Reproduction note

The experiment result JSON files and complete ranking artifacts are preserved in the repository. The official dataset itself remains in the ignored `datasets/downloads/` directory and must be downloaded again in a fresh Kaggle session.

This checkpoint records the sweep and analysis. It does not modify the manuscript and does not claim that the checkpoint has been pushed to GitHub.
