# Router v2a (emb, PCA-32) — nfcorpus, fiqa (ndcg_at_10)

- train split: test (323 queries)
- test split: test (648 queries)
- strategies: bm25, dense, hybrid
- embedding model: sentence-transformers/all-MiniLM-L6-v2 (same as dense baseline)
- model: PCA + GradientBoostingRegressor per strategy

## Test-split scores

| router | score |
|---|---|
| **routed (v2a)** | **0.3549** |
| uniform hybrid | 0.3708 |
| uniform dense | 0.3687 |
| uniform bm25 | 0.2312 |
| domain router {'nfcorpus': 'hybrid'} | n/a (test domains unseen in train) |
| oracle (ceiling) | 0.4343 |

- routing gain over best uniform (hybrid): **-0.0159**
- vs domain router: n/a
- oracle-gap recovery vs best uniform: -25.0%
- gap remaining to oracle: +0.0794

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.1930 | 0.2312 | 0.3111 |
| dense | 0.2204 | 0.3687 | 0.3186 |
| hybrid | 0.2493 | 0.3708 | 0.3376 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 12.3% |
| dense | 26.7% |
| hybrid | 61.0% |

- PCA explained variance (first 5): 0.063, 0.041, 0.027, 0.025, 0.022

## Top feature importances (PCA components)

| component | importance |
|---|---|
| PC2 | 0.059 |
| PC1 | 0.056 |
| PC10 | 0.054 |
| PC9 | 0.051 |
| PC17 | 0.046 |
| PC5 | 0.044 |
| PC14 | 0.043 |
| PC7 | 0.042 |
| PC22 | 0.040 |
| PC25 | 0.039 |

