# Router v2a (emb, PCA-32) — nfcorpus, fiqa (ndcg_at_10)

- train split: test (323 queries)
- test split: test (648 queries)
- strategies: bm25, dense, hybrid
- embedding model: sentence-transformers/all-MiniLM-L6-v2 (same as dense baseline)
- model: PCA + GradientBoostingRegressor per strategy

## Test-split scores

| router | score |
|---|---|
| **routed (v2a)** | **0.3555** |
| uniform hybrid | 0.3730 |
| uniform dense | 0.3687 |
| uniform bm25 | 0.2312 |
| domain router {'nfcorpus': 'hybrid'} | n/a (test domains unseen in train) |
| oracle (ceiling) | 0.4337 |

- routing gain over best uniform (hybrid): **-0.0175**
- vs domain router: n/a
- oracle-gap recovery vs best uniform: -28.9%
- gap remaining to oracle: +0.0782

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.1930 | 0.2312 | 0.3111 |
| dense | 0.2204 | 0.3687 | 0.3186 |
| hybrid | 0.2539 | 0.3730 | 0.3380 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 12.3% |
| dense | 24.8% |
| hybrid | 62.8% |

- PCA explained variance (first 5): 0.063, 0.041, 0.027, 0.025, 0.022

## Top feature importances (PCA components)

| component | importance |
|---|---|
| PC2 | 0.057 |
| PC10 | 0.056 |
| PC1 | 0.053 |
| PC9 | 0.050 |
| PC17 | 0.049 |
| PC5 | 0.045 |
| PC14 | 0.043 |
| PC22 | 0.040 |
| PC7 | 0.040 |
| PC25 | 0.039 |

