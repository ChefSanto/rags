# Router v2a (emb, PCA-32) — fiqa, nfcorpus (ndcg_at_10)

- train split: test (971 queries)
- test split: test (971 queries)
- strategies: bm25, dense, hybrid
- embedding model: sentence-transformers/all-MiniLM-L6-v2 (same as dense baseline)
- model: PCA + GradientBoostingRegressor per strategy

## Test-split scores

| router | score |
|---|---|
| **routed (v2a)** | **0.4008** |
| uniform hybrid | 0.3614 |
| uniform dense | 0.3520 |
| uniform bm25 | 0.2578 |
| domain router {'fiqa': 'hybrid', 'nfcorpus': 'hybrid'} | 0.3614 |
| oracle (ceiling) | 0.4173 |

- routing gain over best uniform (hybrid): **+0.0394**
- vs domain router: +0.0394
- oracle-gap recovery vs best uniform: 70.4%
- gap remaining to oracle: +0.0165

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.2584 | 0.2578 | 0.2578 |
| dense | 0.3532 | 0.3520 | 0.3520 |
| hybrid | 0.3625 | 0.3614 | 0.3614 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 10.7% |
| dense | 43.7% |
| hybrid | 45.6% |

- PCA explained variance (first 5): 0.061, 0.036, 0.027, 0.024, 0.021

## Top feature importances (PCA components)

| component | importance |
|---|---|
| PC5 | 0.057 |
| PC3 | 0.049 |
| PC6 | 0.048 |
| PC15 | 0.043 |
| PC1 | 0.039 |
| PC19 | 0.038 |
| PC32 | 0.037 |
| PC8 | 0.035 |
| PC11 | 0.034 |
| PC25 | 0.034 |

