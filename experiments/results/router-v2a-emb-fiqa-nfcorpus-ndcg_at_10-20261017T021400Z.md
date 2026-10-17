# Router v2a (emb, PCA-32) — fiqa, nfcorpus (ndcg_at_10)

- train split: train (8090 queries)
- test split: test (971 queries)
- strategies: bm25, dense, hybrid
- embedding model: sentence-transformers/all-MiniLM-L6-v2 (same as dense baseline)
- model: PCA + GradientBoostingRegressor per strategy

## Test-split scores

| router | score |
|---|---|
| **routed (v2a)** | **0.3630** |
| uniform hybrid | 0.3614 |
| uniform dense | 0.3520 |
| uniform bm25 | 0.2578 |
| domain router {'fiqa': 'dense', 'nfcorpus': 'hybrid'} | 0.3585 |
| oracle (ceiling) | 0.4173 |

- routing gain over best uniform (hybrid): **+0.0016**
- vs domain router: +0.0045
- oracle-gap recovery vs best uniform: 2.8%
- gap remaining to oracle: +0.0543

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.2518 | 0.2578 | 0.2526 |
| dense | 0.3492 | 0.3520 | 0.3514 |
| hybrid | 0.3515 | 0.3614 | 0.3518 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 5.1% |
| dense | 45.5% |
| hybrid | 49.3% |

- PCA explained variance (first 5): 0.061, 0.036, 0.027, 0.023, 0.020

## Top feature importances (PCA components)

| component | importance |
|---|---|
| PC3 | 0.105 |
| PC1 | 0.094 |
| PC5 | 0.074 |
| PC12 | 0.058 |
| PC8 | 0.045 |
| PC7 | 0.045 |
| PC9 | 0.040 |
| PC4 | 0.040 |
| PC2 | 0.037 |
| PC18 | 0.036 |

