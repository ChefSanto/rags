# Router v2a (emb, PCA-32) — fiqa, nfcorpus (ndcg_at_10)

- train split: train (8090 queries)
- test split: test (971 queries)
- strategies: bm25, dense, hybrid
- embedding model: sentence-transformers/all-MiniLM-L6-v2 (same as dense baseline)
- model: PCA + GradientBoostingRegressor per strategy

## Test-split scores

| router | score |
|---|---|
| **routed (v2a)** | **0.3605** |
| uniform hybrid | 0.3597 |
| uniform dense | 0.3520 |
| uniform bm25 | 0.2578 |
| domain router {'fiqa': 'dense', 'nfcorpus': 'hybrid'} | 0.3583 |
| oracle (ceiling) | 0.4177 |

- routing gain over best uniform (hybrid): **+0.0008**
- vs domain router: +0.0022
- oracle-gap recovery vs best uniform: 1.4%
- gap remaining to oracle: +0.0572

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.2518 | 0.2578 | 0.2526 |
| dense | 0.3499 | 0.3520 | 0.3514 |
| hybrid | 0.3514 | 0.3597 | 0.3519 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 4.6% |
| dense | 48.3% |
| hybrid | 47.1% |

- PCA explained variance (first 5): 0.061, 0.036, 0.027, 0.023, 0.020

## Top feature importances (PCA components)

| component | importance |
|---|---|
| PC3 | 0.104 |
| PC1 | 0.093 |
| PC5 | 0.076 |
| PC12 | 0.057 |
| PC8 | 0.045 |
| PC7 | 0.045 |
| PC9 | 0.041 |
| PC4 | 0.040 |
| PC2 | 0.037 |
| PC18 | 0.037 |

