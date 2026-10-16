# Router v2a (emb+lex, PCA-32) — fiqa, nfcorpus (ndcg_at_10)

- train split: train (8090 queries)
- test split: test (971 queries)
- strategies: bm25, dense, hybrid
- embedding model: sentence-transformers/all-MiniLM-L6-v2 (same as dense baseline)
- model: PCA + GradientBoostingRegressor per strategy

## Test-split scores

| router | score |
|---|---|
| **routed (v2a)** | **0.3624** |
| uniform hybrid | 0.3597 |
| uniform dense | 0.3520 |
| uniform bm25 | 0.2578 |
| domain router {'fiqa': 'dense', 'nfcorpus': 'hybrid'} | 0.3583 |
| oracle (ceiling) | 0.4177 |

- routing gain over best uniform (hybrid): **+0.0026**
- vs domain router: +0.0041
- oracle-gap recovery vs best uniform: 4.6%
- gap remaining to oracle: +0.0553

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.2509 | 0.2578 | 0.2526 |
| dense | 0.3498 | 0.3520 | 0.3514 |
| hybrid | 0.3493 | 0.3597 | 0.3519 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 4.6% |
| dense | 50.4% |
| hybrid | 45.0% |

- PCA explained variance (first 5): 0.897, 0.060, 0.015, 0.009, 0.004

## Top feature importances (PCA components)

| component | importance |
|---|---|
| PC1 | 0.116 |
| PC2 | 0.111 |
| PC10 | 0.107 |
| PC7 | 0.057 |
| PC8 | 0.052 |
| PC12 | 0.043 |
| PC6 | 0.041 |
| PC14 | 0.039 |
| PC9 | 0.034 |
| PC17 | 0.031 |

