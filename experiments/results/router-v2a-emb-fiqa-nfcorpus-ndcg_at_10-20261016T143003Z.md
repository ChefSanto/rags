# Router v2a (emb, PCA-32) — fiqa, nfcorpus (ndcg_at_10)

- train split: test (971 queries)
- test split: test (971 queries)
- strategies: bm25, dense, hybrid
- embedding model: sentence-transformers/all-MiniLM-L6-v2 (same as dense baseline)
- model: PCA + GradientBoostingRegressor per strategy

## Test-split scores

| router | score |
|---|---|
| **routed (v2a)** | **0.4017** |
| uniform hybrid | 0.3597 |
| uniform dense | 0.3520 |
| uniform bm25 | 0.2578 |
| domain router {'fiqa': 'hybrid', 'nfcorpus': 'hybrid'} | 0.3597 |
| oracle (ceiling) | 0.4177 |

- routing gain over best uniform (hybrid): **+0.0420**
- vs domain router: +0.0420
- oracle-gap recovery vs best uniform: 72.5%
- gap remaining to oracle: +0.0159

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.2584 | 0.2578 | 0.2578 |
| dense | 0.3532 | 0.3520 | 0.3520 |
| hybrid | 0.3601 | 0.3597 | 0.3597 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 11.1% |
| dense | 44.8% |
| hybrid | 44.1% |

- PCA explained variance (first 5): 0.061, 0.036, 0.027, 0.024, 0.021

## Top feature importances (PCA components)

| component | importance |
|---|---|
| PC5 | 0.058 |
| PC6 | 0.049 |
| PC3 | 0.048 |
| PC19 | 0.041 |
| PC15 | 0.041 |
| PC32 | 0.038 |
| PC1 | 0.037 |
| PC8 | 0.036 |
| PC30 | 0.034 |
| PC29 | 0.034 |

