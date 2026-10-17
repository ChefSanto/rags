# Router v2a (emb, PCA-32) — fiqa, nfcorpus (ndcg_at_10)

- train split: test (648 queries)
- test split: test (323 queries)
- strategies: bm25, dense, hybrid
- embedding model: sentence-transformers/all-MiniLM-L6-v2 (same as dense baseline)
- model: PCA + GradientBoostingRegressor per strategy

## Test-split scores

| router | score |
|---|---|
| **routed (v2a)** | **0.3288** |
| uniform hybrid | 0.3380 |
| uniform dense | 0.3186 |
| uniform bm25 | 0.3111 |
| domain router {'fiqa': 'hybrid'} | n/a (test domains unseen in train) |
| oracle (ceiling) | 0.3843 |

- routing gain over best uniform (hybrid): **-0.0092**
- vs domain router: n/a
- oracle-gap recovery vs best uniform: -19.9%
- gap remaining to oracle: +0.0555

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.2115 | 0.3111 | 0.2312 |
| dense | 0.4286 | 0.3186 | 0.3687 |
| hybrid | 0.4301 | 0.3380 | 0.3730 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 0.0% |
| dense | 47.7% |
| hybrid | 52.3% |

- PCA explained variance (first 5): 0.055, 0.041, 0.033, 0.029, 0.028

## Top feature importances (PCA components)

| component | importance |
|---|---|
| PC2 | 0.062 |
| PC32 | 0.045 |
| PC28 | 0.044 |
| PC6 | 0.043 |
| PC19 | 0.038 |
| PC24 | 0.038 |
| PC29 | 0.036 |
| PC4 | 0.035 |
| PC18 | 0.034 |
| PC27 | 0.034 |

