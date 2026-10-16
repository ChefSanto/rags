# Router v1 (score regression) — fiqa, nfcorpus (ndcg_at_10)

- train split: train (8090 queries)
- test split: test (971 queries)
- strategies: bm25, dense, hybrid
- model: GradientBoostingRegressor per strategy (n=200, depth=3, lr=0.05, subsample=0.8)

## Test-split scores

| router | score |
|---|---|
| **routed (v1 regression)** | **0.3626** |
| uniform hybrid | 0.3597 |
| uniform dense | 0.3520 |
| uniform bm25 | 0.2578 |
| domain router {'fiqa': 'dense', 'nfcorpus': 'hybrid'} | 0.3583 |
| oracle (ceiling) | 0.4177 |

- routing gain over best uniform (hybrid): **+0.0028**
- vs domain router: +0.0042
- oracle-gap recovery vs best uniform: 4.9%
- gap remaining to oracle: +0.0551

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.2482 | 0.2578 | 0.2526 |
| dense | 0.3488 | 0.3520 | 0.3514 |
| hybrid | 0.3494 | 0.3597 | 0.3519 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 4.3% |
| dense | 56.0% |
| hybrid | 39.6% |

## Mean feature importances (across strategy regressors)

| feature | importance |
|---|---|
| avg_idf | 0.323 |
| max_idf | 0.300 |
| coverage | 0.135 |
| n_terms | 0.121 |
| stopword_hits | 0.065 |
| n_unique | 0.056 |

