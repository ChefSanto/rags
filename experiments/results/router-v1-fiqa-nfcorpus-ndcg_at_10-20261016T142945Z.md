# Router v1 (score regression) — fiqa, nfcorpus (ndcg_at_10)

- train split: test (971 queries)
- test split: test (971 queries)
- strategies: bm25, dense, hybrid
- model: GradientBoostingRegressor per strategy (n=200, depth=3, lr=0.05, subsample=0.8)

## Test-split scores

| router | score |
|---|---|
| **routed (v1 regression)** | **0.3908** |
| uniform hybrid | 0.3597 |
| uniform dense | 0.3520 |
| uniform bm25 | 0.2578 |
| domain router {'fiqa': 'hybrid', 'nfcorpus': 'hybrid'} | 0.3597 |
| oracle (ceiling) | 0.4177 |

- routing gain over best uniform (hybrid): **+0.0310**
- vs domain router: +0.0310
- oracle-gap recovery vs best uniform: 53.6%
- gap remaining to oracle: +0.0269

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.2573 | 0.2578 | 0.2578 |
| dense | 0.3518 | 0.3520 | 0.3520 |
| hybrid | 0.3596 | 0.3597 | 0.3597 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 11.0% |
| dense | 41.1% |
| hybrid | 47.9% |

## Mean feature importances (across strategy regressors)

| feature | importance |
|---|---|
| avg_idf | 0.386 |
| max_idf | 0.310 |
| n_unique | 0.130 |
| n_terms | 0.108 |
| coverage | 0.033 |
| stopword_hits | 0.033 |

