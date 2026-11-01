# Router v1 (score regression) — scifact (ndcg_at_10)

- train split: train (809 queries)
- test split: test (300 queries)
- strategies: bm25, dense, hybrid
- model: GradientBoostingRegressor per strategy (n=200, depth=3, lr=0.05, subsample=0.8)

## Test-split scores

| router | score |
|---|---|
| **routed (v1 regression)** | **0.6882** |
| uniform hybrid | 0.7160 |
| uniform bm25 | 0.6659 |
| uniform dense | 0.6484 |
| domain router {'scifact': 'hybrid'} | 0.7160 |
| oracle (ceiling) | 0.7648 |

- routing gain over best uniform (hybrid): **-0.0277**
- vs domain router: -0.0277
- oracle-gap recovery vs best uniform: -56.8%
- gap remaining to oracle: +0.0766

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.6710 | 0.6659 | 0.6698 |
| dense | 0.6595 | 0.6484 | 0.6602 |
| hybrid | 0.7217 | 0.7160 | 0.7201 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 24.0% |
| dense | 19.0% |
| hybrid | 57.0% |

## Mean feature importances (across strategy regressors)

| feature | importance |
|---|---|
| avg_idf | 0.421 |
| max_idf | 0.228 |
| n_unique | 0.113 |
| n_terms | 0.090 |
| coverage | 0.083 |
| stopword_hits | 0.064 |

