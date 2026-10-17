# Router v1 (score regression) — nfcorpus, fiqa (ndcg_at_10)

- train split: test (323 queries)
- test split: test (648 queries)
- strategies: bm25, dense, hybrid
- model: GradientBoostingRegressor per strategy (n=200, depth=3, lr=0.05, subsample=0.8)

## Test-split scores

| router | score |
|---|---|
| **routed (v1 regression)** | **0.3321** |
| uniform hybrid | 0.3730 |
| uniform dense | 0.3687 |
| uniform bm25 | 0.2312 |
| domain router {'nfcorpus': 'hybrid'} | n/a (test domains unseen in train) |
| oracle (ceiling) | 0.4337 |

- routing gain over best uniform (hybrid): **-0.0410**
- vs domain router: n/a
- oracle-gap recovery vs best uniform: -67.5%
- gap remaining to oracle: +0.1016

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.3586 | 0.2312 | 0.3111 |
| dense | 0.4023 | 0.3687 | 0.3186 |
| hybrid | 0.3741 | 0.3730 | 0.3380 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 27.2% |
| dense | 55.6% |
| hybrid | 17.3% |

## Mean feature importances (across strategy regressors)

| feature | importance |
|---|---|
| avg_idf | 0.479 |
| max_idf | 0.228 |
| coverage | 0.108 |
| n_unique | 0.089 |
| n_terms | 0.069 |
| stopword_hits | 0.027 |

