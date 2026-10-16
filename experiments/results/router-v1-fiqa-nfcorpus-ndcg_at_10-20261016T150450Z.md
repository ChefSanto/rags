# Router v1 (score regression) — fiqa, nfcorpus (ndcg_at_10)

- train split: test (648 queries)
- test split: test (323 queries)
- strategies: bm25, dense, hybrid
- model: GradientBoostingRegressor per strategy (n=200, depth=3, lr=0.05, subsample=0.8)

## Test-split scores

| router | score |
|---|---|
| **routed (v1 regression)** | **0.3238** |
| uniform hybrid | 0.3376 |
| uniform dense | 0.3186 |
| uniform bm25 | 0.3111 |
| domain router {'fiqa': 'hybrid'} | n/a (test domains unseen in train) |
| oracle (ceiling) | 0.3843 |

- routing gain over best uniform (hybrid): **-0.0138**
- vs domain router: n/a
- oracle-gap recovery vs best uniform: -29.5%
- gap remaining to oracle: +0.0605

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.1854 | 0.3111 | 0.2312 |
| dense | 0.1813 | 0.3186 | 0.3687 |
| hybrid | 0.2022 | 0.3376 | 0.3708 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 31.6% |
| dense | 37.8% |
| hybrid | 30.7% |

## Mean feature importances (across strategy regressors)

| feature | importance |
|---|---|
| max_idf | 0.353 |
| avg_idf | 0.342 |
| n_unique | 0.144 |
| n_terms | 0.106 |
| stopword_hits | 0.045 |
| coverage | 0.011 |

