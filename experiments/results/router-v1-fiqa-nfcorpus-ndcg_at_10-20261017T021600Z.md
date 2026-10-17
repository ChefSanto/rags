# Router v1 (score regression) — fiqa, nfcorpus (ndcg_at_10)

- train split: test (648 queries)
- test split: test (323 queries)
- strategies: bm25, dense, hybrid
- model: GradientBoostingRegressor per strategy (n=200, depth=3, lr=0.05, subsample=0.8)

## Test-split scores

| router | score |
|---|---|
| **routed (v1 regression)** | **0.3215** |
| uniform hybrid | 0.3380 |
| uniform dense | 0.3186 |
| uniform bm25 | 0.3111 |
| domain router {'fiqa': 'hybrid'} | n/a (test domains unseen in train) |
| oracle (ceiling) | 0.3843 |

- routing gain over best uniform (hybrid): **-0.0165**
- vs domain router: n/a
- oracle-gap recovery vs best uniform: -35.7%
- gap remaining to oracle: +0.0628

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.1854 | 0.3111 | 0.2312 |
| dense | 0.1813 | 0.3186 | 0.3687 |
| hybrid | 0.2026 | 0.3380 | 0.3730 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 31.3% |
| dense | 34.7% |
| hybrid | 34.1% |

## Mean feature importances (across strategy regressors)

| feature | importance |
|---|---|
| max_idf | 0.358 |
| avg_idf | 0.340 |
| n_unique | 0.140 |
| n_terms | 0.107 |
| stopword_hits | 0.045 |
| coverage | 0.011 |

