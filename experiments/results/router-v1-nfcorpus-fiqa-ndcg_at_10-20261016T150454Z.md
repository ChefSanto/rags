# Router v1 (score regression) — nfcorpus, fiqa (ndcg_at_10)

- train split: test (323 queries)
- test split: test (648 queries)
- strategies: bm25, dense, hybrid
- model: GradientBoostingRegressor per strategy (n=200, depth=3, lr=0.05, subsample=0.8)

## Test-split scores

| router | score |
|---|---|
| **routed (v1 regression)** | **0.3301** |
| uniform hybrid | 0.3708 |
| uniform dense | 0.3687 |
| uniform bm25 | 0.2312 |
| domain router {'nfcorpus': 'hybrid'} | n/a (test domains unseen in train) |
| oracle (ceiling) | 0.4343 |

- routing gain over best uniform (hybrid): **-0.0407**
- vs domain router: n/a
- oracle-gap recovery vs best uniform: -64.0%
- gap remaining to oracle: +0.1042

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.3586 | 0.2312 | 0.3111 |
| dense | 0.4023 | 0.3687 | 0.3186 |
| hybrid | 0.3745 | 0.3708 | 0.3376 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 29.0% |
| dense | 54.8% |
| hybrid | 16.2% |

## Mean feature importances (across strategy regressors)

| feature | importance |
|---|---|
| avg_idf | 0.486 |
| max_idf | 0.227 |
| coverage | 0.106 |
| n_unique | 0.085 |
| n_terms | 0.071 |
| stopword_hits | 0.026 |

