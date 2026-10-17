# Router v1 (score regression) — fiqa, nfcorpus (ndcg_at_10)

- train split: test (971 queries)
- test split: test (971 queries)
- strategies: bm25, dense, hybrid
- model: GradientBoostingRegressor per strategy (n=200, depth=3, lr=0.05, subsample=0.8)

## Test-split scores

| router | score |
|---|---|
| **routed (v1 regression)** | **0.3914** |
| uniform hybrid | 0.3614 |
| uniform dense | 0.3520 |
| uniform bm25 | 0.2578 |
| domain router {'fiqa': 'hybrid', 'nfcorpus': 'hybrid'} | 0.3614 |
| oracle (ceiling) | 0.4173 |

- routing gain over best uniform (hybrid): **+0.0300**
- vs domain router: +0.0300
- oracle-gap recovery vs best uniform: 53.8%
- gap remaining to oracle: +0.0258

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.2573 | 0.2578 | 0.2578 |
| dense | 0.3518 | 0.3520 | 0.3520 |
| hybrid | 0.3612 | 0.3614 | 0.3614 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 8.9% |
| dense | 42.7% |
| hybrid | 48.4% |

## Mean feature importances (across strategy regressors)

| feature | importance |
|---|---|
| avg_idf | 0.383 |
| max_idf | 0.314 |
| n_unique | 0.131 |
| n_terms | 0.106 |
| stopword_hits | 0.033 |
| coverage | 0.033 |

