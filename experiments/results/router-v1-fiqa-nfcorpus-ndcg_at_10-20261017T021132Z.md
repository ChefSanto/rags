# Router v1 (score regression) — fiqa, nfcorpus (ndcg_at_10)

- train split: train (8090 queries)
- test split: test (971 queries)
- strategies: bm25, dense, hybrid
- model: GradientBoostingRegressor per strategy (n=200, depth=3, lr=0.05, subsample=0.8)

## Test-split scores

| router | score |
|---|---|
| **routed (v1 regression)** | **0.3625** |
| uniform hybrid | 0.3614 |
| uniform dense | 0.3520 |
| uniform bm25 | 0.2578 |
| domain router {'fiqa': 'dense', 'nfcorpus': 'hybrid'} | 0.3585 |
| oracle (ceiling) | 0.4173 |

- routing gain over best uniform (hybrid): **+0.0012**
- vs domain router: +0.0041
- oracle-gap recovery vs best uniform: 2.1%
- gap remaining to oracle: +0.0547

## Prediction diagnostics (test)

| strategy | predicted mean | actual mean | train-split mean |
|---|---|---|---|
| bm25 | 0.2482 | 0.2578 | 0.2526 |
| dense | 0.3488 | 0.3520 | 0.3514 |
| hybrid | 0.3489 | 0.3614 | 0.3518 |

## Predicted routing distribution (test)

| strategy | share |
|---|---|
| bm25 | 4.5% |
| dense | 56.8% |
| hybrid | 38.6% |

## Mean feature importances (across strategy regressors)

| feature | importance |
|---|---|
| avg_idf | 0.327 |
| max_idf | 0.298 |
| coverage | 0.134 |
| n_terms | 0.119 |
| stopword_hits | 0.065 |
| n_unique | 0.057 |

