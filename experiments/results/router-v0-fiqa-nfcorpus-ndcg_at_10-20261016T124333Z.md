# Router v0 — fiqa, nfcorpus (ndcg_at_10)

- train split: train (8090 queries)
- test split: test (971 queries)
- strategies: bm25, dense, hybrid
- model: decision tree (max_depth=4, min_samples_leaf=50)

## Test-split scores

| router | score |
|---|---|
| **routed (decision tree)** | **0.2651** |
| uniform hybrid | 0.3597 |
| uniform dense | 0.3520 |
| uniform bm25 | 0.2578 |
| domain router {'fiqa': 'dense', 'nfcorpus': 'hybrid'} | 0.3583 |
| oracle (ceiling) | 0.4177 |

- routing gain over best uniform (hybrid): **-0.0947**
- gap remaining to oracle: +0.1526
- label accuracy (predicted == oracle-best): 51.1%

## Predicted strategy distribution (test)

| strategy | share |
|---|---|
| bm25 | 96.1% |
| dense | 3.9% |
| hybrid | 0.0% |

## Feature importances

| feature | importance |
|---|---|
| n_terms | 0.412 |
| avg_idf | 0.288 |
| max_idf | 0.135 |
| coverage | 0.125 |
| stopword_hits | 0.041 |
| n_unique | 0.000 |

## Decision tree

```
|--- n_terms <= 4.50
|   |--- avg_idf <= 6.18
|   |   |--- coverage <= 0.58
|   |   |   |--- max_idf <= 3.45
|   |   |   |   |--- class: bm25
|   |   |   |--- max_idf >  3.45
|   |   |   |   |--- class: bm25
|   |   |--- coverage >  0.58
|   |   |   |--- avg_idf <= 3.84
|   |   |   |   |--- class: bm25
|   |   |   |--- avg_idf >  3.84
|   |   |   |   |--- class: bm25
|   |--- avg_idf >  6.18
|   |   |--- avg_idf <= 6.90
|   |   |   |--- avg_idf <= 6.33
|   |   |   |   |--- class: bm25
|   |   |   |--- avg_idf >  6.33
|   |   |   |   |--- class: bm25
|   |   |--- avg_idf >  6.90
|   |   |   |--- avg_idf <= 7.16
|   |   |   |   |--- class: bm25
|   |   |   |--- avg_idf >  7.16
|   |   |   |   |--- class: bm25
|--- n_terms >  4.50
|   |--- max_idf <= 6.87
|   |   |--- max_idf <= 4.12
|   |   |   |--- avg_idf <= 2.34
|   |   |   |   |--- class: bm25
|   |   |   |--- avg_idf >  2.34
|   |   |   |   |--- class: bm25
|   |   |--- max_idf >  4.12
|   |   |   |--- n_terms <= 20.50
|   |   |   |   |--- class: bm25
|   |   |   |--- n_terms >  20.50
|   |   |   |   |--- class: dense
|   |--- max_idf >  6.87
|   |   |--- stopword_hits <= 3.50
|   |   |   |--- max_idf <= 9.88
|   |   |   |   |--- class: bm25
|   |   |   |--- max_idf >  9.88
|   |   |   |   |--- class: bm25
|   |   |--- stopword_hits >  3.50
|   |   |   |--- avg_idf <= 2.87
|   |   |   |   |--- class: dense
|   |   |   |--- avg_idf >  2.87
|   |   |   |   |--- class: bm25
```

