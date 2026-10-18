# Router v3 (per-corpus pseudo-labeled) — fiqa, nfcorpus (ndcg_at_10)

- embedding: sentence-transformers/all-MiniLM-L6-v2, PCA-32
- model: GradientBoostingRegressor per strategy, trained per-corpus

## Summary

| domain | test queries | routed | best uniform | oracle | recovery |
|---|---|---|---|---|---|
| fiqa | 648 | 0.3599 | 0.3711 | 0.4338 | -17.7% |
| nfcorpus | 323 | 0.3207 | 0.3368 | 0.3842 | -33.9% |
| **weighted** | 971 | **0.3469** | 0.3597 | 0.4173 | -22.2% |

## Per-domain detail

### fiqa

- train: 15000 pseudo-queries (from pseudo-fiqa)
- test: 648 real test-split queries
- pseudo-label means: bm25 0.6188, dense 0.5065, hybrid 0.6650

| router | score |
|---|---|
| **routed (v3, per-corpus pseudo-trained)** | **0.3599** |
| uniform hybrid | 0.3711 |
| uniform dense | 0.3687 |
| uniform bm25 | 0.2312 |
| pseudo-mean domain router | 0.6650 |
| oracle (ceiling) | 0.4338 |

- gain over best uniform (hybrid): **-0.0111**
- oracle-gap recovery: -17.7%
- predicted distribution: bm25 9.1%, dense 0.2%, hybrid 90.7%

### nfcorpus

- train: 7264 pseudo-queries (from pseudo-nfcorpus)
- test: 323 real test-split queries
- pseudo-label means: bm25 0.9192, dense 0.7450, hybrid 0.9027

| router | score |
|---|---|
| **routed (v3, per-corpus pseudo-trained)** | **0.3207** |
| uniform hybrid | 0.3368 |
| uniform dense | 0.3186 |
| uniform bm25 | 0.3111 |
| pseudo-mean domain router | 0.9192 |
| oracle (ceiling) | 0.3842 |

- gain over best uniform (hybrid): **-0.0161**
- oracle-gap recovery: -33.9%
- predicted distribution: bm25 51.1%, dense 1.5%, hybrid 47.4%

