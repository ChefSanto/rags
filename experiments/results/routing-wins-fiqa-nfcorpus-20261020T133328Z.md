# When does routing help? — fiqa, nfcorpus

| domain | test q | oracle wins | mean gain (win) | mean loss (rest) | break-even p* |
|---|---|---|---|---|---|
| fiqa | 648 | 356 (54.9%) | +0.3444 | -0.3388 | 49.6% |
| nfcorpus | 323 | 160 (49.5%) | +0.3280 | -0.2734 | 45.5% |

## Per-domain detail

### fiqa

- test queries: 648; best uniform: hybrid @ 0.3711
- **oracle wins on 356 queries (54.9%)**, mean gain on those: +0.3444 nDCG
- oracle-best winner distribution on win queries: bm25 100, dense 193, hybrid 63
- mean strategy margin (top1-top2) on win queries: 0.106 vs all: 0.075

| query property | win queries | others |
|---|---|---|
| n_terms | 11.346 | 11.003 |
| n_unique | 10.846 | 10.548 |
| coverage | 0.999 | 0.997 |
| avg_idf | 2.906 | 2.722 |
| max_idf | 6.514 | 5.946 |
| stopword_hits | 2.163 | 2.339 |

- break-even pick accuracy: **p* = 49.6%** — a router must correctly identify winning queries (and pick their best strategy) 50% of the time, given the losses on the rest, to merely tie the uniform baseline.

### nfcorpus

- test queries: 323; best uniform: hybrid @ 0.3368
- **oracle wins on 160 queries (49.5%)**, mean gain on those: +0.3280 nDCG
- oracle-best winner distribution on win queries: bm25 76, dense 61, hybrid 23
- mean strategy margin (top1-top2) on win queries: 0.072 vs all: 0.051

| query property | win queries | others |
|---|---|---|
| n_terms | 3.356 | 3.350 |
| n_unique | 3.325 | 3.307 |
| coverage | 0.984 | 0.805 |
| avg_idf | 4.350 | 3.444 |
| max_idf | 5.848 | 5.128 |
| stopword_hits | 0.425 | 0.442 |

- break-even pick accuracy: **p* = 45.5%** — a router must correctly identify winning queries (and pick their best strategy) 45% of the time, given the losses on the rest, to merely tie the uniform baseline.

