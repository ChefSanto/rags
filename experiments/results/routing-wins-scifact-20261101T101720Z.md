# When does routing help? — scifact

| domain | test q | oracle wins | mean gain (win) | mean loss (rest) | break-even p* |
|---|---|---|---|---|---|
| scifact | 300 | 197 (65.7%) | +0.2787 | -0.5299 | 65.5% |

## Per-domain detail

### scifact

- test queries: 300; best uniform: hybrid @ 0.7160
- **oracle wins on 197 queries (65.7%)**, mean gain on those: +0.2787 nDCG
- oracle-best winner distribution on win queries: bm25 156, dense 32, hybrid 9
- mean strategy margin (top1-top2) on win queries: 0.047 vs all: 0.058

| query property | win queries | others |
|---|---|---|
| n_terms | 12.995 | 13.175 |
| n_unique | 12.447 | 12.233 |
| coverage | 0.976 | 0.971 |
| avg_idf | 3.167 | 2.935 |
| max_idf | 6.994 | 6.609 |
| stopword_hits | 2.355 | 2.534 |

- break-even pick accuracy: **p* = 65.5%** — a router must correctly identify winning queries (and pick their best strategy) 66% of the time, given the losses on the rest, to merely tie the uniform baseline.

