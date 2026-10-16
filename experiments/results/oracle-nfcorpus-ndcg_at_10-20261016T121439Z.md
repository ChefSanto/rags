# Oracle analysis — nfcorpus (ndcg_at_10)

- judged queries compared: 3237
- strategies: bm25, dense, hybrid

- sanity bm25: computed 0.3013 vs result file 0.3013 [OK]
- sanity dense: computed 0.3183 vs result file 0.3183 [OK]
- sanity hybrid: computed 0.3374 vs result file 0.3374 [OK]

## Oracle gap

| quantity | value |
|---|---|
| oracle (per-query best) | 0.3818 |
| best fixed strategy | hybrid @ 0.3374 |
| **oracle gap** | **+0.0444 (+13.2%)** |

## Oracle-best share (ties counted for each)

| strategy | share |
|---|---|
| dense | 60.3% |
| hybrid | 55.5% |
| bm25 | 55.1% |

## Head-to-head win rates

| vs \ wins against | bm25 | dense | hybrid |
|---|---|---|---|
| bm25 | — | 31.3% | 21.9% |
| dense | 37.7% | — | 25.7% |
| hybrid | 40.4% | 34.2% | — |

