# Oracle analysis — nfcorpus (recall_at_10)

- judged queries compared: 3237
- strategies: bm25, dense, hybrid

- sanity bm25: computed 0.6747 vs result file 0.6747 [OK]
- sanity dense: computed 0.6991 vs result file 0.6991 [OK]
- sanity hybrid: computed 0.7244 vs result file 0.7244 [OK]

## Oracle gap

| quantity | value |
|---|---|
| oracle (per-query best) | 0.7674 |
| best fixed strategy | hybrid @ 0.7244 |
| **oracle gap** | **+0.0429 (+5.9%)** |

## Oracle-best share (ties counted for each)

| strategy | share |
|---|---|
| hybrid | 95.7% |
| dense | 93.2% |
| bm25 | 90.7% |

## Head-to-head win rates

| vs \ wins against | bm25 | dense | hybrid |
|---|---|---|---|
| bm25 | — | 6.8% | 2.2% |
| dense | 9.2% | — | 2.3% |
| hybrid | 7.2% | 4.9% | — |

