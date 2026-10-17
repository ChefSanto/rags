# Oracle analysis — fiqa (recall_at_10)

- judged queries compared: 6648
- strategies: bm25, dense, hybrid

- sanity bm25: computed 0.4499 vs result file 0.4499 [OK]
- sanity dense: computed 0.6369 vs result file 0.6369 [OK]
- sanity hybrid: computed 0.6295 vs result file 0.6295 [OK]

## Oracle gap

| quantity | value |
|---|---|
| oracle (per-query best) | 0.6934 |
| best fixed strategy | dense @ 0.6369 |
| **oracle gap** | **+0.0566 (+8.9%)** |

## Oracle-best share (ties counted for each)

| strategy | share |
|---|---|
| dense | 94.3% |
| hybrid | 93.6% |
| bm25 | 75.6% |

## Head-to-head win rates

| vs \ wins against | bm25 | dense | hybrid |
|---|---|---|---|
| bm25 | — | 5.5% | 1.6% |
| dense | 24.2% | — | 4.9% |
| hybrid | 19.5% | 4.2% | — |

