# Oracle analysis — fiqa (ndcg_at_10)

- judged queries compared: 6648
- strategies: bm25, dense, hybrid

- sanity bm25: computed 0.2293 vs result file 0.2293 [OK]
- sanity dense: computed 0.3680 vs result file 0.368 [OK]
- sanity hybrid: computed 0.3610 vs result file 0.361 [OK]

## Oracle gap

| quantity | value |
|---|---|
| oracle (per-query best) | 0.4271 |
| best fixed strategy | dense @ 0.3680 |
| **oracle gap** | **+0.0591 (+16.1%)** |

## Oracle-best share (ties counted for each)

| strategy | share |
|---|---|
| dense | 78.2% |
| hybrid | 61.9% |
| bm25 | 48.9% |

## Head-to-head win rates

| vs \ wins against | bm25 | dense | hybrid |
|---|---|---|---|
| bm25 | — | 14.5% | 7.2% |
| dense | 46.5% | — | 31.3% |
| hybrid | 45.4% | 19.9% | — |

