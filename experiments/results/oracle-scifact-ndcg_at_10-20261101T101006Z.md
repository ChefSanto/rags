# Oracle analysis — scifact (ndcg_at_10)

- judged queries compared: 1109
- strategies: bm25, dense, hybrid

- sanity bm25: computed 0.6688 vs result file 0.6688 [OK]
- sanity dense: computed 0.6570 vs result file 0.657 [OK]
- sanity hybrid: computed 0.7190 vs result file 0.719 [OK]

## Oracle gap

| quantity | value |
|---|---|
| oracle (per-query best) | 0.7719 |
| best fixed strategy | hybrid @ 0.7190 |
| **oracle gap** | **+0.0529 (+7.4%)** |

## Oracle-best share (ties counted for each)

| strategy | share |
|---|---|
| hybrid | 78.1% |
| bm25 | 74.5% |
| dense | 73.4% |

## Head-to-head win rates

| vs \ wins against | bm25 | dense | hybrid |
|---|---|---|---|
| bm25 | — | 23.4% | 9.4% |
| dense | 22.2% | — | 12.7% |
| hybrid | 22.8% | 24.0% | — |

