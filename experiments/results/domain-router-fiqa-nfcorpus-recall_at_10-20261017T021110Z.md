# Domain-routing baseline — fiqa, nfcorpus (recall_at_10)

- combined judged queries: 9885

## Per-domain strategy means

| domain | judged | bm25 | dense | hybrid | best |
|---|---|---|---|---|---|
| fiqa | 6648 | 0.4499 | 0.6369 | 0.6295 | **dense** |
| nfcorpus | 3237 | 0.6747 | 0.6991 | 0.7244 | **hybrid** |

## Routing comparison

| router | combined score |
|---|---|
| domain router (per-domain best) | 0.6656 |
| best uniform: hybrid everywhere | 0.6606 |
| combined oracle (ceiling) | 0.7177 |

**Domain routing beats best uniform by +0.0050 (+0.8%).** The property-based router (H3) must beat the domain router; the remaining gap to the oracle (+0.0521) is what document properties must explain beyond domain membership.

