# Domain-routing baseline — fiqa, nfcorpus (ndcg_at_10)

- combined judged queries: 9885

## Per-domain strategy means

| domain | judged | bm25 | dense | hybrid | best |
|---|---|---|---|---|---|
| fiqa | 6648 | 0.2293 | 0.3680 | 0.3610 | **dense** |
| nfcorpus | 3237 | 0.3013 | 0.3183 | 0.3374 | **hybrid** |

## Routing comparison

| router | combined score |
|---|---|
| domain router (per-domain best) | 0.3580 |
| best uniform: hybrid everywhere | 0.3532 |
| combined oracle (ceiling) | 0.4123 |

**Domain routing beats best uniform by +0.0047 (+1.3%).** The property-based router (H3) must beat the domain router; the remaining gap to the oracle (+0.0543) is what document properties must explain beyond domain membership.

