# Router v5 (soft routing) — fiqa, nfcorpus (ndcg_at_10, emb)

## Weighted summary

| domain | test q | best uniform | learned soft | heuristic soft | oracle soft |
|---|---|---|---|---|---|
| fiqa | 648 | 0.3695 | 0.2691 | 0.3683 | 0.4413 |
| nfcorpus | 323 | 0.3414 | 0.3340 | 0.3454 | 0.3884 |
| **weighted** | 971 | **0.3602** | **0.2907** | 0.3607 | 0.4237 |

Hard-oracle ceiling reference (earlier reports): 0.4177.

## Per-domain detail

### fiqa

- train queries: 5500, test queries: 648

| router | test nDCG@10 |
|---|---|
| uniform bm25 (w=0) | 0.2312 |
| uniform dense (w=1) | 0.3687 |
| **best uniform (w=0.7)** | **0.3695** |
| heuristic soft (predicted scores) | 0.3683 |
| **learned soft (w* regression)** | **0.2691** |
| oracle soft (per-query w*) | 0.4413 |

- uniform sweep: w=0.0: 0.2312, w=0.1: 0.2619, w=0.2: 0.2642, w=0.3: 0.2673, w=0.4: 0.2714, w=0.5: 0.3574, w=0.6: 0.3686, w=0.7: 0.3695, w=0.8: 0.3689, w=0.9: 0.3694, w=1.0: 0.3687
- predicted mean weight (test): 0.323

### nfcorpus

- train queries: 2590, test queries: 323

| router | test nDCG@10 |
|---|---|
| uniform bm25 (w=0) | 0.3185 |
| uniform dense (w=1) | 0.3186 |
| **best uniform (w=0.5)** | **0.3414** |
| heuristic soft (predicted scores) | 0.3454 |
| **learned soft (w* regression)** | **0.3340** |
| oracle soft (per-query w*) | 0.3884 |

- uniform sweep: w=0.0: 0.3185, w=0.1: 0.3302, w=0.2: 0.3309, w=0.3: 0.3306, w=0.4: 0.3300, w=0.5: 0.3414, w=0.6: 0.3325, w=0.7: 0.3293, w=0.8: 0.3287, w=0.9: 0.3261, w=1.0: 0.3186
- predicted mean weight (test): 0.272

