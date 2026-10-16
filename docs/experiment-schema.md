# Experiment Schema (v1.0)

An experiment is a YAML file in `experiments/configs/`. The runner snapshots the
parsed spec into every result JSON, so results stay reproducible even if the
config file later changes.

```yaml
experiment:
  name: smoke-bm25-v1          # unique, used in result filenames
  description: >-              # free text, copied into results

dataset:
  name: synthetic-smoke        # must exist in datasets/manifest.yaml
  version: "1.0"
  path: datasets/synthetic-smoke
  format: beir                 # currently only "beir"

retrieval:
  method: bm25                 # bm25 | dense | hybrid
  top_k: 10
  params: {}                   # method-specific, passed to the retriever

models:
  embedding: null              # null | "hashing" | sentence-transformers model id
  reranker: null               # reserved for future phases

parameters:
  seed: 42                     # recorded in results; hashing/randomness derive from it

evaluation:
  metrics: [recall_at_5, recall_at_10, mrr_at_10]
  k_values: [5, 10]
```

## Retrieval params

| method | params |
|---|---|
| `bm25` | `k1` (default 1.5), `b` (default 0.75) |
| `dense` | none — embedding model comes from `models.embedding` |
| `hybrid` | `dense_weight` in [0,1] (default 0.5), optional nested `bm25:` and `dense:` overrides |

## Metrics

`recall_at_K`, `precision_at_K`, `mrr_at_K`, `ndcg_at_K` for any integer K.
Bare `mrr` means K=10. See `core/evaluation/retrieval.py` for assumptions and
limitations of each metric.

## Rules

1. Configs are version-controlled and immutable in spirit: to change an
   experiment, create a new config (e.g. `-v2`), don't silently edit.
2. `models.embedding: hashing` is for smoke tests only — never report those
   numbers as research results.
3. `dataset.name`/`version` must resolve in `datasets/manifest.yaml`.
