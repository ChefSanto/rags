# NFCorpus Transfer Evaluation — 2026-11-03

## Recovery note

The previous Kaggle runtime was reset, and the original NFCorpus
experiment result and ranking files were lost from its ephemeral storage.

The experiments below were rerun from the official dataset and the
same tracked repository commit. These are regenerated artifacts, not
the original files.

## Dataset integrity

- Dataset: BEIR NFCorpus
- Official archive MD5: `a89dba18a62ef92f7d323ec890a0d38d`
- Corpus documents: 3,633
- Queries: 3,237
- Train: 110,575 judgments / 2,590 queries
- Dev: 11,385 judgments / 324 queries
- Test: 12,334 judgments / 323 queries
- Train/dev/test query overlap: zero
- Active `qrels.tsv`: dev split only
- Dev relevance levels: 1 and 2

## Experiment setup

- Repository commit: `a8dac947fd2196da8713de81532675bfda78a999`
- Retriever: hybrid BM25 + dense
- BM25 k1: 1.5
- BM25 b: 0.75
- Dense encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Top-k: 10
- Seed: 42
- Dataset: NFCorpus
- Evaluation split: official dev only
- Both runs generated rankings for all 3,237 dataset queries.

## Regenerated experiments

| Setting | Weight 0.5 | Weight 0.6 |
|---|---:|---:|
| Recall@5 | 0.6451 | 0.6451 |
| Recall@10 | 0.6944 | 0.7006 |
| Precision@10 | 0.2522 | 0.2512 |
| MRR@10 | 0.5216 | 0.5306 |
| nDCG@10 | 0.3178 | 0.3181 |

Experiment IDs:

- Weight 0.5: `nfcorpus-transfer-w05-dev-v1__20261103T062949Z`
- Weight 0.6: `nfcorpus-transfer-w06-dev-v1__20261103T063219Z`

## Paired query-level bootstrap

- Queries: 324
- Resamples: 10,000
- Random seed: 42
- Mean nDCG@10 at weight 0.5: 0.317823
- Mean nDCG@10 at weight 0.6: 0.318117
- Difference (0.6 minus 0.5): 0.000294
- 95% bootstrap CI: [-0.003908, 0.004438]
- Improved queries: 61
- Worsened queries: 60
- Tied queries: 203

The confidence interval crosses zero. There is no clear evidence that
weight 0.6 outperforms weight 0.5 on NFCorpus dev.

This bootstrap measures query-level uncertainty, not variation across
random seeds.

## Interpretation

The FiQA-selected weight of 0.6 does not demonstrate a measurable
improvement over the 0.5 baseline on NFCorpus dev.

This is evidence of dataset-dependent fusion behavior. It does not
establish that 0.5 is universally optimal.

The NFCorpus test split was inspected in earlier work and must not be
described as an untouched confirmatory holdout. No test results were
used in this transfer comparison.

## Reproducibility limitations

Both experiment records report `git_dirty: true`. The tracked
repository commit is identical, and no tracked source/config changes
were made for these runs. The dirty state reflects generated,
untracked runtime artifacts.

The unauthenticated Hugging Face Hub warning appeared, but both
experiments completed and loaded the model weights successfully.

## Artifact integrity

- `experiments/results/nfcorpus-transfer-w05-dev-v1__20261103T062949Z.json` — SHA256: `b3797155ac35d4a40e0b72e889e75b4fefb90e3fbf2c02cc506cf7a8a10d9fb7`
- `artifacts/runs/nfcorpus-transfer-w05-dev-v1__20261103T062949Z.json` — SHA256: `f5213de7cf03ed341325be14af95864f76a46297b4c7c8602a002b82f89cdb78`
- `experiments/results/nfcorpus-transfer-w06-dev-v1__20261103T063219Z.json` — SHA256: `02055f1d9e8ef23532a9f317563a4af503e1c6639e266afa65af8c9737ed2ea1`
- `artifacts/runs/nfcorpus-transfer-w06-dev-v1__20261103T063219Z.json` — SHA256: `69c2bf4d85ce3885b9d9937ec89ed18868bdaf139008db86ef1c72335a3f03b2`
