# Research Checkpoint — 2 November 2026

## Project

Paper: Why Per-Query Retrieval Routing Fails to Beat Tuned Fusion:
A Mechanism-Level Study.

Repository: ChefSanto/rags

Working directory: /kaggle/working/rags

## 1. Research objective

Investigate whether per-query retrieval routing can outperform
well-tuned retrieval fusion, and identify the mechanisms explaining
when routing succeeds or fails.

Primary datasets:
- FiQA
- NFCorpus

## 2. Verified baseline results

Independent test-only evaluation using standard pytrec_eval nDCG@10.

| Dataset | BM25 | Dense | Hybrid |
|---|---:|---:|---:|
| FiQA | 0.231182 | 0.368671 | 0.387637 |
| NFCorpus | 0.311080 | 0.318632 | 0.336759 |

Exact pinned artifacts:

FiQA:
- BM25: fiqa-bm25-v1__20261018T153258Z
- Dense: fiqa-dense-v1__20261018T154116Z
- Hybrid: fiqa-hybrid-w07-v1__20261101T150744Z

NFCorpus:
- BM25: nfcorpus-bm25-v1__20261018T155734Z
- Dense: nfcorpus-dense-v1__20261018T155825Z
- Hybrid: nfcorpus-hybrid-v1__20261018T155922Z

All six artifacts exist and were independently evaluated.
All saved ranking lists contain 10 documents per query.
All test queries were covered.

Important:
- NFCorpus must use standard exponential relevance gains.
- Existing project custom nDCG uses raw relevance grades and is not
  suitable for graded NFCorpus results.
- Existing HybridRetriever and router_v5 weighted RRF are different
  fusion methods and must be reported separately.
- The exact provenance of the pinned fusion weights still needs review.

## 3. Dataset integrity

### FiQA

Location:
datasets/downloads/fiqa

- Documents: 57,638
- Train queries: 5,500
- Dev queries: 500
- Test queries: 648
- No overlap between query splits.
- All document titles are empty.

The router's body-only corpus statistics are consistent with the
BEIR title-plus-body representation for FiQA.

### NFCorpus

Official BEIR archive restored and checksum verified.

Expected MD5:
a89dba18a62ef92f7d323ec890a0d38d

Location:
datasets/downloads/nfcorpus

- Documents: 3,633
- Queries: 3,237
- Train queries: 2,590
- Dev queries: 324
- Test queries: 323
- Train judgments: 110,575
- Dev judgments: 11,385
- Test judgments: 12,334
- No query ID overlap between splits.
- No unknown query or document IDs.
- Restored test qrels exactly match the previously used test qrels.

NFCorpus train judgments have relevance grade 1 only.
Dev and test contain relevance grades 1 and 2.

## 4. Feature extraction audit

BM25 implementation:
core/retrieval/bm25.py

BM25 indexes Document.text, which the BEIR loader constructs
using title plus body.

router_v0.py corpus_stats currently reads raw doc["text"] only.

FiQA:
- All titles empty.
- No title/body mismatch.

NFCorpus:
- All 3,633 documents have non-empty titles.
- 2,905 documents contain title-only terms.
- Body vocabulary: 25,969 unique terms.
- Combined title/body vocabulary: 26,359 unique terms.
- 390 additional unique terms appear through titles.
- 3,043 terms have changed document frequencies.

This confirms a real inconsistency between NFCorpus BM25 indexing
and router feature statistics.

Do not interpret previous NFCorpus router results as fully validated
until this is corrected and the relevant experiments are rerun.

## 5. Other important source-audit findings

- router_v0 dynamically selects latest artifacts. Pin exact artifact IDs.
- router_v0 originally uses merged qrels for per-query scores.
  Prefer explicit split qrels.
- router_v1 uses a test-set oracle as an upper-bound diagnostic.
- Selecting the best uniform strategy on test is optimistic.
- router_v5 uses weighted RRF and should not be confused with the
  existing normalized-score HybridRetriever.
- reviewer_stats.py selects hybrid artifacts using registry order.
  Pin the exact hybrid artifact.
- All NFCorpus graded nDCG calculations must use standard gains.
- Existing artifacts have git_dirty=true. Commit metadata alone
  does not fully reproduce their source state.

## 6. Next actions

1. Correct router corpus statistics to use the same title-plus-body
   text representation as BM25.
2. Verify that the correction does not alter FiQA features.
3. Regenerate corpus statistics and router features.
4. Pin all baseline artifacts explicitly.
5. Ensure train/dev/test qrels are used separately.
6. Tune feature/model choices on dev, not test.
7. Evaluate the final router on test.
8. Evaluate FiQA-trained routing on NFCorpus test as a
   cross-dataset transfer experiment.
9. Recalculate all NFCorpus metrics with standard nDCG@10.
10. Preserve oracle results as diagnostic upper bounds only.

## 7. Current stopping point

Dataset restoration and integrity checks are complete.

The latest completed experiment is the NFCorpus title/body
feature audit.

No router correction or new routing experiment has been run
after this audit.

Resume from the feature-extraction correction, not from
baseline generation.
