# Property-Based Retrieval Routing — Research Proposal (draft v0.1)

> Can measurable document properties be used to select the retrieval strategy
> that maximizes RAG performance?

Status: hypothesis. This document is a protocol sketch, not a conclusion.

## 1. Motivation

Fixed retrieval strategies behave differently across document types: lexical
methods excel on terminology-dense text; dense methods excel on paraphrase-like
queries; tables/figures-heavy documents break both. If cheap, measurable
document properties predict which strategy wins *per document*, a router can
beat every fixed baseline without training a heavyweight model.

## 2. Hypotheses

- **H1**: No single fixed strategy (BM25 / dense / hybrid) dominates across
  document categories (finance reports, scientific papers, tech docs,
  tables-heavy, figures-heavy, text-heavy).
- **H2**: An oracle per-document strategy selector ("oracle routing") achieves
  significantly higher aggregate Recall@10 / nDCG@10 than the best fixed
  strategy. (Upper bound — if H2 fails, stop.)
- **H3**: A property-based router using only profiler-v0 features (pages,
  text_density, tables, figures, image_density, scanned_ratio) recovers a
  significant fraction of the oracle gap (target: ≥70%) using a simple
  decision rule (gradient boosting / small decision tree).
- **H4 (domain routing baseline)**: Domain/category metadata alone routes
  worse than property-based routing — properties carry information beyond
  domain labels.

## 3. Method

1. **Corpus with properties.** Assemble a multi-category corpus; profile every
   document with `core/profiling/document.py` (record `profiler_version`).
2. **Per-document strategy scores.** Run each fixed strategy over the full
   query set; per document, each strategy gets a quality score (e.g. recall
   contribution of that document's relevant queries).
3. **Oracle** = per-document argmax strategy. **Router** = classifier mapping
   property vector → strategy, trained on a train split, evaluated on a held-out
   split and, ideally, held-out *categories*.
4. **Metrics**: Recall@5/10, nDCG@10 primary; MRR secondary; per-category
   breakdowns mandatory; macro-average over categories reported alongside
   micro-average.
5. **Statistics**: paired bootstrap CIs on the aggregate difference
   (router − best-fixed); per-category differences with Holm correction.

## 4. Datasets (anti-cherry-picking rules)

- Minimum 4 document categories from at least 3 sources, fixed **before**
  looking at routing results.
- Initial: FiQA (finance), NFCorpus (science), arXiv PDFs (figures/tables-heavy
  science), technical documentation (e.g. a docs corpus).
- Categories must be *document-property-based*, not just topic-based: e.g.
  "tables-heavy" defined by profiler thresholds, so finance docs can appear in
  both "text-heavy finance" and "tables-heavy finance" strata.
- Any dataset added after seeing results must be reported as post-hoc.

## 5. Failure analysis (roadmap §11)

Every query where the router's chosen strategy underperforms the oracle choice
gets classified: retrieval failure / ranking failure / routing failure (chosen
strategy was top-1 but still failed = not routing's fault) / multimodal
interpretation failure. Error taxonomy feeds Paper 4.

## 6. Risks

- **Profiler signal too weak** → extend profiler (OCR ratio, table structure)
  and rerun; version changes are cheap by design.
- **Category leakage** → property thresholds stratify within domains to
  decorrelate domain from property effects (H4 controls for this directly).
- **Query set bias** → BEIR qrels are human-judged but small; report
  per-query variance, not just means.

  ## 7. Results so far (Phase 4–5, 2026-10-16)

Baselines: all-MiniLM-L6-v2 dense, own BM25 (inverted index, max_df skip),
min-max score fusion hybrid (dense_weight 0.5). Evaluation: BEIR all-splits
(train+dev+test merged), judged queries only, k=10. Canonical runs on Kaggle;
artifacts in repo `experiments/results/`.

**Per-dataset baselines (nDCG@10):**

| domain | BM25 | dense | hybrid | best fixed |
|---|---|---|---|---|
| FiQA (finance, 6648 judged) | 0.2293 | 0.3680 | 0.3610 | dense |
| NFCorpus (science, 3237 judged) | 0.3013 | 0.3183 | 0.3374 | hybrid |

- **H1 supported**: best fixed strategy flips across domains (dense wins
  finance, hybrid wins science); BM25's head-to-head win rate vs dense nearly
  doubles (14.5% → 31.3%) on terminology-heavy science.
- **H2 supported on both datasets**: per-query oracle gap +16.1% (FiQA,
  0.4271 vs 0.3680) and +13.2% (NFCorpus, 0.3818 vs 0.3374) over best fixed.
  Recall@10 oracle gaps: +8.9% / +5.9%. Routing signal is not domain-specific.
- **H4 baseline measured (domain router)**: routing by domain label alone
  buys only +1.3% nDCG (+0.8% recall) over the best uniform strategy
  (0.3580 vs 0.3532; combined oracle ceiling 0.4123). Domain membership
  explains almost none of the oracle headroom.
- **The prize for H3**: the +5.4% nDCG gap between the domain router and
  the combined oracle is what document/query properties must explain beyond
  domain membership. First router (v0): per-query features → interpretable
  decision tree, trained on train split, evaluated on test split.

  - **H3 first test (router v0, query features -> decision tree): NEGATIVE.**
  Routed test nDCG 0.2651 vs best uniform 0.3597; tree collapsed to BM25
  (96% of predictions). Mechanisms: (1) argmax labels with alphabetical
  tie-break bias toward bm25 under dense train-split annotations; (2)
  train/test query distribution shift (train splits favor lexical
  matching); (3) classification ignores score margins, so routing regret
  is unmodeled. Conclusion: v0's framing rejected; v1 uses per-strategy
  score regression with predicted-argmax routing.

- **H3 second test (router v1, per-strategy score regression): marginal.**
  Routed 0.3626 vs best uniform 0.3597 (+0.0028) and domain router
  0.3583 (+0.0042), but only 4.9% oracle-gap recovery. Predictions are
  calibrated at the mean (dense ~ hybrid > bm25) but per-query deviations
  — the actual routing signal — are unpredictable from lexical-statistical
  query features. Conclusion: this feature family is ruled out; next
  candidates are query embeddings (v2a) and per-document properties (v2b).

  - **H3 third test (router v2a, query-embedding features): negative.**
  Pure-embedding routing recovered 1.4% of the oracle gap (vs v1 lexical
  4.9%); emb+lex 4.6% but confounded by unscaled PCA. Two independent
  query-side feature families converge on the same ceiling: strategy
  levels predictable, per-query deviations not. Next: in-sample
  diagnostic to separate "signal invisible to query features" from
  "irreducible noise"; then document-side properties (v2b).

- **In-sample diagnostic: signal exists, shift is the failure.**
  Training and testing on the same split recovers 53.6% (v1 lexical) and
  72.5% (v2a embedding) of the oracle gap — versus 4.9% / 1.4% with honest
  train->test splits. The routing signal is present in query features
  (embeddings > lexical) but BEIR train splits are not test-distributed.
  Open question: test-distributed training labels (pseudo-labeled queries
  per corpus) — router v3.

  - **Transfer study: the routing signal is local, not universal.**
  Cross-domain transfer at test distribution is negative (v2a embedding:
  -11.5% / -25.0% oracle recovery fiqa<->nfcorpus), with cross-domain
  score miscalibration as the mechanism. Combined with in-sample
  (53.6-72.5%) vs cross-split (1.4-4.9%) results: routing rules do not
  transfer across splits or corpora. Consequence: routers must be
  trained per-corpus on test-distributed data -> router v3
  (pseudo-labeled queries).

- **Generation stage live (working RAG, 2026-10-17).** End-to-end
  query -> BM25(top-5) -> Qwen2.5-3B-Instruct -> grounded answers on
  100 FiQA queries. First answer-level metric: relevance-weighted
  grounding 0.4467 (MiniLM cosine, no reference point yet). Qualitative:
  ~1/3 coherent grounded answers, ~1/3 prompt-echo, ~1/3 corpus-noise
  propagation (FiQA is partly noisy source text; 3B model amplifies).
  Grounding comparison across retrieval strategies is the next
  experiment — first RAG-level evidence for the routing thesis.

  - **Answer-level evidence for routing (2026-10-17).** Working RAG
  (BM25/dense/hybrid top-5 -> Qwen2.5-3B) on 100 FiQA queries:
  answer grounding 0.501 / 0.574 / 0.631 tracks retrieval nDCG
  (0.206 / 0.322 / 0.334) with preserved ordering. Retrieval strategy
  choice propagates to answer quality. Grounding is a proxy metric;
  correctness validation (judge model / human sample) is future work.
  Batched-generation left-padding bug found via transformers warning and
  fixed (BM25 grounding +0.054 after fix).

- **H3 fourth test (router v3, per-corpus pseudo-labeled): negative, with
  mechanism.** Routed 0.2603 vs best uniform 0.3616 (recovery -180.8%).
  Diagnosis: document-derived pseudo-queries are lexical copies of their
  source docs, so pseudo-label means collapse to BM25 (0.94-0.96) and the
  trained router routes 95-98% to BM25. Conclusion: pseudo-labels must be
  question-distributed, not document-literal. Next: v4 uses an LLM
  (Qwen2.5-3B) to generate natural questions per document, then the same
  per-corpus pipeline.

  - **H3 final test (router v4, LLM-generated question pseudo-labels): much
  improved, still short of best-uniform.** Weighted routed 0.3469 vs best
  uniform 0.3597 (recovery -22.2%, vs v3's -180.8%). Pseudo-label means
  compressed (hybrid now competitive), routing distribution plausible. The
  per-query deviation signal remains non-transferable from any constructed
  training distribution. Query-side routing conclusion: real but not
  harvestable with these strategies/features/labels.

## 8. Deliverables

- Benchmark: multi-category corpus + profiler profiles + per-strategy scores
  (versioned in this repo).
- Router model + decision rules (small, inspectable).
- Report: methodology, results, stats, error analysis, limitations
  (template: docs/report-template.md).
