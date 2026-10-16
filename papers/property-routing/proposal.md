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

## 7. Deliverables

- Benchmark: multi-category corpus + profiler profiles + per-strategy scores
  (versioned in this repo).
- Router model + decision rules (small, inspectable).
- Report: methodology, results, stats, error analysis, limitations
  (template: docs/report-template.md).
