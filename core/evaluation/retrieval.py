"""Retrieval evaluation metrics.

Run format: {query_id: [doc_id, ...]} ranked best-first. Relevance threshold
for binary metrics is rel >= 1; nDCG uses graded relevance as gain.

Assumptions and limitations (roadmap §11):
- recall/precision/MRR treat any rel >= 1 as relevant; graded judgments are
  collapsed. Use nDCG when grade differences matter.
- metrics are macro-averaged over queries present in qrels.
- queries with no judged relevant documents are excluded (they would make
  recall undefined, not zero).
"""

from __future__ import annotations

import math

Run = dict[str, list[str]]
Qrels = dict[str, dict[str, int]]


def _judged(run: Run, qrels: Qrels) -> list[str]:
    return [qid for qid in run if qid in qrels and any(r >= 1 for r in qrels[qid].values())]


def recall_at_k(run: Run, qrels: Qrels, k: int) -> float:
    qids = _judged(run, qrels)
    if not qids:
        return 0.0
    hits = sum(
        1
        for qid in qids
        if any(d in qrels[qid] and qrels[qid][d] >= 1 for d in run[qid][:k])
    )
    return hits / len(qids)


def precision_at_k(run: Run, qrels: Qrels, k: int) -> float:
    qids = _judged(run, qrels)
    if not qids:
        return 0.0
    total = 0.0
    for qid in qids:
        relevant = {d for d, r in qrels[qid].items() if r >= 1}
        total += len([d for d in run[qid][:k] if d in relevant]) / k
    return total / len(qids)


def mrr_at_k(run: Run, qrels: Qrels, k: int) -> float:
    qids = _judged(run, qrels)
    if not qids:
        return 0.0
    total = 0.0
    for qid in qids:
        for rank, doc_id in enumerate(run[qid][:k], start=1):
            if qrels[qid].get(doc_id, 0) >= 1:
                total += 1.0 / rank
                break
    return total / len(qids)


def _dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(run: Run, qrels: Qrels, k: int) -> float:
    qids = _judged(run, qrels)
    if not qids:
        return 0.0
    total = 0.0
    for qid in qids:
        gains = [qrels[qid].get(d, 0) for d in run[qid][:k]]
        ideal = sorted((r for r in qrels[qid].values() if r >= 1), reverse=True)[:k]
        denom = _dcg(ideal)
        total += _dcg(gains) / denom if denom else 0.0
    return total / len(qids)


# --- per-query variants (oracle / routing analysis) ---
# Same formulas as above, returned per query. mean(per_query_x) == x for all.


def per_query_recall_at_k(run: Run, qrels: Qrels, k: int) -> dict[str, float]:
    out = {}
    for qid in _judged(run, qrels):
        relevant = {d for d, r in qrels[qid].items() if r >= 1}
        out[qid] = 1.0 if any(d in relevant for d in run[qid][:k]) else 0.0
    return out


def per_query_ndcg_at_k(run: Run, qrels: Qrels, k: int) -> dict[str, float]:
    out = {}
    for qid in _judged(run, qrels):
        gains = [qrels[qid].get(d, 0) for d in run[qid][:k]]
        ideal = sorted((r for r in qrels[qid].values() if r >= 1), reverse=True)[:k]
        denom = _dcg(ideal)
        out[qid] = _dcg(gains) / denom if denom else 0.0
    return out


# name (without _at_k suffix) -> function(run, qrels, k)
METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def parse_metric(name: str) -> tuple[str, int]:
    """'recall_at_5' -> ('recall', 5). Bare names get k=10."""
    base, sep, k = name.rpartition("_at_")
    if not sep:
        return name, 10
    return base, int(k)


def compute_metrics(
    run: Run, qrels: Qrels, metric_names: list[str]
) -> dict[str, float]:
    out = {}
    for name in metric_names:
        base, k = parse_metric(name)
        if base not in METRICS:
            raise ValueError(f"Unknown metric '{name}'. Available: "
                             f"{', '.join(f'{m}_at_K' for m in METRICS)}")
        out[name] = round(METRICS[base](run, qrels, k), 4)
    return out
