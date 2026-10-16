"""Answer-level metrics that do not require gold answer text.

v0 grounding metric: semantic similarity between the generated answer and the
retrieved documents, weighted by qrels relevance (relevant docs weigh 1.0,
others 0.1). Rationale: a grounded RAG answer should be semantically close
to the documents that are actually relevant to the query. This is a proxy,
not faithfulness — an LLM-judge stage comes later.

Embeddings are assumed L2-normalized (both hashing and sentence-transformers
paths normalize), so dot product = cosine similarity.
"""

from __future__ import annotations

import numpy as np

from ..models.embeddings import get_embedder


def answer_grounding(
    answers: dict[str, str],
    run: dict[str, list[str]],
    docs_by_id: dict[str, str],
    qrels: dict[str, dict[str, int]],
    top_k: int | None = None,
    embedder_name: str | None = None,
) -> float:
    embedder = get_embedder(embedder_name)
    doc_ids = sorted({d for ids in run.values() for d in ids if d in docs_by_id})
    if not doc_ids:
        return 0.0
    doc_vec = embedder.encode([docs_by_id[d] for d in doc_ids])
    dmap = {d: v for d, v in zip(doc_ids, doc_vec)}

    qids = [q for q in answers if run.get(q)]
    if not qids:
        return 0.0
    a_vec = embedder.encode([answers[q] for q in qids])

    scores = []
    for q, av in zip(qids, a_vec):
        acc, wsum = 0.0, 0.0
        for d in run[q][:top_k]:
            w = 1.0 if qrels.get(q, {}).get(d, 0) >= 1 else 0.1
            acc += w * float(av @ dmap[d])
            wsum += w
        scores.append(acc / wsum if wsum else 0.0)
    return float(np.mean(scores))
