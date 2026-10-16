"""Hybrid retrieval: min-max normalized score fusion of two retrievers."""

from __future__ import annotations

from ..datasets.base import Document
from .base import Retriever


def _normalize(scores: list[tuple[str, float]]) -> dict[str, float]:
    if not scores:
        return {}
    vals = [s for _, s in scores]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return {doc_id: (s - lo) / span for doc_id, s in scores}


class HybridRetriever:
    def __init__(self, lexical: Retriever, dense: Retriever, dense_weight: float = 0.5):
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be in [0, 1]")
        self.lexical = lexical
        self.dense = dense
        self.dense_weight = dense_weight

    def index(self, documents: list[Document]) -> None:
        self.lexical.index(documents)
        self.dense.index(documents)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        # fetch more than top_k from each side so fusion sees a wide candidate set
        n = top_k * 3
        lex = _normalize(self.lexical.search(query, n))
        den = _normalize(self.dense.search(query, n))
        fused: dict[str, float] = {}
        for doc_id in set(lex) | set(den):
            fused[doc_id] = (1.0 - self.dense_weight) * lex.get(doc_id, 0.0) + \
                            self.dense_weight * den.get(doc_id, 0.0)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
