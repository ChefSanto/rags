"""Build a retriever from an ExperimentSpec."""

from __future__ import annotations

from ..experiments.schema import ExperimentSpec
from .bm25 import BM25
from .dense import DenseRetriever
from .hybrid import HybridRetriever


def build_retriever(spec: ExperimentSpec):
    method = spec.retrieval.method
    p = dict(spec.retrieval.params)

    if method == "bm25":
        return BM25(**p)

    if method == "dense":
        return DenseRetriever(embedding_model=spec.models.embedding)

    if method == "hybrid":
        dense_weight = float(p.pop("dense_weight", 0.5))
        bm25_params = p.pop("bm25", {})
        dense_params = p.pop("dense", {})
        lexical = BM25(**bm25_params)
        dense = DenseRetriever(embedding_model=dense_params.get("embedding", spec.models.embedding))
        return HybridRetriever(lexical, dense, dense_weight=dense_weight)

    raise ValueError(f"Unknown retrieval method: {method}")
