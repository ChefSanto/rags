"""Dense retrieval: cosine/dot-product search over document embeddings."""

from __future__ import annotations

import numpy as np

from ..datasets.base import Document
from ..models.embeddings import get_embedder


class DenseRetriever:
    def __init__(self, embedding_model: str | None = None):
        self.embedder = get_embedder(embedding_model)

    def index(self, documents: list[Document]) -> None:
        self._doc_ids = [d.id for d in documents]
        self._matrix = self.embedder.encode([d.text for d in documents])

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q = self.embedder.encode([query])
        scores = (self._matrix @ q[0]).tolist()
        ranked = sorted(zip(self._doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
