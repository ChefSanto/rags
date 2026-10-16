"""Okapi BM25 lexical retrieval (self-contained, no external dependencies)."""

from __future__ import annotations

import math
import re
from collections import Counter

from ..datasets.base import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def index(self, documents: list[Document]) -> None:
        self._doc_ids = [d.id for d in documents]
        self._doc_tokens = [tokenize(d.text) for d in documents]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avgdl = sum(self._doc_len) / max(len(self._doc_len), 1)
        self._tf = [Counter(t) for t in self._doc_tokens]

        n_docs = len(documents)
        df: Counter = Counter()
        for tokens in self._doc_tokens:
            df.update(set(tokens))
        self._idf = {
            t: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for t, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        scores = []
        for i, tf in enumerate(self._tf):
            dl = self._doc_len[i]
            norm = 1.0 - self.b + self.b * dl / (self._avgdl or 1.0)
            s = 0.0
            for t in q_tokens:
                f = tf.get(t, 0)
                if f:
                    s += self._idf.get(t, 0.0) * f * (self.k1 + 1.0) / (f + self.k1 * norm)
            scores.append((self._doc_ids[i], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
