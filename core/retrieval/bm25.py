"""Okapi BM25 lexical retrieval (self-contained, no external dependencies).

Uses an inverted index: each query only scores documents containing at least
one query term, so a search costs O(postings touched) instead of
O(corpus size). With FiQA (~57k docs) the naive scan takes many minutes;
the indexed version takes seconds.
"""

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
        doc_tokens = [tokenize(d.text) for d in documents]
        doc_len = [len(t) for t in doc_tokens]
        avgdl = sum(doc_len) / max(len(doc_len), 1)

        # per-document length normalization, computed once at index time
        self._norm = [
            self.k1 * (1.0 - self.b + self.b * dl / (avgdl or 1.0)) for dl in doc_len
        ]

        n_docs = len(documents)
        df: Counter = Counter()
        tf = [Counter(t) for t in doc_tokens]
        for tokens in doc_tokens:
            df.update(set(tokens))
        self._idf = {
            t: math.log((n_docs - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()
        }

        # inverted index: token -> list of (doc_index, term_freq)
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for i, counts in enumerate(tf):
            for t, f in counts.items():
                self._postings.setdefault(t, []).append((i, f))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = {}
        for t in set(tokenize(query)):  # set: repeated query terms score once
            idf = self._idf.get(t)
            if idf is None:
                continue
            for i, f in self._postings.get(t, ()):
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1.0) / (
                    f + self._norm[i]
                )
        # sort by score desc, ties broken by doc index (deterministic — matches
        # the order a full-corpus scan would produce)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [(self._doc_ids[i], s) for i, s in ranked]
