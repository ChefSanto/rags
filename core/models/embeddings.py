"""Embedding model wrapper with an offline fallback.

`get_embedder(name)` resolves:
- "hashing" or None  -> HashingEmbedder (deterministic feature hashing; CPU-only;
                       for smoke tests ONLY, never for reported research numbers)
- anything else      -> sentence-transformers model, e.g.
                       "sentence-transformers/all-MiniLM-L6-v2"

Research runs must set models.embedding to a real model so results are
meaningful and comparable. The hashing fallback exists so CI / smoke tests run
without downloads or GPU.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedder (offline fallback)."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerAdapter:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # deferred import

        self.model = SentenceTransformer(model_name)
        self.name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


def get_embedder(name: str | None):
    if name in (None, "hashing"):
        return HashingEmbedder()
    return SentenceTransformerAdapter(name)
