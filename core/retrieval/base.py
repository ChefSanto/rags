"""Retriever protocol shared by all strategies."""

from __future__ import annotations

from typing import Protocol

from ..datasets.base import Document


class Retriever(Protocol):
    def index(self, documents: list[Document]) -> None: ...
    def search(self, query: str, top_k: int) -> list[tuple[str, float]]: ...
