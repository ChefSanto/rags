"""Dataset types shared by all loaders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    id: str
    text: str


@dataclass
class Query:
    id: str
    text: str


@dataclass
class RetrievalDataset:
    """A retrieval benchmark: documents, queries, and relevance judgments.

    qrels maps query_id -> {document_id: relevance}. relevance >= 1 counts as
    relevant for recall/precision/MRR; the raw value is used for nDCG gains.
    """

    name: str
    version: str
    documents: list[Document]
    queries: list[Query]
    qrels: dict[str, dict[str, int]]
