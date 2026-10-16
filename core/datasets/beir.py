"""Loader for BEIR-style retrieval datasets.

Expected directory layout:

    <path>/
    ├── corpus.jsonl    # {"_id": str, "text": str, "title": str (optional)}
    ├── queries.jsonl   # {"_id": str, "text": str}
    └── qrels.tsv       # header line, then: query_id <TAB> doc_id <TAB> relevance

This covers BEIR datasets (FiQA, NFCorpus, ...) as well as our synthetic
smoke dataset. Register datasets in datasets/manifest.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path

from .base import Document, Query, RetrievalDataset


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_beir(path: str | Path, name: str | None = None, version: str | None = None) -> RetrievalDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {path}. "
            f"Check datasets/manifest.yaml and docs/datasets.md."
        )

    docs = []
    for d in _read_jsonl(path / "corpus.jsonl"):
        text = d["text"]
        if d.get("title"):
            text = f"{d['title']}. {text}"
        docs.append(Document(id=str(d["_id"]), text=text.strip()))

    queries = [Query(id=str(q["_id"]), text=q["text"]) for q in _read_jsonl(path / "queries.jsonl")]

    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels.tsv", encoding="utf-8") as f:
        first = f.readline()
        if not first.lower().startswith("query"):  # tolerate missing header
            parts = first.rstrip("\n").split("\t")
            if len(parts) >= 3:
                qrels.setdefault(parts[0], {})[parts[1]] = int(parts[2])
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                qrels.setdefault(parts[0], {})[parts[1]] = int(parts[2])

    return RetrievalDataset(
        name=name or path.name,
        version=version or "unknown",
        documents=docs,
        queries=queries,
        qrels=qrels,
    )
