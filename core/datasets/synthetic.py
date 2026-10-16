"""Deterministic synthetic smoke-test dataset.

Purpose: verify the experiment runner end-to-end (config → retrieval → metrics
→ result JSON) without downloading anything. NOT for research — keyword overlap
makes scores artificially high. Real benchmarks come from BEIR
(datasets/manifest.yaml).

Documents are drawn from topical keyword clusters across three domains;
queries target one cluster each; qrels mark same-cluster documents as relevant.
Fully deterministic given a seed.
"""

from __future__ import annotations

import random

from .base import Document, Query, RetrievalDataset

TOPICS: dict[str, dict[str, list[str]]] = {
    "capital-markets": {
        "domain": "finance",
        "words": ["stock", "bond", "equity", "dividend", "portfolio", "market", "trading"],
    },
    "macroeconomics": {
        "domain": "finance",
        "words": ["inflation", "interest", "monetary", "fiscal", "gdp", "unemployment"],
    },
    "banking-regulation": {
        "domain": "finance",
        "words": ["capital", "liquidity", "basel", "leverage", "supervision", "compliance"],
    },
    "neural-networks": {
        "domain": "science",
        "words": ["neural", "gradient", "layer", "training", "backpropagation", "optimizer"],
    },
    "materials-science": {
        "domain": "science",
        "words": ["polymer", "crystal", "alloy", "tensile", "composite", "microstructure"],
    },
    "devtools": {
        "domain": "tech",
        "words": ["compiler", "runtime", "repository", "pipeline", "api", "deployment"],
    },
}

_FILLER = [
    "analysis", "framework", "method", "result", "study", "system", "approach",
    "design", "data", "model", "process", "evaluation", "report", "section",
]

_TEMPLATES = [
    "This document discusses {w1} and {w2} in the context of {w3}. "
    "The {w4} framework combines {w1} with {w5} to improve outcomes.",
    "We present a study of {w1}, {w2}, and {w3}. Results show that {w4} "
    "outperforms prior work when combined with {w5}.",
    "An overview of {w1}: key concepts include {w2}, {w3}, and {w4}. "
    "The methodology extends {w5} with a novel evaluation.",
    "{w1} is central to this report. We compare {w2} against {w3} and "
    "propose improvements to {w4} using {w5}.",
]


def _doc_text(rng: random.Random, words: list[str]) -> str:
    w = rng.sample(words, 5)
    return rng.choice(_TEMPLATES).format(w1=w[0], w2=w[1], w3=w[2], w4=w[3], w5=w[4])


def _query_text(rng: random.Random, words: list[str]) -> str:
    w = rng.sample(words, 3)
    return f"What is the relationship between {w[0]} and {w[1]} in {w[2]}?"


def generate_synthetic(seed: int = 42, docs_per_topic: int = 12, queries_per_topic: int = 4) -> RetrievalDataset:
    rng = random.Random(seed)
    all_words = [w for t in TOPICS.values() for w in t["words"]]

    documents: list[Document] = []
    queries: list[Query] = []
    qrels: dict[str, dict[str, int]] = {}

    for topic, spec in TOPICS.items():
        topic_doc_ids = []
        for i in range(docs_per_topic):
            doc_id = f"d-{topic}-{i:02d}"
            text = _doc_text(rng, spec["words"])
            # sprinkle distractor terms from other topics to make it non-trivial
            for noise in rng.sample(all_words, 2):
                text += f" Related work also mentions {noise}."
            documents.append(Document(id=doc_id, text=text))
            topic_doc_ids.append(doc_id)

        primary = rng.sample(topic_doc_ids, 3)  # rel=2: "best" answers
        for j in range(queries_per_topic):
            qid = f"q-{topic}-{j:02d}"
            queries.append(Query(id=qid, text=_query_text(rng, spec["words"])))
            qrels[qid] = {
                **{d: 2 for d in primary},
                **{d: 1 for d in topic_doc_ids if d not in primary},
            }

    # noise documents: no query should retrieve these as relevant
    for i in range(6):
        words = rng.sample(_FILLER, 5)
        documents.append(
            Document(id=f"d-noise-{i:02d}", text=_doc_text(rng, words))
        )

    return RetrievalDataset(
        name="synthetic-smoke",
        version="1.0",
        documents=documents,
        queries=queries,
        qrels=qrels,
    )
