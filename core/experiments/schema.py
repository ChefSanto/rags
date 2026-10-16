"""Machine-readable experiment specification.

An experiment is fully described by a YAML file (see experiments/configs/ and
docs/experiment-schema.md). This module parses and validates it. The schema is
versioned implicitly by git — the runner snapshots the parsed spec into every
result file so experiments cannot silently change when configs change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

SCHEMA_VERSION = "1.0"

RETRIEVAL_METHODS = ("bm25", "dense", "hybrid")


@dataclass
class DatasetSpec:
    name: str
    version: str
    path: str  # directory containing corpus.jsonl / queries.jsonl / qrels.tsv
    format: str = "beir"


@dataclass
class RetrievalSpec:
    method: str  # one of RETRIEVAL_METHODS
    top_k: int = 10
    params: dict = field(default_factory=dict)


@dataclass
class ModelSpec:
    embedding: str | None = None  # e.g. "sentence-transformers/all-MiniLM-L6-v2" or "hashing"
    reranker: str | None = None


@dataclass
class EvaluationSpec:
    metrics: list[str] = field(
        default_factory=lambda: ["recall_at_5", "recall_at_10", "mrr_at_10"]
    )
    k_values: list[int] = field(default_factory=lambda: [5, 10])


@dataclass
class ExperimentSpec:
    name: str
    description: str
    dataset: DatasetSpec
    retrieval: RetrievalSpec
    models: ModelSpec = field(default_factory=ModelSpec)
    parameters: dict = field(default_factory=lambda: {"seed": 42})
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)

    @property
    def seed(self) -> int:
        return int(self.parameters.get("seed", 42))


def load_spec(config_path: str | Path) -> ExperimentSpec:
    """Parse and validate a YAML experiment config."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    try:
        exp = raw["experiment"]
        spec = ExperimentSpec(
            name=exp["name"],
            description=exp.get("description", ""),
            dataset=DatasetSpec(**raw["dataset"]),
            retrieval=RetrievalSpec(**raw["retrieval"]),
            models=ModelSpec(**raw.get("models", {})),
            parameters=raw.get("parameters", {"seed": 42}),
            evaluation=EvaluationSpec(**raw.get("evaluation", {})),
        )
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid experiment config {config_path}: {e}") from e

    if spec.retrieval.method not in RETRIEVAL_METHODS:
        raise ValueError(
            f"Unknown retrieval method '{spec.retrieval.method}'. "
            f"Available: {', '.join(RETRIEVAL_METHODS)}"
        )
    return spec


def spec_to_dict(spec: ExperimentSpec) -> dict:
    d = asdict(spec)
    d["schema_version"] = SCHEMA_VERSION
    return d
