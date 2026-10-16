"""The experiment runner: config -> dataset -> retriever -> run -> metrics -> result.

Usage (see scripts/run_experiment.py):
    python scripts/run_experiment.py --config experiments/configs/smoke_bm25.yaml

Every execution writes one JSON file to experiments/results/ containing the
full config snapshot, metrics, and reproducibility metadata. Results are
committed to git so experiments are traceable and comparable over time.
Ranked runs (full per-query doc lists) are written to artifacts/runs/
(gitignored) and referenced from the result file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..datasets.beir import load_beir
from ..evaluation.retrieval import compute_metrics
from ..retrieval.factory import build_retriever
from .metadata import collect
from .schema import load_spec, spec_to_dict

DEFAULT_RESULTS_DIR = "experiments/results"


def run_experiment(
    config_path: str | Path,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    repo_root: str | Path | None = None,
) -> dict:
    spec = load_spec(config_path)
    dataset = load_beir(spec.dataset.path, name=spec.dataset.name, version=spec.dataset.version)

    print(f"indexing {len(dataset.documents)} documents...")
    retriever = build_retriever(spec)
    retriever.index(dataset.documents)

    run: dict[str, list[str]] = {}
    n = len(dataset.queries)
    for qi, query in enumerate(dataset.queries, start=1):
        hits = retriever.search(query.text, top_k=spec.retrieval.top_k)
        run[query.id] = [doc_id for doc_id, _ in hits]
        if qi % 50 == 0 or qi == n:
            print(f"  queried {qi}/{n}")
    print("evaluating...")

    metrics = compute_metrics(run, dataset.qrels, spec.evaluation.metrics)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"{spec.name}__{timestamp}"

    # ranked runs are too large for git; saved as artifacts (gitignored) and
    # referenced from the result file — oracle/routing analyses read these
    artifact_dir = Path(repo_root or ".") / "artifacts" / "runs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_artifact = artifact_dir / f"{experiment_id}.json"
    run_artifact.write_text(json.dumps(run) + "\n", encoding="utf-8")

    result = {
        "experiment_id": experiment_id,
        "name": spec.name,
        "description": spec.description,
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "retriever": spec.retrieval.method,
        "retrieval_params": spec.retrieval.params,
        "seed": spec.seed,
        "num_documents": len(dataset.documents),
        "num_queries": len(dataset.queries),
        "metrics": metrics,
        "run_artifact": str(run_artifact),
        "config": spec_to_dict(spec),
        "metadata": collect(spec.seed, repo_root=repo_root),
    }

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{spec.name}__{timestamp}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["result_path"] = str(out_path)
    return result
