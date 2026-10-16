"""The experiment runner: config -> dataset -> retriever -> run -> metrics -> result.

Usage (see scripts/run_experiment.py):
    python scripts/run_experiment.py --config experiments/configs/smoke_bm25.yaml

Every execution writes one JSON file to experiments/results/ containing the
full config snapshot, metrics, and reproducibility metadata. Results are
committed to git so experiments are traceable and comparable over time.
Ranked runs (full per-query doc lists) are written to artifacts/runs/
(gitignored) and referenced from the result file. When the config enables
`generator`, the top-ranked documents are additionally fed to an LLM and
answers are stored under artifacts/answers/ with a grounding metric.
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

    # --- optional generation stage: answer queries from retrieved docs ---
    if spec.generator is not None:
        from ..evaluation.answers import answer_grounding
        from ..generation.generator import get_generator
        from ..generation.prompts import build_prompt

        print(f"generating answers with {spec.generator.model} "
              f"(limit={spec.generator.limit})...")
        docs_by_id = {d.id: d.text for d in dataset.documents}
        qtext = {q.id: q.text for q in dataset.queries}
        items = list(run.items())
        if spec.generator.limit:
            items = items[: spec.generator.limit]
        prompts = {
            qid: build_prompt(
                qtext[qid],
                [docs_by_id[d] for d in docids if d in docs_by_id],
                template=spec.generator.prompt_template,
                max_context_chars=spec.generator.max_context_chars,
            )
            for qid, docids in items
        }
        answers = get_generator(spec.generator).generate(prompts)

        ans_dir = Path(repo_root or ".") / "artifacts" / "answers"
        ans_dir.mkdir(parents=True, exist_ok=True)
        ans_path = ans_dir / f"{experiment_id}.json"
        ans_path.write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8")
        result["answers_artifact"] = str(ans_path)
        result["metrics"]["answer_grounding"] = round(
            answer_grounding(answers, run, docs_by_id, dataset.qrels,
                             embedder_name=spec.models.embedding), 4)
        result["metrics"]["answer_avg_chars"] = round(
            sum(len(a) for a in answers.values()) / max(len(answers), 1), 1)

    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["result_path"] = str(out_path)
    return result
