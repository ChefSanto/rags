"""Unit tests for retrieval metrics and result registry robustness.

Run: pytest
"""

from core.evaluation.retrieval import compute_metrics
from core.experiments.registry import comparison_rows, load_results

RUN = {"q1": ["d2", "d1", "d3"], "q2": ["d4", "d5", "d6"]}
QRELS = {"q1": {"d1": 2, "d3": 1}, "q2": {"d6": 1}}


def test_recall():
    m = compute_metrics(RUN, QRELS, ["recall_at_1", "recall_at_3"])
    assert m["recall_at_1"] == 0.0  # top hits d2, d4 are not relevant
    assert m["recall_at_3"] == 1.0


def test_precision():
    m = compute_metrics(RUN, QRELS, ["precision_at_3"])
    assert m["precision_at_3"] == 0.5  # (2/3 + 1/3) / 2


def test_mrr():
    m = compute_metrics(RUN, QRELS, ["mrr_at_3"])
    # q1: first relevant at rank 2 -> 1/2 ; q2: rank 3 -> 1/3
    assert m["mrr_at_3"] == round((0.5 + 1 / 3) / 2, 4)


def test_ndcg():
    m = compute_metrics(RUN, QRELS, ["ndcg_at_3"])
    # q1: (2/log2(3) + 1/log2(4)) / (2/log2(2) + 1/log2(3)) ; q2: 0.5
    q1 = (2 / 1.584962500721156 + 0.5) / (2 + 1 / 1.584962500721156)
    assert m["ndcg_at_3"] == round((q1 + 0.5) / 2, 4)


def test_registry_skips_foreign_files(tmp_path):
    (tmp_path / "good.json").write_text(
        '{"name": "exp", "dataset": "d", "dataset_version": "1", '
        '"retriever": "bm25", "metrics": {"recall_at_5": 1.0}}'
    )
    (tmp_path / "foreign.json").write_text('{"some": "other schema"}')
    (tmp_path / "broken.json").write_text("{not json")
    results = load_results(tmp_path)
    assert len(results) == 1
    rows = comparison_rows(results)
    assert rows[0]["experiment"] == "exp"
    assert rows[0]["recall_at_5"] == 1.0
