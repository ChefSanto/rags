#!/usr/bin/env python
"""Generate the tiny deterministic smoke-test dataset (BEIR format).

    python scripts/make_synthetic_dataset.py

Writes datasets/synthetic-smoke/{corpus.jsonl,queries.jsonl,qrels.tsv}.
Deterministic given the seed. NOT for research — see module docstring in
core/datasets/synthetic.py.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.datasets.synthetic import generate_synthetic  # noqa: E402

OUT_DIR = ROOT / "datasets" / "synthetic-smoke"


def main() -> None:
    ds = generate_synthetic(seed=42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in ds.documents:
            f.write(json.dumps({"_id": d.id, "text": d.text}) + "\n")

    with open(OUT_DIR / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in ds.queries:
            f.write(json.dumps({"_id": q.id, "text": q.text}) + "\n")

    with open(OUT_DIR / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\tdoc-id\trelevance\n")
        for qid, rels in sorted(ds.qrels.items()):
            for doc_id, rel in sorted(rels.items()):
                f.write(f"{qid}\t{doc_id}\t{rel}\n")

    print(f"wrote {len(ds.documents)} documents, {len(ds.queries)} queries "
          f"to {OUT_DIR}")


if __name__ == "__main__":
    main()
