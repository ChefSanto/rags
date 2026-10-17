#!/usr/bin/env python
"""Generate deterministic pseudo-queries from a BEIR corpus (router v3, H3).

Motivation: the routing signal is local (see router-v2a reports) — routers must
be trained per-corpus on test-distributed data. Pseudo-queries provide that
training signal without human labels: document-derived queries scored against
pseudo-qrels (source doc = relevant).

For each document, up to --per-doc pseudo-queries:
  - its first sentence, if short ("title-like")
  - random sentences from its text
Pseudo-qrels mark the source document relevant (1).

Output is BEIR format in <out_dir>/ (queries.jsonl, qrels.tsv, corpus.jsonl
symlink), so the standard experiment runner scores strategies on them.
Also writes the three strategy configs (bm25 / dense / hybrid) for the
pseudo-dataset, ready for run_experiment.py.

Honest risk (stated for the report): document-derived queries are lexically
easy and may over-favor BM25. That bias is itself measurable — the routed
result vs uniform comparison exposes it. (Confirmed by router-v3 report:
pseudo-label means collapse to BM25 0.94-0.96; see make_llm_queries.py for the
question-distributed fix.)

Usage:
    python scripts/make_pseudo_queries.py datasets/downloads/fiqa \
        --dataset-name fiqa --out datasets/pseudo-fiqa --per-doc 2 --limit 20000
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SENT_RE = re.compile(r"[^.!?\n]{20,220}[.!?]")

STRATEGY_PARAM_BLOCKS = {
    "bm25": "    k1: 1.5\n    b: 0.75",
    "dense": None,
    "hybrid": ("    dense_weight: 0.5\n    bm25:\n      k1: 1.5\n      b: 0.75\n"
               "    dense:\n      embedding: sentence-transformers/all-MiniLM-L6-v2"),
}

CONFIG_TEMPLATE = """experiment:
  name: {dset}-pseudo-{strat}-v1
  description: >
    Pseudo-labeled routing training data for {dset}: strategy scores on
    document-derived pseudo-queries (router v3, H3).

dataset:
  name: pseudo-{dset}
  version: "1.0"
  path: {outdir}

retrieval:
  method: {strat}
  top_k: 10
{params_yaml}
models:
  embedding: sentence-transformers/all-MiniLM-L6-v2

parameters:
  seed: 42

evaluation:
  metrics:
    - recall_at_5
    - recall_at_10
    - precision_at_10
    - mrr_at_10
    - ndcg_at_10
  k_values: [5, 10]
"""


def pseudo_queries(text: str, rng: random.Random, per_doc: int) -> list[str]:
    cands = [m.group(0).strip() for m in _SENT_RE.finditer(text)]
    if not cands:
        return []
    out = []
    if len(cands[0]) <= 120:
        out.append(cands[0])
        cands = cands[1:]
    while cands and len(out) < per_doc:
        out.append(cands.pop(rng.randrange(len(cands))))
    return out


def write_configs(dataset_name: str, out: Path) -> None:
    """Write the three strategy configs for a pseudo-dataset (shared with make_llm_queries)."""
    for strat in ("bm25", "dense", "hybrid"):
        block = STRATEGY_PARAM_BLOCKS[strat]
        params_yaml = "  params: {}" if block is None else "  params:\n" + block
        cfg = CONFIG_TEMPLATE.format(dset=dataset_name, strat=strat,
                                     outdir=str(out), params_yaml=params_yaml)
        cfg_path = ROOT / "experiments" / "configs" / f"pseudo_{dataset_name}_{strat}.yaml"
        cfg_path.write_text(cfg, encoding="utf-8")
        print("wrote", cfg_path.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus_dir", help="BEIR dataset dir containing corpus.jsonl")
    ap.add_argument("--dataset-name", required=True, help="real dataset name, e.g. fiqa")
    ap.add_argument("--out", required=True, help="output dir for the pseudo dataset")
    ap.add_argument("--per-doc", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="max documents to use")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # corpus: symlink (fall back to full copy) so load_beir finds it next to queries
    link = out / "corpus.jsonl"
    if not link.exists():
        try:
            os.symlink(str(Path(args.corpus_dir).resolve() / "corpus.jsonl"), str(link))
        except OSError:
            import shutil
            shutil.copy(Path(args.corpus_dir) / "corpus.jsonl", link)

    n_docs = 0
    n_queries = 0
    with open(Path(args.corpus_dir) / "corpus.jsonl", encoding="utf-8") as fc, \
            open(out / "queries.jsonl", "w", encoding="utf-8") as fq, \
            open(out / "qrels.tsv", "w", encoding="utf-8") as fr:
        fr.write("query-id\tdoc-id\trelevance\n")
        for line in fc:
            if args.limit and n_docs >= args.limit:
                break
            doc = json.loads(line)
            qs = pseudo_queries(doc["text"], rng, args.per_doc)
            for j, q in enumerate(qs):
                qid = f"pq-{doc['_id']}-{j}"
                fq.write(json.dumps({"_id": qid, "text": q}) + "\n")
                fr.write(f"{qid}\t{doc['_id']}\t1\n")
                n_queries += 1
            n_docs += 1

    write_configs(args.dataset_name, out)

    print(f"done: {n_docs} docs -> {n_queries} pseudo-queries in {out}")


if __name__ == "__main__":
    main()
