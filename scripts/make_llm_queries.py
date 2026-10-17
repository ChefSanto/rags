#!/usr/bin/env python
"""Generate natural-language pseudo-queries with a small LLM (router v4, H3).

Why this exists — the v3 finding (experiments/results/router-v3-*.md):
document-derived pseudo-queries are lexical copies of their source documents,
so BM25 wins the pseudo-task by construction (pseudo-label means 0.94-0.96)
and a router trained on them collapses to BM25 on real queries. The label
DISTRIBUTION was wrong. v4 fixes it: Qwen2.5-3B-Instruct writes short,
natural questions per document — greedy decoding, deterministic given the model.

Output is BEIR format in <out_dir>/, overwriting the v3 sentence-based
pseudo-queries when pointed at the same directory. The pseudo configs and
router_v3.py need no changes: re-run the pseudo experiments and the analyses
pick up the newest results per strategy.

Usage (Kaggle GPU):
    python scripts/make_llm_queries.py datasets/downloads/fiqa \
        --dataset-name fiqa --out datasets/pseudo-fiqa --per-doc 2 --limit 10000
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_pseudo_queries as mpq  # noqa: E402  (config template + symlink logic)
from core.generation.generator import HFGenerator  # noqa: E402

PROMPT = (
    "Read the passage and write {n} short factual questions that the passage "
    "can answer. Number them 1. 2. 3. Use natural question words (what, how, "
    "why, when, which). Do not copy sentences from the passage.\n\n"
    "Passage:\n{text}\n\nQuestions:"
)

_NUM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


def parse_questions(output: str, per_doc: int) -> list[str]:
    qs = []
    for line in output.splitlines():
        q = _NUM_RE.sub("", line).strip()
        if 20 <= len(q) <= 220:
            qs.append(q)
        if len(qs) >= per_doc:
            break
    return qs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus_dir", help="BEIR dataset dir containing corpus.jsonl")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--out", required=True, help="pseudo dataset dir (same as v3 to overwrite)")
    ap.add_argument("--per-doc", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="max documents")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    link = out / "corpus.jsonl"
    if not link.exists():
        try:
            os.symlink(str(Path(args.corpus_dir).resolve() / "corpus.jsonl"), str(link))
        except OSError:
            import shutil
            shutil.copy(Path(args.corpus_dir) / "corpus.jsonl", link)

    docs = []
    with open(Path(args.corpus_dir) / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if args.limit and len(docs) >= args.limit:
                break
            d = json.loads(line)
            docs.append({"id": str(d["_id"]), "text": str(d["text"])[:1500]})
    print(f"loaded {len(docs)} docs; generating questions with {args.model}")

    gen = HFGenerator(args.model, max_new_tokens=args.max_new_tokens,
                      temperature=0.0, chat=True)
    prompts = {d["id"]: PROMPT.format(n=args.per_doc, text=d["text"]) for d in docs}

    # chunk through HFGenerator.generate in batches to bound GPU memory
    answers = {}
    ids = list(prompts)
    for i in range(0, len(ids), args.batch_size * 8):
        chunk = ids[i : i + args.batch_size * 8]
        answers.update(gen.generate({q: prompts[q] for q in chunk}))
        print(f"  generated {min(i + args.batch_size * 8, len(ids))}/{len(ids)}")

    n_queries = 0
    n_docs_q = 0
    with open(out / "queries.jsonl", "w", encoding="utf-8") as fq, \
            open(out / "qrels.tsv", "w", encoding="utf-8") as fr:
        fr.write("query-id\tdoc-id\trelevance\n")
        for d in docs:
            qs = parse_questions(answers.get(d["id"], ""), args.per_doc)
            if not qs:
                continue
            n_docs_q += 1
            for j, q in enumerate(qs):
                qid = f"pq-{d['id']}-{j}"
                fq.write(json.dumps({"_id": qid, "text": q}) + "\n")
                fr.write(f"{qid}\t{d['id']}\t1\n")
                n_queries += 1

    mpq.write_configs(args.dataset_name, out)
    print(f"done: {n_docs_q} docs produced {n_queries} questions in {out}")


if __name__ == "__main__":
    main()
