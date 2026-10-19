#!/usr/bin/env python
"""Build a blind review sheet for human answer evaluation.

Merges answer artifacts from several experiments (e.g. the three
fiqa-*-qwen-v1 runs) into a single shuffled HTML page. Each card shows the
question, ONE answer, and the top-3 retrieved source texts — the strategy
that produced it is hidden. The reviewer grades each card
(Correct / Partial / Wrong) and clicks Export, which downloads a CSV of
(qid, hidden_strategy, grade). Unshuffling offline yields per-strategy
correct-rates for the paper's human-eval row.

Usage:
    python scripts/make_review_sheet.py \
        --answers 'artifacts/answers/fiqa-bm25-qwen-v1__*.json' \
                   'artifacts/answers/fiqa-dense-qwen-v1__*.json' \
                   'artifacts/answers/fiqa-hybrid-qwen-v1__*.json' \
        --runs 'artifacts/runs/fiqa-bm25-qwen-v1__*.json' \
               'artifacts/runs/fiqa-dense-qwen-v1__*.json' \
               'artifacts/runs/fiqa-hybrid-qwen-v1__*.json' \
        --dataset datasets/downloads/fiqa \
        --per-strategy 30 --out review.html
"""

import argparse
import glob
import html
import json
import random
import sys
from pathlib import Path

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Answer review (blind)</title>
<style>
 body{{font:15px/1.5 -apple-system,system-ui,sans-serif;background:#0f1115;color:#e6e9ef;margin:0;padding:20px 24px}}
 h1{{font-size:18px}} .dim{{color:#9aa3b2;font-size:13px}}
 .card{{background:#171a21;border:1px solid #262b36;border-radius:12px;padding:16px;margin:16px 0}}
 .q{{font-weight:600}} .a{{background:#0c0e12;border-left:3px solid #5b9dff;padding:10px;margin-top:10px;white-space:pre-wrap}}
 .src{{font-size:12px;color:#9aa3b2;border-top:1px dashed #262b36;margin-top:10px;padding-top:8px;white-space:pre-wrap}}
 select{{background:#0c0e12;color:#e6e9ef;border:1px solid #2c3340;border-radius:6px;padding:6px;margin-top:10px}}
 button{{background:#5b9dff;color:#08111f;border:0;border-radius:8px;padding:10px 18px;font-weight:600;cursor:pointer;margin-top:20px}}
 #progress{{margin-left:12px}}
</style></head><body>
<h1>Blind answer review</h1>
<div class="dim">Grade each answer <b>against its sources</b>: Correct = answers the question and is grounded;
Partial = related but misses the key point or adds a minor unsupported claim;
Wrong = noise echo, contradiction, or off-topic. The generating strategy is hidden until export.</div>
{cards}
<button onclick="exportCsv()">Export grades (CSV)</button><span id="progress" class="dim"></span>
<script>
let graded = 0;
function bump() {{
  graded = document.querySelectorAll('select').length - [...document.querySelectorAll('select')].filter(s => !s.value).length;
  document.getElementById('progress').textContent = graded + ' / ' + document.querySelectorAll('select').length + ' graded';
}}
document.querySelectorAll('select').forEach(s => s.onchange = bump); bump();
function exportCsv() {{
  let csv = 'qid,strategy,grade\\n';
  document.querySelectorAll('.card').forEach(c => {{
    const s = c.querySelector('select');
    if (s.value) csv += c.dataset.qid + ',' + c.dataset.strat + ',' + s.value + '\\n';
  }});
  const blob = new Blob([csv], {{type: 'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'grades.csv'; a.click();
}}
</script></body></html>"""

CARD = """<div class="card" data-qid="{qid}" data-strat="{strat}">
<div class="q">{i}. {q}</div>
<div class="a">{a}</div>
<div class="src"><b>Sources:</b>\n{srcs}</div>
<select><option value="">grade…</option><option>Correct</option><option>Partial</option><option>Wrong</option></select>
</div>"""


def load_jsonl_map(path, key, val):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[str(d[key])] = d[val]
    return out


def newest(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match: {pattern}")
    return files[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", required=True, help="BEIR dataset dir (corpus + queries)")
    ap.add_argument("--per-strategy", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="review.html")
    args = ap.parse_args()

    if len(args.answers) != len(args.runs):
        raise SystemExit("--answers and --runs must correspond 1:1")

    queries = load_jsonl_map(Path(args.dataset) / "queries.jsonl", "_id", "text")
    corpus = load_jsonl_map(Path(args.dataset) / "corpus.jsonl", "_id", "text")

    answers, runs = {}, {}
    for a_path, r_path in zip(args.answers, args.runs):
        exp = json.load(open(newest(a_path)))
        strat = exp["retriever"]
        answers[strat] = json.load(open(exp["answers_artifact"]))
        runs[strat] = json.load(open(newest(r_path)))

    strategies = sorted(answers)
    common = sorted(set.intersection(*[set(v) for v in answers.values()]))
    rng = random.Random(args.seed)
    sample = rng.sample(common, min(args.per_strategy, len(common)))
    rng.shuffle(sample)

    cards = []
    for i, qid in enumerate(sample, 1):
        for strat in strategies:
            ans = answers[strat].get(qid, "")
            docs = runs[strat].get(qid, [])[:3]
            srcs = "\n---\n".join(f"[{d}] {corpus.get(d, '(missing)')[:600]}" for d in docs)
            cards.append(CARD.format(i=i, qid=qid, strat=strat, q=html.escape(queries[qid]),
                                     a=html.escape(ans), srcs=html.escape(srcs)))

    Path(args.out).write_text(PAGE.format(cards="\n".join(cards)), encoding="utf-8")
    print(f"wrote {args.out}: {len(sample)} questions x {len(strategies)} strategies "
          f"= {len(cards)} cards (order shuffled, seed {args.seed})")


if __name__ == "__main__":
    main()
