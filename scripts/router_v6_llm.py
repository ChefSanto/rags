#!/usr/bin/env python
"""Router v6: LLM-as-router (zero-shot) — the missing cell in the matrix.

Every router so far was gradient-boosted trees on query features. Modern routing
work (RAGRouter, RouteLLM) uses LLMs as routers. This experiment asks a small
LLM to name the best retrieval strategy per query, routes accordingly using the
committed run artifacts, and scores on the real test split.

Requires GROQ_API_KEY in the environment (already in ~/.zshrc) or OPENAI_API_KEY
with --provider openai. Answers are cached in artifacts/llm-router-cache.json so
reruns are free and stable.

    python scripts/router_v6_llm.py --datasets fiqa nfcorpus
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core.evaluation.retrieval import per_query_ndcg_at_k  # noqa: E402
from core.experiments.registry import load_results  # noqa: E402
import router_v0 as rv  # noqa: E402

STRATEGIES = ("bm25", "dense", "hybrid")
SYNONYMS = {"bm_25": "bm25", "bm25_retrieval": "bm25", "lexical": "bm25",
            "sparse": "bm25", "dense_retrieval": "dense", "semantic": "dense",
            "embedding": "dense", "neural": "dense", "fusion": "hybrid",
            "combined": "hybrid", "both": "hybrid", "mixed": "hybrid"}
CACHE_PATH = ROOT / "artifacts" / "llm-router-cache.json"

PROMPT_HEAD = (
    "You route search queries to retrieval strategies.\n\n"
    "Strategies:\n"
    "- bm25: exact keyword/terminology matching; best for rare technical terms, codes, precise names\n"
    "- dense: semantic matching; best for natural-language questions about meaning and concepts\n\n"
    "For each numbered query answer with exactly one word: bm25 or dense. "
    "Commit to one -- do not hedge.\n\n"
)


def build_prompt(queries: list[str]) -> str:
    lines = "\n".join(f'{i + 1}. "{q}"' for i, q in enumerate(queries))
    return PROMPT_HEAD + lines + "\n\nAnswers:"


def parse_answers(text: str, n: int) -> list[str]:
    got = {}
    for i, w in re.findall(r"^\s*(\d+)[.)]\s*([A-Za-z0-9_]+)", text, re.M):
        i = int(i)
        if 1 <= i <= n:
            got[i] = w.lower()
    out = []
    for i in range(1, n + 1):
        w = SYNONYMS.get(got.get(i, ""), got.get(i, ""))
        out.append(w if w in STRATEGIES else "hybrid")
    return out


def call_llm(prompt: str, args) -> str:
    base = "https://api.groq.com/openai/v1" if args.provider == "groq" else "https://api.openai.com/v1"
    env_name = "GROQ_API_KEY" if args.provider == "groq" else "OPENAI_KEY"
    key = os.environ.get(env_name, "")
    if not key:
        raise SystemExit(f"{env_name} not in environment")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps({
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max(300, 8 * args.batch_size),
            "reasoning_effort": "low",
        }).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key, "User-Agent": "curl/8.5.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            j = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{args.provider} error {e.code}: {e.read().decode()[:400]}")
    return j["choices"][0]["message"]["content"]


def load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))



def _artifact_path(path_str):
    """Run artifacts may carry absolute paths from the machine that produced
    them (e.g. /kaggle/working/...); fall back to the local committed file."""
    p = Path(path_str)
    return p if p.exists() else ROOT / "artifacts" / "runs" / p.name


def _local_run(path_str):
    return json.loads(_artifact_path(path_str).read_text())

def load_runs3(name: str, results_dir: str) -> dict:
    results = [r for r in load_results(results_dir)
               if r.get("dataset") == name and r.get("run_artifact")]
    by = {}
    for r in sorted(results, key=lambda r: r["experiment_id"]):
        if r["retriever"] in STRATEGIES and _artifact_path(r["run_artifact"]).exists():
            by[r["retriever"]] = r
    if set(by) != set(STRATEGIES):
        raise SystemExit(f"{name}: need bm25+dense+hybrid run artifacts, have {sorted(by)}")
    return {s: _local_run(r["run_artifact"]) for s, r in by.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--provider", choices=["groq", "openai"], default="groq")
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--results-dir", default=str(ROOT / "experiments" / "results"))
    args = ap.parse_args()

    results = load_results(args.results_dir)
    cache = load_cache()
    sections, summary = [], []

    for name in args.datasets:
        runs = load_runs3(name, args.results_dir)
        ds = load_beir_cached = None
        ds_dir = next(r for r in results if r.get("dataset") == name)["config"]["dataset"]["path"]
        from core.datasets.beir import load_beir
        ds = load_beir(ds_dir, name=name, version="1.0")
        qrels = ds.qrels
        texts = {q.id: q.text for q in ds.queries}
        test_q = [q for q in rv.load_split_qrels(ds_dir, args.test_split)
                  if all(q in runs[s] for s in STRATEGIES) and q in qrels and q in texts]

        strat_of = {}
        todo = [q for q in test_q if f"{args.model}:{q}" not in cache]
        print(f"{name}: {len(test_q)} test queries, {len(todo)} to ask the LLM")
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            prompt = build_prompt([texts[q] for q in batch])
            answers = parse_answers(call_llm(prompt, args), len(batch))
            for q, a in zip(batch, answers):
                cache[f"{args.model}:{q}"] = a
            save_cache(cache)
            print(f"  routed {min(i + args.batch_size, len(todo))}/{len(todo)}")
            time.sleep(6)
        strat_of = {q: cache[f"{args.model}:{q}"] for q in test_q}

        routed_scores, oracle_scores = [], []
        per_strat = {s: per_query_ndcg_at_k(runs[s], qrels, 10) for s in STRATEGIES}
        dist = {s: 0 for s in STRATEGIES}
        for q in test_q:
            s = strat_of[q]
            dist[s] += 1
            routed_scores.append(per_strat[s].get(q, 0.0))
            oracle_scores.append(max(per_strat[x].get(q, 0.0) for x in STRATEGIES))
        routed = sum(routed_scores) / len(routed_scores)
        best_uniform = max(sum(per_strat[s].get(q, 0.0) for q in test_q) / len(test_q)
                           for s in STRATEGIES)
        oracle = sum(oracle_scores) / len(oracle_scores)

        sections.append(
            f"### {name}\n\n- model: `{args.model}` ({args.provider})\n"
            f"- test queries: {len(test_q)}\n\n"
            f"| router | test nDCG@10 |\n|---|---|\n"
            f"| best uniform strategy | {best_uniform:.4f} |\n"
            f"| **LLM-routed (zero-shot)** | **{routed:.4f}** |\n"
            f"| oracle (per-query best) | {oracle:.4f} |\n\n"
            f"- routed distribution: " +
            ", ".join(f"{s} {dist[s]/len(test_q):.1%}" for s in STRATEGIES) + "\n"
            f"- routing gain over best uniform: **{routed - best_uniform:+.4f}**\n"
        )
        summary.append((name, len(test_q), best_uniform, routed, oracle))

    total = sum(n for _, n, *_ in summary)
    bu = sum(n * r[2] for r, n in [(r, r[1]) for r in summary]) / total
    ro = sum(n * r[3] for r, n in [(r, r[1]) for r in summary]) / total
    orc = sum(n * r[4] for r, n in [(r, r[1]) for r in summary]) / total
    lines = [f"# Router v6 (LLM-as-router, zero-shot) — {', '.join(args.datasets)}", "",
             f"- provider/model: `{args.provider}` / `{args.model}`", "",
             "| domain | test q | best uniform | LLM-routed | oracle |", "|---|---|---|---|---|"]
    for n_, q, u, r_, o in summary:
        lines.append(f"| {n_} | {q} | {u:.4f} | {r_:.4f} | {o:.4f} |")
    lines += [f"| **weighted** | {total} | **{bu:.4f}** | **{ro:.4f}** | {orc:.4f} |", "",
              f"**Gain over best uniform: {ro - bu:+.4f}**", "", "## Per-domain detail", ""] + sections
    report = "\n".join(lines)
    print(report)
    out = Path(args.results_dir) / (
        f"router-v6-llm-{'-'.join(args.datasets)}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"report saved: {out}")


if __name__ == "__main__":
    main()
