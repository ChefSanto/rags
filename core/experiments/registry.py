"""File-based experiment registry.

Results live as JSON files in experiments/results/. This module scans them into
an index and renders comparison tables. When the Cloudflare dashboard arrives
(Phase 8), D1 syncs from these same files — the registry interface stays.

See docs/decisions/0001-local-first-results.md.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_RESULTS_DIR = "experiments/results"


REQUIRED_RESULT_KEYS = ("name", "dataset", "dataset_version", "retriever", "metrics")


def load_results(results_dir: str | Path = DEFAULT_RESULTS_DIR) -> list[dict]:
    """Scan result JSONs, skipping files that don't match the result schema.

    A malformed or foreign file must never break comparisons — it's skipped
    with a warning so one bad upload can't poison the registry.
    """
    rows = []
    for p in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[registry] skipping unreadable file: {p.name}")
            continue
        if not all(k in data for k in REQUIRED_RESULT_KEYS):
            print(f"[registry] skipping file without result schema: {p.name}")
            continue
        rows.append(data)
    return rows


def comparison_rows(results: list[dict]) -> list[dict]:
    """Flatten results into table rows (one per experiment)."""
    table = []
    for r in results:
        meta = r.get("metadata", {})
        commit = meta.get("git_commit") or "no-git"
        row = {
            "experiment": r.get("name", r.get("experiment_id", "?")),
            "dataset": f'{r.get("dataset")}@{r.get("dataset_version")}',
            "retriever": r.get("retriever"),
            "seed": r.get("seed"),
            "git_commit": commit[:8] + ("*" if meta.get("git_dirty") else ""),
        }
        row.update(r.get("metrics", {}))
        table.append(row)
    return table


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "(no results yet)"
    metric_keys = []
    for r in rows:
        for k in r:
            if k not in ("experiment", "dataset", "retriever", "seed", "git_commit"):
                if k not in metric_keys:
                    metric_keys.append(k)
    headers = ["experiment", "dataset", "retriever", "seed", "git_commit"] + metric_keys

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)
