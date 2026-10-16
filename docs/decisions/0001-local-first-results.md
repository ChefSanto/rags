# ADR 0001: Local-first, file-based experiment results

Date: 2026-10-15
Status: accepted

## Context

Roadmap Phase 7 introduces a Cloudflare D1 experiment database, scheduled
*before* the retrieval baselines (Phase 4–6). Building the DB schema first
risks coupling research code to an infrastructure that doesn't exist yet, and
the dashboard (Phase 8) is the only real consumer of that schema.

## Decision

Experiment results are JSON files in `experiments/results/`, committed to git.
A registry module (`core/experiments/registry.py`) indexes them and renders
comparisons. The D1 layer, when it arrives, syncs from these files; the
registry interface (`load_results`, `comparison_rows`) is the seam.

## Consequences

- The scientific loop works with zero infrastructure: clone, install, run.
- Results are reviewable in pull requests and blameable in git history.
- Result files must stay small (metrics + metadata, no raw runs). Large
  per-query runs/artifacts go to R2 later with a pointer in the result file.
- Comparison tooling reads a directory, not a network — it works offline,
  on Kaggle, and in CI without change.

## Alternatives considered

- **D1 from day one**: rejected — couples Phase 4 to Phase 7/8, needs network
  for every local run, and schemas always change once real experiments exist.
- **SQLite locally, D1 later**: rejected for now — one more moving part; JSON
  files are sufficient until result volume forces a real query layer.
