# AI Lab

A reproducible experimentation platform for RAG research — and the artifact behind
the paper *Why Per-Query Retrieval Routing Fails to Beat Tuned Fusion: A
Mechanism-Level Study* (paper source in `papers/property-routing/`).

The repository holds three things in one place:

1. **A research lab** — declarative experiments (YAML configs), a reproducible
   runner with full provenance, retrieval (BM25 / dense / hybrid fusion),
   evaluation (recall, precision, MRR, nDCG + per-query variants), and a
   complete experimental study of per-query retrieval routing: seven router
   paradigms, break-even and decisive-stratum analyses, and a blind human
   evaluation. All results and run artifacts are committed; every analysis in
   the paper reproduces from a fresh clone (`scripts/kaggle_bootstrap.sh`,
   then the analysis scripts in `scripts/`).
2. **A deployed RAG product** — a Cloudflare Workers + Vectorize + D1
   pipeline (`worker/`) with a chat dashboard (`dashboard/`): dual-index
   retrieval over the FiQA corpus and user-uploaded documents, browser PDF
   upload, URL ingestion (`scripts/ingest_url.py`), reranking, and an
   app-key gate.
3. **A document-intelligence prototype** — a WordPress plugin
   (`wordpress/ai-lab-docintel/`) for structured field extraction from PDFs
   (upload -> LLM extraction -> human review -> CSV), calling the same worker.

## Quick orientation

| You want to... | Go to |
|---|---|
| Read the paper | `papers/property-routing/paper.tex` (compiled PDF on arXiv) |
| Reproduce the paper's analyses | Appendix A of the paper; `scripts/reviewer_stats.py`, `scripts/routing_win_analysis.py` |
| Use the deployed RAG | https://ai-lab-dashboard.pages.dev (requires the app key) |
| Run experiments | `experiments/configs/` (YAML), `scripts/run_experiment.py` |
| See all results | `experiments/results/` (committed JSONs + analysis reports) |

## Layout

- `core/` — retrieval, evaluation, generation, experiment schema
- `scripts/` — runner, baselines/router analyses (v0–v6), oracle/domain routers,
  pseudo-query generators, reviewer statistics, review-sheet and figure generators
- `experiments/` — configs and committed results
- `artifacts/runs/` — ranked run artifacts for all baselines (analyses read these)
- `worker/`, `dashboard/`, `wordpress/` — the deployed product pieces
- `papers/` — the routing study (proposal, reports, LaTeX)
- `datasets/` — manifest; downloads are fetched by the bootstrap script

Only the v6 LLM-router experiment requires an external API (Groq free tier);
all other analyses run from committed artifacts with no keys.

## Status

Complete for the routing study (three BEIR domains, sealed statistics,
human evaluation) and functional as a product (chat, upload, URLs, extraction).
See `papers/property-routing/` for the study and roadmap context.
