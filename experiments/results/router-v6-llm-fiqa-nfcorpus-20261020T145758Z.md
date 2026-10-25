# Router v6 (LLM-as-router, zero-shot) — fiqa, nfcorpus

- provider/model: `groq` / `openai/gpt-oss-20b`

| domain | test q | best uniform | LLM-routed | oracle |
|---|---|---|---|---|
| fiqa | 648 | 0.3711 | 0.3409 | 0.4338 |
| nfcorpus | 323 | 0.3368 | 0.3338 | 0.3842 |
| **weighted** | 971 | **0.3597** | **0.3386** | 0.4173 |

**Gain over best uniform: -0.0211**

## Per-domain detail

### fiqa

- model: `openai/gpt-oss-20b` (groq)
- test queries: 648

| router | test nDCG@10 |
|---|---|
| best uniform strategy | 0.3711 |
| **LLM-routed (zero-shot)** | **0.3409** |
| oracle (per-query best) | 0.4338 |

- routed distribution: bm25 25.3%, dense 41.5%, hybrid 33.2%
- routing gain over best uniform: **-0.0301**

### nfcorpus

- model: `openai/gpt-oss-20b` (groq)
- test queries: 323

| router | test nDCG@10 |
|---|---|
| best uniform strategy | 0.3368 |
| **LLM-routed (zero-shot)** | **0.3338** |
| oracle (per-query best) | 0.3842 |

- routed distribution: bm25 35.0%, dense 41.8%, hybrid 23.2%
- routing gain over best uniform: **-0.0029**

