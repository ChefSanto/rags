# Router v6 (LLM-as-router, zero-shot) — fiqa, nfcorpus

- provider/model: `groq` / `openai/gpt-oss-20b`

| domain | test q | best uniform | LLM-routed | oracle |
|---|---|---|---|---|
| fiqa | 648 | 0.3711 | 0.3711 | 0.4338 |
| nfcorpus | 323 | 0.3368 | 0.3374 | 0.3842 |
| **weighted** | 971 | **0.3597** | **0.3599** | 0.4173 |

**Gain over best uniform: +0.0002**

## Per-domain detail

### fiqa

- model: `openai/gpt-oss-20b` (groq)
- test queries: 648

| router | test nDCG@10 |
|---|---|
| best uniform strategy | 0.3711 |
| **LLM-routed (zero-shot)** | **0.3711** |
| oracle (per-query best) | 0.4338 |

- routed distribution: bm25 0.0%, dense 0.0%, hybrid 100.0%
- routing gain over best uniform: **+0.0000**

### nfcorpus

- model: `openai/gpt-oss-20b` (groq)
- test queries: 323

| router | test nDCG@10 |
|---|---|
| best uniform strategy | 0.3368 |
| **LLM-routed (zero-shot)** | **0.3374** |
| oracle (per-query best) | 0.3842 |

- routed distribution: bm25 0.9%, dense 2.5%, hybrid 96.6%
- routing gain over best uniform: **+0.0006**

