#!/usr/bin/env node
/**
 * Upload a BEIR corpus.jsonl into the ai-lab Vectorize index.
 * Embeddings via Workers AI @cf/baai/bge-small-en-v1.5 (384 dims) — the SAME
 * model the worker queries with, so upload and query stay in one space.
 *
 * Prerequisites:
 *   export CF_ACCOUNT_ID=...     # dashboard or: wrangler whoami
 *   export CF_API_TOKEN=...      # API token with Workers AI + Vectorize edit
 *
 * Usage:
 *   node scripts/upload_corpus.mjs ../datasets/downloads/fiqa/corpus.jsonl --limit 5000
 *
 * Note: full FiQA is 57,638 docs (~1,150 embedding calls). Free-tier Workers AI
 * has a daily quota — start with --limit 5000. Re-running re-upserts by id (safe).
 */
const ACCOUNT = process.env.CF_ACCOUNT_ID;
const TOKEN = process.env.CF_API_TOKEN;
const INDEX = process.env.VECTORIZE_INDEX || "ai-lab-corpus";

if (!ACCOUNT || !TOKEN) {
  console.error("set CF_ACCOUNT_ID and CF_API_TOKEN first");
  process.exit(1);
}

const [, , file, ...rest] = process.argv;
if (!file) {
  console.error("usage: node upload_corpus.mjs <corpus.jsonl> [--limit N]");
  process.exit(1);
}
const limitIdx = rest.indexOf("--limit");
const limit = limitIdx >= 0 ? parseInt(rest[limitIdx + 1], 10) : Infinity;
const skipIdx = rest.indexOf("--skip");
const skip = skipIdx >= 0 ? parseInt(rest[skipIdx + 1], 10) : 0;

const AI_URL = `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT}/ai/run/@cf/baai/bge-small-en-v1.5`;
const VZ_URL = `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT}/vectorize/v2/indexes/${INDEX}/upsert`;
const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };

import { createReadStream } from "node:fs";
import readline from "node:readline";

const rl = readline.createInterface({ input: createReadStream(file), crlfDelay: Infinity });
const docs = [];
for await (const line of rl) {
  if (!line.trim()) continue;
  const d = JSON.parse(line);
  docs.push({ id: String(d._id), text: String(d.text).slice(0, 2000) });
  if (docs.length >= skip + limit) break;
}
console.log(`loaded ${docs.length} docs; skipping ${skip}, processing from doc ${skip}`);

const todo = docs.slice(skip, skip + limit);
const BATCH = 50;
for (let i = 0; i < todo.length; i += BATCH) {
  const batch = todo.slice(i, i + BATCH);

  const er = await fetch(AI_URL, {
    method: "POST",
    headers,
    body: JSON.stringify({ text: batch.map((d) => d.text) }),
  });
  const ej = await er.json();
  if (!er.ok) {
    throw new Error(`embed failed at ${i}: ${er.status} ${JSON.stringify(ej).slice(0, 300)}`);
  }
  const vectors = ej.result.data;

  const lines = batch
    .map((d, k) => JSON.stringify({ id: d.id, values: vectors[k], metadata: { text: d.text } }))
    .join("\n");
  const ur = await fetch(VZ_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/x-ndjson" },
    body: lines,
  });
  const uj = await ur.json();
  if (!ur.ok) {
    throw new Error(`upsert failed at ${i}: ${ur.status} ${JSON.stringify(uj).slice(0, 300)}`);
  }
  console.log(`upserted ${Math.min(i + BATCH, todo.length)}/${todo.length}`);
}
console.log("done");
