/**
 * AI Lab worker — the serving layer of the working RAG.
 *
 * POST /query  {question} -> embed -> Vectorize retrieve -> Llama generate -> log to D1
 * GET  /health
 *
 * Retrieval/generation models are Cloudflare-native (BGE-small embeddings,
 * Llama-3.1-8B-instruct) — NOT the lab's benchmarked MiniLM/Qwen stack. The
 * lab remains the source of benchmark truth.
 */

const PROMPT = (ctx, q) =>
  `Answer the question using only the context below. ` +
  `If the context is insufficient, say you don't know.\n\n` +
  `Context:\n${ctx}\n\nQuestion: ${q}\nAnswer:`;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (request.method === "POST" && url.pathname === "/query") {
      return handleQuery(request, env);
    }
    if (url.pathname === "/health") {
      return Response.json({ ok: true, service: "ai-lab-worker" });
    }
    return Response.json(
      { error: "POST /query with JSON body {question}" },
      { status: 404 }
    );
  },
};

async function handleQuery(request, env) {
  const t0 = Date.now();
  let body;
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "invalid JSON body" }, { status: 400 });
  }
  const question = (body.question || "").trim();
  if (!question) {
    return Response.json({ error: "missing 'question'" }, { status: 400 });
  }

  // 1. embed the question — same model used at upload time
  const emb = await env.AI.run("@cf/baai/bge-small-en-v1.5", { text: [question] });
  const vector = emb.data[0];

  // 2. retrieve
  const topK = parseInt(env.TOP_K || "5", 10);
  const res = await env.VECTORIZE.query(vector, { topK, returnMetadata: "all" });
  const hits = res.matches || [];

  // 3. build context from retrieved texts (stored as metadata at upload)
  const budget = parseInt(env.MAX_CONTEXT_CHARS || "3000", 10);
  let context = "";
  let used = 0;
  const sources = [];
  for (let i = 0; i < hits.length; i++) {
    const text = (hits[i].metadata && hits[i].metadata.text) || "";
    const chunk = `[${i + 1}] ${text}`;
    if (used + chunk.length > budget) break;
    context += (context ? "\n\n" : "") + chunk;
    used += chunk.length;
    sources.push({ id: hits[i].id, score: hits[i].score, preview: text.slice(0, 200) });
  }

  // 4. generate
  const gen = await env.AI.run("@cf/meta/llama-3.1-8b-instruct", {
    messages: [
      {
        role: "system",
        content: "You are a careful assistant that answers only from the provided context.",
      },
      { role: "user", content: PROMPT(context || "(no documents retrieved)", question) },
    ],
    max_tokens: 300,
  });
  const answer = (gen.response || "").trim();
  const elapsed_ms = Date.now() - t0;

  // 5. log to D1 (best-effort — a logging failure must not break the query)
  try {
    await env.DB.prepare(
      `INSERT INTO query_logs (question, answer, num_sources, elapsed_ms, created_at)
       VALUES (?, ?, ?, ?, datetime('now'))`
    )
      .bind(question, answer, sources.length, elapsed_ms)
      .run();
  } catch (e) {
    console.error("d1 log failed:", e);
  }

  return Response.json({ answer, sources, elapsed_ms });
}
