/**
 * AI Lab worker — serving layer of the working RAG.
 * POST /query | GET /logs | GET /health   (CORS-enabled for the dashboard)
 */

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const PROMPT = (ctx, q) =>
  `Answer the question using only the context below. ` +
  `If the context is insufficient, say you don't know.\n\n` +
  `Context:\n${ctx}\n\nQuestion: ${q}\nAnswer:`;

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS });
    }
    const url = new URL(request.url);
    let resp;
    try {
      if (request.method === "POST" && url.pathname === "/query") {
        resp = await handleQuery(request, env);
      } else if (url.pathname === "/logs") {
        resp = await handleLogs(url, env);
      } else if (url.pathname === "/health") {
        resp = Response.json({ ok: true, service: "ai-lab-worker" });
      } else {
        resp = Response.json(
          { error: "POST /query | GET /logs | GET /health" },
          { status: 404 }
        );
      }
    } catch (e) {
      resp = Response.json(
        { error: String((e && e.message) || e) },
        { status: 500 }
      );
    }
    const h = new Headers(resp.headers);
    for (const [k, v] of Object.entries(CORS)) h.set(k, v);
    return new Response(resp.body, { status: resp.status, headers: h });
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

  // 1. embed — same model used at upload time
  const emb = await env.AI.run("@cf/baai/bge-small-en-v1.5", { text: [question] });
  const vector = emb.data[0];

  // 2. retrieve
  const topK = parseInt(env.TOP_K || "5", 10);
  const res = await env.VECTORIZE.query(vector, { topK, returnMetadata: "all" });
  const hits = res.matches || [];

  // 3. context from retrieved texts
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
  const gen = await env.AI.run("@cf/meta/llama-3.2-3b-instruct", {
    messages: [
      {
        role: "system",
        content: "You are a careful assistant that answers only from the provided context.",
      },
      { role: "user", content: PROMPT(context || "(no documents retrieved)", question) },
    ],
    max_tokens: parseInt(env.MAX_TOKENS || "300", 10),
  });
  const answer = (gen.response || "").trim();
  const elapsed_ms = Date.now() - t0;

  // 5. log (best-effort)
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

async function handleLogs(url, env) {
  const limit = Math.min(parseInt(url.searchParams.get("limit") || "50", 10), 200);
  const { results } = await env.DB.prepare(
    `SELECT id, question, answer, num_sources, elapsed_ms, created_at
     FROM query_logs ORDER BY id DESC LIMIT ?`
  )
    .bind(limit)
    .all();
  return Response.json({ logs: results });
}
