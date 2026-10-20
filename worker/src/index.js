/**
 * AI Lab worker — serving layer of the working RAG.
 * POST /query      single-shot question (no memory)
 * POST /chat       conversational turn (creates/resumes a chat)
 * GET  /chats      list chats
 * GET  /chat?id=   load one chat's messages
 * POST /ingest-text | GET /logs | GET /health
 */

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, X-Ingest-Key",
};

const SYSTEM = {
  role: "system",
  content:
    "You are a careful assistant that answers only from the provided Context. " +
    "If the context is insufficient, say you don't know. " +
    "Use the Conversation history only to resolve follow-up references " +
    "(pronouns, 'the second method'); factual claims must come from the Context.",
};

const CONTEXT_PROMPT = (ctx, q) =>
  `${ctx}\n\nContext:\n${ctx}\n\nQuestion: ${q}\nAnswer:`;

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
      } else if (request.method === "POST" && url.pathname === "/chat") {
        resp = await handleChat(request, env);
      } else if (url.pathname === "/chats") {
        resp = await handleChatList(env);
      } else if (url.pathname === "/chat") {
        resp = await handleChatGet(url, env);
      } else if (request.method === "POST" && url.pathname === "/ingest-text") {
        resp = await handleIngest(request, env);
      } else if (url.pathname === "/logs") {
        resp = await handleLogs(url, env);
      } else if (url.pathname === "/health") {
        resp = Response.json({ ok: true, service: "ai-lab-worker" });
      } else {
        resp = Response.json(
          { error: "POST /query | POST /chat | GET /chats | GET /chat | POST /ingest-text | GET /logs | GET /health" },
          { status: 404 }
        );
      }
    } catch (e) {
      resp = Response.json({ error: String((e && e.message) || e) }, { status: 500 });
    }
    const h = new Headers(resp.headers);
    for (const [k, v] of Object.entries(CORS)) h.set(k, v);
    return new Response(resp.body, { status: resp.status, headers: h });
  },
};

async function retrieve(env, question, topK) {
  const emb = await env.AI.run("@cf/baai/bge-small-en-v1.5", { text: [question] });
  const vector = emb.data[0];
  const fetchN = topK * 2;
  const [fiqaRes, persRes] = await Promise.all([
    env.VECTORIZE.query(vector, { topK: fetchN, returnMetadata: "all" }),
    env.VECTORIZE_PERSONAL.query(vector, { topK: fetchN, returnMetadata: "all" })
      .catch(() => ({ matches: [] })),
  ]);
  const hits = [...(fiqaRes.matches || []), ...(persRes.matches || [])]
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

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
  return { context, sources };
}

async function generate(env, messages, maxTokens) {
  const gen = await env.AI.run("@cf/meta/llama-3.2-3b-instruct", {
    messages,
    max_tokens: parseInt(env.MAX_TOKENS || "300", 10),
  });
  return (gen.response || "").trim();
}

async function logQuery(env, question, answer, n, ms) {
  try {
    await env.DB.prepare(
      `INSERT INTO query_logs (question, answer, num_sources, elapsed_ms, created_at)
       VALUES (?, ?, ?, ?, datetime('now'))`
    ).bind(question, answer, n, ms).run();
  } catch (e) {
    console.error("d1 log failed:", e);
  }
}

async function handleQuery(request, env) {
  const t0 = Date.now();
  const body = await request.json().catch(() => null);
  const question = ((body && body.question) || "").trim();
  if (!question) return Response.json({ error: "missing 'question'" }, { status: 400 });

  const { context, sources } = await retrieve(env, question, parseInt(env.TOP_K || "5", 10));
  const answer = await generate(env, [SYSTEM, { role: "user", content: CONTEXT_PROMPT(context || "(no documents retrieved)", question) }]);
  const elapsed_ms = Date.now() - t0;
  await logQuery(env, question, answer, sources.length, elapsed_ms);
  return Response.json({ answer, sources, elapsed_ms });
}

async function handleChat(request, env) {
  const t0 = Date.now();
  const body = await request.json().catch(() => null);
  const message = ((body && body.message) || "").trim();
  if (!message) return Response.json({ error: "missing 'message'" }, { status: 400 });

  let chatId = (body.chat_id || "").trim() || null;
  const now = new Date().toISOString();
  if (!chatId) {
    chatId = crypto.randomUUID();
    const title = message.length > 40 ? message.slice(0, 40) + "…" : message;
    await env.DB.prepare("INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)")
      .bind(chatId, title, now).run();
  }
  await env.DB.prepare("INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, 'user', ?, ?)")
    .bind(chatId, message, now).run();

  // prior turns for prompt context: last 6 turns (12 messages), excluding the one just inserted
  const { results: raw } = await env.DB.prepare(
    "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT 13"
  ).bind(chatId).all();
  const prior = raw.slice(1).reverse();

  const { context, sources } = await retrieve(env, message, parseInt(env.TOP_K || "5", 10));

  let hist = "";
  if (prior.length) {
    hist = "Conversation so far:\n" + prior.map((m) => `${m.role}: ${m.content}`).join("\n") + "\n\n";
  }
  const userContent =
    hist +
    `Context:\n${context || "(no documents retrieved)"}\n\nCurrent question: ${message}\nAnswer:`;

  const answer = await generate(env, [SYSTEM, { role: "user", content: userContent }]);
  const elapsed_ms = Date.now() - t0;

  await env.DB.prepare("INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)")
    .bind(chatId, answer, new Date().toISOString()).run();
  await logQuery(env, message, answer, sources.length, elapsed_ms);

  const { results: chatRow } = await env.DB.prepare("SELECT title FROM chats WHERE id = ?")
    .bind(chatId).all();
  return Response.json({
    chat_id: chatId,
    title: chatRow && chatRow.length ? chatRow[0].title : "",
    answer, sources, elapsed_ms,
  });
}

async function handleChatList(env) {
  const { results } = await env.DB.prepare(
    `SELECT c.id, c.title, c.created_at, COUNT(m.id) AS n,
            COALESCE(MAX(m.created_at), c.created_at) AS last_at
     FROM chats c LEFT JOIN messages m ON m.chat_id = c.id
     GROUP BY c.id ORDER BY last_at DESC`
  ).all();
  return Response.json({ chats: results });
}

async function handleChatGet(url, env) {
  const id = url.searchParams.get("id");
  if (!id) return Response.json({ error: "missing id" }, { status: 400 });
  const { results: chat } = await env.DB.prepare("SELECT id, title, created_at FROM chats WHERE id = ?").bind(id).all();
  if (!chat || !chat.length) return Response.json({ error: "chat not found" }, { status: 404 });
  const { results: messages } = await env.DB.prepare(
    "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id ASC"
  ).bind(id).all();
  return Response.json({ chat: chat[0], messages });
}

async function handleIngest(request, env) {
  if (env.INGEST_KEY) {
    if (request.headers.get("X-Ingest-Key") !== env.INGEST_KEY) {
      return Response.json({ error: "invalid ingest key" }, { status: 401 });
    }
  }
  const body = await request.json().catch(() => null);
  const chunks = body && body.chunks;
  if (!Array.isArray(chunks) || chunks.length === 0) {
    return Response.json({ error: "body must be {chunks: [{id, text}, ...]}" }, { status: 400 });
  }
  if (chunks.length > 500) {
    return Response.json({ error: "max 500 chunks per request" }, { status: 400 });
  }
  let uploaded = 0;
  for (let i = 0; i < chunks.length; i += 50) {
    const batch = chunks.slice(i, i + 50);
    const texts = batch.map((c) => String(c.text).slice(0, 2000));
    const emb = await env.AI.run("@cf/baai/bge-small-en-v1.5", { text: texts });
    await env.VECTORIZE_PERSONAL.upsert(
      batch.map((c, k) => ({
        id: String(c.id).slice(0, 200),
        values: emb.data[k],
        metadata: { text: texts[k] },
      }))
    );
    uploaded += batch.length;
  }
  return Response.json({ uploaded });
}

async function handleLogs(url, env) {
  const limit = Math.min(parseInt(url.searchParams.get("limit") || "50", 10), 200);
  const { results } = await env.DB.prepare(
    `SELECT id, question, answer, num_sources, elapsed_ms, created_at
     FROM query_logs ORDER BY id DESC LIMIT ?`
  ).bind(limit).all();
  return Response.json({ logs: results });
}
