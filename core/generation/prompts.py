"""Prompt construction for the generation stage."""

DEFAULT_TEMPLATE = (
    "Answer the question using only the context below. "
    "If the context is insufficient, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def build_prompt(
    question: str,
    doc_texts: list[str],
    template: str | None = None,
    max_context_chars: int = 3000,
) -> str:
    """Render the default (or custom) RAG prompt for one query.

    Documents are numbered and appended until the context budget is used.
    """
    tmpl = template or DEFAULT_TEMPLATE
    parts: list[str] = []
    used = 0
    for i, text in enumerate(doc_texts):
        chunk = f"[{i + 1}] {text}"
        if used + len(chunk) > max_context_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    context = "\n\n".join(parts) if parts else "(no documents retrieved)"
    return tmpl.format(context=context, question=question)
