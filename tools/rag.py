import os
from openai import OpenAI
from tools.kb import retrieve_chunks

# Lazy-loaded client — created on first use so import never fails
# even if OPENAI_API_KEY hasn't been loaded yet.
_openai_client = None

def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client

# ── Prompt template ───────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a research assistant with deep expertise in academic papers.
Answer the user's question using ONLY the context chunks provided below.
Rules:
- Be precise and concise.
- Cite your sources inline using [Paper Title] notation.
- If the context doesn't contain enough information to answer, say so honestly.
- Do NOT make up facts or use outside knowledge."""

_CONTEXT_TEMPLATE = """Here are the most relevant excerpts from the research papers in the knowledge base:

{chunks}

---
Question: {question}
"""


def _format_chunks(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    Each chunk shows its source paper so the LLM can cite it.
    """
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk["metadata"].get("title", "Unknown Paper")
        url   = chunk["metadata"].get("url", "")
        text  = chunk["text"]
        formatted.append(
            f"[{i}] From: {title}\n"
            f"    URL: {url}\n"
            f"    Excerpt: {text}\n"
        )
    return "\n".join(formatted)


def _build_citations(chunks: list[dict]) -> list[dict]:
    """
    Build a deduplicated list of cited papers from the retrieved chunks.
    Used to return clean citation metadata alongside the answer.
    """
    seen = set()
    citations = []
    for chunk in chunks:
        pid = chunk["metadata"].get("paper_id")
        if pid and pid not in seen:
            seen.add(pid)
            citations.append({
                "paper_id": pid,
                "title":    chunk["metadata"].get("title"),
                "authors":  chunk["metadata"].get("authors"),
                "url":      chunk["metadata"].get("url"),
            })
    return citations


def answer_question(
    question: str,
    n_chunks: int = 5,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Full RAG pipeline: question → retrieve → prompt → LLM → answer + citations.

    Args:
        question: The user's natural-language research question.
        n_chunks: Number of context chunks to retrieve from ChromaDB.
        model:    OpenAI model to use for generation.

    Returns:
        Dict with:
            "answer"    – LLM-generated answer string
            "citations" – List of source paper dicts (paper_id, title, authors, url)
            "chunks"    – Raw retrieved chunks (useful for debugging / evaluation)
    """

    # ── Step 1: Retrieve relevant chunks ──────────────────────────────────────
    chunks = retrieve_chunks(question, n_results=n_chunks)

    if not chunks:
        return {
            "answer":    "No relevant papers found in the knowledge base. Try ingesting some papers first.",
            "citations": [],
            "chunks":    [],
        }

    # ── Step 2: Build the prompt ──────────────────────────────────────────────
    context_block = _format_chunks(chunks)
    user_message  = _CONTEXT_TEMPLATE.format(
        chunks=context_block,
        question=question,
    )

    # ── Step 3: Call the LLM ──────────────────────────────────────────────────
    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,   # low temp = factual, consistent answers
    )

    answer = response.choices[0].message.content.strip()

    # ── Step 4: Package citations ─────────────────────────────────────────────
    citations = _build_citations(chunks)

    return {
        "answer":    answer,
        "citations": citations,
        "chunks":    chunks,     # kept for evaluation / debugging
    }
