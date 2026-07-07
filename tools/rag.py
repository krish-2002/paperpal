import os
import time
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
- Before answering, think step-by-step about what the context says.
- Be precise and concise.
- Cite your sources inline using [Paper Title] notation.
- If the context doesn't contain enough information to answer, say so honestly.
- Do NOT make up facts or use outside knowledge."""

_CONTEXT_TEMPLATE = """Here are the most relevant excerpts from the research papers in the knowledge base:

{chunks}

---
Question: {question}
"""

# ── Abstention Threshold ──────────────────────────────────────────────────────
# If the best retrieved chunk has an RRF score below this threshold,
# the system refuses to answer rather than hallucinating from weak context.
# RRF scores are stored as negative distances, so we compare abs values.
_ABSTENTION_THRESHOLD = 0.015  # Empirically tuned for RRF with k=60


def _compute_confidence(chunks: list[dict]) -> float:
    """
    Compute a confidence score (0.0 to 1.0) based on retrieval quality.
    
    Factors:
      1. Best chunk's RRF score (higher = more relevant retrieval)
      2. Score spread (if top chunks have similar scores, context is consistent)
      3. Source diversity (chunks from same paper = more focused = higher confidence)
    """
    if not chunks:
        return 0.0
    
    # RRF scores are stored as negative distances
    scores = [abs(c["distance"]) for c in chunks]
    
    # Factor 1: Best score (normalized to 0-1 range)
    # Typical RRF scores range from ~0.016 to ~0.033 for good matches
    best_score = max(scores)
    score_factor = min(best_score / 0.035, 1.0)  # Cap at 1.0
    
    # Factor 2: Consistency (how similar are the top scores)
    if len(scores) >= 2:
        avg_score = sum(scores) / len(scores)
        consistency = min(avg_score / best_score, 1.0) if best_score > 0 else 0
    else:
        consistency = 0.5
    
    # Factor 3: Source focus (more chunks from same paper = higher confidence)
    paper_ids = [c["metadata"].get("paper_id", "") for c in chunks]
    unique_papers = len(set(paper_ids))
    focus_factor = 1.0 - (unique_papers - 1) / max(len(chunks), 1) * 0.3
    
    # Weighted combination
    confidence = (score_factor * 0.5) + (consistency * 0.3) + (focus_factor * 0.2)
    return round(min(max(confidence, 0.0), 1.0), 2)


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
    session_id: str,
    question: str,
    chat_history: list[dict] = None,
    n_chunks: int = 8,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Full RAG pipeline: question → retrieve → confidence check → prompt → LLM → answer + citations.

    Returns:
        Dict with:
            "answer"     – LLM-generated answer string
            "citations"  – List of source paper dicts
            "chunks"     – Raw retrieved chunks
            "confidence" – Float 0.0-1.0 confidence score
            "timings"    – Dict with latency breakdown in seconds
            "tokens"     – Approximate token usage
    """
    timings = {}

    # ── Step 1: Retrieve relevant chunks ──────────────────────────────────────
    t0 = time.time()
    chunks = retrieve_chunks(session_id, question, n_results=n_chunks)
    timings["retrieval"] = round(time.time() - t0, 2)

    if not chunks:
        return {
            "answer":     "No relevant papers found in the knowledge base. Try ingesting some papers first.",
            "citations":  [],
            "chunks":     [],
            "confidence": 0.0,
            "timings":    timings,
            "tokens":     0,
        }

    # ── Step 1.5: Abstention Check ────────────────────────────────────────────
    best_score = max(abs(c["distance"]) for c in chunks)
    confidence = _compute_confidence(chunks)
    
    if best_score < _ABSTENTION_THRESHOLD:
        return {
            "answer":     "⚠️ **Low confidence — abstaining from answer.**\n\nThe retrieved context is too weakly related to your question. "
                          "This usually means the knowledge base doesn't contain papers relevant to this topic.\n\n"
                          "**Try:** Ingesting papers that specifically cover this topic, then ask again.",
            "citations":  [],
            "chunks":     chunks,
            "confidence": confidence,
            "timings":    timings,
            "tokens":     0,
        }

    # ── Step 2: Build the prompt ──────────────────────────────────────────────
    context_block = _format_chunks(chunks)
    user_message  = _CONTEXT_TEMPLATE.format(
        chunks=context_block,
        question=question,
    )

    # ── Step 3: Call the LLM ──────────────────────────────────────────────────
    t0 = time.time()
    
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})
    
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,   # low temp = factual, consistent answers
    )
    timings["llm"] = round(time.time() - t0, 2)

    answer = response.choices[0].message.content.strip()
    
    # Extract token usage
    tokens_used = 0
    if response.usage:
        tokens_used = response.usage.total_tokens

    # ── Step 4: Package citations ─────────────────────────────────────────────
    citations = _build_citations(chunks)

    return {
        "answer":     answer,
        "citations":  citations,
        "chunks":     chunks,
        "confidence": confidence,
        "timings":    timings,
        "tokens":     tokens_used,
    }
