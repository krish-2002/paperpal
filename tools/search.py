import arxiv
import numpy as np
from utils.embedder import Embedder

_embedder = None

def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def search_arxiv(query: str, max_results: int = 10, top_k: int = 5) -> list[dict]:
    """
    Search ArXiv for papers matching `query`, then re-rank the results
    semantically so the most relevant papers float to the top.

    Args:
        query:       Natural-language search query (e.g. "attention mechanism transformers").
        max_results: How many papers to fetch from ArXiv before re-ranking.
        top_k:       How many papers to return after re-ranking.

    Returns:
        List of dicts (sorted best-first), each containing:
            "paper_id" – ArXiv ID (e.g. "2301.07041")
            "title"    – Paper title
            "authors"  – Comma-separated author names
            "abstract" – Full abstract text
            "url"      – Direct PDF link
            "score"    – Semantic similarity score (0–1)
    """

    # ── Step 1: Fetch candidates from ArXiv ──────────────────────────────────
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = list(client.results(search))

    if not papers:
        return []

    # ── Step 2: Embed the query and all abstracts in one batch ────────────────
    # We put the query first so we can grab it by index [0],
    # then slice [1:] for the abstracts.
    texts = [query] + [p.summary for p in papers]
    embeddings = _get_embedder().embed(texts)

    query_vec = embeddings[0]
    abstract_vecs = embeddings[1:]

    # ── Step 3: Score each paper against the query ───────────────────────────
    results = []
    for paper, abstract_vec in zip(papers, abstract_vecs):
        score = _cosine_similarity(query_vec, abstract_vec)
        results.append({
            "paper_id": paper.get_short_id(),
            "title":    paper.title,
            "authors":  ", ".join(str(a) for a in paper.authors),
            "abstract": paper.summary,
            "url":      paper.pdf_url,
            "score":    round(score, 4),
        })

    # ── Step 4: Re-rank by score, return top_k ───────────────────────────────
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
