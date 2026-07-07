import chromadb
from pathlib import Path
from utils.embedder import Embedder
import numpy as np
from rank_bm25 import BM25Okapi

# Absolute path so ChromaDB works regardless of working directory
_DB_PATH = str(Path(__file__).parent.parent / "chroma_db")

# ── Persistent ChromaDB client ──────────────────────────────────────────────
_client = chromadb.PersistentClient(path=_DB_PATH)

# One collection holds every chunk from every paper we ingest.
_collection = _client.get_or_create_collection(
    name="research_papers",
    metadata={"hnsw:space": "cosine"},
)

# Lazy-loaded embedder — model loads on first use, not at import time.
_embedder: Embedder | None = None

def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder

# ── BM25 In-Memory Index (Per-Session) ───────────────────────────────────────
_bm25_cache = {}

def _init_bm25(session_id: str):
    """Load documents for a specific session into an in-memory BM25 index."""
    if session_id not in _bm25_cache:
        data = _collection.get(where={"session_id": session_id}, include=["documents", "metadatas"])
        if data and data["documents"]:
            docs = data["documents"]
            metas = data["metadatas"]
            tokenized_corpus = [doc.lower().split(" ") for doc in docs]
            bm25 = BM25Okapi(tokenized_corpus)
            _bm25_cache[session_id] = {"bm25": bm25, "docs": docs, "metas": metas}
        else:
            _bm25_cache[session_id] = {"bm25": None, "docs": [], "metas": []}

def _rebuild_bm25(session_id: str):
    """Force rebuild BM25 after new documents are ingested."""
    cache = _bm25_cache.get(session_id)
    if cache and cache["docs"]:
        tokenized_corpus = [doc.lower().split(" ") for doc in cache["docs"]]
        cache["bm25"] = BM25Okapi(tokenized_corpus)

# ── Save ─────────────────────────────────────────────────────────────────────

def save_chunks(session_id: str, paper_id: str, chunks: list[str], metadata: dict) -> None:
    """
    Embed a list of text chunks and upsert them into ChromaDB.
    Also updates the in-memory BM25 index.
    """
    if not chunks:
        return

    embeddings = _get_embedder().embed(chunks)
    ids = [f"{session_id}_{paper_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "chunk_index": i, "session_id": session_id} for i in range(len(chunks))]

    _collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    # Update BM25
    _init_bm25(session_id)
    cache = _bm25_cache[session_id]
    cache["docs"].extend(chunks)
    cache["metas"].extend(metadatas)
    _rebuild_bm25(session_id)


# ── Retrieve (Hybrid Search + RRF) ──────────────────────────────────────────

def retrieve_chunks(session_id: str, query: str, n_results: int = 5) -> list[dict]:
    """
    Retrieve chunks using Hybrid Search:
      1. Vector Search (Semantic) via ChromaDB
      2. BM25 Search (Keyword) via rank_bm25
      3. Fuse rankings using Reciprocal Rank Fusion (RRF)
    """
    _init_bm25(session_id)
    
    # 1. Vector Search
    query_embedding = _get_embedder().embed([query])[0]
    
    # Fetch extra results to give RRF more overlap to work with
    fetch_k = n_results * 2 
    
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        where={"session_id": session_id},
        include=["documents", "metadatas", "distances"],
    )

    vector_ranked = []
    if results and results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            vector_ranked.append({"text": doc, "metadata": meta})

    # 2. BM25 Search
    bm25_ranked = []
    cache = _bm25_cache.get(session_id, {"bm25": None, "docs": [], "metas": []})
    bm25 = cache["bm25"]
    if bm25 is not None:
        tokenized_query = query.lower().split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_n_idx = np.argsort(doc_scores)[::-1][:fetch_k]
        
        for idx in top_n_idx:
            if doc_scores[idx] > 0:  # Only include if it actually matched keywords
                bm25_ranked.append({
                    "text": cache["docs"][idx], 
                    "metadata": cache["metas"][idx]
                })

    # 3. Reciprocal Rank Fusion (RRF)
    # RRF Score = sum( 1 / (k + rank) ) where k is typically 60
    rrf_scores = {}
    
    for rank, item in enumerate(vector_ranked):
        text = item["text"]
        if text not in rrf_scores:
            rrf_scores[text] = {"metadata": item["metadata"], "score": 0.0}
        rrf_scores[text]["score"] += 1.0 / (60 + rank)
        
    for rank, item in enumerate(bm25_ranked):
        text = item["text"]
        if text not in rrf_scores:
            rrf_scores[text] = {"metadata": item["metadata"], "score": 0.0}
        rrf_scores[text]["score"] += 1.0 / (60 + rank)
        
    # Sort by RRF score descending
    fused_results = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:n_results]
    
    # Return in the expected format
    return [
        {
            "text": text, 
            "metadata": data["metadata"], 
            "distance": -data["score"] # Negative score so "lower distance" is better, keeping compatibility
        } 
        for text, data in fused_results
    ]

# ── Clear Session ────────────────────────────────────────────────────────────

def clear_session_kb(session_id: str) -> None:
    """Delete all chunks for a specific session."""
    _collection.delete(where={"session_id": session_id})
    if session_id in _bm25_cache:
        del _bm25_cache[session_id]
