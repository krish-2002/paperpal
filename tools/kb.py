import chromadb
from pathlib import Path
from utils.embedder import Embedder

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
# This prevents Claude Desktop from timing out during the MCP handshake.
_embedder: Embedder | None = None

def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


# ── Save ─────────────────────────────────────────────────────────────────────

def save_chunks(paper_id: str, chunks: list[str], metadata: dict) -> None:
    """
    Embed a list of text chunks and upsert them into ChromaDB.

    Args:
        paper_id: Unique identifier for the paper (e.g. ArXiv ID "2301.07041").
        chunks:   List of text chunks produced by chunker.chunk_text().
        metadata: Dict of paper-level info (title, authors, url, …) attached
                  to every chunk so we can surface it in citations later.
    """
    if not chunks:
        return

    embeddings = _get_embedder().embed(chunks)

    # ChromaDB needs a unique string ID for every vector.
    # We combine the paper ID with the chunk index: "2301.07041_chunk_0"
    ids = [f"{paper_id}_chunk_{i}" for i in range(len(chunks))]

    # Each chunk gets the paper-level metadata PLUS its own index,
    # so we can reconstruct the reading order if needed.
    metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]

    _collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, n_results: int = 5) -> list[dict]:
    """
    Embed a query and return the top-n most semantically similar chunks.

    Args:
        query:     The user's natural-language question.
        n_results: How many chunks to return (default 5).

    Returns:
        List of dicts, each with keys:
            "text"     – the chunk text
            "metadata" – paper-level info (title, url, …)
            "distance" – cosine distance (lower = more similar)
    """
    query_embedding = _get_embedder().embed([query])[0]

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten ChromaDB's nested response into a clean list of dicts.
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})

    return chunks
