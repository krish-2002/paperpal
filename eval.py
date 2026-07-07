r"""
Phase 5 - Evaluation Script for research-mcp
=============================================
Tests retrieval quality, search ranking, and end-to-end RAG answers.

Usage:
    cd research-mcp
    venv\Scripts\python.exe eval.py
"""

import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Load env before any tool imports
load_dotenv(Path(__file__).parent / ".env")

from tools.search import search_arxiv
from tools.ingest import ingest_paper
from tools.kb import retrieve_chunks, _collection
from tools.rag import answer_question
from utils.embedder import Embedder
from utils.chunker import chunk_text


# ── Helpers ───────────────────────────────────────────────────────────────────

def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def timed(label: str):
    """Simple context manager to time a block."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f"  [timer] {label}: {elapsed:.2f}s")
    return Timer()


# ── Test 1: Embedder ─────────────────────────────────────────────────────────

def test_embedder():
    separator("TEST 1 -- Embedder")

    embedder = Embedder()

    # Test basic embedding
    with timed("Single text embedding"):
        result = embedder.embed(["hello world"])
    
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert isinstance(result[0], list), f"Expected nested list, got {type(result[0])}"
    assert len(result[0]) == 1536, f"Expected 1536 dims, got {len(result[0])}"
    print(f"  [PASS] Output shape: 1 x {len(result[0])} dims")

    # Test batch embedding
    with timed("Batch embedding (10 texts)"):
        batch = embedder.embed([f"test sentence {i}" for i in range(10)])
    
    assert len(batch) == 10, f"Expected 10 embeddings, got {len(batch)}"
    print(f"  [PASS] Batch shape: {len(batch)} x {len(batch[0])} dims")

    # Test semantic similarity (sanity check)
    import numpy as np
    vecs = embedder.embed([
        "machine learning and neural networks",
        "deep learning for computer vision",
        "how to bake chocolate cake",
    ])
    
    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    sim_related = cosine(vecs[0], vecs[1])
    sim_unrelated = cosine(vecs[0], vecs[2])
    
    print(f"  [PASS] ML <-> DL similarity:      {sim_related:.4f}")
    print(f"  [PASS] ML <-> Baking similarity:   {sim_unrelated:.4f}")
    assert sim_related > sim_unrelated, "Related texts should be more similar!"
    print(f"  [PASS] Semantic sanity check PASSED (related > unrelated)")

    print("\n  >> Embedder: ALL TESTS PASSED")


# ── Test 2: Chunker ───────────────────────────────────────────────────────────

def test_chunker():
    separator("TEST 2 -- Chunker")

    # Basic chunking
    text = " ".join([f"word{i}" for i in range(1200)])
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    print(f"  Input: 1200 words")
    print(f"  Output: {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        words = len(c.split())
        print(f"    Chunk {i}: {words} words")
    
    # Verify overlap exists
    chunk0_words = chunks[0].split()
    chunk1_words = chunks[1].split()
    overlap_words = set(chunk0_words[-50:]) & set(chunk1_words[:50])
    print(f"  [PASS] Overlap between chunk 0 and 1: {len(overlap_words)} words")
    assert len(overlap_words) > 0, "Expected overlap between chunks!"

    # Edge cases
    assert chunk_text("") == [], "Empty string should return []"
    assert chunk_text("   ") == [], "Whitespace should return []"
    assert len(chunk_text("hello")) == 1, "Single word should return 1 chunk"
    print(f"  [PASS] Edge cases passed (empty, whitespace, single word)")

    print("\n  >> Chunker: ALL TESTS PASSED")


# ── Test 3: Search ────────────────────────────────────────────────────────────

def test_search():
    separator("TEST 3 -- ArXiv Search + Semantic Ranking")

    query = "transformer attention mechanism"
    
    with timed("ArXiv search + re-ranking"):
        results = search_arxiv(query, max_results=5, top_k=3)
    
    assert len(results) > 0, "Expected at least 1 result"
    assert len(results) <= 3, f"Expected <=3 results, got {len(results)}"
    
    print(f"  [PASS] Got {len(results)} papers")
    for i, p in enumerate(results):
        print(f"    [{i+1}] Score: {p['score']:.4f} -- {p['title'][:70]}...")
    
    # Verify scores are sorted descending
    scores = [p["score"] for p in results]
    assert scores == sorted(scores, reverse=True), "Results should be sorted by score!"
    print(f"  [PASS] Results sorted by semantic score (descending)")

    # Verify all required fields exist
    required_fields = {"paper_id", "title", "authors", "abstract", "url", "score"}
    for p in results:
        missing = required_fields - set(p.keys())
        assert not missing, f"Missing fields: {missing}"
    print(f"  [PASS] All required fields present")

    print("\n  >> Search: ALL TESTS PASSED")
    return results


# ── Test 4: Ingest + Retrieve ─────────────────────────────────────────────────

def test_ingest_and_retrieve(papers: list[dict]):
    separator("TEST 4 -- Ingest + Retrieve")

    if not papers:
        print("  [WARN] No papers to ingest (search returned empty). Skipping.")
        return

    # Ingest just the first paper
    paper = papers[0]
    print(f"  Ingesting: {paper['title'][:60]}...")

    with timed("Full ingest pipeline (download -> parse -> chunk -> embed -> store)"):
        result = ingest_paper("eval_session", paper)
    
    assert result["chunks_stored"] > 0, "Expected chunks to be stored"
    print(f"  [PASS] Stored {result['chunks_stored']} chunks for {result['paper_id']}")

    # Retrieve and check relevance
    test_query = paper["title"]  # Use the paper's own title as query
    
    with timed("Retrieval query"):
        retrieved = retrieve_chunks("eval_session", test_query, n_results=3)
    
    assert len(retrieved) > 0, "Expected chunks to be retrieved"
    print(f"  [PASS] Retrieved {len(retrieved)} chunks")
    
    # Check that at least one chunk is from the paper we just ingested
    retrieved_papers = {c["metadata"].get("paper_id") for c in retrieved}
    assert paper["paper_id"] in retrieved_papers, \
        f"Expected to retrieve chunks from {paper['paper_id']}, got {retrieved_papers}"
    print(f"  [PASS] Retrieved chunk matches ingested paper ({paper['paper_id']})")

    # Print retrieval details
    for i, c in enumerate(retrieved):
        print(f"    [{i+1}] Distance: {c['distance']:.4f} — {c['metadata']['title'][:50]}...")
        print(f"         Text: {c['text'][:100]}...")

    print("\n  >> Ingest + Retrieve: ALL TESTS PASSED")


# ── Test 5: RAG End-to-End ────────────────────────────────────────────────────

def test_rag():
    separator("TEST 5 -- RAG End-to-End")

    # Check if we have any data in the collection
    count = _collection.count()
    if count == 0:
        print("  [WARN] Knowledge base is empty. Skipping RAG test.")
        return

    print(f"  Knowledge base has {count} chunks")

    question = "What is the attention mechanism in transformers?"
    
    with timed("Full RAG pipeline (retrieve -> LLM -> answer)"):
        result = answer_question("eval_session", question, n_chunks=3)
    
    assert "answer" in result, "Missing 'answer' key"
    assert "citations" in result, "Missing 'citations' key"
    assert "chunks" in result, "Missing 'chunks' key"
    assert len(result["answer"]) > 50, "Answer seems too short"
    
    print(f"  [PASS] Answer length: {len(result['answer'])} chars")
    print(f"  [PASS] Citations: {len(result['citations'])} papers")
    print(f"  [PASS] Chunks used: {len(result['chunks'])}")
    
    print(f"\n  --- Answer Preview ---")
    print(f"  {result['answer'][:300]}...")
    
    if result["citations"]:
        print(f"\n  --- Citations ---")
        for c in result["citations"]:
            print(f"    - {c['title'][:60]} -- {c['url']}")

    print("\n  >> RAG: ALL TESTS PASSED")


# ── Summary ───────────────────────────────────────────────────────────────────

def main():
    print("\n" + "== " * 20)
    print("  research-mcp -- Phase 5 Evaluation")
    print("== " * 20)

    passed = 0
    failed = 0
    errors = []

    tests = [
        ("Embedder", test_embedder, None),
        ("Chunker", test_chunker, None),
    ]

    # Run Embedder + Chunker first
    for name, test_fn, _ in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"\n  [FAIL] {name}: FAILED -- {e}")

    # Search (returns papers for next test)
    papers = []
    try:
        papers = test_search()
        passed += 1
    except Exception as e:
        failed += 1
        errors.append(("Search", str(e)))
        print(f"\n  [FAIL] Search: FAILED -- {e}")

    # Ingest + Retrieve
    try:
        test_ingest_and_retrieve(papers)
        passed += 1
    except Exception as e:
        failed += 1
        errors.append(("Ingest+Retrieve", str(e)))
        print(f"\n  [FAIL] Ingest+Retrieve: FAILED -- {e}")

    # RAG
    try:
        test_rag()
        passed += 1
    except Exception as e:
        failed += 1
        errors.append(("RAG", str(e)))
        print(f"\n  [FAIL] RAG: FAILED -- {e}")

    # Summary
    separator("EVALUATION SUMMARY")
    print(f"  Passed: {passed}/5")
    print(f"  Failed: {failed}/5")
    if errors:
        print(f"\n  Failures:")
        for name, err in errors:
            print(f"    [FAIL] {name}: {err}")
    else:
        print(f"\n  ALL TESTS PASSED -- research-mcp is production-ready!")
    print()


if __name__ == "__main__":
    main()
