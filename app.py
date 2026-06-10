import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env before any tool imports
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

# ── Lazy imports ──────────────────────────────────────────────────────────
# Heavy ML libraries (torch, transformers, sentence-transformers) take
# 30-60s to load on cold start. We defer them to first API call so
# the Flask server starts instantly and the dashboard is usable right away.

_tools_loaded = False

def _load_tools():
    global _tools_loaded, search_arxiv, ingest_papers, answer_question, _collection
    if _tools_loaded:
        return
    print("  Loading ML models (first request only)...")
    from tools.search import search_arxiv as _search
    from tools.ingest import ingest_papers as _ingest
    from tools.rag import answer_question as _answer
    from tools.kb import _collection as _coll
    search_arxiv = _search
    ingest_papers = _ingest
    answer_question = _answer
    _collection = _coll
    _tools_loaded = True
    print("  Models loaded -- ready!")


# ── Serve the dashboard ──────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── API Routes ────────────────────────────────────────────────────────────

@app.route("/api/search", methods=["POST"])
def api_search():
    """Search ArXiv for papers matching a query."""
    _load_tools()
    data = request.get_json()
    query = data.get("query", "")
    max_results = data.get("max_results", 10)
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        results = search_arxiv(query, max_results=max_results, top_k=top_k)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """Ingest papers into the knowledge base."""
    _load_tools()
    data = request.get_json()
    papers = data.get("papers", [])

    if not papers:
        return jsonify({"error": "No papers provided"}), 400

    try:
        summaries = ingest_papers(papers)
        return jsonify({"results": summaries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Ask a research question using RAG."""
    _load_tools()
    data = request.get_json()
    question = data.get("question", "")
    n_chunks = data.get("n_chunks", 5)

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        result = answer_question(question, n_chunks=n_chunks)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/kb", methods=["GET"])
def api_kb():
    """List all papers currently in the knowledge base."""
    _load_tools()
    try:
        all_data = _collection.get(include=["metadatas"])
        papers = {}
        for meta in all_data["metadatas"]:
            pid = meta.get("paper_id", "unknown")
            if pid not in papers:
                papers[pid] = {
                    "paper_id": pid,
                    "title": meta.get("title", "Unknown"),
                    "authors": meta.get("authors", "Unknown"),
                    "url": meta.get("url", ""),
                    "chunks": 0,
                }
            papers[pid]["chunks"] += 1

        return jsonify({"papers": list(papers.values()), "total_chunks": len(all_data["metadatas"])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  PaperPal -- Research Assistant Dashboard")
    print("  http://localhost:5000")
    print("=" * 50)
    print("  Server is starting... (models load on first request)")
    print("")
    app.run(debug=True, port=5000)
