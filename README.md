# PaperPal

**AI-powered research assistant** that searches ArXiv, ingests academic papers, and answers questions grounded in the papers -- with citations.

Built as an **MCP server** (for Claude Desktop) and a **web dashboard** for standalone use.

---

## How It Works

```
User question
     |
search_papers   -->  ArXiv API  -->  semantic re-ranking
     |
ingest_papers   -->  PDF download  -->  parse  -->  chunk  -->  embed  -->  ChromaDB
     |
ask             -->  retrieve chunks  -->  OpenAI LLM  -->  answer + citations
```

---

## Features

- **Semantic Search** -- Search ArXiv with AI-powered re-ranking (not just keyword matching)
- **Paper Ingestion** -- Download PDFs, extract text, chunk, embed, and store in a vector database
- **RAG Q&A** -- Ask questions and get answers grounded in your ingested papers with inline citations
- **MCP Server** -- Connect to Claude Desktop as a tool server
- **Web Dashboard** -- Clean dark-themed UI with Search, Knowledge Base, and Ask tabs

---

## Quick Start

### 1. Clone & setup
```bash
git clone https://github.com/YOUR_USERNAME/paperpal.git
cd paperpal
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Add your OpenAI API key
Create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the Web Dashboard
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

### 4. Or connect to Claude Desktop
Add this to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "paperpal": {
      "command": "python",
      "args": ["/path/to/paperpal/server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-your-key-here"
      }
    }
  }
}
```

---

## Project Structure

```
paperpal/
|-- app.py              # Flask web dashboard
|-- server.py           # MCP server for Claude Desktop
|-- eval.py             # Evaluation test suite
|-- requirements.txt    # Pinned dependencies
|-- .env                # API keys (not committed)
|
|-- tools/
|   |-- search.py       # ArXiv search + semantic re-ranking
|   |-- ingest.py       # PDF download -> parse -> chunk -> embed -> store
|   |-- rag.py          # Question -> retrieve -> LLM -> answer + citations
|   |-- kb.py           # ChromaDB setup, save_chunks(), retrieve_chunks()
|
|-- utils/
|   |-- embedder.py     # sentence-transformers wrapper
|   |-- chunker.py      # Sliding-window word chunker
|
|-- static/
|   |-- index.html      # Dashboard HTML
|   |-- style.css       # Dark navy + amber academic theme
|   |-- script.js       # Frontend logic
|
|-- chroma_db/          # Persistent vector store (auto-created)
```

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_papers` | Search ArXiv and return top papers ranked by semantic relevance |
| `ingest_papers` | Download, parse, chunk, embed, and store papers in ChromaDB |
| `ask` | Answer a research question using retrieved paper chunks + OpenAI |

---

## Tech Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB (persistent, local)
- **PDF Parsing**: PyMuPDF
- **LLM**: OpenAI (gpt-4o-mini)
- **Protocol**: MCP (Model Context Protocol)
- **Web**: Flask + vanilla HTML/CSS/JS
- **Search**: ArXiv API with cosine similarity re-ranking

---

## Evaluation Results

All 5 test suites pass:

| Test | Result |
|------|--------|
| Embedder | 384-dim vectors, semantic sanity check passed |
| Chunker | Correct chunking with 50-word overlap |
| ArXiv Search | Sorted by semantic score, all fields present |
| Ingest + Retrieve | PDF ingested, correct chunks retrieved |
| RAG End-to-End | Grounded answer with citations generated |

Run the evaluation yourself:
```bash
python eval.py
```

---

## License

MIT
