# 📚 PaperPal

> **Your AI-powered research companion** — Search ArXiv, ingest academic papers, and get answers grounded in real research with citations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiLz4=)](https://modelcontextprotocol.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🔥 What is PaperPal?

PaperPal is a **research assistant** that combines semantic search, PDF ingestion, and RAG (Retrieval-Augmented Generation) to help you explore academic literature. It works as both a **standalone web dashboard** and an **MCP server** for Claude Desktop.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Semantic Search** | Search ArXiv with AI-powered re-ranking — not just keywords |
| 📄 **Paper Ingestion** | Download PDFs, extract text, chunk, embed & store automatically |
| 🧠 **RAG Q&A** | Ask questions, get answers grounded in your papers with citations |
| 🤖 **MCP Server** | Plug directly into Claude Desktop as a tool server |
| 🎨 **Web Dashboard** | Beautiful dark-themed UI with Search, KB & Ask tabs |

---

## 🏗️ Architecture

```
        User Question
             |
  +----------v-----------+
  |   🔍 search_papers   |----> ArXiv API ----> Semantic Re-ranking
  +----------+-----------+
             |
  +----------v-----------+
  |   📄 ingest_papers   |----> PDF Download -> Parse -> Chunk -> Embed -> ChromaDB
  +----------+-----------+
             |
  +----------v-----------+
  |   🧠 ask             |----> Retrieve Chunks -> OpenAI LLM -> Answer + Citations
  +----------------------+
```

---

## 🚀 Quick Start

### 1️⃣ Clone & Setup
```bash
git clone https://github.com/krish-2002/paperpal.git
cd paperpal
python -m venv venv
```

**Activate the virtual environment:**
```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 2️⃣ Add Your API Key
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-key-here
```

### 3️⃣ Launch the Dashboard
```bash
python app.py
```
Open 👉 **http://localhost:5000** in your browser.

### 4️⃣ Or Connect to Claude Desktop
Add to your `claude_desktop_config.json`:
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

## 📁 Project Structure

```
paperpal/
├── 🌐 app.py              # Flask web dashboard
├── 🔌 server.py            # MCP server for Claude Desktop
├── 🧪 eval.py              # Evaluation test suite (5/5 passing)
├── 📋 requirements.txt     # Pinned dependencies
├── 🔒 .env                 # API keys (not committed)
│
├── 🛠️ tools/
│   ├── search.py           # ArXiv search + semantic re-ranking
│   ├── ingest.py           # PDF -> parse -> chunk -> embed -> store
│   ├── rag.py              # Retrieve -> LLM -> answer with citations
│   └── kb.py               # ChromaDB knowledge base operations
│
├── ⚙️ utils/
│   ├── embedder.py         # sentence-transformers wrapper
│   └── chunker.py          # Sliding-window word chunker
│
├── 🎨 static/
│   ├── index.html          # Dashboard HTML
│   ├── style.css           # Dark navy + amber academic theme
│   └── script.js           # Frontend logic
│
└── 💾 chroma_db/           # Persistent vector store (auto-created)
```

---

## 🛠️ MCP Tools

| Tool | Description | Input |
|------|-------------|-------|
| 🔍 `search_papers` | Search ArXiv, return papers ranked by semantic relevance | `query`, `top_k` |
| 📄 `ingest_papers` | Download + parse + chunk + embed + store in ChromaDB | `papers[]` |
| 🧠 `ask` | Answer a research question using RAG with citations | `question` |

---

## 🧬 Tech Stack

| Layer | Technology |
|-------|-----------|
| 🧠 **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| 💾 **Vector DB** | ChromaDB (persistent, local) |
| 📄 **PDF Parsing** | PyMuPDF |
| 🤖 **LLM** | OpenAI (`gpt-4o-mini`) |
| 🔌 **Protocol** | MCP (Model Context Protocol) |
| 🌐 **Web** | Flask + vanilla HTML/CSS/JS |
| 🔍 **Search** | ArXiv API + cosine similarity re-ranking |

---

## 🧪 Evaluation Results

All **5/5** test suites passing:

| Test | Status | Details |
|------|--------|---------|
| Embedder | ✅ Pass | 384-dim vectors, semantic sanity check passed |
| Chunker | ✅ Pass | Correct chunking with 50-word overlap |
| ArXiv Search | ✅ Pass | Sorted by semantic score, all fields present |
| Ingest + Retrieve | ✅ Pass | PDF ingested, correct chunks retrieved (distance: 0.24) |
| RAG End-to-End | ✅ Pass | Grounded answer with citations in 7.5s |

Run the evaluation yourself:
```bash
python eval.py
```

---

## ⚙️ Tuning Parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `chunk_size` | `chunker.py` | 500 words | Larger = more context per chunk |
| `overlap` | `chunker.py` | 50 words | Larger = less boundary context loss |
| `n_results` | `kb.py` | 5 chunks | More chunks = richer context for LLM |
| `max_results` | `search.py` | 10 | More candidates = better re-ranking |
| `model` | `rag.py` | `gpt-4o-mini` | Swap for `gpt-4o` for higher quality |
| Distance metric | `kb.py` | `cosine` | Can switch to `l2` for comparison |

---

## 📝 Notes

- 📊 **Tables & images** in PDFs are not extracted — text content only
- 🔒 **No data leaves locally** except the OpenAI API call in `rag.py`
- 💾 ChromaDB persists to `chroma_db/` — delete the folder to reset
- 🔄 Re-ingesting the same paper is safe (upsert, no duplicates)

---

## 📜 License

MIT

---

<p align="center">
  Built with ❤️ using MCP + ChromaDB + sentence-transformers
</p>
