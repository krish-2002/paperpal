---
title: PaperPal
emoji: 📚
colorFrom: blue
colorTo: amber
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---
# 📚 PaperPal

> **Your AI-powered enterprise research companion** — Search universally via Semantic Scholar, ingest academic papers (bypassing paywalls), and get answers grounded in real research with hybrid search and hallucination guardrails.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange)](https://gradio.app/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🔥 What is PaperPal?

PaperPal is an **enterprise-grade RAG (Retrieval-Augmented Generation) assistant** designed specifically for academic literature. It handles the complete pipeline from searching for papers (across 200M+ publications) to parsing complex PDFs, indexing them, and answering questions with inline citations.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Universal Search & Re-ranking** | Searches Semantic Scholar and uses OpenAI embeddings to compute cosine-similarity, ranking papers by *actual relevance* instead of citation count. |
| 📄 **Robust PDF Ingestion** | Uses `pymupdf4llm` to extract clean Markdown (preserving tables/math). Automatically routes downloads through ArXiv when possible to bypass publisher firewalls. |
| 🧠 **Hybrid Search + RRF** | Combines Vector Search (meaning) and BM25 (exact keywords) using Reciprocal Rank Fusion for high-precision document retrieval. |
| 🛡️ **Anti-Hallucination Guardrails** | Computes a **Confidence Score** for every retrieval. If context is too weak, an **Abstention Threshold** prevents the LLM from hallucinating an answer. |
| 📊 **Observability Dashboard** | Built-in analytics tab tracks end-to-end latency, token usage, and API costs per session. |
| 📤 **Local PDF Upload** | Seamless fallback workflow for paywalled papers (IEEE, Elsevier, Springer). |

---

## 🏗️ Architecture

```
        User Question
             |
  +----------v-----------+
  |   🔍 Search Papers   |----> Semantic Scholar API ----> OpenAI Semantic Re-ranking
  +----------+-----------+
             |
  +----------v-----------+
  |   📄 Ingest Papers   |----> ArXiv Fallback / Local PDF Upload -> Parse -> Chunk -> Embed -> ChromaDB
  +----------+-----------+
             |
  +----------v-----------+
  |   🧠 Ask Question    |----> Hybrid Search (Vector+BM25) -> RRF Fusion -> Confidence Check -> LLM -> Answer
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

### 2️⃣ Add Your API Keys
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-key-here
SEMANTIC_SCHOLAR_API_KEY=your-api-key-here  # Optional but recommended
```

### 3️⃣ Launch the Dashboard
```bash
python hf_app.py
```
Open 👉 **http://localhost:7860** in your browser to access the Gradio UI.

---

## 📁 Project Structure

```
paperpal/
├── 🌐 hf_app.py            # Gradio Web Dashboard (Main Entrypoint)
├── 🔌 app.py               # Legacy Flask Entrypoint / API
├── 🧪 eval.py              # Automated Evaluation Harness
├── 📋 requirements.txt     # Pinned dependencies
├── 🔒 .env                 # API keys (not committed)
│
├── 🛠️ tools/
│   ├── search.py           # Semantic Scholar search + semantic re-ranking
│   ├── ingest.py           # PDF -> parse -> chunk -> embed -> store
│   ├── rag.py              # Retrieve -> Confidence Check -> LLM -> answer with citations
│   └── kb.py               # ChromaDB operations + Hybrid Search (BM25 + Vector)
│
├── ⚙️ utils/
│   ├── embedder.py         # OpenAI Embeddings wrapper (`text-embedding-3-small`)
│   └── chunker.py          # Sliding-window word chunker (400 words, 100 overlap)
│
└── 💾 chroma_db/           # Persistent vector store (auto-created)
```

---

## 🧬 Tech Stack

| Layer | Technology | Why we use it |
|-------|-----------|---------------|
| 🧠 **Embeddings** | OpenAI (`text-embedding-3-small`) | Fast, cost-effective, high-dimensional semantic capture |
| 💾 **Vector DB** | ChromaDB (local SQLite) | Lightweight, purely Python, uses HNSW for fast ANN search |
| 🔍 **Keyword Search** | Rank_BM25 | Handles exact acronyms and part numbers that vectors miss |
| 📄 **PDF Parsing** | PyMuPDF4LLM | Preserves academic formatting, tables, and math equations |
| 🤖 **LLM** | OpenAI (`gpt-4o-mini`) | Excellent reasoning speed and cost ratio for summarization |
| 🌐 **Web UI** | Gradio | Rapid UI prototyping with built-in state management |
| 📚 **Data Source** | Semantic Scholar | Indexes 200M+ papers across all disciplines (unlike ArXiv) |

---

## ⚙️ Tuning Parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `chunk_size` | `chunker.py` | 400 words | Larger = more context, weaker vectors. Smaller = tighter vectors, lost context. |
| `overlap` | `chunker.py` | 100 words | Prevents cutting critical sentences in half across boundaries. |
| `n_results` | `rag.py` | 8 chunks | Determines how much total context is fed to the LLM (8 * 400 = 3,200 words). |
| `temperature` | `rag.py` | 0.2 | Sweet spot for factual extraction without being overly rigid or hallucinating. |
| `_ABSTENTION_THRESHOLD` | `rag.py` | 0.015 | Hard floor for retrieval quality. If best chunk is below this, LLM refuses to answer. |

---

## 📝 Notes & Limitations

- **Image Extraction:** Embedded images and figures in PDFs are ignored; only layout-preserved text, math, and tables are extracted.
- **BM25 Persistence:** Currently, BM25 indexing is done in-memory. If the server restarts, keyword search indices rebuild automatically on next query.
- **Publisher Firewalls:** IEEE, Elsevier, and Springer often block automated PDF downloads (403 Forbidden). The app provides a "Open in Browser" link so users can download via university login and use the **Upload Local PDF** tab.

---

## 📜 License

MIT
