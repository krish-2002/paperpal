"""
PaperPal -- Hugging Face Spaces Gradio App
==========================================
Gradio-based interface for the PaperPal research assistant.
Deploy to HF Spaces or run locally with: python hf_app.py
"""

import os
import json
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load .env if running locally
load_dotenv(Path(__file__).parent / ".env")

import gradio as gr

# ── Lazy-loaded tools (same pattern as app.py) ────────────────────────────

_tools_loaded = False

def _load_tools():
    global _tools_loaded, search_semantic_scholar, ingest_paper, ingest_papers, ingest_local_pdf
    global answer_question, _collection, clear_session_kb
    if _tools_loaded:
        return
    from tools.search import search_semantic_scholar as _search
    from tools.ingest import ingest_paper as _ingest_one, ingest_papers as _ingest_many, ingest_local_pdf as _ingest_local
    from tools.rag import answer_question as _answer
    from tools.kb import _collection as _coll, clear_session_kb as _clear_kb
    search_semantic_scholar = _search
    ingest_paper = _ingest_one
    ingest_papers = _ingest_many
    ingest_local_pdf = _ingest_local
    answer_question = _answer
    _collection = _coll
    clear_session_kb = _clear_kb
    _tools_loaded = True


# ── Global Session Analytics ──────────────────────────────────────────────

_session_stats = {
    "queries": [],           # List of {question, confidence, retrieval_time, llm_time, tokens, timestamp}
    "total_tokens": 0,
    "papers_ingested": 0,
    "papers_searched": 0,
}


# ── Handlers ──────────────────────────────────────────────────────────────

def handle_search(query, top_k, last_search_results):
    if not query.strip():
        return "Please enter a search query.", "", last_search_results

    _load_tools()

    try:
        t0 = time.time()
        results = search_semantic_scholar(query, max_results=int(top_k) * 2, top_k=int(top_k))
        search_time = round(time.time() - t0, 2)
        
        last_search_results = results
        _session_stats["papers_searched"] += len(results)

        if not results:
            return "No papers found. Try a different query.", "", last_search_results

        md = f"### Found {len(results)} papers *(search took {search_time}s)*\n\n"
        for i, p in enumerate(results):
            md += f"**{i+1}. [{p['title']}]({p['url']})**\n"
            md += f"- *Authors:* {p['authors']}\n"
            md += f"- *Relevance:* `{p['score']:.1%}`\n"
            md += f"- *Abstract:* {p['abstract'][:200]}...\n\n"
            md += "---\n\n"

        choices = [f"{i+1}. {p['title'][:80]}" for i, p in enumerate(results)]
        return md, gr.update(choices=choices, value=choices, interactive=True), last_search_results

    except Exception as e:
        return f"**Error:** {str(e)}", "", last_search_results

def handle_ingest_papers(selected_papers, session_id, last_search_results):
    if not selected_papers:
        return "Please select papers to ingest from the search results.", handle_kb_refresh(session_id)

    _load_tools()

    try:
        indices = []
        for sel in selected_papers:
            idx = int(sel.split(".")[0]) - 1
            indices.append(idx)

        papers_to_ingest = [last_search_results[i] for i in indices if i < len(last_search_results)]

        if not papers_to_ingest:
            return "No valid papers selected.", handle_kb_refresh(session_id)

        # Ingest papers one by one so a single failure doesn't kill the whole batch
        md = "### Ingestion Results\n\n"
        success_count = 0
        for paper in papers_to_ingest:
            try:
                r = ingest_paper(session_id, paper)
                md += f"✅ **{r['title'][:70]}...** — `{r['chunks_stored']}` chunks stored\n\n"
                success_count += 1
                _session_stats["papers_ingested"] += 1
            except Exception as e:
                pdf_url = paper.get('url', '')
                md += f"❌ **{paper['title'][:70]}...** — Failed: {str(e)}\n"
                md += f"   → [📥 Open PDF in Browser]({pdf_url}) — download manually, then use **Upload Local PDF**\n\n"
        
        if success_count == 0:
            md += "\n> ⚠️ **Tip:** Click the 'Open PDF in Browser' links above to download via your university login, then upload using the **Upload Local PDF** tab.\n"

        return md, handle_kb_refresh(session_id)

    except Exception as e:
        return f"**Error:** {str(e)}", handle_kb_refresh(session_id)

def handle_local_upload(file_obj, session_id):
    if not file_obj:
        return "Please upload a PDF file.", handle_kb_refresh(session_id)
        
    _load_tools()
    
    try:
        # file_obj is typically a temp file path in gradio
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        
        result = ingest_local_pdf(session_id, file_path)
        _session_stats["papers_ingested"] += 1
        
        md = "### Local PDF Ingested!\n\n"
        md += f"- **{result['title']}** -- `{result['chunks_stored']}` chunks stored\n"
        
        return md, handle_kb_refresh(session_id)
        
    except Exception as e:
        return f"**Error:** {str(e)}", handle_kb_refresh(session_id)

def handle_kb_refresh(session_id):
    _load_tools()

    try:
        all_data = _collection.get(where={"session_id": session_id}, include=["metadatas"])
        if not all_data or not all_data["metadatas"]:
            return "*Knowledge base is empty. Upload or search for papers.*"

        papers = {}
        for meta in all_data["metadatas"]:
            pid = meta.get("paper_id", "unknown")
            if pid not in papers:
                papers[pid] = {
                    "title": meta.get("title", "Unknown"),
                    "chunks": 0,
                }
            papers[pid]["chunks"] += 1

        md = f"**Total Papers:** {len(papers)} | **Total Chunks:** {len(all_data['metadatas'])}\n\n"
        for i, (pid, info) in enumerate(papers.items(), 1):
            title_short = info['title'][:60] + "..." if len(info['title']) > 60 else info['title']
            md += f"**{i}. {title_short}**\n*Chunks: {info['chunks']}*\n\n"

        return md

    except Exception as e:
        return f"**Error:** {str(e)}"

def handle_clear_kb(session_id):
    _load_tools()
    try:
        clear_session_kb(session_id)
        return "Knowledge base cleared successfully.", handle_kb_refresh(session_id)
    except Exception as e:
        return f"**Error:** {str(e)}", handle_kb_refresh(session_id)

def handle_ask(user_message, chat_history, session_id):
    if not user_message.strip():
        return chat_history, ""

    _load_tools()

    # Build API-ready history from Gradio's messages format
    api_history = []
    if chat_history:
        for msg in chat_history:
            api_history.append({"role": msg["role"], "content": msg["content"]})

    # Append the user message to chat display
    chat_history.append({"role": "user", "content": user_message})

    try:
        count = _collection.count()
        if count == 0:
            chat_history.append({"role": "assistant", "content": "No papers in knowledge base. Please upload or search for papers first."})
            return chat_history, ""
    except:
        pass

    try:
        t_total = time.time()
        result = answer_question(session_id, user_message, chat_history=api_history, n_chunks=8)
        total_time = round(time.time() - t_total, 2)
        
        confidence = result.get("confidence", 0.0)
        timings = result.get("timings", {})
        tokens = result.get("tokens", 0)
        
        # Track in session analytics
        _session_stats["queries"].append({
            "question": user_message[:80],
            "confidence": confidence,
            "retrieval_time": timings.get("retrieval", 0),
            "llm_time": timings.get("llm", 0),
            "tokens": tokens,
            "total_time": total_time,
        })
        _session_stats["total_tokens"] += tokens

        # Confidence badge
        if confidence >= 0.7:
            badge = f"🟢 **Confidence: {confidence:.0%}** (High)"
        elif confidence >= 0.4:
            badge = f"🟡 **Confidence: {confidence:.0%}** (Medium)"
        else:
            badge = f"🔴 **Confidence: {confidence:.0%}** (Low)"

        md = f"{badge} | ⏱️ {total_time}s"
        if tokens > 0:
            md += f" | 🪙 {tokens} tokens"
        md += "\n\n---\n\n"
        md += result["answer"]
        md += "\n\n---\n\n"

        if result.get("citations"):
            md += "### Sources\n\n"
            for i, c in enumerate(result["citations"], 1):
                title = c.get("title", "Unknown")
                url = c.get("url", "")
                authors = c.get("authors", "")
                
                # If local file, don't link it like a web URL
                if url and not url.startswith("file://"):
                    md += f"{i}. **[{title}]({url})**"
                else:
                    md += f"{i}. **{title}**"
                    
                if authors and authors != "Local Upload":
                    md += f" -- *{authors}*"
                md += "\n"

        # Append assistant response
        chat_history.append({"role": "assistant", "content": md})
        return chat_history, ""

    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"**Error:** {str(e)}"})
        return chat_history, ""


def handle_analytics(password):
    """Generate the analytics dashboard content, protected by password."""
    if password != "admin123":
        return "🔒 **Access Denied:** Incorrect password."

    stats = _session_stats
    queries = stats["queries"]
    
    md = "## 📊 Global Analytics Dashboard\n\n"
    
    # ── Summary Cards ──
    md += "### Overview\n\n"
    md += f"| Metric | Value |\n|---|---|\n"
    md += f"| Total Queries | {len(queries)} |\n"
    md += f"| Papers Searched | {stats['papers_searched']} |\n"
    md += f"| Papers Ingested | {stats['papers_ingested']} |\n"
    md += f"| Total Tokens Used | {stats['total_tokens']:,} |\n"
    
    if stats['total_tokens'] > 0:
        # text-embedding-3-small: $0.02/1M tokens, gpt-4o-mini: $0.15/1M input + $0.60/1M output
        est_cost = stats['total_tokens'] * 0.0003 / 1000  # rough average
        md += f"| Est. API Cost | ${est_cost:.4f} |\n"
    md += "\n"
    
    if not queries:
        md += "*No queries yet.*\n"
        return md
    
    # ── Latency Breakdown ──
    md += "### ⏱️ Latency Breakdown (Last 10 Queries)\n\n"
    md += "| # | Question | Retrieval | LLM | Total | Confidence | Tokens |\n"
    md += "|---|---|---|---|---|---|---|\n"
    
    for i, q in enumerate(queries[-10:], 1):
        question_short = q["question"][:40] + "..." if len(q["question"]) > 40 else q["question"]
        conf = q["confidence"]
        if conf >= 0.7:
            conf_badge = f"🟢 {conf:.0%}"
        elif conf >= 0.4:
            conf_badge = f"🟡 {conf:.0%}"
        else:
            conf_badge = f"🔴 {conf:.0%}"
        
        md += f"| {i} | {question_short} | {q['retrieval_time']}s | {q['llm_time']}s | {q['total_time']}s | {conf_badge} | {q['tokens']} |\n"
    
    md += "\n"
    
    # ── Averages ──
    if queries:
        avg_retrieval = sum(q["retrieval_time"] for q in queries) / len(queries)
        avg_llm = sum(q["llm_time"] for q in queries) / len(queries)
        avg_total = sum(q["total_time"] for q in queries) / len(queries)
        avg_confidence = sum(q["confidence"] for q in queries) / len(queries)
        avg_tokens = sum(q["tokens"] for q in queries) / len(queries)
        
        md += "### 📈 Averages\n\n"
        md += f"| Metric | Average |\n|---|---|\n"
        md += f"| Retrieval Latency | {avg_retrieval:.2f}s |\n"
        md += f"| LLM Latency | {avg_llm:.2f}s |\n"
        md += f"| Total Latency | {avg_total:.2f}s |\n"
        md += f"| Confidence | {avg_confidence:.0%} |\n"
        md += f"| Tokens/Query | {avg_tokens:.0f} |\n"
    
    return md


# ── Custom CSS ────────────────────────────────────────────────────────────

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    font-family: 'Inter', sans-serif !important;
}
.sidebar {
    border-right: 1px solid #e5e7eb;
    padding-right: 20px;
}
.tab-nav button {
    font-size: 16px !important;
    font-weight: 500 !important;
}
footer { display: none !important; }
"""


# ── Build the Gradio Interface ───────────────────────────────────────────

with gr.Blocks(title="PaperPal - AI Research Assistant") as demo:
    
    # Session States
    session_id = gr.State(lambda: uuid.uuid4().hex)
    last_search_results = gr.State([])

    gr.Markdown(
        """
        # 📚 PaperPal
        **AI-Powered Research Assistant** -- Search universally for Open Access academic papers, upload local PDFs, and ask questions with citations.
        """
    )

    with gr.Row():
        
        # ── Left Sidebar (Knowledge Base & Upload) ──────────────────────────
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### 📂 My Knowledge Base")
            kb_display = gr.Markdown(value="*Loading...*")
            
            with gr.Row():
                refresh_kb_btn = gr.Button("🔄 Refresh", size="sm", variant="secondary")
                clear_kb_btn = gr.Button("🗑️ Clear KB", size="sm", variant="stop")
            
            clear_status = gr.Markdown()
            
            gr.Markdown("---")
            gr.Markdown("### 📤 Upload Local PDF")
            gr.Markdown("Upload downloaded PDFs from IEEE, Springer, or your computer to bypass paywalls.")
            pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Ingest PDF", variant="primary")
            upload_status = gr.Markdown()
            
            # Wire up the upload
            upload_btn.click(
                fn=handle_local_upload,
                inputs=[pdf_upload, session_id],
                outputs=[upload_status, kb_display]
            )
            refresh_kb_btn.click(fn=handle_kb_refresh, inputs=[session_id], outputs=[kb_display])
            clear_kb_btn.click(fn=handle_clear_kb, inputs=[session_id], outputs=[clear_status, kb_display])
            
            # Load KB on start
            demo.load(fn=handle_kb_refresh, inputs=[session_id], outputs=[kb_display])


        # ── Main Content Area ───────────────────────────────────────────────
        with gr.Column(scale=3):
            with gr.Tabs():
                
                # ── Ask Tab (Chatbot) ──
                with gr.TabItem("💬 Chat", id="ask"):
                    gr.Markdown("Ask a research question. Answers are generated using **Hybrid Search** and grounded in your ingested papers.")
                    
                    chatbot = gr.Chatbot(height=500)
                    with gr.Row():
                        ask_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g. What is the role of attention mechanisms in transformers?",
                            scale=4
                        )
                        ask_btn = gr.Button("Send", variant="primary", scale=1)

                    ask_btn.click(
                        fn=handle_ask,
                        inputs=[ask_input, chatbot, session_id],
                        outputs=[chatbot, ask_input],
                    )
                    ask_input.submit(
                        fn=handle_ask,
                        inputs=[ask_input, chatbot, session_id],
                        outputs=[chatbot, ask_input],
                    )
                
                # ── Search Tab ──
                with gr.TabItem("🔍 Universal Search", id="search"):
                    gr.Markdown("Search across IEEE, Nature, PubMed, ACM, and ArXiv for Open Access papers.")

                    with gr.Row():
                        search_input = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g. latent diffusion models",
                            scale=4,
                        )
                        top_k_slider = gr.Slider(
                            label="Results",
                            minimum=1, maximum=10, value=5, step=1,
                            scale=1,
                        )

                    search_btn = gr.Button("Search Papers", variant="primary")
                    search_output = gr.Markdown(label="Results")

                    gr.Markdown("### Ingest Papers into Knowledge Base")
                    ingest_select = gr.CheckboxGroup(
                        label="Select papers to ingest",
                        choices=[],
                        interactive=False,
                    )
                    ingest_btn = gr.Button("Ingest Selected Papers", variant="secondary")
                    ingest_output = gr.Markdown(label="Ingest Status")

                    search_btn.click(
                        fn=handle_search,
                        inputs=[search_input, top_k_slider, last_search_results],
                        outputs=[search_output, ingest_select, last_search_results],
                    )
                    search_input.submit(
                        fn=handle_search,
                        inputs=[search_input, top_k_slider, last_search_results],
                        outputs=[search_output, ingest_select, last_search_results],
                    )
                    ingest_btn.click(
                        fn=handle_ingest_papers,
                        inputs=[ingest_select, session_id, last_search_results],
                        outputs=[ingest_output, kb_display],
                    )

                # ── Analytics Tab ──
                with gr.TabItem("📊 Admin Analytics", id="analytics"):
                    gr.Markdown("Monitor global pipeline performance. **Admin access only.**")
                    admin_pass = gr.Textbox(label="Admin Password", type="password")
                    analytics_btn = gr.Button("View Analytics", variant="secondary")
                    analytics_output = gr.Markdown(value="*Enter password to view global stats.*")
                    
                    analytics_btn.click(
                        fn=handle_analytics,
                        inputs=[admin_pass],
                        outputs=[analytics_output],
                    )
                    admin_pass.submit(
                        fn=handle_analytics,
                        inputs=[admin_pass],
                        outputs=[analytics_output],
                    )


# ── Launch ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        )
    )
