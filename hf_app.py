"""
PaperPal -- Hugging Face Spaces Gradio App
==========================================
Gradio-based interface for the PaperPal research assistant.
Deploy to HF Spaces or run locally with: python hf_app.py
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env if running locally
load_dotenv(Path(__file__).parent / ".env")

import gradio as gr

# ── Lazy-loaded tools (same pattern as app.py) ────────────────────────────

_tools_loaded = False

def _load_tools():
    global _tools_loaded, search_arxiv, ingest_paper, ingest_papers
    global answer_question, _collection
    if _tools_loaded:
        return
    from tools.search import search_arxiv as _search
    from tools.ingest import ingest_paper as _ingest_one, ingest_papers as _ingest_many
    from tools.rag import answer_question as _answer
    from tools.kb import _collection as _coll
    search_arxiv = _search
    ingest_paper = _ingest_one
    ingest_papers = _ingest_many
    answer_question = _answer
    _collection = _coll
    _tools_loaded = True


# ── State: store last search results ─────────────────────────────────────

last_search_results = []


# ── Search Handler ────────────────────────────────────────────────────────

def handle_search(query, top_k):
    global last_search_results
    if not query.strip():
        return "Please enter a search query.", ""

    _load_tools()

    try:
        results = search_arxiv(query, max_results=int(top_k) * 2, top_k=int(top_k))
        last_search_results = results

        if not results:
            return "No papers found. Try a different query.", ""

        # Build a nice markdown output
        md = f"### Found {len(results)} papers\n\n"
        for i, p in enumerate(results):
            md += f"**{i+1}. [{p['title']}]({p['url']})**\n"
            md += f"- *Authors:* {p['authors']}\n"
            md += f"- *Score:* `{p['score']}`\n"
            md += f"- *ID:* `{p['paper_id']}`\n"
            md += f"- *Abstract:* {p['abstract'][:200]}...\n\n"
            md += "---\n\n"

        # Build choices for ingest dropdown
        choices = [f"{i+1}. {p['title'][:80]}" for i, p in enumerate(results)]
        return md, gr.update(choices=choices, value=choices, interactive=True)

    except Exception as e:
        return f"**Error:** {str(e)}", ""


# ── Ingest Handler ────────────────────────────────────────────────────────

def handle_ingest(selected_papers):
    if not selected_papers:
        return "Please select papers to ingest from the search results."

    _load_tools()

    try:
        # Map selected display strings back to paper dicts
        indices = []
        for sel in selected_papers:
            idx = int(sel.split(".")[0]) - 1
            indices.append(idx)

        papers_to_ingest = [last_search_results[i] for i in indices if i < len(last_search_results)]

        if not papers_to_ingest:
            return "No valid papers selected."

        results = ingest_papers(papers_to_ingest)

        md = "### Ingestion Complete!\n\n"
        for r in results:
            md += f"- **{r['title'][:70]}...** -- `{r['chunks_stored']}` chunks stored\n"

        return md

    except Exception as e:
        return f"**Error:** {str(e)}"


# ── KB Handler ────────────────────────────────────────────────────────────

def handle_kb():
    _load_tools()

    try:
        all_data = _collection.get(include=["metadatas"])
        if not all_data["metadatas"]:
            return "### Knowledge base is empty\n\nUse the **Search** tab to find and ingest papers first."

        papers = {}
        for meta in all_data["metadatas"]:
            pid = meta.get("paper_id", "unknown")
            if pid not in papers:
                papers[pid] = {
                    "title": meta.get("title", "Unknown"),
                    "authors": meta.get("authors", "Unknown"),
                    "chunks": 0,
                }
            papers[pid]["chunks"] += 1

        md = f"### Knowledge Base: {len(papers)} papers, {len(all_data['metadatas'])} chunks\n\n"
        md += "| # | Paper ID | Title | Chunks |\n"
        md += "|---|----------|-------|--------|\n"
        for i, (pid, info) in enumerate(papers.items(), 1):
            title_short = info['title'][:60] + "..." if len(info['title']) > 60 else info['title']
            md += f"| {i} | `{pid}` | {title_short} | {info['chunks']} |\n"

        return md

    except Exception as e:
        return f"**Error:** {str(e)}"


# ── Ask Handler ───────────────────────────────────────────────────────────

def handle_ask(question):
    if not question.strip():
        return "Please enter a question."

    _load_tools()

    # Check if KB has data
    try:
        count = _collection.count()
        if count == 0:
            return ("### No papers in knowledge base\n\n"
                    "Please **search** and **ingest** papers first using the Search tab.")
    except:
        pass

    try:
        result = answer_question(question, n_chunks=5)

        md = "### Answer\n\n"
        md += result["answer"]
        md += "\n\n---\n\n"

        if result.get("citations"):
            md += "### Sources\n\n"
            for i, c in enumerate(result["citations"], 1):
                title = c.get("title", "Unknown")
                url = c.get("url", "")
                authors = c.get("authors", "")
                if url:
                    md += f"{i}. **[{title}]({url})**"
                else:
                    md += f"{i}. **{title}**"
                if authors:
                    md += f" -- *{authors}*"
                md += "\n"

        return md

    except Exception as e:
        return f"**Error:** {str(e)}"


# ── Custom CSS ────────────────────────────────────────────────────────────

custom_css = """
.gradio-container {
    max-width: 950px !important;
    font-family: 'Inter', sans-serif !important;
}
.tab-nav button {
    font-size: 16px !important;
    font-weight: 500 !important;
}
footer { display: none !important; }
"""


# ── Build the Gradio Interface ───────────────────────────────────────────

with gr.Blocks(
    title="PaperPal - AI Research Assistant",
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:

    gr.Markdown(
        """
        # PaperPal
        **AI-Powered Research Assistant** -- Search ArXiv, ingest papers, and ask questions with citations.
        """
    )

    with gr.Tabs():

        # ── Search Tab ────────────────────────────────────────────────
        with gr.TabItem("Search Papers", id="search"):
            gr.Markdown("Search ArXiv for academic papers with semantic re-ranking.")

            with gr.Row():
                search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g. retrieval augmented generation",
                    scale=4,
                )
                top_k_slider = gr.Slider(
                    label="Results",
                    minimum=1, maximum=10, value=5, step=1,
                    scale=1,
                )

            search_btn = gr.Button("Search", variant="primary", size="lg")
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
                inputs=[search_input, top_k_slider],
                outputs=[search_output, ingest_select],
            )
            search_input.submit(
                fn=handle_search,
                inputs=[search_input, top_k_slider],
                outputs=[search_output, ingest_select],
            )
            ingest_btn.click(
                fn=handle_ingest,
                inputs=[ingest_select],
                outputs=[ingest_output],
            )

        # ── Knowledge Base Tab ────────────────────────────────────────
        with gr.TabItem("Knowledge Base", id="kb"):
            gr.Markdown("View all papers currently stored in your vector database.")
            kb_btn = gr.Button("Refresh Knowledge Base", variant="secondary")
            kb_output = gr.Markdown()
            kb_btn.click(fn=handle_kb, outputs=[kb_output])

        # ── Ask Tab ───────────────────────────────────────────────────
        with gr.TabItem("Ask a Question", id="ask"):
            gr.Markdown("Ask a research question -- answers are grounded in your ingested papers with citations.")
            ask_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What are the main challenges in RAG systems?",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary", size="lg")
            ask_output = gr.Markdown(label="Answer")

            ask_btn.click(
                fn=handle_ask,
                inputs=[ask_input],
                outputs=[ask_output],
            )
            ask_input.submit(
                fn=handle_ask,
                inputs=[ask_input],
                outputs=[ask_output],
            )


# ── Launch ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
