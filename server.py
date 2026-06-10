import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load .env FIRST before any other imports so OPENAI_API_KEY is available
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from tools.search import search_arxiv
from tools.ingest import ingest_papers
from tools.rag import answer_question

# ── Create the MCP server ─────────────────────────────────────────────────────
app = Server("research-mcp")


# ── Register tools ────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Advertise all available tools to the MCP client."""
    return [
        types.Tool(
            name="search_papers",
            description=(
                "Search ArXiv for academic papers matching a query. "
                "Returns the top papers ranked by semantic relevance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query, e.g. 'attention mechanism in transformers'",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of candidates to fetch from ArXiv before re-ranking (default 10)",
                        "default": 10,
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top papers to return after re-ranking (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="ingest_papers",
            description=(
                "Download and ingest a list of papers into the knowledge base. "
                "Pass the output of search_papers directly. "
                "Each paper is downloaded, parsed, chunked, embedded, and stored in ChromaDB."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "papers": {
                        "type": "array",
                        "description": "List of paper dicts from search_papers (must include paper_id, title, authors, url)",
                        "items": {"type": "object"},
                    },
                },
                "required": ["papers"],
            },
        ),
        types.Tool(
            name="ask",
            description=(
                "Ask a research question and get an answer grounded in the ingested papers. "
                "Returns the answer text plus cited sources."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The research question to answer",
                    },
                    "n_chunks": {
                        "type": "integer",
                        "description": "Number of context chunks to retrieve from the knowledge base (default 5)",
                        "default": 5,
                    },
                },
                "required": ["question"],
            },
        ),
    ]


# ── Handle tool calls ─────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Route incoming tool calls to the correct function and return results."""

    if name == "search_papers":
        results = search_arxiv(
            query=arguments["query"],
            max_results=arguments.get("max_results", 10),
            top_k=arguments.get("top_k", 5),
        )
        lines = []
        for i, p in enumerate(results, 1):
            lines.append(
                f"{i}. {p['title']}\n"
                f"   Authors : {p['authors']}\n"
                f"   Score   : {p['score']}\n"
                f"   URL     : {p['url']}\n"
                f"   Abstract: {p['abstract'][:300]}...\n"
            )
        output = "\n".join(lines) if lines else "No results found."
        return [types.TextContent(type="text", text=output)]

    elif name == "ingest_papers":
        summaries = ingest_papers(arguments["papers"])
        lines = [
            f"✓ {s['paper_id']} — {s['title']} ({s['chunks_stored']} chunks stored)"
            for s in summaries
        ]
        output = "\n".join(lines) if lines else "No papers ingested."
        return [types.TextContent(type="text", text=output)]

    elif name == "ask":
        result = answer_question(
            question=arguments["question"],
            n_chunks=arguments.get("n_chunks", 5),
        )
        citation_lines = "\n".join(
            f"  [{i+1}] {c['title']} — {c['url']}"
            for i, c in enumerate(result["citations"])
        )
        output = (
            f"{result['answer']}\n\nSources:\n{citation_lines}"
            if result["citations"]
            else result["answer"]
        )
        return [types.TextContent(type="text", text=output)]

    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main())
