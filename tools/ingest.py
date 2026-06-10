import io
import urllib.request

import fitz  # PyMuPDF

from utils.chunker import chunk_text
from tools.kb import save_chunks


def _download_pdf(url: str) -> bytes:
    """Download a PDF from a URL and return raw bytes."""
    with urllib.request.urlopen(url) as response:
        return response.read()


def _parse_pdf(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF given its raw bytes.
    Joins pages with a newline so chunk boundaries don't
    accidentally merge the last sentence of one page with
    the first of the next.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def ingest_paper(paper: dict, chunk_size: int = 500, overlap: int = 50) -> dict:
    """
    Full ingestion pipeline for a single paper:
        download PDF → parse text → chunk → embed → store in ChromaDB

    Args:
        paper:      A dict produced by search.search_arxiv(), containing at
                    minimum: paper_id, title, authors, abstract, url.
        chunk_size: Words per chunk (passed to chunker).
        overlap:    Overlap words between chunks (passed to chunker).

    Returns:
        A summary dict with paper_id, title, and how many chunks were stored.
    """

    paper_id = paper["paper_id"]
    pdf_url   = paper["url"]

    # ── Step 1: Download ──────────────────────────────────────────────────────
    print(f"[ingest] Downloading {paper_id} ...")
    pdf_bytes = _download_pdf(pdf_url)

    # ── Step 2: Parse ─────────────────────────────────────────────────────────
    print(f"[ingest] Parsing PDF ...")
    full_text = _parse_pdf(pdf_bytes)

    # ── Step 3: Chunk ─────────────────────────────────────────────────────────
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
    print(f"[ingest] Split into {len(chunks)} chunks.")

    # ── Step 4: Embed + Store ─────────────────────────────────────────────────
    # Build the metadata dict that will be attached to every chunk in ChromaDB.
    # This is what surfaces as citation info when a chunk is retrieved.
    metadata = {
        "paper_id": paper_id,
        "title":    paper["title"],
        "authors":  paper["authors"],
        "url":      pdf_url,
    }

    print(f"[ingest] Embedding and storing chunks ...")
    save_chunks(paper_id=paper_id, chunks=chunks, metadata=metadata)

    print(f"[ingest] Done -- {paper_id}")
    return {
        "paper_id":    paper_id,
        "title":       paper["title"],
        "chunks_stored": len(chunks),
    }


def ingest_papers(papers: list[dict], **kwargs) -> list[dict]:
    """
    Convenience wrapper to ingest a list of papers (e.g. from search_arxiv).
    Returns a list of summary dicts, one per paper.
    """
    return [ingest_paper(p, **kwargs) for p in papers]
