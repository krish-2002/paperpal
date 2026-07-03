import io
from pathlib import Path
import requests as req_lib

import fitz  # PyMuPDF

from utils.chunker import chunk_text
from tools.kb import save_chunks


_BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}


def _download_pdf(url: str) -> bytes:
    """Download a PDF from a URL and return raw bytes. Uses requests for better redirect/cookie handling."""
    print(f"[ingest] Downloading from: {url}")
    
    # Use requests library — it handles redirects and cookies automatically,
    # which is critical for publisher sites that bounce through auth pages
    response = req_lib.get(url, headers=_BROWSER_HEADERS, timeout=30, allow_redirects=True)
    response.raise_for_status()
    
    pdf_bytes = response.content
    
    # Validate we actually got a PDF and not an HTML error page
    if len(pdf_bytes) < 100:
        raise ValueError(f"Downloaded file is too small ({len(pdf_bytes)} bytes) — likely blocked by publisher.")
    
    if pdf_bytes[:4] != b'%PDF':
        # Sometimes publishers return an HTML page instead of a PDF
        snippet = pdf_bytes[:200].decode('utf-8', errors='ignore')
        if '<html' in snippet.lower():
            raise ValueError(f"Publisher returned an HTML page instead of a PDF. This paper's PDF is likely paywalled. Use 'Upload Local PDF' instead.")
        raise ValueError(f"Downloaded file is not a valid PDF (starts with: {pdf_bytes[:20]})")
    
    print(f"[ingest] Downloaded {len(pdf_bytes)} bytes, valid PDF confirmed.")
    return pdf_bytes


import pymupdf4llm

def _parse_pdf(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF given its raw bytes using PyMuPDF4LLM.
    Converts tables, lists, and headers perfectly to Markdown so the LLM
    can easily read them.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    md_text = pymupdf4llm.to_markdown(doc)
    return md_text


def ingest_paper(paper: dict, chunk_size: int = 400, overlap: int = 100) -> dict:
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

def ingest_local_pdf(file_path: str, chunk_size: int = 400, overlap: int = 100) -> dict:
    """
    Ingest a local PDF file, completely bypassing ArXiv or internet downloads.
    """
    import os
    import pymupdf4llm
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"PDF not found at {file_path}")
        
    filename = file_path_obj.name
    paper_id = f"local_{filename}"
    
    print(f"[ingest] Parsing local PDF: {filename} ...")
    md_text = pymupdf4llm.to_markdown(str(file_path_obj))
    
    print(f"[ingest] Chunking local PDF ...")
    chunks = chunk_text(md_text, chunk_size=chunk_size, overlap=overlap)
    print(f"[ingest] Split into {len(chunks)} chunks.")
    
    metadata = {
        "paper_id": paper_id,
        "title":    filename,
        "authors":  "Local Upload",
        "url":      f"file://{filename}",
    }
    
    print(f"[ingest] Embedding and storing chunks ...")
    save_chunks(paper_id=paper_id, chunks=chunks, metadata=metadata)
    
    print(f"[ingest] Done -- {paper_id}")
    return {
        "paper_id":    paper_id,
        "title":       filename,
        "chunks_stored": len(chunks),
    }

