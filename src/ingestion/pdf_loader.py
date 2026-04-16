"""
PDF document loader — ingests clinical guideline PDFs from /guides/.

Uses PyMuPDF (fitz) for text extraction. Falls back to a graceful error
message if PyMuPDF is not installed (it is optional in dev environments).

Strategy:
    1. Extract text page by page, preserving page numbers as metadata
    2. Apply the same semantic chunking logic as markdown_loader.py:
       - Tables → preserved whole
       - Long sections → sliding window with sentence-boundary respect
    3. Return List[DocumentChunk] identical in structure to Markdown chunks

PDF files expected in /guides/:
    - ASCO_CAP_HER2_2023.pdf
    - Breast.Bmk_1.6.0.0.REL.CAPCP-2025.pdf
    - ESMO_ECS_HER2-low_2023.pdf
    - Rakha_International_Consensus_2026.pdf
    - (and others as discovered)

Usage:
    from src.ingestion.pdf_loader import load_pdf_document, load_all_pdf_docs
    chunks = load_pdf_document(Path("guides/ASCO_CAP_HER2_2023.pdf"))
"""
from __future__ import annotations

import re
from pathlib import Path

from src.domain.models import ContentType, DocumentChunk
from src.ingestion.markdown_loader import (
    _detect_content_type,
    _sliding_window_chunks,
    _split_into_sentences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TABLE_PATTERN  = re.compile(r'^\s*[A-Za-z0-9][^\n]{0,60}\s{2,}[A-Za-z0-9+%]', re.MULTILINE)
_HEADER_PATTERN = re.compile(r'^(?:\d+[\.\d]*\s+|[A-Z][A-Z\s]{4,}\n)', re.MULTILINE)


def _chunk_id_from_pdf(doc_name: str, page: int, idx: int) -> str:
    safe = re.sub(r'[^a-z0-9]', '_', doc_name.lower())
    return f"{safe}_p{page:04d}_c{idx:04d}"


def _split_pdf_page(
    page_text: str,
    doc_name: str,
    page_num: int,
    chunk_size: int,
    overlap: int,
    start_counter: int = 0,
) -> tuple[list[DocumentChunk], int]:
    """
    Split a single PDF page text into DocumentChunks.
    Applies sliding window chunking and returns (chunks, updated_counter).
    """
    chunks: list[DocumentChunk] = []
    counter = start_counter

    # Very short pages → single chunk
    if len(page_text.split()) < 60:
        if page_text.strip():
            ct = _detect_content_type(page_text)
            cid = _chunk_id_from_pdf(doc_name, page_num, counter)
            chunks.append(DocumentChunk(
                chunk_id=cid,
                source_doc=doc_name,
                section=f"Page {page_num}",
                content=page_text.strip(),
                content_type=ct,
            ))
            counter += 1
        return chunks, counter

    # Apply sliding window (returns (list, new_counter))
    sub_chunks, counter = _sliding_window_chunks(
        page_text,
        section=f"Page {page_num}",
        doc_name=doc_name,
        start_counter=counter,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    chunks.extend(sub_chunks)
    return chunks, counter


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_pdf_document(
    path: Path,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """
    Load a single PDF file and return a list of DocumentChunks.

    Requires PyMuPDF (pip install pymupdf).
    Gracefully returns empty list if the file cannot be opened.
    """
    try:
        import fitz  # PyMuPDF  # noqa: PLC0415
    except ImportError:
        print(
            f"[PDFLoader] WARNING: PyMuPDF (fitz) not installed. "
            f"Skipping {path.name}. Run: pip install pymupdf"
        )
        return []

    doc_name = path.name
    chunks: list[DocumentChunk] = []

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        print(f"[PDFLoader] Cannot open {path}: {exc}")
        return []

    counter = 0
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if not page_text or not page_text.strip():
            continue

        page_chunks, counter = _split_pdf_page(
            page_text, doc_name, page_num, chunk_size, overlap, counter
        )
        chunks.extend(page_chunks)

    doc.close()
    print(f"  [PDFLoader] {doc_name}: {len(chunks)} chunks from {page_num} pages")
    return chunks


def load_all_pdf_docs(
    guides_dir: Path | str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """
    Load all PDF files from the given directory.

    Returns:
        Flat list of DocumentChunks from all PDFs (may be empty if no PDFs found
        or if PyMuPDF is not installed).
    """
    guides_dir = Path(guides_dir)
    if not guides_dir.exists():
        print(f"[PDFLoader] guides_dir not found: {guides_dir}")
        return []

    pdf_files = sorted(guides_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[PDFLoader] No PDF files found in {guides_dir}")
        return []

    all_chunks: list[DocumentChunk] = []
    for pdf_path in pdf_files:
        chunks = load_pdf_document(pdf_path, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)

    print(f"[PDFLoader] Total: {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
    return all_chunks
