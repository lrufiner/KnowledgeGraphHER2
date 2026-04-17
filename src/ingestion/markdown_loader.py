"""
Semantic-aware Markdown document loader and chunker.

Strategy:
  1. Algorithm blocks (```...``` or NODO-N: blocks) → preserved whole
  2. Tables (| ... |) → preserved whole as JSON-convertible text
  3. Section-aware text → sliding window with sentence-boundary respect

Priority: annex_guidelines.md and annex_ontology.md are loaded first
(per project decision — these are the preprocessed reference documents).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.domain.models import ContentType, DocumentChunk


# ---------------------------------------------------------------------------
# Content-type patterns
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Technical appendix docs — software/tutorial content, skip clinical extraction
# ---------------------------------------------------------------------------

TECHNICAL_APPENDIX_DOCS: frozenset[str] = frozenset({
    "apendice_frameworks_graphrag.md",
    "apendice_langchain_langgraph.md",
})


# ---------------------------------------------------------------------------
# Content-type patterns
# ---------------------------------------------------------------------------

_CONTENT_TYPE_PATTERNS = {
    ContentType.ALGORITHM: re.compile(
        r'(?:NODO[-_]\d|ENTRADA:|ACCIÓN:|SÍ\s*→|→|algorithm|flowchart|decision|paso\s+\d)',
        re.IGNORECASE
    ),
    ContentType.CRITERIA: re.compile(
        r'(?:IHC\s*[0-9+]+|ISH|ratio|señales/célula|score|tinción|staining)',
        re.IGNORECASE
    ),
    ContentType.RECOMMENDATION: re.compile(
        r'(?:recomienda|should|must|guideline|recomend|se debe)',
        re.IGNORECASE
    ),
    ContentType.FRACTAL_MAPPING: re.compile(
        r'(?:fractal|lacun|dimension|D0|D1|multifractal|entropía)',
        re.IGNORECASE
    ),
    ContentType.QA: re.compile(
        r'(?:calidad|quality|control|validation|QA|EQA|proficiency|aseguramiento)',
        re.IGNORECASE
    ),
    ContentType.ONTOLOGY: re.compile(
        r'(?:owl:|rdfs:|her2:|frac:|ncit:|namespace|@prefix|subClassOf|equivalentClass)',
        re.IGNORECASE
    ),
    ContentType.TABLE: re.compile(
        r'^\|.+\|$',
        re.MULTILINE
    ),
}


def _detect_content_type(text: str) -> ContentType:
    """Return the dominant ContentType for a text segment."""
    scores = {ct: len(p.findall(text)) for ct, p in _CONTENT_TYPE_PATTERNS.items()}
    best_ct = max(scores, key=lambda k: scores[k])
    return best_ct if scores[best_ct] > 0 else ContentType.GENERAL


# ---------------------------------------------------------------------------
# Sentence splitter (lightweight, no NLTK dependency)
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])')


def _split_into_sentences(text: str) -> list[str]:
    return _SENT_SPLIT.split(text.strip())


# ---------------------------------------------------------------------------
# Block extractors
# ---------------------------------------------------------------------------

def _extract_code_blocks(text: str, section: str, doc_name: str,
                          start_counter: int) -> tuple[list[DocumentChunk], str, int]:
    """Extract fenced code blocks (``` ... ```) as intact chunks."""
    chunks = []
    counter = start_counter
    clean_text = text

    pattern = re.compile(r'(```[\w]*\n[\s\S]*?```)', re.MULTILINE)
    for match in pattern.findall(text):
        cid = f"{Path(doc_name).stem}_c{counter:04d}"
        chunks.append(DocumentChunk(
            chunk_id=cid,
            source_doc=doc_name,
            section=section,
            content=match.strip(),
            content_type=_detect_content_type(match),
            metadata={"block_type": "code_block"},
        ))
        counter += 1
        clean_text = clean_text.replace(match, f"[CODE_BLOCK_{cid}]")

    return chunks, clean_text, counter


def _extract_table_blocks(text: str, section: str, doc_name: str,
                           start_counter: int) -> tuple[list[DocumentChunk], str, int]:
    """Extract Markdown tables as intact chunks."""
    chunks = []
    counter = start_counter
    clean_text = text

    # Match consecutive lines starting with |
    table_pattern = re.compile(r'(\|[^\n]+\|\n(?:\|[-| :]+\|\n)?(?:\|[^\n]+\|\n?)+)')
    for match in table_pattern.findall(text):
        cid = f"{Path(doc_name).stem}_c{counter:04d}"
        chunks.append(DocumentChunk(
            chunk_id=cid,
            source_doc=doc_name,
            section=section,
            content=match.strip(),
            content_type=ContentType.TABLE,
            metadata={"block_type": "table"},
        ))
        counter += 1
        clean_text = clean_text.replace(match, f"[TABLE_{cid}]")

    return chunks, clean_text, counter


def _extract_algorithm_blocks(text: str, section: str, doc_name: str,
                               start_counter: int) -> tuple[list[DocumentChunk], str, int]:
    """
    Extract algorithm/decision-tree text blocks.
    Algorithm blocks in this corpus start with 'NODO-' or 'ENTRADA:' lines,
    or are contained within triple-backtick code blocks already handled above.
    Here we capture multi-line structured blocks like the ISH/IHC algorithms.
    """
    chunks = []
    counter = start_counter
    clean_text = text

    # Multi-line algorithm blocks starting with NODO, ENTRADA, or SI/NO decision lines
    algo_pattern = re.compile(
        r'((?:(?:NODO[-_]\d+|ENTRADA:|ACCIÓN:|COMENTARIO[-_]\w*):.*\n'
        r'(?:.*\n){0,30}?)'
        r'(?=\n#{1,3}\s|\Z))',
        re.MULTILINE
    )
    for match_obj in algo_pattern.finditer(text):
        match = match_obj.group(0)
        if len(match.strip()) < 80:
            continue
        cid = f"{Path(doc_name).stem}_c{counter:04d}"
        chunks.append(DocumentChunk(
            chunk_id=cid,
            source_doc=doc_name,
            section=section,
            content=match.strip(),
            content_type=ContentType.ALGORITHM,
            metadata={"block_type": "algorithm"},
        ))
        counter += 1
        clean_text = clean_text.replace(match, f"[ALGO_BLOCK_{cid}]")

    return chunks, clean_text, counter


def _sliding_window_chunks(text: str, section: str, doc_name: str,
                            start_counter: int,
                            chunk_size: int, overlap: int) -> tuple[list[DocumentChunk], int]:
    """Split free text into overlapping sentence-respecting windows."""
    chunks = []
    counter = start_counter
    sentences = _split_into_sentences(text)

    current: list[str] = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        if current_len + len(words) > chunk_size and current:
            chunk_text = " ".join(current)
            if len(chunk_text.strip()) >= 50:
                cid = f"{Path(doc_name).stem}_c{counter:04d}"
                chunks.append(DocumentChunk(
                    chunk_id=cid,
                    source_doc=doc_name,
                    section=section,
                    content=chunk_text,
                    content_type=_detect_content_type(chunk_text),
                ))
                counter += 1
            # Keep overlap portion
            overlap_words = " ".join(current).split()[-overlap:]
            current = overlap_words
            current_len = len(overlap_words)
        current.extend(words)
        current_len += len(words)

    # Flush remainder
    if current:
        chunk_text = " ".join(current)
        if len(chunk_text.strip()) >= 50:
            cid = f"{Path(doc_name).stem}_c{counter:04d}"
            chunks.append(DocumentChunk(
                chunk_id=cid,
                source_doc=doc_name,
                section=section,
                content=chunk_text,
                content_type=_detect_content_type(chunk_text),
            ))
            counter += 1

    return chunks, counter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_markdown_document(
    path: Path,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """
    Load and semantically chunk a Markdown document.

    Processing order (per project decision):
    1. Extract code/algorithm/table blocks intact
    2. Remaining text split into section-aware sliding-window chunks
    """
    text = path.read_text(encoding="utf-8")
    doc_name = path.name
    all_chunks: list[DocumentChunk] = []
    counter = 0

    # Split into sections by heading
    section_parts = re.split(r'\n(#{1,3}\s+.+)\n', text)
    current_section = "Introducción"

    for segment in section_parts:
        # Heading line → update current section name
        if re.match(r'#{1,3}\s+', segment):
            current_section = re.sub(r'^#{1,3}\s+', '', segment).strip()
            continue

        if not segment.strip():
            continue

        # 1) Extract code blocks
        code_chunks, segment, counter = _extract_code_blocks(
            segment, current_section, doc_name, counter)
        all_chunks.extend(code_chunks)

        # 2) Extract table blocks
        table_chunks, segment, counter = _extract_table_blocks(
            segment, current_section, doc_name, counter)
        all_chunks.extend(table_chunks)

        # 3) Extract algorithm blocks
        algo_chunks, segment, counter = _extract_algorithm_blocks(
            segment, current_section, doc_name, counter)
        all_chunks.extend(algo_chunks)

        # 4) Sliding window on remainder
        text_chunks, counter = _sliding_window_chunks(
            segment, current_section, doc_name, counter, chunk_size, overlap)
        all_chunks.extend(text_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Priority loading order (project convention)
# ---------------------------------------------------------------------------

PRIORITY_ORDER = [
    "annex_guidelines.md",   # preprocessed unified reference
    "annex_ontology.md",     # ontology definitions
    "her2_kg_pipeline_guide.md",  # algorithms §6.1–6.3
    # remaining .md files follow
]


def load_all_markdown_docs(
    docs_dir: Path,
    chunk_size: int = 500,
    overlap: int = 100,
    exclude: set[str] | None = None,
) -> list[DocumentChunk]:
    """
    Load all .md files from docs_dir in priority order.
    Returns a flat list of DocumentChunk objects.

    Args:
        exclude: Set of filenames to skip (e.g. non-clinical appendices).
    """
    excluded = exclude or set()
    all_chunks: list[DocumentChunk] = []
    processed: set[str] = set()

    # Priority files first
    for fname in PRIORITY_ORDER:
        if fname in excluded:
            continue
        fpath = docs_dir / fname
        if fpath.exists():
            chunks = load_markdown_document(fpath, chunk_size, overlap)
            if fname in TECHNICAL_APPENDIX_DOCS:
                for c in chunks:
                    c.content_type = ContentType.TECHNICAL_APPENDIX
            all_chunks.extend(chunks)
            processed.add(fname)

    # Remaining .md files (alphabetical)
    for fpath in sorted(docs_dir.glob("*.md")):
        if fpath.name not in processed and fpath.name not in excluded:
            chunks = load_markdown_document(fpath, chunk_size, overlap)
            if fpath.name in TECHNICAL_APPENDIX_DOCS:
                for c in chunks:
                    c.content_type = ContentType.TECHNICAL_APPENDIX
            all_chunks.extend(chunks)

    return all_chunks
