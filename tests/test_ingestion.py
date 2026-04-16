"""Tests for Markdown loader and chunker."""
import pytest
from pathlib import Path
from src.domain.models import ContentType
from src.ingestion.markdown_loader import (
    load_markdown_document, _detect_content_type, _split_into_sentences,
)


class TestContentTypeDetection:
    def test_detects_criteria(self):
        text = "IHC score 2+ requires ISH reflex testing. tinción circumferencial."
        ct = _detect_content_type(text)
        assert ct == ContentType.CRITERIA

    def test_detects_fractal(self):
        text = "The fractal dimension D0 measures architectural complexity. Lacunarity is high."
        ct = _detect_content_type(text)
        assert ct == ContentType.FRACTAL_MAPPING

    def test_detects_recommendation(self):
        text = "The guideline recommends that pathologists should use validated assays."
        ct = _detect_content_type(text)
        assert ct in (ContentType.RECOMMENDATION, ContentType.CRITERIA, ContentType.GENERAL)

    def test_general_fallback(self):
        ct = _detect_content_type("This is a neutral sentence with no keywords.")
        assert ct == ContentType.GENERAL


class TestSentenceSplitter:
    def test_splits_on_period(self):
        text = "This is sentence one. This is sentence two. And sentence three."
        parts = _split_into_sentences(text)
        assert len(parts) >= 2

    def test_handles_empty_string(self):
        parts = _split_into_sentences("")
        assert parts == [""]


class TestMarkdownLoader:
    def test_loads_real_annex_guidelines(self):
        """Integration test — requires docs/annex_guidelines.md to exist."""
        docs_dir = Path("./docs")
        fpath = docs_dir / "annex_guidelines.md"
        if not fpath.exists():
            pytest.skip("annex_guidelines.md not found — integration test skipped")

        chunks = load_markdown_document(fpath, chunk_size=500, overlap=100)
        assert len(chunks) > 5, f"Expected >5 chunks, got {len(chunks)}"

        # All chunks should have non-empty content
        for chunk in chunks:
            assert len(chunk.content.strip()) >= 10, f"Short chunk: {chunk.chunk_id}"
            assert chunk.source_doc == "annex_guidelines.md"

    def test_loads_real_annex_ontology(self):
        """Integration test — requires docs/annex_ontology.md to exist."""
        docs_dir = Path("./docs")
        fpath = docs_dir / "annex_ontology.md"
        if not fpath.exists():
            pytest.skip("annex_ontology.md not found")

        chunks = load_markdown_document(fpath)
        assert len(chunks) > 0
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs!"

    def test_chunk_ids_are_unique(self):
        docs_dir = Path("./docs")
        if not docs_dir.exists():
            pytest.skip("docs dir not found")
        from src.ingestion.markdown_loader import load_all_markdown_docs
        chunks = load_all_markdown_docs(docs_dir)
        if not chunks:
            pytest.skip("No markdown files found")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), f"Duplicate chunk IDs found!"
