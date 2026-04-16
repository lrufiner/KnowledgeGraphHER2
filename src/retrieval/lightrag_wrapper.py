"""
LightRAG integration wrapper for the HER2 Knowledge Graph.

Implements §6.2c of the implementation plan: community-based retrieval
using LightRAG with Neo4j backend for global (multi-hop) queries.

LightRAG provides two complementary retrieval modes:
  - local  : entity-level lookup (matches neo4j-graphrag vector search)
  - global : community-level reasoning across the entire knowledge graph

This wrapper is optional — the system degrades gracefully when LightRAG
is not installed or when NEO4J_URI is not configured.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import — LightRAG is not a hard dependency
# ---------------------------------------------------------------------------

try:
    from lightrag import LightRAG, QueryParam  # type: ignore[import]
    from lightrag.kg.neo4j_impl import Neo4JStorage  # type: ignore[import]
    _LIGHTRAG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIGHTRAG_AVAILABLE = False
    LightRAG = None  # type: ignore[assignment,misc]
    QueryParam = None  # type: ignore[assignment]
    Neo4JStorage = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class LightRAGResult:
    """Result from a LightRAG query."""
    query: str
    mode: str                        # "local" | "global" | "hybrid" | "naive"
    answer: str
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    sources: list[str]
    is_fallback: bool = False        # True when LightRAG used fallback path


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------


class LightRAGWrapper:
    """
    Wraps LightRAG to provide global/community-level retrieval from the HER2 KG.

    Instantiate once per session; the inner RAG index is lazily configured.
    When LightRAG is not installed, all methods return graceful fallbacks.

    Example:
        wrapper = LightRAGWrapper.from_env()
        result = wrapper.query("What are the ISH Group 3 workup requirements?",
                                mode="hybrid")
        print(result.answer)
    """

    # Map from human-friendly mode names to LightRAG QueryParam modes
    _MODE_MAP: dict[str, str] = {
        "local": "local",
        "global": "global",
        "hybrid": "hybrid",
        "naive": "naive",
    }

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        working_dir: str = "./her2_lightrag",
        llm_model: str = "qwen3:8b",
        embed_model: str = "nomic-embed-text",
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._working_dir = working_dir
        self._llm_model = llm_model
        self._embed_model = embed_model
        self._rag: Any = None  # LightRAG instance, lazily created

    @classmethod
    def from_env(cls) -> "LightRAGWrapper":
        """Create from environment variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)."""
        import os
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            working_dir=os.getenv("LIGHTRAG_WORKING_DIR", "./her2_lightrag"),
            llm_model=os.getenv("LLM_MODEL", "qwen3:8b"),
            embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        )

    @property
    def is_available(self) -> bool:
        """True if the LightRAG library is installed."""
        return _LIGHTRAG_AVAILABLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> LightRAGResult:
        """
        Run a query against the LightRAG index.

        Args:
            query_text: Natural language question.
            mode: "local" | "global" | "hybrid" | "naive"
                  - local  → entity-level (faster)
                  - global → community summary (slower, more comprehensive)
                  - hybrid → both (recommended for clinical queries)
            top_k: Maximum entities/chunks to retrieve.

        Returns:
            LightRAGResult — falls back gracefully if LightRAG not available.
        """
        if not _LIGHTRAG_AVAILABLE:
            return self._fallback_result(query_text, mode)

        try:
            rag = self._get_or_build_rag()
            param = QueryParam(mode=self._MODE_MAP.get(mode, "hybrid"), top_k=top_k)
            answer = rag.query(query_text, param=param)
            return LightRAGResult(
                query=query_text,
                mode=mode,
                answer=str(answer),
                entities=[],   # LightRAG embeds these in the answer text
                relations=[],
                sources=[],
                is_fallback=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LightRAG query failed: %s — falling back", exc)
            return self._fallback_result(query_text, mode)

    def insert_documents(self, texts: list[str]) -> None:
        """
        Insert documents into the LightRAG index.
        Call this after ingesting new guidelines.
        """
        if not _LIGHTRAG_AVAILABLE:
            logger.info("LightRAG not installed — skipping document insertion.")
            return
        try:
            rag = self._get_or_build_rag()
            for text in texts:
                rag.insert(text)
            logger.info("Inserted %d documents into LightRAG index.", len(texts))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LightRAG insert failed: %s", exc)

    def query_her2_classification(self, ihc_score: str, ish_group: str = "") -> LightRAGResult:
        """
        Convenience method: build a focused HER2 classification query.
        Uses hybrid mode for maximum coverage.
        """
        parts = [f"IHC {ihc_score}"]
        if ish_group:
            parts.append(f"ISH {ish_group}")
        case_str = ", ".join(parts)
        query_text = (
            f"For a breast cancer case with {case_str}: "
            f"what is the correct HER2 classification per ASCO/CAP 2023, "
            f"and what treatment options are available?"
        )
        return self.query(query_text, mode="hybrid")

    def query_therapeutic_eligibility(self, category: str) -> LightRAGResult:
        """
        Convenience method: retrieve therapeutic eligibility for a HER2 category.
        Uses global mode to aggregate community-level knowledge.
        """
        query_text = (
            f"What therapeutic agents are patients with {category} status eligible for? "
            f"Include T-DXd / trastuzumab deruxtecan, pertuzumab, and other HER2-targeted therapies."
        )
        return self.query(query_text, mode="global")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_build_rag(self) -> Any:
        """Lazily create the LightRAG instance."""
        if self._rag is not None:
            return self._rag

        import os
        os.makedirs(self._working_dir, exist_ok=True)

        self._rag = LightRAG(
            working_dir=self._working_dir,
            graph_storage="Neo4JStorage",
            kg_triplet_extract_max_gleaning=2,
            addon_params={
                "uri": self._uri,
                "username": self._user,
                "password": self._password,
            },
            # LLM function — uses Ollama
            llm_model_func=self._ollama_llm,
            embedding_func=self._ollama_embed,
        )
        logger.info("LightRAG instance created (working_dir=%s)", self._working_dir)
        return self._rag

    @staticmethod
    def _ollama_llm(prompt: str, **kwargs: Any) -> str:  # pragma: no cover
        """Synchronous Ollama call for LightRAG's llm_model_func."""
        import os
        try:
            import ollama
            model = os.getenv("LLM_MODEL", "qwen3:8b")
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as exc:
            logger.warning("Ollama LLM call failed: %s", exc)
            return ""

    @staticmethod
    def _ollama_embed(texts: list[str], **kwargs: Any) -> list[list[float]]:  # pragma: no cover
        """Synchronous Ollama embedding call for LightRAG's embedding_func."""
        import os
        try:
            import ollama
            model = os.getenv("EMBED_MODEL", "nomic-embed-text")
            embeddings = []
            for text in texts:
                resp = ollama.embeddings(model=model, prompt=text)
                embeddings.append(resp["embedding"])
            return embeddings
        except Exception as exc:
            logger.warning("Ollama embed call failed: %s", exc)
            return [[0.0] * 768 for _ in texts]

    def _fallback_result(self, query_text: str, mode: str) -> LightRAGResult:
        return LightRAGResult(
            query=query_text,
            mode=mode,
            answer=(
                "LightRAG is not available. "
                "Install it with: pip install lightrag-hku[neo4j] "
                "to enable community-level retrieval."
            ),
            entities=[],
            relations=[],
            sources=[],
            is_fallback=True,
        )
