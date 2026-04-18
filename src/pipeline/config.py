"""
Pipeline configuration — LLM provider abstraction + environment loading.

Supports three providers via LangChain BaseChatModel:
  - Claude Sonnet 4 (Anthropic API)       → mode = "claude"
  - GPT-4o-mini (OpenAI API)              → mode = "openai"
  - Ollama local models                   → mode = "ollama"

Usage:
    cfg = PipelineConfig.from_env()
    llm = cfg.get_llm()           # ready-to-use ChatModel
    embedder = cfg.get_embedder() # ready-to-use Embeddings
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from project root (two levels up from this file)
_ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_ENV_PATH, override=False)


class PipelineConfig(BaseModel):
    # ── LLM settings ────────────────────────────────────────────────────────
    llm_mode:    str = Field(default="ollama",
                             description="claude | openai | ollama")
    # Claude
    anthropic_api_key: Optional[str] = None
    claude_model:      str = "claude-sonnet-4-5"
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_model:   str = "gpt-4o-mini"
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model:    str = "qwen3:8b"

    # ── Embedding settings ───────────────────────────────────────────────────
    embedding_mode:  str = "openai"   # openai | ollama
    embedding_model: str = "text-embedding-3-small"
    embedding_dim:   int = 1536

    # ── Neo4j settings ───────────────────────────────────────────────────────
    neo4j_uri:       str = "bolt://localhost:7687"
    neo4j_username:  str = "neo4j"
    neo4j_password:  str = "password"
    neo4j_database:  Optional[str] = None

    # ── Document paths ───────────────────────────────────────────────────────
    docs_dir:   str = "./docs"
    guides_dir: str = "./guides"
    output_dir: str = "./output"

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size:    int = 500
    chunk_overlap: int = 100

    # ── Extraction ───────────────────────────────────────────────────────────
    llm_temperature:     float = 0.0
    llm_max_tokens:      int   = 3000
    min_confidence_threshold: float = 0.75

    # ── LangSmith (optional observability) ──────────────────────────────────
    langsmith_tracing:   bool = False
    langsmith_api_key:   Optional[str] = None
    langsmith_project:   str = "her2-kg"

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Load config from environment variables (with sensible defaults)."""
        return cls(
            llm_mode         = os.getenv("HER2_KG_LLM_MODE", "ollama"),
            anthropic_api_key= os.getenv("ANTHROPIC_API_KEY"),
            claude_model     = os.getenv("HER2_KG_CLAUDE_MODEL", "claude-sonnet-4-5"),
            openai_api_key   = os.getenv("OPENAI_API_KEY"),
            openai_model     = os.getenv("HER2_KG_OPENAI_MODEL", "gpt-4o-mini"),
            ollama_base_url  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model     = os.getenv("HER2_KG_OLLAMA_MODEL", "qwen3:8b"),
            embedding_mode   = os.getenv("HER2_KG_EMBEDDING_MODE", "openai"),
            embedding_model  = os.getenv("HER2_KG_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim    = int(os.getenv("HER2_KG_EMBEDDING_DIM",
                                  "768" if os.getenv("HER2_KG_EMBEDDING_MODE", "openai") == "ollama"
                                  else "1536")),
            neo4j_uri        = os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username   = os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password   = os.getenv("NEO4J_PASSWORD", "password"),
            neo4j_database   = os.getenv("NEO4J_DATABASE") or None,
            docs_dir         = os.getenv("HER2_KG_DOCS_DIR", "./docs"),
            guides_dir       = os.getenv("HER2_KG_GUIDES_DIR", "./guides"),
            output_dir       = os.getenv("HER2_KG_OUTPUT_DIR", "./output"),
            langsmith_tracing= os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
            langsmith_api_key= os.getenv("LANGCHAIN_API_KEY"),
            langsmith_project= os.getenv("LANGCHAIN_PROJECT", "her2-kg"),
        )

    def get_llm(self, json_mode: bool = True):
        """Return a LangChain BaseChatModel for the configured provider.

        Args:
            json_mode: For Ollama only — forces JSON output format.
                       Set False for free-text tasks like narration.
        """
        if self.llm_mode == "claude":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.claude_model,
                api_key=self.anthropic_api_key,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )
        elif self.llm_mode == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.openai_model,
                api_key=self.openai_api_key,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )
        elif self.llm_mode == "ollama":
            from langchain_ollama import ChatOllama
            kwargs: dict = {
                "model": self.ollama_model,
                "base_url": self.ollama_base_url,
                "temperature": self.llm_temperature,
                "num_predict": 3000,
                "num_ctx": 8192,
            }
            if json_mode:
                kwargs["format"] = "json"
            return ChatOllama(**kwargs)
        else:
            raise ValueError(f"Unknown LLM mode: {self.llm_mode!r}. "
                             f"Valid options: claude | openai | ollama")

    def get_embedder(self):
        """Return a LangChain Embeddings object."""
        if self.embedding_mode == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=self.openai_api_key,
            )
        elif self.embedding_mode == "ollama":
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=self.ollama_base_url,
            )
        else:
            raise ValueError(f"Unknown embedding mode: {self.embedding_mode!r}")

    def get_neo4j_driver(self):
        """Return a Neo4j driver configured for local or AuraDB."""
        from neo4j import GraphDatabase
        return GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password),
        )

    def configure_langsmith(self) -> None:
        """Enable LangSmith tracing if configured."""
        if self.langsmith_tracing and self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"]     = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"]     = self.langsmith_project

    def __repr__(self) -> str:
        return (
            f"PipelineConfig(llm_mode={self.llm_mode!r}, "
            f"model={self._active_model()!r}, "
            f"neo4j={self.neo4j_uri!r})"
        )

    def _active_model(self) -> str:
        if self.llm_mode == "claude":  return self.claude_model
        if self.llm_mode == "openai":  return self.openai_model
        return self.ollama_model
