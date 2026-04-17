"""
Neo4j vector index creation and embedding upsert.

Creates vector indexes for:
  - Entity embeddings  → index name: "entity_embeddings"  (on any node with an embedding)
  - Chunk embeddings   → index name: "chunk_embeddings"   (on Chunk nodes)

The embedding dimension defaults to 1536 (OpenAI text-embedding-3-small).
For Ollama embeddings (e.g. nomic-embed-text), set dimension=768.

Usage:
    from src.graph.vector_indexer import create_vector_indexes, upsert_embedding
    create_vector_indexes(driver, dim=1536)
    upsert_embedding(driver, node_id="Score3Plus", embedding=[...], label="IHCScore")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from neo4j import Driver


# ---------------------------------------------------------------------------
# Index creation
# ---------------------------------------------------------------------------

_ENTITY_INDEX_DDL = """
CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
FOR (n:_EmbeddableEntity)
ON (n.embedding)
OPTIONS {{
    indexConfig: {{
        `vector.dimensions`: {dim},
        `vector.similarity_function`: 'cosine'
    }}
}}
"""

_CHUNK_INDEX_DDL = """
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
FOR (n:Chunk)
ON (n.embedding)
OPTIONS {{
    indexConfig: {{
        `vector.dimensions`: {dim},
        `vector.similarity_function`: 'cosine'
    }}
}}
"""


def create_vector_indexes(driver: Driver, dim: int = 1536, verbose: bool = True) -> None:
    """
    Create (or verify) the vector indexes on Neo4j.

    Neo4j 5.11+ supports CREATE VECTOR INDEX syntax.
    This is idempotent — safe to call when indexes already exist.
    """
    with driver.session() as session:
        try:
            session.run(_ENTITY_INDEX_DDL.format(dim=dim))
            if verbose:
                print(f"  [VectorIndex] entity_embeddings index ready (dim={dim})")
        except Exception as e:
            if verbose:
                print(f"  [VectorIndex] entity_embeddings: {e!s:.80}")

        try:
            session.run(_CHUNK_INDEX_DDL.format(dim=dim))
            if verbose:
                print(f"  [VectorIndex] chunk_embeddings index ready (dim={dim})")
        except Exception as e:
            if verbose:
                print(f"  [VectorIndex] chunk_embeddings: {e!s:.80}")


# ---------------------------------------------------------------------------
# Embedding upsert helpers
# ---------------------------------------------------------------------------

_SET_NODE_EMBEDDING = """
MATCH (n {id: $node_id})
SET n.embedding = $embedding,
    n:_EmbeddableEntity
"""

_SET_CHUNK_EMBEDDING = """
MATCH (n:Chunk {chunk_id: $chunk_id})
SET n.embedding = $embedding
"""


def upsert_entity_embedding(
    driver: Driver,
    node_id: str,
    embedding: Sequence[float],
) -> None:
    """Store an embedding vector on any node with the given id."""
    with driver.session() as session:
        session.run(_SET_NODE_EMBEDDING, node_id=node_id, embedding=list(embedding))


def upsert_chunk_embedding(
    driver: Driver,
    chunk_id: str,
    embedding: Sequence[float],
) -> None:
    """Store an embedding vector on a Chunk node."""
    with driver.session() as session:
        session.run(_SET_CHUNK_EMBEDDING, chunk_id=chunk_id, embedding=list(embedding))


# ---------------------------------------------------------------------------
# Batch embedding utility
# ---------------------------------------------------------------------------

def embed_all_entities(
    driver: Driver,
    embedder: Any,
    batch_size: int = 50,
    verbose: bool = True,
) -> int:
    """
    Embed all entity nodes that don't yet have an embedding.

    Args:
        driver:     Neo4j driver
        embedder:   LangChain Embeddings instance (e.g. OpenAIEmbeddings)
        batch_size: number of entities to embed per API call
        verbose:    print progress

    Returns:
        Total number of entities embedded.
    """
    # Fetch nodes without embeddings (all label types except Chunk and internal labels)
    _FETCH_UNEMBEDDED = """
    MATCH (n)
    WHERE n.label IS NOT NULL
      AND n.embedding IS NULL
      AND NOT n:Chunk
      AND NOT n:_EmbeddableEntity
    RETURN n.id AS id, n.label AS label, n.definition AS definition
    LIMIT 5000
    """
    with driver.session() as session:
        records = session.run(_FETCH_UNEMBEDDED).data()

    if not records:
        if verbose:
            print("  [VectorIndexer] No entities to embed.")
        return 0

    texts = [
        f"{r['label']}: {r['definition'] or ''}" for r in records
    ]
    ids   = [r["id"] for r in records]

    total = 0
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids   = ids[i:i + batch_size]
        embeddings  = embedder.embed_documents(batch_texts)
        for node_id, emb in zip(batch_ids, embeddings):
            upsert_entity_embedding(driver, node_id, emb)
        total += len(batch_ids)
        if verbose:
            print(f"  [VectorIndexer] Embedded {total}/{len(texts)} entities...")

    return total


def embed_all_chunks(
    driver: Driver,
    embedder: Any,
    batch_size: int = 50,
    verbose: bool = True,
) -> int:
    """
    Embed all Chunk nodes without an embedding.

    Returns:
        Total number of chunks embedded.
    """
    _FETCH_CHUNKS = """
    MATCH (n:Chunk)
    WHERE n.embedding IS NULL
    RETURN n.chunk_id AS chunk_id, coalesce(n.text, n.content, '') AS content
    LIMIT 5000
    """
    with driver.session() as session:
        records = session.run(_FETCH_CHUNKS).data()

    if not records:
        if verbose:
            print("  [VectorIndexer] No chunks to embed.")
        return 0

    texts      = [r["content"] or "" for r in records]
    chunk_ids  = [r["chunk_id"] for r in records]

    total = 0
    for i in range(0, len(texts), batch_size):
        batch_texts    = texts[i:i + batch_size]
        batch_ids      = chunk_ids[i:i + batch_size]
        embeddings     = embedder.embed_documents(batch_texts)
        for chunk_id, emb in zip(batch_ids, embeddings):
            upsert_chunk_embedding(driver, chunk_id, emb)
        total += len(batch_ids)
        if verbose:
            print(f"  [VectorIndexer] Embedded {total}/{len(texts)} chunks...")

    return total


# ---------------------------------------------------------------------------
# Type annotation workaround
# ---------------------------------------------------------------------------
from typing import Any  # noqa: E402 — must come after class body
