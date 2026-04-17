"""Compute and store embeddings for all entities and chunks in Neo4j.

Idempotent — only embeds nodes without an existing embedding.
Uses nomic-embed-text via Ollama (768-dim).
"""
from src.pipeline.config import PipelineConfig
from src.graph.vector_indexer import create_vector_indexes, embed_all_entities, embed_all_chunks

cfg = PipelineConfig.from_env()
driver = cfg.get_neo4j_driver()

print(f"Embedding mode: {cfg.embedding_mode}, dim: {cfg.embedding_dim}")
print(f"Neo4j: {cfg.neo4j_uri}\n")

try:
    create_vector_indexes(driver, dim=cfg.embedding_dim, verbose=True)
    embedder = cfg.get_embedder()
    n_ent = embed_all_entities(driver, embedder, verbose=True)
    n_chk = embed_all_chunks(driver, embedder, verbose=True)
    print(f"\nDone: {n_ent} entities + {n_chk} chunks embedded.")
finally:
    driver.close()
