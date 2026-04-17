"""Recreate vector indexes with correct 768-dim for nomic-embed-text."""
import os
os.environ.setdefault("NEO4J_PASSWORD", "password")
os.environ.setdefault("HER2_KG_EMBEDDING_MODE", "ollama")

from src.pipeline.config import PipelineConfig
from src.graph.vector_indexer import create_vector_indexes

cfg = PipelineConfig.from_env()
driver = cfg.get_neo4j_driver()

# Drop old 1536-dim indexes
with driver.session() as s:
    s.run("DROP INDEX entity_embeddings IF EXISTS")
    s.run("DROP INDEX chunk_embeddings IF EXISTS")
    print("Old indexes dropped.")

# Recreate with 768-dim
create_vector_indexes(driver, dim=768, verbose=True)
print(f"Embedding dim from config: {cfg.embedding_dim}")
driver.close()
print("Done.")
