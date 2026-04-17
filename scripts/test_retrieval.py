"""Test vector retrieval end-to-end."""
import os
os.environ.setdefault("HER2_KG_EMBEDDING_MODE", "ollama")
os.environ.setdefault("NEO4J_PASSWORD", "password")

from src.pipeline.config import PipelineConfig
from src.retrieval.entity_retriever import EntityRetriever

cfg = PipelineConfig.from_env()
driver = cfg.get_neo4j_driver()
embedder = cfg.get_embedder()

# 1. Test retrieve (by entity_ids)
ret = EntityRetriever(driver)

print("=== Entity context retrieval ===")
results = ret.retrieve("ihc 2+ equivocal", entity_ids=["Score2Plus", "ISH_Group3"], top_k=5)
for r in results:
    print(f"  {r.get('entity_id')} | {str(r.get('context',''))[:70]}")

# 2. Test vector similarity directly via Cypher
print("\n=== Vector search (Chunk) ===")
query_text = "IHC 2+ reflex ISH testing required"
vec = embedder.embed_query(query_text)
with driver.session() as s:
    rows = s.run(
        """
        CALL db.index.vector.queryNodes('chunk_embeddings', 5, $vec)
        YIELD node, score
        RETURN node.chunk_id AS id, score, left(node.text, 80) AS snippet
        """,
        vec=vec
    ).data()
for row in rows:
    print(f"  [{row['score']:.3f}] {row['id']} — {row['snippet']}")

print("\n=== Vector search (Entity) ===")
query_text2 = "T-DXd trastuzumab deruxtecan HER2 low eligibility"
vec2 = embedder.embed_query(query_text2)
with driver.session() as s:
    rows2 = s.run(
        """
        CALL db.index.vector.queryNodes('entity_embeddings', 5, $vec)
        YIELD node, score
        RETURN node.id AS id, score, node.label AS label
        """,
        vec=vec2
    ).data()
for row in rows2:
    print(f"  [{row['score']:.3f}] {row['id']} ({row['label']})")

driver.close()
print("\nRetrieval OK.")
