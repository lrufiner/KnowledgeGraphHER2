"""
Quick patch: create MENTIONS edges in the current Neo4j graph.
Run once after pipeline has already executed without MENTIONS creation.
"""
from src.pipeline.config import PipelineConfig

cfg = PipelineConfig.from_env()
driver = cfg.get_neo4j_driver()

with driver.session() as s:
    # Check LLM-extracted entities (source_doc = chunk_id, not 'ontology:seed')
    r = s.run(
        "MATCH (e) WHERE NOT e:Chunk AND NOT e:ToySpecimen "
        "AND e.source_doc IS NOT NULL AND e.source_doc <> 'ontology:seed' "
        "RETURN count(e) AS c"
    )
    print("LLM-extracted entities with chunk source_doc:", r.single()["c"])

    # Sample
    r2 = s.run(
        "MATCH (e) WHERE NOT e:Chunk AND NOT e:ToySpecimen "
        "AND e.source_doc IS NOT NULL AND e.source_doc <> 'ontology:seed' "
        "RETURN e.id AS id, e.source_doc AS src LIMIT 5"
    )
    for row in r2:
        print(" ", dict(row))

    # Create MENTIONS edges: Chunk.chunk_id == entity.source_doc
    r3 = s.run(
        "MATCH (c:Chunk), (e) "
        "WHERE NOT e:Chunk AND NOT e:ToySpecimen "
        "AND c.chunk_id = e.source_doc "
        "MERGE (c)-[r:MENTIONS]->(e) "
        "RETURN count(r) AS cnt"
    )
    print("MENTIONS edges merged:", r3.single()["cnt"])

    # Verify
    r4 = s.run("MATCH ()-[:MENTIONS]->() RETURN count(*) AS c")
    print("Total MENTIONS edges now:", r4.single()["c"])

driver.close()
print("Done.")
