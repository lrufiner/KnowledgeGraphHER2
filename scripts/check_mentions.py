"""Check MENTIONS edges distribution."""
import os
os.environ.setdefault("NEO4J_PASSWORD", "password")
from src.pipeline.config import PipelineConfig
cfg = PipelineConfig.from_env()
driver = cfg.get_neo4j_driver()
with driver.session() as s:
    rows = s.run(
        "MATCH (c:Chunk)-[:MENTIONS]->(e) RETURN e.id AS eid, count(c) AS cnt ORDER BY cnt DESC LIMIT 10"
    ).data()
    print("Top MENTIONS targets:")
    for row in rows:
        print(f"  {row['eid']}: {row['cnt']} chunks")

    total = s.run("MATCH ()-[:MENTIONS]->() RETURN count(*) AS n").single()["n"]
    print(f"\nTotal MENTIONS: {total}")

    # Check chunk text property
    sample = s.run(
        "MATCH (c:Chunk) WHERE c.text IS NOT NULL RETURN c.chunk_id AS id, left(c.text,50) AS t LIMIT 2"
    ).data()
    for row in sample:
        print(f"  Chunk {row['id']}: {row['t']}")
driver.close()
