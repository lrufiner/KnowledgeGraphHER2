"""
Neo4j graph builder — constructs the HER2 Knowledge Graph from resolved entities,
relations, and seed data.

Uses MERGE to be idempotent (safe to re-run on an existing database).
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j import Driver

from src.domain.models import ResolvedEntity, ResolvedRelation
from src.domain.ontology import SEED_ENTITIES, SEED_RELATIONS, TOY_FRACTAL_SPECIMENS


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

SCHEMA_CONSTRAINTS = [
    # Uniqueness constraints (one per label)
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ClinicalCategory)   REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:IHCScore)           REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ISHGroup)           REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:TherapeuticAgent)   REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:ClinicalTrial)      REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Guideline)          REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:FractalMetric)      REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Assay)              REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Biomarker)          REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:DiagnosticDecision) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Threshold)          REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:StainingPattern)    REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:PathologicalFeature) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:QualityMeasure)     REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Chunk)              REQUIRE n.chunk_id IS UNIQUE",
]

SCHEMA_INDEXES = [
    "CREATE INDEX IF NOT EXISTS FOR (n:ClinicalCategory) ON (n.label)",
    "CREATE INDEX IF NOT EXISTS FOR (n:IHCScore)         ON (n.label)",
    "CREATE INDEX IF NOT EXISTS FOR (n:FractalMetric)    ON (n.label)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Guideline)        ON (n.label)",
    "CREATE INDEX IF NOT EXISTS FOR (n:Chunk)            ON (n.source_doc)",
]


def initialize_schema(driver: Driver) -> None:
    """Create constraints and indexes. Safe to call multiple times."""
    with driver.session() as session:
        for stmt in SCHEMA_CONSTRAINTS:
            session.run(stmt)
        for stmt in SCHEMA_INDEXES:
            session.run(stmt)
    print("[Schema] Constraints and indexes created/verified.")


# ---------------------------------------------------------------------------
# Seed data loader
# ---------------------------------------------------------------------------

_UPSERT_NODE_TEMPLATE = """
MERGE (n:{label} {{id: $id}})
SET n += $props
RETURN n.id AS id
"""

_UPSERT_REL_TEMPLATE = """
MATCH (s {{id: $sid}})
MATCH (o {{id: $oid}})
MERGE (s)-[r:{predicate}]->(o)
SET r += $rprops
"""


def _node_props(entity_dict: dict) -> dict:
    """Extract writable properties (exclude 'type' and 'id')."""
    skip = {"type", "id"}
    return {k: v for k, v in entity_dict.items() if k not in skip and v is not None}


def load_seed_data(driver: Driver) -> dict[str, int]:
    """
    Load all seed entities and relations from ontology.py into Neo4j.
    Returns stats dict.
    """
    stats: dict[str, int] = {"nodes": 0, "relations": 0}
    now = datetime.utcnow().isoformat()

    with driver.session() as session:
        # Nodes
        for ent in SEED_ENTITIES:
            label = ent.get("type", "Entity")
            props = _node_props(ent)
            props["source_doc"] = "ontology:seed"
            props["created_at"] = now
            props["confidence"] = 1.0
            query = _UPSERT_NODE_TEMPLATE.format(label=label)
            session.run(query, id=ent["id"], props=props)
            stats["nodes"] += 1

        # Relations
        for rel in SEED_RELATIONS:
            predicate = rel["predicate"]
            rprops = {k: v for k, v in rel.items()
                      if k not in {"subject_id", "object_id", "predicate"} and v is not None}
            rprops["created_at"] = now
            query = _UPSERT_REL_TEMPLATE.format(predicate=predicate)
            try:
                session.run(
                    query,
                    sid=rel["subject_id"],
                    oid=rel["object_id"],
                    rprops=rprops,
                )
                stats["relations"] += 1
            except Exception as e:
                print(f"  [!] Relation skipped ({rel['subject_id']}->{rel['object_id']}): {e}")

    print(f"[Seed] Loaded {stats['nodes']} nodes, {stats['relations']} relations.")
    return stats


# ---------------------------------------------------------------------------
# Toy fractal specimens
# ---------------------------------------------------------------------------

def load_toy_fractal_specimens(driver: Driver) -> int:
    """
    Load toy/artificial fractal specimens into Neo4j for testing.
    Creates ToySpecimen nodes linked to IHCScore via HAS_FRACTAL_PROFILE relation.
    """
    count = 0
    now = datetime.utcnow().isoformat()

    with driver.session() as session:
        for spec in TOY_FRACTAL_SPECIMENS:
            sid = spec["specimen_id"]
            session.run(
                """
                MERGE (s:ToySpecimen {id: $id})
                SET s.D0 = $D0, s.D1 = $D1, s.Lacunarity = $lac,
                    s.DeltaAlpha = $da, s.MultiscaleEntropy = $me,
                    s.note = $note, s.created_at = $ts,
                    s.source = 'DigPatho_Internal_2025'
                """,
                id=sid, D0=spec["D0"], D1=spec["D1"],
                lac=spec["Lacunarity"], da=spec["DeltaAlpha"],
                me=spec["MultiscaleEntropy"], note=spec["note"], ts=now,
            )
            # Link to IHCScore
            session.run(
                """
                MATCH (spec:ToySpecimen {id: $sid})
                MATCH (ihc:IHCScore {id: $ihc_id})
                MERGE (spec)-[:HAS_IHC_SCORE]->(ihc)
                """,
                sid=sid, ihc_id=spec["ihc_score"],
            )
            count += 1

    print(f"[Toy] Loaded {count} toy fractal specimens.")
    return count


# ---------------------------------------------------------------------------
# Extracted entity/relation upsert
# ---------------------------------------------------------------------------

def upsert_entities(driver: Driver, entities: list[ResolvedEntity]) -> int:
    """Insert/update resolved entities from extraction phase."""
    count = 0
    now = datetime.utcnow().isoformat()

    with driver.session() as session:
        for ent in entities:
            label = ent.type.value
            props = ent.to_neo4j_dict()
            props["created_at"] = now
            query = _UPSERT_NODE_TEMPLATE.format(label=label)
            try:
                session.run(query, id=ent.id, props=props)
                count += 1
            except Exception as e:
                print(f"  [!] Entity upsert failed ({ent.id}): {e}")

    return count


def upsert_relations(driver: Driver, relations: list[ResolvedRelation]) -> int:
    """Insert/update resolved relations from extraction phase."""
    count = 0
    now = datetime.utcnow().isoformat()

    with driver.session() as session:
        for rel in relations:
            predicate = rel.predicate.value
            # proposedEquivalence is exploratory by definition
            is_hyp = True if predicate == "proposedEquivalence" else rel.is_hypothesis
            rprops: dict[str, Any] = {
                "confidence":        rel.confidence,
                "evidence":          rel.evidence or "",
                "source_chunk":      rel.source_chunk or "",
                "guideline_version": rel.guideline_version or "",
                "conditions":        rel.conditions or "",
                "is_hypothesis":     is_hyp,
                "created_at":        now,
            }
            query = _UPSERT_REL_TEMPLATE.format(predicate=predicate)
            try:
                session.run(
                    query,
                    sid=rel.subject_id,
                    oid=rel.object_id,
                    rprops=rprops,
                )
                count += 1
            except Exception as e:
                print(f"  [!] Relation upsert failed ({rel.subject_id}->{rel.object_id}): {e}")

    return count


def create_mentions_edges(driver: Driver, entities: list) -> int:
    """
    Create (:Chunk)-[:MENTIONS]->(:Entity) edges for GraphRAG entity-centric retrieval.

    Each ResolvedEntity.source_doc stores the chunk_id that produced it during
    LLM extraction.  This function wires those provenance links into the graph so
    that neo4j-graphrag VectorCypherRetriever can traverse chunk→entity paths.
    """
    count = 0
    with driver.session() as session:
        for ent in entities:
            chunk_id = getattr(ent, "source_doc", None)
            if not chunk_id:
                continue
            try:
                result = session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MATCH (e {id: $entity_id})
                    WHERE NOT e:Chunk AND NOT e:ToySpecimen
                    MERGE (c)-[r:MENTIONS]->(e)
                    RETURN count(r) AS merged
                    """,
                    chunk_id=chunk_id,
                    entity_id=ent.id,
                )
                row = result.single()
                if row and row["merged"] > 0:
                    count += 1
            except Exception:
                pass
    return count


def upsert_chunk_node(driver: Driver, chunk_id: str, source_doc: str,
                       section: str, content: str, content_type: str) -> None:
    """Store a Chunk node (for provenance and vector indexing later)."""
    with driver.session() as session:
        session.run(
            """
            MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.source_doc    = $source_doc,
                c.section       = $section,
                c.text          = $content,
                c.content_type  = $content_type,
                c.created_at    = $ts
            """,
            chunk_id=chunk_id, source_doc=source_doc,
            section=section, content=content[:5000],  # cap to 5000 chars
            content_type=content_type,
            ts=datetime.utcnow().isoformat(),
        )


# ---------------------------------------------------------------------------
# Stats query
# ---------------------------------------------------------------------------

def get_graph_stats(driver: Driver) -> dict[str, Any]:
    """Return basic stats about the current KG state."""
    with driver.session() as session:
        node_counts = {}
        for label in [
            "ClinicalCategory", "IHCScore", "ISHGroup", "TherapeuticAgent",
            "ClinicalTrial", "Guideline", "FractalMetric", "Assay",
            "DiagnosticDecision", "Chunk", "ToySpecimen",
        ]:
            res = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
            node_counts[label] = res.single()["c"]

        rel_res = session.run("MATCH ()-[r]->() RETURN count(r) AS c")
        total_rels = rel_res.single()["c"]

    return {
        "node_counts": node_counts,
        "total_nodes": sum(node_counts.values()),
        "total_relations": total_rels,
    }
