"""
Entity-centric retrieval from the HER2 Knowledge Graph.

Fetches context chunks and related entities given a set of entity IDs,
following the pattern defined in §6.2a of the implementation plan.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j import Driver

# ---------------------------------------------------------------------------
# Cypher query (from §6.2a)
# ---------------------------------------------------------------------------

_ENTITY_RETRIEVAL_QUERY = """
MATCH (chunk:Chunk)-[:MENTIONS]->(entity)
WHERE entity.id IN $entity_ids
OPTIONAL MATCH (entity)-[r]->(related)
RETURN chunk.text           AS context,
       entity.label         AS entity,
       entity.id            AS entity_id,
       type(r)              AS relation,
       related.label        AS related_entity,
       r.confidence         AS confidence,
       chunk.source_doc     AS source_doc,
       chunk.source_quote   AS source_quote
ORDER BY r.confidence DESC
LIMIT 20
"""

_ENTITY_BY_LABEL_QUERY = """
MATCH (e)
WHERE toLower(e.label) CONTAINS toLower($label)
  AND NOT e:Chunk
RETURN e.id AS id, e.label AS label, labels(e)[0] AS node_type
ORDER BY length(e.label)
LIMIT 10
"""


class EntityRetriever:
    """Retrieve entity-centric context from Neo4j for grounded LLM generation."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        entity_ids: list[str],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Retrieve context chunks and neighbour entities for *entity_ids*.

        Args:
            query:      Natural language query (informational, not yet used for
                        Cypher). Reserved for future hybrid re-ranking.
            entity_ids: List of entity node IDs to pivot on.
            top_k:      Maximum rows to return (applied in Cypher LIMIT).

        Returns:
            List of row dicts with keys: context, entity, entity_id, relation,
            related_entity, confidence, source_doc, source_quote.
        """
        if not entity_ids:
            return []

        with self._driver.session() as session:
            result = session.run(
                _ENTITY_RETRIEVAL_QUERY.replace("LIMIT 20", f"LIMIT {top_k}"),
                entity_ids=entity_ids,
            )
            return [dict(r) for r in result]

    def find_entities_by_label(self, label: str) -> list[dict[str, Any]]:
        """Case-insensitive substring search for entity nodes by label.

        Args:
            label: Partial or full entity label.

        Returns:
            List of dicts with id, label, node_type.
        """
        with self._driver.session() as session:
            result = session.run(_ENTITY_BY_LABEL_QUERY, label=label)
            return [dict(r) for r in result]

    def retrieve_by_labels(
        self,
        labels: list[str],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Convenience wrapper: resolve label strings → entity IDs → retrieve.

        Args:
            labels: List of entity label strings.
            top_k:  Maximum rows to return overall.

        Returns:
            Combined list of context rows for all matched entities.
        """
        entity_ids: list[str] = []
        for lbl in labels:
            matches = self.find_entities_by_label(lbl)
            entity_ids.extend(m["id"] for m in matches)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_ids: list[str] = []
        for eid in entity_ids:
            if eid not in seen:
                seen.add(eid)
                unique_ids.append(eid)

        return self.retrieve("", unique_ids, top_k=top_k)
