"""
Diagnostic pathway retrieval from the HER2 Knowledge Graph.

Fetches IHC→ISH→ClinicalCategory traversal paths from Neo4j,
following the pattern defined in §6.2b of the implementation plan.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j import Driver

# ---------------------------------------------------------------------------
# Cypher queries (from §6.2b)
# ---------------------------------------------------------------------------

_PATHWAY_QUERY = """
MATCH path = (start:IHCScore {id: $ihc_score})
              -[:implies|requiresReflexTest|leadsTo*1..5]->(end:ClinicalCategory)
RETURN [node IN nodes(path) | {id: node.id, label: node.label, types: labels(node)}] AS nodes,
       [r    IN relationships(path) | {type: type(r), props: properties(r)}]         AS steps,
       end.id    AS final_classification,
       end.label AS final_label
ORDER BY length(path)
LIMIT 5
"""

_ISH_PATHWAY_QUERY = """
MATCH path = (grp:ISHGroup {id: $ish_group})
              -[:leadsTo*1..4]->(end:ClinicalCategory)
RETURN [node IN nodes(path) | {id: node.id, label: node.label}] AS nodes,
       [r    IN relationships(path) | {type: type(r), label: r.label}]  AS steps,
       end.id    AS final_classification,
       end.label AS final_label
ORDER BY length(path)
LIMIT 5
"""

_ALGORITHM_PATHWAY_QUERY = """
MATCH (start:DiagnosticDecision {id: $entry_id, algorithm_id: $algorithm_id})
MATCH path = (start)-[:leadsTo*1..8]->(n)
RETURN DISTINCT
       [node IN nodes(path) | {id: node.id, question: node.question,
                                node_type: node.node_type, result: node.result,
                                category: node.category, guideline_source: node.guideline_source}] AS nodes,
       length(path) AS depth
ORDER BY depth
LIMIT 20
"""


class PathwayRetriever:
    """Retrieve diagnostic pathways from the HER2 KG for clinical decision support."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pathway(
        self,
        ihc_score: str,
        ish_group: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve the dignostic pathway for a given IHC score.

        If *ihc_score* is equivocal (2+) and *ish_group* is provided, also
        fetches the ISH-branch pathway.

        Args:
            ihc_score: IHCScore node ID, e.g. 'Score2Plus'.
            ish_group: Optional ISHGroup node ID, e.g. 'Group3'.

        Returns:
            Dict with 'ihc_path', optional 'ish_path', and 'final_classification'.
        """
        result: dict[str, Any] = {"ihc_score": ihc_score, "ish_group": ish_group}

        with self._driver.session() as session:
            # IHC branch
            rows = list(session.run(_PATHWAY_QUERY, ihc_score=ihc_score))
            result["ihc_path"] = [dict(r) for r in rows]

            # ISH branch (only when provided)
            if ish_group:
                rows_ish = list(session.run(_ISH_PATHWAY_QUERY, ish_group=ish_group))
                result["ish_path"] = [dict(r) for r in rows_ish]
            else:
                result["ish_path"] = []

        # Best final classification: first non-null hit from IHC path
        for row in result["ihc_path"]:
            cls = row.get("final_classification")
            if cls:
                result["final_classification"] = cls
                result["final_label"] = row.get("final_label", cls)
                break
        else:
            # fall back to ISH
            for row in result.get("ish_path", []):
                cls = row.get("final_classification")
                if cls:
                    result["final_classification"] = cls
                    result["final_label"] = row.get("final_label", cls)
                    break
            else:
                result["final_classification"] = None
                result["final_label"] = None

        return result

    def get_algorithm_pathway(
        self,
        algorithm_id: str,
        entry_node_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return ordered decision nodes from the algorithm decision-tree.

        Args:
            algorithm_id:  e.g. 'IHC_ASCO_CAP_2023'.
            entry_node_id: Start node ID; defaults to algorithm_id + '_ENTRY'
                           heuristic if not given.

        Returns:
            List of path rows, each containing 'nodes' (ordered) and 'depth'.
        """
        if entry_node_id is None:
            # Heuristic: look up the entry node for the algorithm
            prefix = algorithm_id.split("_")[0]  # e.g. IHC
            entry_node_id = f"{prefix}_ENTRY"

        with self._driver.session() as session:
            rows = list(
                session.run(
                    _ALGORITHM_PATHWAY_QUERY,
                    entry_id=entry_node_id,
                    algorithm_id=algorithm_id,
                )
            )
        return [dict(r) for r in rows]

    def get_all_ihc_pathways(self) -> list[dict[str, Any]]:
        """Return pathways for all IHC scores in the KG."""
        cypher = "MATCH (s:IHCScore) RETURN s.id AS score_id ORDER BY s.score DESC"
        pathways: list[dict[str, Any]] = []
        with self._driver.session() as session:
            scores = [r["score_id"] for r in session.run(cypher)]
        for score_id in scores:
            pw = self.get_pathway(score_id)
            if pw.get("final_classification"):
                pathways.append(pw)
        return pathways
