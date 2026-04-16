"""
Shared LangChain tools for the HER2 multi-agent system.

All tools are implemented as @tool-decorated functions that accept a
Neo4j driver injected at call-site through a closure factory so they
remain serialisable (no driver stored in the function object itself).
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_tools(driver: Any) -> list:
    """
    Build a list of bound LangChain tools that share *driver*.

    Usage::

        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, pw))
        tools = make_tools(driver)

    Returns a list of @tool instances ready for binding to an LLM or
    agent executor.
    """

    @tool
    def execute_cypher(query: str, params: dict | None = None) -> list[dict]:
        """Execute an arbitrary read-only Cypher query against the HER2 KG.

        Args:
            query: Valid Cypher SELECT / MATCH query.
            params: Optional dict of query parameters.

        Returns:
            List of result dicts (one per row).
        """
        params = params or {}
        with driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    @tool
    def traverse_decision_tree(
        algorithm_id: str,
        start_node_id: str = "IHC_ENTRY",
        max_hops: int = 10,
    ) -> list[dict]:
        """Return the ordered list of decision nodes reachable from *start_node_id*
        in the given algorithm, following leadsTo edges (BFS).

        Args:
            algorithm_id: e.g. 'IHC_ASCO_CAP_2023'.
            start_node_id: ID of the entry node.
            max_hops: Safety limit on traversal depth.

        Returns:
            Ordered list of node dicts with id, question, type, etc.
        """
        cypher = """
        MATCH path = (start:DiagnosticDecision {id: $start_id, algorithm_id: $algo_id})
              -[:leadsTo*1..%d]->(n)
        RETURN DISTINCT n.id AS id, n.question AS question,
               n.node_type AS node_type, n.node_order AS node_order,
               n.result AS result, n.category AS category,
               n.guideline_source AS guideline_source
        ORDER BY n.node_order
        """ % max_hops
        with driver.session() as session:
            records = session.run(
                cypher,
                start_id=start_node_id,
                algo_id=algorithm_id,
            )
            return [dict(r) for r in records]

    @tool
    def get_ihc_algorithm() -> list[dict]:
        """Return all nodes of the IHC_ASCO_CAP_2023 decision algorithm in order.

        Returns:
            List of DiagnosticDecision node dicts.
        """
        cypher = """
        MATCH (n:DiagnosticDecision {algorithm_id: 'IHC_ASCO_CAP_2023'})
        RETURN n.id AS id, n.question AS question, n.node_type AS node_type,
               n.node_order AS node_order, n.result AS result, n.category AS category,
               n.if_yes AS if_yes, n.if_no AS if_no
        ORDER BY n.node_order
        """
        with driver.session() as session:
            return [dict(r) for r in session.run(cypher)]

    @tool
    def get_ish_algorithm() -> list[dict]:
        """Return all nodes of the ISH_ASCO_CAP_2023 decision algorithm in order.

        Returns:
            List of DiagnosticDecision node dicts.
        """
        cypher = """
        MATCH (n:DiagnosticDecision {algorithm_id: 'ISH_ASCO_CAP_2023'})
        RETURN n.id AS id, n.question AS question, n.node_type AS node_type,
               n.node_order AS node_order, n.result AS result, n.category AS category
        ORDER BY n.node_order
        """
        with driver.session() as session:
            return [dict(r) for r in session.run(cypher)]

    @tool
    def vector_search_entities(query_text: str, top_k: int = 5) -> list[dict]:
        """Retrieve entity nodes from the KG using vector similarity search.

        Args:
            query_text: Natural language description of the concept to find.
            top_k:      Number of results to return.

        Returns:
            List of entity dicts with id, label, uri, score.
        """
        # Uses Neo4j db.index.vector.queryNodes (Neo4j 5.11+)
        cypher = """
        CALL db.index.vector.queryNodes('entity_embeddings', $top_k, null)
        YIELD node AS entity, score
        RETURN entity.id AS id, entity.label AS label,
               entity.uri AS uri, score
        ORDER BY score DESC
        """
        # NOTE: The full implementation passes the embedding vector.
        # This stub shows the Cypher pattern; embeddings are generated
        # by the pipeline (see vector_indexer.py).
        with driver.session() as session:
            result = session.run(cypher, top_k=top_k)
            return [dict(r) for r in result]

    @tool
    def run_validation_rules(
        ihc_score: str | None = None,
        ish_group: str | None = None,
        fractal_d0: float | None = None,
    ) -> list[dict]:
        """Run the defined clinical validation rules against the provided case data.

        Returns a list of triggered validation issues (may be empty).
        """
        from src.domain.validation_rules import VALIDATION_RULES

        case: dict[str, Any] = {}
        if ihc_score is not None:
            case["ihc_score"] = ihc_score
        if ish_group is not None:
            case["ish_group"] = ish_group
        if fractal_d0 is not None:
            case["fractal_d0"] = fractal_d0

        issues: list[dict] = []
        for rule in VALIDATION_RULES:
            triggered = rule.check(case)
            if triggered:
                issues.append({
                    "rule_id":     rule.rule_id,
                    "severity":    rule.severity.value,
                    "description": rule.description,
                    "triggered":   True,
                })
        return issues

    @tool
    def get_diagnostic_pathway(
        ihc_score: str,
        ish_group: str | None = None,
    ) -> dict:
        """Retrieve the diagnostic pathway for a given IHC score and optional ISH group.

        Returns a dict with nodes, edges, final_category, and applicable guidelines.
        """
        cypher = """
        MATCH path = (score:IHCScore {id: $ihc_id})-[:requiresReflexTest|implies|leadsTo*1..5]->(n)
        RETURN [node in nodes(path) | {id: node.id, label: node.label, type: labels(node)}] AS nodes,
               $ihc_id AS start_score
        LIMIT 1
        """
        with driver.session() as session:
            records = list(session.run(cypher, ihc_id=ihc_score))
            if records:
                return dict(records[0])
        return {"start_score": ihc_score, "nodes": [], "ish_group": ish_group}

    return [
        execute_cypher,
        traverse_decision_tree,
        get_ihc_algorithm,
        get_ish_algorithm,
        vector_search_entities,
        run_validation_rules,
        get_diagnostic_pathway,
    ]
