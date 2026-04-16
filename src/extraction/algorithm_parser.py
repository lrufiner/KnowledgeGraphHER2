"""
Algorithm parser — loads IHC/ISH decision tree structures into Neo4j.

Converts the data structures in algorithm_definitions.py into
DiagnosticDecision nodes and leadsTo edges in the Neo4j KG.

Each algorithm node becomes:
    (:DiagnosticDecision {id, question, type, algorithm_id, node_order, ...})

Edges between decision nodes:
    (a)-[:leadsTo {condition: "YES"|"NO"|"start", label}]->(b)

Result/outcome links:
    (decision_node)-[:leadsTo {condition}]->(IHCScore or ClinicalCategory)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j import Driver

from src.domain.algorithm_definitions import ALL_ALGORITHMS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UPSERT_DECISION_NODE = """
MERGE (n:DiagnosticDecision {id: $id})
SET n.question        = $question,
    n.node_type       = $node_type,
    n.algorithm_id    = $algorithm_id,
    n.node_order      = $node_order,
    n.guideline_source = $guideline_source,
    n.label           = $label,
    n.result          = $result,
    n.category        = $category,
    n.action          = $action,
    n.note            = $note,
    n.source_doc      = 'algorithm:' + $algorithm_id
RETURN n.id AS id
"""

_UPSERT_LEADS_TO = """
MATCH (a {id: $from_id})
MATCH (b {id: $to_id})
MERGE (a)-[r:leadsTo {condition: $condition, algorithm_id: $algorithm_id}]->(b)
SET r.label = $label
"""


def _node_to_props(node: dict[str, Any]) -> dict[str, Any]:
    """Flatten a node dict to Neo4j properties."""
    def _str(v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, dict):
            return str(v)
        return str(v)

    return {
        "id":              node["id"],
        "question":        node.get("question", ""),
        "node_type":       node.get("type", "decision"),
        "algorithm_id":    node.get("algorithm_id", ""),
        "node_order":      node.get("node_order", 0),
        "guideline_source": node.get("guideline_source", ""),
        "label":           node.get("label", node["id"]),
        "result":          node.get("result"),
        "category":        node.get("category"),
        "action":          _str(node.get("action")),
        "note":            node.get("note"),
    }


def _resolve_branch(
    branch: str | dict | None,
    algorithm_id: str,
) -> tuple[str | None, str | None, str | None]:
    """
    Resolve a branch value to (to_id, condition_label, note).

    Branch can be:
      - str  → directly the next node ID
      - dict → {result, category, label, action, ...} (terminal/result)
      - None → nothing
    """
    if branch is None:
        return None, None, None
    if isinstance(branch, str):
        return branch, None, None
    # Dict: extract the target IHCScore or ClinicalCategory node
    to_id = branch.get("result") or branch.get("category")
    label = branch.get("label")
    return to_id, label, None


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_and_load_algorithm(
    driver: Driver,
    algorithm_id: str,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Parse one algorithm from ALL_ALGORITHMS and load it into Neo4j.

    Returns:
        stats dict with 'nodes' and 'edges' counts.
    """
    algo = ALL_ALGORITHMS.get(algorithm_id)
    if algo is None:
        raise ValueError(f"Algorithm '{algorithm_id}' not found in ALL_ALGORITHMS")

    nodes = algo["nodes"]
    stats = {"nodes": 0, "edges": 0}

    with driver.session() as session:
        # 1. Upsert all DiagnosticDecision nodes
        for node in nodes:
            props = _node_to_props(node)
            session.run(_UPSERT_DECISION_NODE, **props)
            stats["nodes"] += 1

        # 2. Create leadsTo edges from explicit edge list (IHC algorithm)
        for edge in algo.get("edges", []):
            from_id   = edge["from_id"]
            to_id     = edge["to_id"]
            condition = edge.get("condition", "")
            label     = edge.get("label", "")
            session.run(
                _UPSERT_LEADS_TO,
                from_id=from_id, to_id=to_id,
                condition=condition, label=label,
                algorithm_id=algorithm_id,
            )
            stats["edges"] += 1

        # 3. Create leadsTo edges from if_yes / if_no / next fields
        for node in nodes:
            node_id = node["id"]

            # next → simple sequential edge
            if "next" in node:
                session.run(
                    _UPSERT_LEADS_TO,
                    from_id=node_id, to_id=node["next"],
                    condition="start", label="",
                    algorithm_id=algorithm_id,
                )
                stats["edges"] += 1

            for branch_key, condition_label in [("if_yes", "YES"), ("if_no", "NO")]:
                branch = node.get(branch_key)
                if branch is None:
                    continue
                to_id, edge_label, _ = _resolve_branch(branch, algorithm_id)
                if to_id:
                    try:
                        session.run(
                            _UPSERT_LEADS_TO,
                            from_id=node_id, to_id=to_id,
                            condition=condition_label,
                            label=edge_label or "",
                            algorithm_id=algorithm_id,
                        )
                        stats["edges"] += 1
                    except Exception:
                        # Target node may not exist (e.g. ClinicalCategory — already exists as seed)
                        pass  # Already seeded; MATCH will succeed at query time

    if verbose:
        print(f"  [{algorithm_id}] {stats['nodes']} decision nodes, {stats['edges']} edges")

    return stats


def parse_and_load_all_algorithms(
    driver: Driver,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Parse and load all algorithms in ALL_ALGORITHMS into Neo4j.

    Returns aggregate stats.
    """
    total = {"nodes": 0, "edges": 0}
    if verbose:
        print(f"[AlgorithmParser] Loading {len(ALL_ALGORITHMS)} algorithms into Neo4j...")

    for algo_id in ALL_ALGORITHMS:
        stats = parse_and_load_algorithm(driver, algo_id, verbose=verbose)
        total["nodes"] += stats["nodes"]
        total["edges"] += stats["edges"]

    if verbose:
        print(f"[AlgorithmParser] Done: {total['nodes']} nodes, {total['edges']} edges total")

    return total
