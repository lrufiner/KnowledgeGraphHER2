"""
IHC and ISH diagnostic algorithm definitions — pure data structures.

These encode the decision tree algorithms from §6.1–6.3 of her2_kg_pipeline_guide.md
and ASCO/CAP 2023 guidelines as explicit tree structures that can be loaded into Neo4j
as DiagnosticDecision nodes.

Each algorithm is a list of node dicts with:
    id:        Unique node ID within the algorithm
    question:  Clinical decision text
    type:      "entry" | "decision" | "result" | "action_required"
    algorithm_id: Which algorithm this node belongs to
    if_yes:    (optional) next node ID or {result, category, action}
    if_no:     (optional) next node ID or {result, category, action}
    result:    (for result nodes) IHCScore or ClinicalCategory ID
    category:  (for result nodes) ClinicalCategory ID
    action:    (for action_required nodes) text description
    node_order: int — display order in the algorithm
    guideline_source: guideline ID
"""
from __future__ import annotations

from typing import Any

# ── IHC Algorithm (§6.1, ASCO/CAP 2023) ─────────────────────────────────────

IHC_ALGORITHM_ID = "IHC_ASCO_CAP_2023"

IHC_ALGORITHM_NODES: list[dict[str, Any]] = [
    {
        "id":           "IHC_ENTRY",
        "type":         "entry",
        "question":     "Begin IHC scoring: assay performed on invasive tumor component with appropriate controls",
        "algorithm_id": IHC_ALGORITHM_ID,
        "node_order":   0,
        "guideline_source": "ASCO_CAP_2023",
        "next":         "IHC_NODE1",
    },
    {
        "id":           "IHC_NODE1",
        "type":         "decision",
        "question":     "Is there complete, intense circumferential membrane staining (strong) in >10% of tumor cells?",
        "algorithm_id": IHC_ALGORITHM_ID,
        "node_order":   1,
        "guideline_source": "ASCO_CAP_2023",
        "if_yes":       {"result": "Score3Plus", "category": "HER2_Positive", "label": "IHC 3+"},
        "if_no":        "IHC_NODE2",
    },
    {
        "id":           "IHC_NODE2",
        "type":         "decision",
        "question":     "Is there complete, weak-to-moderate circumferential membrane staining in >10% of tumor cells?",
        "algorithm_id": IHC_ALGORITHM_ID,
        "node_order":   2,
        "guideline_source": "ASCO_CAP_2023",
        "if_yes":       {
            "result": "Score2Plus",
            "category": "HER2_Equivocal",
            "label": "IHC 2+",
            "action": "Order reflex ISH testing (mandatory)",
        },
        "if_no":        "IHC_NODE3",
    },
    {
        "id":           "IHC_NODE3",
        "type":         "decision",
        "question":     "Is there incomplete and/or weak/faint membrane staining in >10% of tumor cells?",
        "algorithm_id": IHC_ALGORITHM_ID,
        "node_order":   3,
        "guideline_source": "ASCO_CAP_2023",
        "if_yes":       {
            "result": "Score1Plus",
            "category": "HER2_Low",
            "label": "IHC 1+",
            "note": "HER2-Low subcategory of HER2-Negative",
        },
        "if_no":        "IHC_NODE4",
    },
    {
        "id":           "IHC_NODE4",
        "type":         "decision",
        "question":     "Is there faint/barey perceptible incomplete membrane staining in >0% to ≤10% of tumor cells?",
        "algorithm_id": IHC_ALGORITHM_ID,
        "node_order":   4,
        "guideline_source": "Rakha_International_2026",
        "if_yes":       {
            "result": "Score0Plus",
            "category": "HER2_Ultralow",
            "label": "IHC 0+",
            "note": "HER2-Ultralow — new subcategory from Rakha 2026",
        },
        "if_no":        "IHC_NODE5",
    },
    {
        "id":           "IHC_NODE5",
        "type":         "result",
        "question":     "No membrane staining observed in any invasive tumor cells",
        "algorithm_id": IHC_ALGORITHM_ID,
        "node_order":   5,
        "guideline_source": "ASCO_CAP_2023",
        "result":       "Score0",
        "category":     "HER2_Null",
        "label":        "IHC 0",
        "note":         "HER2-Null — not eligible for T-DXd",
    },
]

IHC_ALGORITHM_EDGES: list[dict[str, Any]] = [
    # ENTRY → NODE1
    {"from_id": "IHC_ENTRY",  "to_id": "IHC_NODE1",  "condition": "start"},
    # NODE1
    {"from_id": "IHC_NODE1",  "to_id": "Score3Plus",  "condition": "YES",
     "label": "IHC 3+ → HER2-Positive"},
    {"from_id": "IHC_NODE1",  "to_id": "IHC_NODE2",   "condition": "NO"},
    # NODE2
    {"from_id": "IHC_NODE2",  "to_id": "Score2Plus",  "condition": "YES",
     "label": "IHC 2+ → Equivocal / reflex ISH"},
    {"from_id": "IHC_NODE2",  "to_id": "IHC_NODE3",   "condition": "NO"},
    # NODE3
    {"from_id": "IHC_NODE3",  "to_id": "Score1Plus",  "condition": "YES",
     "label": "IHC 1+ → HER2-Low"},
    {"from_id": "IHC_NODE3",  "to_id": "IHC_NODE4",   "condition": "NO"},
    # NODE4
    {"from_id": "IHC_NODE4",  "to_id": "Score0Plus",  "condition": "YES",
     "label": "IHC 0+ → HER2-Ultralow"},
    {"from_id": "IHC_NODE4",  "to_id": "IHC_NODE5",   "condition": "NO"},
    # NODE5 → terminal
    {"from_id": "IHC_NODE5",  "to_id": "Score0",       "condition": "terminal",
     "label": "IHC 0 → HER2-Null"},
]


# ── ISH Algorithm — Groups 2–4 Workup (§6.2, ASCO/CAP 2023) ─────────────────

ISH_ALGORITHM_ID = "ISH_ASCO_CAP_2023"

ISH_ALGORITHM_NODES: list[dict[str, Any]] = [
    # ─ Group 1 (no workup needed) ─
    {
        "id":           "ISH_GROUP1_RESULT",
        "type":         "result",
        "question":     "ISH Group 1: HER2/CEP17 ratio ≥2.0 AND average HER2 signals ≥4.0/cell",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   10,
        "guideline_source": "ASCO_CAP_2023",
        "result":       "Group1",
        "category":     "HER2_Positive",
        "label":        "ISH Group 1 → Amplified",
    },
    # ─ Group 2 workup ─
    {
        "id":           "ISH_NODE_G2_ENTRY",
        "type":         "entry",
        "question":     "ISH Group 2: ratio ≥2.0 AND average HER2 signals <4.0/cell — IHC workup required",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   20,
        "guideline_source": "ASCO_CAP_2023",
        "next":         "ISH_NODE_G2_IHC_CHECK",
    },
    {
        "id":           "ISH_NODE_G2_IHC_CHECK",
        "type":         "decision",
        "question":     "Concurrent IHC result for Group 2: Is IHC 3+?",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   21,
        "guideline_source": "ASCO_CAP_2023",
        "if_yes":       {"category": "HER2_Positive", "label": "Group2 + IHC 3+ → Positive"},
        "if_no":        "ISH_NODE_G2_IHC2",
    },
    {
        "id":           "ISH_NODE_G2_IHC2",
        "type":         "decision",
        "question":     "Concurrent IHC result for Group 2: Is IHC 2+?",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   22,
        "guideline_source": "ASCO_CAP_2023",
        "if_yes":       {
            "action": "Recount ≥20 additional cells; interpret as Group 1 or 5",
            "note":   "Comment-A required if remains Group 2 after recount",
        },
        "if_no":        "ISH_NODE_G2_IHC_LOW",
    },
    {
        "id":           "ISH_NODE_G2_IHC_LOW",
        "type":         "result",
        "question":     "Concurrent IHC result for Group 2: IHC 0 or 1+",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   23,
        "guideline_source": "ASCO_CAP_2023",
        "category":     "HER2_Negative",
        "label":        "Group2 + IHC 0/1+ → Negative (Comment-A)",
        "note":         "Comment-A: limited evidence for anti-HER2 therapy",
    },
    # ─ Group 3 workup ─
    {
        "id":           "ISH_NODE_G3_ENTRY",
        "type":         "entry",
        "question":     "ISH Group 3: ratio <2.0 AND average HER2 signals ≥6.0/cell — IHC workup required",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   30,
        "guideline_source": "ASCO_CAP_2023",
        "next":         "ISH_NODE_G3_IHC_CHECK",
    },
    {
        "id":           "ISH_NODE_G3_IHC_CHECK",
        "type":         "decision",
        "question":     "Concurrent IHC result for Group 3: Is IHC 3+?",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   31,
        "guideline_source": "ASCO_CAP_2023",
        "if_yes":       {"category": "HER2_Positive", "label": "Group3 + IHC 3+ → Positive"},
        "if_no":        "ISH_NODE_G3_IHC2",
    },
    {
        "id":           "ISH_NODE_G3_IHC2",
        "type":         "decision",
        "question":     "Concurrent IHC result for Group 3: Is IHC 2+?",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   32,
        "guideline_source": "ASCO_CAP_2023",
        "if_yes":       {
            "action": "Recount ≥20 additional cells; interpret as Group 1 or 5",
            "note":   "Comment-A required if remains ambiguous",
        },
        "if_no":        "ISH_NODE_G3_IHC_LOW",
    },
    {
        "id":           "ISH_NODE_G3_IHC_LOW",
        "type":         "result",
        "question":     "Concurrent IHC result for Group 3: IHC 0 or 1+",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   33,
        "guideline_source": "ASCO_CAP_2023",
        "category":     "HER2_Negative",
        "label":        "Group3 + IHC 0/1+ → Negative (Comment-A)",
        "note":         "Comment-A: limited evidence for anti-HER2 therapy",
    },
    # ─ Group 4 workup ─
    {
        "id":           "ISH_NODE_G4_ENTRY",
        "type":         "entry",
        "question":     "ISH Group 4: ratio <2.0 AND average HER2 signals ≥4.0 and <6.0/cell — IHC workup required",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   40,
        "guideline_source": "ASCO_CAP_2023",
        "next":         "ISH_NODE_G4_IHC_CHECK",
    },
    {
        "id":           "ISH_NODE_G4_IHC_CHECK",
        "type":         "decision",
        "question":     "Concurrent IHC result for Group 4: Is IHC 3+?",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   41,
        "guideline_source": "ASCO_CAP_2023",
        "if_yes":       {"category": "HER2_Positive", "label": "Group4 + IHC 3+ → Positive"},
        "if_no":        "ISH_NODE_G4_IHC2",
    },
    {
        "id":           "ISH_NODE_G4_IHC2",
        "type":         "decision",
        "question":     "Concurrent IHC result for Group 4: Is IHC 2+?",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   42,
        "guideline_source": "ASCO_CAP_2023",
        "if_yes":       {
            "action": "Recount ≥20 additional cells; interpret as Group 1 or 5",
            "note":   "Comment-A required if remains ambiguous",
        },
        "if_no":        "ISH_NODE_G4_IHC_LOW",
    },
    {
        "id":           "ISH_NODE_G4_IHC_LOW",
        "type":         "result",
        "question":     "Concurrent IHC result for Group 4: IHC 0 or 1+",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   43,
        "guideline_source": "ASCO_CAP_2023",
        "category":     "HER2_Negative",
        "label":        "Group4 + IHC 0/1+ → Negative",
    },
    # ─ Group 5 (no workup needed) ─
    {
        "id":           "ISH_GROUP5_RESULT",
        "type":         "result",
        "question":     "ISH Group 5: ratio <2.0 AND average HER2 signals <4.0/cell — Not Amplified",
        "algorithm_id": ISH_ALGORITHM_ID,
        "node_order":   50,
        "guideline_source": "ASCO_CAP_2023",
        "result":       "Group5",
        "category":     "HER2_Negative",
        "label":        "ISH Group 5 → Not Amplified",
    },
]


# ── Rakha Scoring Matrix (§6.3, Rakha 2026) ──────────────────────────────────

RAKHA_ALGORITHM_ID = "Rakha_Scoring_Matrix_2026"

RAKHA_MATRIX_NODES: list[dict[str, Any]] = [
    {
        "id":           "RAKHA_ENTRY",
        "type":         "entry",
        "question":     "Rakha 2026 scoring matrix: apply after IHC assay on invasive component",
        "algorithm_id": RAKHA_ALGORITHM_ID,
        "node_order":   0,
        "guideline_source": "Rakha_International_2026",
        "next":         "RAKHA_NODE1",
    },
    {
        "id":           "RAKHA_NODE1",
        "type":         "decision",
        "question":     "Is staining complete circumferential (full ring) AND intense?",
        "algorithm_id": RAKHA_ALGORITHM_ID,
        "node_order":   1,
        "guideline_source": "Rakha_International_2026",
        "if_yes":       "RAKHA_NODE1_YES",
        "if_no":        "RAKHA_NODE2",
    },
    {
        "id":           "RAKHA_NODE1_YES",
        "type":         "decision",
        "question":     "Is positive staining in >10% of tumor cells?",
        "algorithm_id": RAKHA_ALGORITHM_ID,
        "node_order":   11,
        "guideline_source": "Rakha_International_2026",
        "if_yes":       {
            "result": "Score3Plus", "category": "HER2_Positive",
            "label": "3+ (POSITIVE)", "score": 3,
        },
        "if_no":        {
            "result": "Score2Plus", "category": "HER2_Equivocal",
            "label": "2+ (EQUIVOCAL)", "score": 2,
            "action": "Reflex ISH required",
        },
    },
    {
        "id":           "RAKHA_NODE2",
        "type":         "decision",
        "question":     "Is staining complete circumferential AND weak-to-moderate intensity?",
        "algorithm_id": RAKHA_ALGORITHM_ID,
        "node_order":   2,
        "guideline_source": "Rakha_International_2026",
        "if_yes":       "RAKHA_NODE2_YES",
        "if_no":        "RAKHA_NODE3",
    },
    {
        "id":           "RAKHA_NODE2_YES",
        "type":         "decision",
        "question":     "Is positive staining in >10% of tumor cells?",
        "algorithm_id": RAKHA_ALGORITHM_ID,
        "node_order":   21,
        "guideline_source": "Rakha_International_2026",
        "if_yes":       {
            "result": "Score2Plus", "category": "HER2_Equivocal",
            "label": "2+ (EQUIVOCAL)", "score": 2,
            "action": "Reflex ISH required",
        },
        "if_no":        {
            "result": "Score1Plus", "category": "HER2_Low",
            "label": "1+ (NEGATIVE/LOW)", "score": 1,
        },
    },
    {
        "id":           "RAKHA_NODE3",
        "type":         "decision",
        "question":     "Is staining incomplete (partial membrane) AND any intensity in >10% of cells?",
        "algorithm_id": RAKHA_ALGORITHM_ID,
        "node_order":   3,
        "guideline_source": "Rakha_International_2026",
        "if_yes":       {
            "result": "Score1Plus", "category": "HER2_Low",
            "label": "1+ (NEGATIVE/LOW)", "score": 1,
        },
        "if_no":        "RAKHA_NODE4",
    },
    {
        "id":           "RAKHA_NODE4",
        "type":         "decision",
        "question":     "Is there faint/barely perceptible incomplete staining in >0% to ≤10% of cells?",
        "algorithm_id": RAKHA_ALGORITHM_ID,
        "node_order":   4,
        "guideline_source": "Rakha_International_2026",
        "if_yes":       {
            "result": "Score0Plus", "category": "HER2_Ultralow",
            "label": "0+ (NEGATIVE/ULTRALOW)", "score": 0,
            "note": "New Rakha 2026 subcategory: HER2-Ultralow",
        },
        "if_no":        {
            "result": "Score0", "category": "HER2_Null",
            "label": "0 (NEGATIVE/NULL)", "score": 0,
        },
    },
]


# ── Combined registry of all algorithms ──────────────────────────────────────

ALL_ALGORITHMS: dict[str, dict] = {
    IHC_ALGORITHM_ID: {
        "id":          IHC_ALGORITHM_ID,
        "label":       "IHC HER2 Scoring Algorithm (ASCO/CAP 2023)",
        "guideline":   "ASCO_CAP_2023",
        "nodes":       IHC_ALGORITHM_NODES,
        "edges":       IHC_ALGORITHM_EDGES,
        "description": "Stepwise IHC decision tree from §6.1 of her2_kg_pipeline_guide.md",
    },
    ISH_ALGORITHM_ID: {
        "id":          ISH_ALGORITHM_ID,
        "label":       "ISH HER2 Groups Workup Algorithm (ASCO/CAP 2023)",
        "guideline":   "ASCO_CAP_2023",
        "nodes":       ISH_ALGORITHM_NODES,
        "edges":       [],  # edges implicit in if_yes / if_no
        "description": "ISH Groups 1–5 workup algorithm from §6.2 of her2_kg_pipeline_guide.md",
    },
    RAKHA_ALGORITHM_ID: {
        "id":          RAKHA_ALGORITHM_ID,
        "label":       "Rakha 2026 Scoring Matrix",
        "guideline":   "Rakha_International_2026",
        "nodes":       RAKHA_MATRIX_NODES,
        "edges":       [],
        "description": "Stepwise IHC scoring from the Rakha 2026 international consensus §6.3",
    },
}

# Convenience aliases used by the Streamlit dashboard
IHC_ALGORITHM = ALL_ALGORITHMS[IHC_ALGORITHM_ID]
ISH_ALGORITHM = ALL_ALGORITHMS[ISH_ALGORITHM_ID]
