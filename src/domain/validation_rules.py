"""
Clinical validation rules for the HER2 Knowledge Graph.
Each rule is a Cypher query that must return {valid: true} when the KG is consistent.
Derived from ASCO/CAP 2023, CAP 2025, and Rakha 2026.
"""
from src.domain.models import ValidationSeverity

VALIDATION_RULES: list[dict] = [
    # ── CRITICAL rules ────────────────────────────────────────────────────────
    {
        "rule_id":  "IHC3Plus_implies_Positive",
        "severity": ValidationSeverity.CRITICAL,
        "source":   "ASCO/CAP 2023; CAP 2025; Rakha 2026",
        "cypher": """
            MATCH (s:IHCScore {id: 'Score3Plus'})-[:implies]->(c:ClinicalCategory)
            WHERE c.id = 'HER2_Positive'
            RETURN count(*) > 0 AS valid
        """,
        "message_pass": "IHC 3+ correctly implies HER2-Positive",
        "message_fail": "CRITICAL: IHC 3+ does not imply HER2-Positive — core ontological rule violated",
    },
    {
        "rule_id":  "IHC2Plus_requires_ISH",
        "severity": ValidationSeverity.CRITICAL,
        "source":   "ASCO/CAP 2023",
        "cypher": """
            MATCH (s:IHCScore {id: 'Score2Plus'})
            WHERE EXISTS { (s)-[:requiresReflexTest]->() }
            RETURN count(s) > 0 AS valid
        """,
        "message_pass": "IHC 2+ correctly requires reflex ISH",
        "message_fail": "CRITICAL: IHC 2+ has no requiresReflexTest relation",
    },
    {
        "rule_id":  "Positive_disjoint_Negative",
        "severity": ValidationSeverity.CRITICAL,
        "source":   "Ontological axiom (Annex A)",
        "cypher": """
            MATCH (p:ClinicalCategory {id: 'HER2_Positive'}),
                  (n:ClinicalCategory {id: 'HER2_Negative'})
            WHERE NOT EXISTS { (p)-[:refinesCategory]->(n) }
              AND NOT EXISTS { (n)-[:refinesCategory]->(p) }
            RETURN count(*) > 0 AS valid
        """,
        "message_pass": "HER2-Positive and HER2-Negative are correctly disjoint",
        "message_fail": "CRITICAL: HER2-Positive and HER2-Negative are not disjoint",
    },
    {
        "rule_id":  "ISH_Group1_positive",
        "severity": ValidationSeverity.CRITICAL,
        "source":   "ASCO/CAP 2023",
        "cypher": """
            MATCH (g:ISHGroup {id: 'Group1'})-[:implies]->(c:ClinicalCategory)
            WHERE c.id = 'HER2_Positive'
            RETURN count(*) > 0 AS valid
        """,
        "message_pass": "ISH Group 1 correctly implies HER2-Positive",
        "message_fail": "CRITICAL: ISH Group 1 does not imply HER2-Positive",
    },
    {
        "rule_id":  "ISH_Group5_negative",
        "severity": ValidationSeverity.CRITICAL,
        "source":   "ASCO/CAP 2023",
        "cypher": """
            MATCH (g:ISHGroup {id: 'Group5'})-[:implies]->(c:ClinicalCategory)
            WHERE c.id = 'HER2_Negative'
            RETURN count(*) > 0 AS valid
        """,
        "message_pass": "ISH Group 5 correctly implies HER2-Negative",
        "message_fail": "CRITICAL: ISH Group 5 does not imply HER2-Negative",
    },
    # ── HIGH rules ───────────────────────────────────────────────────────────
    {
        "rule_id":  "HER2Low_eligible_TDXd",
        "severity": ValidationSeverity.HIGH,
        "source":   "DESTINY-Breast04 (Modi 2022)",
        "cypher": """
            MATCH (c:ClinicalCategory {id: 'HER2_Low'})-[:eligibleFor]->(t:TherapeuticAgent)
            WHERE t.id = 'TrastuzumabDeruxtecan'
            RETURN count(*) > 0 AS valid
        """,
        "message_pass": "HER2-Low is correctly eligible for T-DXd",
        "message_fail": "HIGH: HER2-Low not linked as eligible for TrastuzumabDeruxtecan",
    },
    {
        "rule_id":  "HER2Ultralow_eligible_TDXd",
        "severity": ValidationSeverity.HIGH,
        "source":   "DESTINY-Breast06 (Bardia 2024)",
        "cypher": """
            MATCH (c:ClinicalCategory {id: 'HER2_Ultralow'})-[:eligibleFor]->(t:TherapeuticAgent)
            WHERE t.id = 'TrastuzumabDeruxtecan'
            RETURN count(*) > 0 AS valid
        """,
        "message_pass": "HER2-Ultralow is correctly eligible for T-DXd (HR+ setting)",
        "message_fail": "HIGH: HER2-Ultralow not linked as eligible for TrastuzumabDeruxtecan",
    },
    {
        "rule_id":  "HER2Null_not_eligible_TDXd",
        "severity": ValidationSeverity.HIGH,
        "source":   "Current evidence (2026)",
        "cypher": """
            MATCH (c:ClinicalCategory {id: 'HER2_Null'})-[:notEligibleFor]->(t:TherapeuticAgent)
            WHERE t.id = 'TrastuzumabDeruxtecan'
            RETURN count(*) > 0 AS valid
        """,
        "message_pass": "HER2-Null is correctly not eligible for T-DXd",
        "message_fail": "HIGH: HER2-Null not marked as notEligibleFor TrastuzumabDeruxtecan",
    },
    {
        "rule_id":  "Fractal_marked_as_hypothesis",
        "severity": ValidationSeverity.HIGH,
        "source":   "Design constraint — fractal equivalences are exploratory",
        "cypher": """
            OPTIONAL MATCH (fm:FractalMetric)-[r:proposedEquivalence]->(:ClinicalCategory)
            WHERE r.is_hypothesis <> true OR r.is_hypothesis IS NULL
            RETURN count(r) = 0 AS valid
        """,
        "message_pass": "All fractal proposedEquivalence relations correctly marked as hypotheses",
        "message_fail": "HIGH: Some fractal proposedEquivalence relations are missing the is_hypothesis flag",
    },
    {
        "rule_id":  "All_5_categories_present",
        "severity": ValidationSeverity.HIGH,
        "source":   "Ontology completeness",
        "cypher": """
            MATCH (c:ClinicalCategory)
            WHERE c.id IN ['HER2_Positive','HER2_Negative','HER2_Low','HER2_Ultralow','HER2_Null','HER2_Equivocal']
            RETURN count(DISTINCT c.id) = 6 AS valid
        """,
        "message_pass": "All 6 HER2 clinical categories are present",
        "message_fail": "HIGH: One or more HER2 clinical categories are missing from the graph",
    },
    {
        "rule_id":  "All_5_ISH_groups_present",
        "severity": ValidationSeverity.HIGH,
        "source":   "ASCO/CAP 2023 — ISH 5-group system",
        "cypher": """
            MATCH (g:ISHGroup)
            WHERE g.id IN ['Group1','Group2','Group3','Group4','Group5']
            RETURN count(DISTINCT g.id) = 5 AS valid
        """,
        "message_pass": "All 5 ISH groups are present",
        "message_fail": "HIGH: One or more ISH groups are missing from the graph",
    },
    {
        "rule_id":  "Equivocal_has_no_direct_therapy",
        "severity": ValidationSeverity.MEDIUM,
        "source":   "Clinical logic — equivocal status requires ISH before therapy decision",
        "cypher": """
            OPTIONAL MATCH (c:ClinicalCategory {id: 'HER2_Equivocal'})-[:eligibleFor]->(t:TherapeuticAgent)
            RETURN count(t) = 0 AS valid
        """,
        "message_pass": "HER2-Equivocal correctly has no direct therapy eligibility",
        "message_fail": "MEDIUM: HER2-Equivocal is linked to therapy eligibility without ISH confirmation",
    },
    {
        "rule_id":  "No_orphan_entities",
        "severity": ValidationSeverity.MEDIUM,
        "source":   "Graph quality",
        "cypher": """
            MATCH (n)
            WHERE n:ClinicalCategory OR n:IHCScore OR n:ISHGroup
            WITH n
            WHERE NOT EXISTS { (n)-[]-() }
            RETURN count(n) = 0 AS valid
        """,
        "message_pass": "No orphan clinical/IHC/ISH nodes found",
        "message_fail": "MEDIUM: Some ClinicalCategory, IHCScore or ISHGroup nodes have no relations",
    },
]
