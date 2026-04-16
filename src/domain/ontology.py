"""
HER2 Ontology constants — namespaces, canonical URIs, and class hierarchy.
Derived from Annex A (annex_ontology.md).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# RDF Namespaces
# ---------------------------------------------------------------------------

NAMESPACES = {
    "her2":   "http://digpatho.sinc.unl.edu.ar/ontology/her2#",
    "frac":   "http://digpatho.sinc.unl.edu.ar/ontology/fractal#",
    "ncit":   "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#",
    "snomed": "http://snomed.info/id/",
    "loinc":  "http://loinc.org/rdf#",
    "owl":    "http://www.w3.org/2002/07/owl#",
    "rdfs":   "http://www.w3.org/2000/01/rdf-schema#",
    "xsd":    "http://www.w3.org/2001/XMLSchema#",
}

HER2_BASE  = NAMESPACES["her2"]
FRAC_BASE  = NAMESPACES["frac"]
NCIT_BASE  = NAMESPACES["ncit"]
SNOMED_BASE = NAMESPACES["snomed"]

# ---------------------------------------------------------------------------
# Canonical URI lookup table (curated from Annex A + her2_kg_pipeline_guide.md)
# NOTE: NCIt codes are estimates — verify against ncit.nci.nih.gov before publication.
# ---------------------------------------------------------------------------

CANONICAL_URIS: dict[str, str] = {
    # ── IHC Scores ──────────────────────────────────────────────────────────
    "HER2 Score 0":    "NCIt:C173786",
    "HER2 Score 0+":   "NCIt:C173787",
    "HER2 Score 1+":   "NCIt:C173788",
    "HER2 Score 2+":   "NCIt:C173789",
    "HER2 Score 3+":   "NCIt:C173790",
    "Score0":          "NCIt:C173786",
    "Score0Plus":      "NCIt:C173787",
    "Score1Plus":      "NCIt:C173788",
    "Score2Plus":      "NCIt:C173789",
    "Score3Plus":      "NCIt:C173790",
    "IHC 0":           "NCIt:C173786",
    "IHC 0+":          "NCIt:C173787",
    "IHC 1+":          "NCIt:C173788",
    "IHC 2+":          "NCIt:C173789",
    "IHC 3+":          "NCIt:C173790",

    # ── Clinical Categories ──────────────────────────────────────────────────
    "HER2-Positive":  "NCIt:C68748",
    "HER2-Negative":  "NCIt:C68749",
    "HER2-Low":       "NCIt:C173791",
    "HER2-Ultralow":  "NCIt:C173792",
    "HER2-Null":      "NCIt:C173793",
    "HER2-Equivocal": "NCIt:C173794",
    "HER2 Positive":  "NCIt:C68748",
    "HER2 Negative":  "NCIt:C68749",
    "HER2 Low":       "NCIt:C173791",
    "HER2 Ultralow":  "NCIt:C173792",
    "HER2 Null":      "NCIt:C173793",
    "HER2 Equivocal": "NCIt:C173794",

    # ── ISH Groups ───────────────────────────────────────────────────────────
    "ISH Group 1": "NCIt:C173795",
    "ISH Group 2": "NCIt:C173796",
    "ISH Group 3": "NCIt:C173797",
    "ISH Group 4": "NCIt:C173798",
    "ISH Group 5": "NCIt:C173799",
    "Group1": "NCIt:C173795",
    "Group2": "NCIt:C173796",
    "Group3": "NCIt:C173797",
    "Group4": "NCIt:C173798",
    "Group5": "NCIt:C173799",

    # ── Therapies ─────────────────────────────────────────────────────────────
    "Trastuzumab":           "NCIt:C1647",
    "Trastuzumab Deruxtecan": "NCIt:C155379",
    "T-DXd":                 "NCIt:C155379",
    "Pertuzumab":            "NCIt:C64636",
    "T-DM1":                 "NCIt:C82492",
    "Ado-Trastuzumab Emtansine": "NCIt:C82492",

    # ── Fractal Metrics ───────────────────────────────────────────────────────
    "Fractal Dimension D0":      "NCIt:C25730",
    "Fractal Dimension D1":      "NCIt:C25731",
    "Capacity Dimension D0":     "NCIt:C25730",
    "Information Dimension D1":  "NCIt:C25731",
    "Lacunarity":                "NCIt:C173800",
    "Multifractal Spectrum":     "NCIt:C173801",
    "Multifractal Spread":       "NCIt:C173801",
    "Multiscale Entropy":        "NCIt:C173802",
    "D0": "NCIt:C25730",
    "D1": "NCIt:C25731",

    # ── Pathological Features ─────────────────────────────────────────────────
    "Architectural Complexity":    "NCIt:C19754",
    "Tumor Heterogeneity":         "NCIt:C16947",
    "Intratumoral Heterogeneity":  "NCIt:C16947",
    "HER2 Gene Amplification":     "NCIt:C116178",
    "HER2 Protein Overexpression": "NCIt:C44573",
    "Spatial Heterogeneity":       "NCIt:C16947",

    # ── Biomarkers ────────────────────────────────────────────────────────────
    "ERBB2":  "NCIt:C17382",
    "HER2":   "NCIt:C17382",
    "CEP17":  "NCIt:C173803",
    "ER":     "NCIt:C17687",
    "PgR":    "NCIt:C17043",
    "Ki67":   "NCIt:C17203",

    # ── Assays ────────────────────────────────────────────────────────────────
    "Immunohistochemistry":          "NCIt:C17089",
    "IHC":                           "NCIt:C17089",
    "In Situ Hybridization":         "NCIt:C17236",
    "ISH":                           "NCIt:C17236",
    "Fluorescence In Situ Hybridization": "NCIt:C17236",
    "FISH":                          "NCIt:C17236",
    "Chromogenic In Situ Hybridization": "NCIt:C173804",
    "CISH":                          "NCIt:C173804",
    "VENTANA HER2 (4B5)":            "NCIt:C173805",
    "HercepTest":                    "NCIt:C173806",

    # ── Guidelines ───────────────────────────────────────────────────────────
    "ASCO_CAP_2023":           "her2:ASCO_CAP_2023",
    "ASCO/CAP 2023":           "her2:ASCO_CAP_2023",
    "CAP_2025":                "her2:CAP_Biomarker_2025",
    "CAP 2025":                "her2:CAP_Biomarker_2025",
    "ESMO_2023":               "her2:ESMO_2023",
    "ESMO 2023":               "her2:ESMO_2023",
    "Rakha_2026":              "her2:Rakha_International_2026",
    "Rakha 2026":              "her2:Rakha_International_2026",

    # ── Clinical Trials ───────────────────────────────────────────────────────
    "DESTINY-Breast04":  "her2:DESTINY_Breast04",
    "DB-04":             "her2:DESTINY_Breast04",
    "DESTINY-Breast06":  "her2:DESTINY_Breast06",
    "DB-06":             "her2:DESTINY_Breast06",
    "DAISY":             "her2:DAISY_Trial",
}

SNOMED_URIS: dict[str, str] = {
    "Invasive Breast Carcinoma": "snomed:413448000",
    "Ductal Carcinoma In Situ":  "snomed:399935008",
    "Metastatic Breast Cancer":  "snomed:408643008",
    "HER2 Protein":              "snomed:442478000",
    "Immunohistochemistry":      "snomed:117299004",
    "In Situ Hybridization":     "snomed:404217000",
}

# ---------------------------------------------------------------------------
# Pre-defined seed entities (from Annex A class hierarchy)
# Used in BUILD phase to ensure core nodes always exist.
# ---------------------------------------------------------------------------

SEED_ENTITIES: list[dict] = [
    # IHC Scores
    {"id": "Score0",    "label": "IHC Score 0",  "type": "IHCScore",
     "definition": "No membrane staining in tumor cells (HER2-Null)",
     "ncit_uri": "NCIt:C173786"},
    {"id": "Score0Plus","label": "IHC Score 0+", "type": "IHCScore",
     "definition": "Faint/barely perceptible incomplete membrane staining in >0% to ≤10% of tumor cells (HER2-Ultralow)",
     "ncit_uri": "NCIt:C173787"},
    {"id": "Score1Plus","label": "IHC Score 1+", "type": "IHCScore",
     "definition": "Incomplete, faint/barely perceptible membrane staining in >10% of tumor cells (HER2-Low)",
     "ncit_uri": "NCIt:C173788"},
    {"id": "Score2Plus","label": "IHC Score 2+", "type": "IHCScore",
     "definition": "Complete membrane staining of weak-to-moderate intensity in >10% of tumor cells (Equivocal — requires ISH)",
     "ncit_uri": "NCIt:C173789"},
    {"id": "Score3Plus","label": "IHC Score 3+", "type": "IHCScore",
     "definition": "Complete, intense circumferential membrane staining in >10% of tumor cells (HER2-Positive)",
     "ncit_uri": "NCIt:C173790"},
    # Clinical Categories
    {"id": "HER2_Positive",  "label": "HER2-Positive",  "type": "ClinicalCategory",
     "definition": "IHC 3+ or ISH-amplified (Group 1 or workup-confirmed)", "ncit_uri": "NCIt:C68748"},
    {"id": "HER2_Negative",  "label": "HER2-Negative",  "type": "ClinicalCategory",
     "definition": "IHC 0, 0+, or 1+ OR IHC 2+/ISH non-amplified", "ncit_uri": "NCIt:C68749"},
    {"id": "HER2_Low",       "label": "HER2-Low",       "type": "ClinicalCategory",
     "definition": "IHC 1+ OR IHC 2+/ISH non-amplified; eligible for T-DXd",
     "ncit_uri": "NCIt:C173791", "interobserver_variability": 0.25},
    {"id": "HER2_Ultralow",  "label": "HER2-Ultralow",  "type": "ClinicalCategory",
     "definition": "IHC 0+ (faint/incomplete ≤10% cells); eligible for T-DXd (HR+)",
     "ncit_uri": "NCIt:C173792", "interobserver_variability": 0.37},
    {"id": "HER2_Null",      "label": "HER2-Null",      "type": "ClinicalCategory",
     "definition": "IHC 0 / no membrane staining; not eligible for T-DXd",
     "ncit_uri": "NCIt:C173793"},
    {"id": "HER2_Equivocal", "label": "HER2-Equivocal", "type": "ClinicalCategory",
     "definition": "IHC 2+ — requires ISH reflex testing for definitive status",
     "ncit_uri": "NCIt:C173794"},
    # ISH Groups
    {"id": "Group1","label": "ISH Group 1","type": "ISHGroup",
     "definition": "ratio ≥2.0 AND signals ≥4.0/cell → Amplified (Positive)",
     "ncit_uri": "NCIt:C173795",
     "ratio_threshold": 2.0, "signals_per_cell_threshold": 4.0, "group_number": 1},
    {"id": "Group2","label": "ISH Group 2","type": "ISHGroup",
     "definition": "ratio ≥2.0 AND signals <4.0/cell → Requires IHC workup",
     "ncit_uri": "NCIt:C173796",
     "ratio_threshold": 2.0, "signals_per_cell_threshold": 4.0, "group_number": 2},
    {"id": "Group3","label": "ISH Group 3","type": "ISHGroup",
     "definition": "ratio <2.0 AND signals ≥6.0/cell → Requires IHC workup",
     "ncit_uri": "NCIt:C173797",
     "ratio_threshold": 2.0, "signals_per_cell_threshold": 6.0, "group_number": 3},
    {"id": "Group4","label": "ISH Group 4","type": "ISHGroup",
     "definition": "ratio <2.0 AND signals ≥4.0 and <6.0/cell → Requires IHC workup",
     "ncit_uri": "NCIt:C173798",
     "ratio_threshold": 2.0, "signals_per_cell_threshold": 4.0, "group_number": 4},
    {"id": "Group5","label": "ISH Group 5","type": "ISHGroup",
     "definition": "ratio <2.0 AND signals <4.0/cell → Not amplified (Negative)",
     "ncit_uri": "NCIt:C173799",
     "ratio_threshold": 2.0, "signals_per_cell_threshold": 4.0, "group_number": 5},
    # Therapeutic Agents
    {"id": "Trastuzumab",          "label": "Trastuzumab",          "type": "TherapeuticAgent",
     "definition": "Anti-HER2 monoclonal antibody", "ncit_uri": "NCIt:C1647"},
    {"id": "TrastuzumabDeruxtecan","label": "Trastuzumab Deruxtecan (T-DXd)","type": "TherapeuticAgent",
     "definition": "Anti-HER2 antibody-drug conjugate with topoisomerase I inhibitor payload",
     "ncit_uri": "NCIt:C155379", "fda_approved": True, "approval_date": "2025"},
    {"id": "Pertuzumab","label": "Pertuzumab","type": "TherapeuticAgent",
     "definition": "Anti-HER2 monoclonal antibody targeting domain II", "ncit_uri": "NCIt:C64636"},
    {"id": "TDM1","label": "T-DM1 (Ado-Trastuzumab Emtansine)","type": "TherapeuticAgent",
     "definition": "Anti-HER2 antibody-drug conjugate with microtubule inhibitor payload",
     "ncit_uri": "NCIt:C82492"},
    # Clinical Trials
    {"id": "DESTINY_Breast04","label": "DESTINY-Breast04","type": "ClinicalTrial",
     "definition": "Phase 3 RCT demonstrating T-DXd efficacy in HER2-low metastatic breast cancer"},
    {"id": "DESTINY_Breast06","label": "DESTINY-Breast06","type": "ClinicalTrial",
     "definition": "Phase 3 RCT extending T-DXd benefit to HER2-ultralow in HR+ metastatic breast cancer"},
    {"id": "DAISY_Trial","label": "DAISY Trial","type": "ClinicalTrial",
     "definition": "Phase 2 trial showing T-DXd activity across HER2 expression spectrum including HER2-null"},
    # Guidelines
    {"id": "ASCO_CAP_2023","label": "ASCO/CAP HER2 Guideline 2023","type": "Guideline",
     "definition": "Wolff et al. 2023 — Updates IHC/ISH criteria for HER2-low/ultralow"},
    {"id": "CAP_Biomarker_2025","label": "CAP Biomarker Template 2025","type": "Guideline",
     "definition": "CAP reporting template v1.6.0.0 (March 2025) for breast biomarkers"},
    {"id": "ESMO_2023","label": "ESMO HER2-Low Expert Consensus 2023","type": "Guideline",
     "definition": "Tarantino et al. 2023 — ESMO expert consensus on HER2-low definition and management"},
    {"id": "Rakha_International_2026","label": "Rakha International Consensus 2026","type": "Guideline",
     "definition": "Rakha et al. 2026 — International expert consensus on HER2 reporting including ultralow"},
    # Fractal Metrics
    {"id": "FractalDimension_D0","label": "Fractal Dimension D0 (Capacity)","type": "FractalMetric",
     "definition": "Box-counting fractal dimension; high values indicate complex glandular architecture",
     "ncit_uri": "NCIt:C25730",
     "value_range_low": 1.0, "value_range_high": 2.0},
    {"id": "FractalDimension_D1","label": "Fractal Dimension D1 (Information)","type": "FractalMetric",
     "definition": "Information dimension; high values indicate heterogeneous information distribution",
     "ncit_uri": "NCIt:C25731",
     "value_range_low": 1.0, "value_range_high": 2.0},
    {"id": "Lacunarity","label": "Lacunarity","type": "FractalMetric",
     "definition": "Measure of spatial texture; low values = dense, high values = porous",
     "ncit_uri": "NCIt:C173800",
     "value_range_low": 0.0, "value_range_high": 1.0},
    {"id": "MultifractalSpread","label": "Multifractal Spread (Δα)","type": "FractalMetric",
     "definition": "Width of the multifractal spectrum; high values indicate high spatial variability",
     "ncit_uri": "NCIt:C173801",
     "value_range_low": 0.0, "value_range_high": 2.0},
    {"id": "MultiscaleEntropy","label": "Multiscale Entropy","type": "FractalMetric",
     "definition": "Temporal complexity measure; high values indicate complex dynamics",
     "value_range_low": 0.0, "value_range_high": 3.0},
]

# ---------------------------------------------------------------------------
# Seed relations (pre-defined from Annex A + guidelines)
# ---------------------------------------------------------------------------

SEED_RELATIONS: list[dict] = [
    # IHC → ClinicalCategory (direct implications)
    {"subject_id": "Score3Plus", "predicate": "implies",
     "object_id": "HER2_Positive", "confidence": 1.0,
     "evidence": "ASCO/CAP 2023; CAP 2025; Rakha 2026 — all guidelines",
     "guideline_version": "ASCO_CAP_2023"},
    {"subject_id": "Score2Plus", "predicate": "implies",
     "object_id": "HER2_Equivocal", "confidence": 1.0,
     "evidence": "Requires ISH reflex testing",
     "guideline_version": "ASCO_CAP_2023"},
    {"subject_id": "Score1Plus", "predicate": "implies",
     "object_id": "HER2_Negative", "confidence": 1.0,
     "evidence": "ASCO/CAP 2023", "guideline_version": "ASCO_CAP_2023"},
    {"subject_id": "Score1Plus", "predicate": "refinesCategory",
     "object_id": "HER2_Low", "confidence": 1.0,
     "evidence": "HER2-Low subcategory", "guideline_version": "ASCO_CAP_2023"},
    {"subject_id": "Score0Plus", "predicate": "implies",
     "object_id": "HER2_Negative", "confidence": 1.0,
     "evidence": "ASCO/CAP 2023; CAP 2025", "guideline_version": "CAP_Biomarker_2025"},
    {"subject_id": "Score0Plus", "predicate": "refinesCategory",
     "object_id": "HER2_Ultralow", "confidence": 1.0,
     "evidence": "HER2-Ultralow subcategory (Rakha 2026)",
     "guideline_version": "Rakha_International_2026"},
    {"subject_id": "Score0", "predicate": "implies",
     "object_id": "HER2_Null", "confidence": 1.0,
     "evidence": "No membrane staining", "guideline_version": "ASCO_CAP_2023"},
    # IHC 2+ requires ISH reflex
    {"subject_id": "Score2Plus", "predicate": "requiresReflexTest",
     "object_id": "ISH", "confidence": 1.0,
     "evidence": "ASCO/CAP 2023 — mandatory reflex ISH for IHC 2+",
     "guideline_version": "ASCO_CAP_2023"},
    # ISH Group 1 → Positive
    {"subject_id": "Group1", "predicate": "implies",
     "object_id": "HER2_Positive", "confidence": 1.0,
     "evidence": "ratio ≥2.0 AND signals ≥4.0 → amplified",
     "guideline_version": "ASCO_CAP_2023"},
    # ISH Group 5 → Negative
    {"subject_id": "Group5", "predicate": "implies",
     "object_id": "HER2_Negative", "confidence": 1.0,
     "evidence": "ratio <2.0 AND signals <4.0 → not amplified",
     "guideline_version": "ASCO_CAP_2023"},
    # Therapeutic eligibility
    {"subject_id": "HER2_Low", "predicate": "eligibleFor",
     "object_id": "TrastuzumabDeruxtecan", "confidence": 1.0,
     "evidence": "DESTINY-Breast04 (Modi 2022)", "conditions": "metastatic setting"},
    {"subject_id": "HER2_Ultralow", "predicate": "eligibleFor",
     "object_id": "TrastuzumabDeruxtecan", "confidence": 1.0,
     "evidence": "DESTINY-Breast06 (Bardia 2024)", "conditions": "HR+, metastatic setting"},
    {"subject_id": "HER2_Positive", "predicate": "notEligibleFor",
     "object_id": "TrastuzumabDeruxtecan", "confidence": 1.0,
     "evidence": "Standard anti-HER2 therapies indicated instead"},
    {"subject_id": "HER2_Null", "predicate": "notEligibleFor",
     "object_id": "TrastuzumabDeruxtecan", "confidence": 1.0,
     "evidence": "Current evidence 2026; no T-DXd benefit shown"},
    # HER2-Low/Ultralow/Null are subtypes of HER2-Negative
    {"subject_id": "HER2_Low",      "predicate": "refinesCategory",
     "object_id": "HER2_Negative", "confidence": 1.0,
     "evidence": "Ontological axiom (Annex A)"},
    {"subject_id": "HER2_Ultralow", "predicate": "refinesCategory",
     "object_id": "HER2_Negative", "confidence": 1.0,
     "evidence": "Ontological axiom (Annex A)"},
    {"subject_id": "HER2_Null",     "predicate": "refinesCategory",
     "object_id": "HER2_Negative", "confidence": 1.0,
     "evidence": "Ontological axiom (Annex A)"},
    # Guideline supersedings
    {"subject_id": "ASCO_CAP_2023", "predicate": "SUPERSEDED_BY",
     "object_id": "Rakha_International_2026", "confidence": 0.8,
     "evidence": "Rakha 2026 extends ASCO/CAP 2023 with ultralow categories",
     "conditions": "HER2-low/ultralow scope only"},
    # Evidence support
    {"subject_id": "HER2_Low", "predicate": "supportedByEvidence",
     "object_id": "DESTINY_Breast04", "confidence": 1.0,
     "evidence": "Phase 3 RCT; primary evidence source"},
    {"subject_id": "HER2_Ultralow", "predicate": "supportedByEvidence",
     "object_id": "DESTINY_Breast06", "confidence": 1.0,
     "evidence": "Phase 3 RCT; primary evidence source"},
    # Fractal–clinical proposed equivalences (HYPOTHESIS — marked as is_hypothesis=True)
    {"subject_id": "FractalDimension_D0", "predicate": "proposedEquivalence",
     "object_id": "HER2_Positive", "confidence": 0.6,
     "evidence": "D0 >1.85 associated with complex architecture (DigPatho Internal 2025)",
     "is_hypothesis": True, "conditions": "D0 > 1.85"},
    {"subject_id": "FractalDimension_D0", "predicate": "proposedEquivalence",
     "object_id": "HER2_Null", "confidence": 0.6,
     "evidence": "D0 <1.45 associated with minimal architecture (DigPatho Internal 2025)",
     "is_hypothesis": True, "conditions": "D0 < 1.45"},
    {"subject_id": "Lacunarity", "predicate": "proposedEquivalence",
     "object_id": "Score3Plus", "confidence": 0.55,
     "evidence": "Low lacunarity (<0.15) associated with dense IHC 3+ pattern",
     "is_hypothesis": True, "conditions": "Lacunarity < 0.15"},
    {"subject_id": "MultifractalSpread", "predicate": "inconsistentWith",
     "object_id": "HER2_Null", "confidence": 0.65,
     "evidence": "High Δα (>0.80) implies spatial heterogeneity, unlikely in HER2-null",
     "is_hypothesis": True, "conditions": "Δα > 0.80"},
]

# ---------------------------------------------------------------------------
# Toy / artificial fractal examples for system testing
# (DigPatho_Internal_2025 — purely synthetic data)
# ---------------------------------------------------------------------------

TOY_FRACTAL_SPECIMENS: list[dict] = [
    {"specimen_id": "TOY_001", "ihc_score": "Score3Plus",
     "D0": 1.91, "D1": 1.78, "Lacunarity": 0.08, "DeltaAlpha": 0.42, "MultiscaleEntropy": 1.62,
     "note": "Toy example: high D0, low Lacunarity → consistent with IHC 3+"},
    {"specimen_id": "TOY_002", "ihc_score": "Score2Plus",
     "D0": 1.72, "D1": 1.61, "Lacunarity": 0.28, "DeltaAlpha": 0.61, "MultiscaleEntropy": 1.31,
     "note": "Toy example: intermediate D0 → consistent with IHC 2+ (equivocal)"},
    {"specimen_id": "TOY_003", "ihc_score": "Score1Plus",
     "D0": 1.57, "D1": 1.48, "Lacunarity": 0.44, "DeltaAlpha": 0.55, "MultiscaleEntropy": 1.05,
     "note": "Toy example: low-intermediate D0 → consistent with IHC 1+"},
    {"specimen_id": "TOY_004", "ihc_score": "Score0Plus",
     "D0": 1.43, "D1": 1.35, "Lacunarity": 0.62, "DeltaAlpha": 0.39, "MultiscaleEntropy": 0.88,
     "note": "Toy example: low D0, high Lacunarity → consistent with IHC 0+ (ultralow)"},
    {"specimen_id": "TOY_005", "ihc_score": "Score0",
     "D0": 1.31, "D1": 1.24, "Lacunarity": 0.74, "DeltaAlpha": 0.28, "MultiscaleEntropy": 0.62,
     "note": "Toy example: very low D0 → consistent with IHC 0 (null)"},
    # Inconsistency example: high D0 but IHC 0 — should trigger alert
    {"specimen_id": "TOY_006_INCONSISTENT", "ihc_score": "Score0",
     "D0": 1.88, "D1": 1.75, "Lacunarity": 0.09, "DeltaAlpha": 0.82, "MultiscaleEntropy": 1.71,
     "note": "Toy INCONSISTENCY: high D0 with IHC 0 — fractal-clinical alert should be raised"},
]

# ---------------------------------------------------------------------------
# Class hierarchy (derived from Annex A ontology)
# Maps each abstract class to its concrete subclasses / instances.
# Used by the Streamlit dashboard Ontology Summary panel.
# ---------------------------------------------------------------------------

CLASS_HIERARCHY: dict[str, list[str]] = {
    "HER2Entity": [
        "IHCScore", "ClinicalCategory", "ISHGroup",
        "TherapeuticAgent", "ClinicalTrial", "Guideline", "FractalMetric",
    ],
    "IHCScore": ["Score0", "Score0Plus", "Score1Plus", "Score2Plus", "Score3Plus"],
    "ClinicalCategory": [
        "HER2_Positive", "HER2_Negative", "HER2_Equivocal",
        "HER2_Low", "HER2_Ultralow", "HER2_Null",
    ],
    "ISHGroup": ["Group1", "Group2", "Group3", "Group4", "Group5"],
    "TherapeuticAgent": ["Trastuzumab", "TrastuzumabDeruxtecan", "Pertuzumab", "TDM1"],
    "ClinicalTrial": ["DESTINY_Breast04", "DESTINY_Breast06", "DAISY_Trial"],
    "Guideline": [
        "ASCO_CAP_2023", "CAP_Biomarker_2025",
        "ESMO_2023", "Rakha_International_2026",
    ],
    "FractalMetric": [
        "FractalDimension_D0", "FractalDimension_D1",
        "Lacunarity", "MultifractalSpread", "MultiscaleEntropy",
    ],
}
