"""
RDF/OWL exporter — serializes the Neo4j KG to Turtle and JSON-LD.

Strategy:
  1. Fetch all nodes (entities) from Neo4j
  2. Fetch all relationships from Neo4j
  3. Build an rdflib Graph using the HER2 ontology namespaces (from ontology.py)
  4. Serialize to:
       - her2_knowledge_graph.ttl    (RDF/Turtle)
       - her2_knowledge_graph.jsonld (JSON-LD)

OWL axioms added:
  - owl:Class declarations for each node label
  - rdfs:subClassOf for refinesCategory edges
  - owl:disjointWith for HER2_Positive / HER2_Negative
  - owl:ObjectProperty declarations for all edge types

Does NOT require the neosemantics (n10s) plugin — uses pure Python rdflib.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j import Driver
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from src.domain.ontology import NAMESPACES


# ---------------------------------------------------------------------------
# Namespace bindings
# ---------------------------------------------------------------------------

HER2  = Namespace(NAMESPACES["her2"])
FRAC  = Namespace(NAMESPACES["frac"])
NCIT  = Namespace(NAMESPACES["ncit"])
SNOMED = Namespace(NAMESPACES["snomed"])
LOINC = Namespace(NAMESPACES["loinc"])


# ---------------------------------------------------------------------------
# Predicate → OWL ObjectProperty URI mapping
# ---------------------------------------------------------------------------

PREDICATE_URIS: dict[str, URIRef] = {
    "implies":                      HER2["implies"],
    "impliesIfISHAmplified":        HER2["impliesIfISHAmplified"],
    "impliesIfISHNonAmplified":     HER2["impliesIfISHNonAmplified"],
    "requiresReflexTest":           HER2["requiresReflexTest"],
    "requiresIHCWorkup":            HER2["requiresIHCWorkup"],
    "eligibleFor":                  HER2["eligibleFor"],
    "notEligibleFor":               HER2["notEligibleFor"],
    "definedIn":                    HER2["definedIn"],
    "hasQualityRequirement":        HER2["hasQualityRequirement"],
    "associatedWith":               HER2["associatedWith"],
    "proposedCorrelation":          HER2["proposedCorrelation"],
    "proposedEquivalence":          HER2["proposedCorrelation"],  # legacy alias
    "inconsistentWith":             HER2["inconsistentWith"],
    "hasThreshold":                 HER2["hasThreshold"],
    "leadsTo":                      HER2["leadsTo"],
    "conditionedOn":                HER2["conditionedOn"],
    "overrides":                    HER2["overrides"],
    "SUPERSEDED_BY":                HER2["supersededBy"],
    "supportedByEvidence":          HER2["supportedByEvidence"],
    "hasStainingPattern":           HER2["hasStainingPattern"],
    "refinesCategory":              RDFS.subClassOf,
    "contradictsIfConcurrent":      HER2["contradictsIfConcurrent"],
}

_LABEL_TO_CLASS: dict[str, URIRef] = {
    "ClinicalCategory":    HER2["ClinicalCategory"],
    "IHCScore":            HER2["IHCScore"],
    "ISHGroup":            HER2["ISHGroup"],
    "StainingPattern":     HER2["StainingPattern"],
    "TherapeuticAgent":    HER2["TherapeuticAgent"],
    "ClinicalTrial":       HER2["ClinicalTrial"],
    "Biomarker":           HER2["Biomarker"],
    "Guideline":           HER2["Guideline"],
    "QualityMeasure":      HER2["QualityMeasure"],
    "FractalMetric":       HER2["FractalMetric"],
    "PathologicalFeature": HER2["PathologicalFeature"],
    "Assay":               HER2["Assay"],
    "DiagnosticDecision":  HER2["DiagnosticDecision"],
    "Threshold":           HER2["Threshold"],
    "Chunk":               HER2["DocumentChunk"],
}


def _node_uri(node_id: str) -> URIRef:
    """Convert a Neo4j node id to a HER2 namespace URI."""
    safe = node_id.replace(" ", "_").replace("/", "_").replace(":", "_")
    return HER2[safe]


def _add_owl_class_declarations(g: Graph, label: str, class_uri: URIRef) -> None:
    g.add((class_uri, RDF.type, OWL.Class))
    g.add((class_uri, RDFS.label, Literal(label)))


def _add_owl_property_declarations(g: Graph) -> None:
    for pred_name, pred_uri in PREDICATE_URIS.items():
        if pred_uri != RDFS.subClassOf:
            g.add((pred_uri, RDF.type, OWL.ObjectProperty))
            g.add((pred_uri, RDFS.label, Literal(pred_name)))


def _add_owl_disjoint_axioms(g: Graph) -> None:
    """OWL disjointWith between HER2-Positive and HER2-Negative (core axiom)."""
    pos = _node_uri("HER2_Positive")
    neg = _node_uri("HER2_Negative")
    g.add((pos, OWL.disjointWith, neg))


# ---------------------------------------------------------------------------
# Neo4j → rdflib
# ---------------------------------------------------------------------------

_FETCH_NODES = """
MATCH (n)
WHERE NOT n:Chunk AND NOT n:_EmbeddableEntity
RETURN
    labels(n)[0] AS label,
    n.id         AS id,
    properties(n) AS props
"""

_FETCH_RELS = """
MATCH (s)-[r]->(o)
WHERE NOT s:Chunk AND NOT o:Chunk
  AND NOT s:_EmbeddableEntity AND NOT o:_EmbeddableEntity
  AND s.id IS NOT NULL AND o.id IS NOT NULL
RETURN
    s.id AS subject_id,
    type(r) AS predicate,
    o.id AS object_id,
    properties(r) AS props
"""


def _build_rdf_graph(driver: Driver) -> Graph:
    """Fetch Neo4j data and build an rdflib Graph."""
    g = Graph()

    # Bind namespaces
    g.bind("her2",  HER2)
    g.bind("frac",  FRAC)
    g.bind("ncit",  NCIT)
    g.bind("owl",   OWL)
    g.bind("rdfs",  RDFS)
    g.bind("xsd",   XSD)

    # OWL class + property boilerplate
    for label, class_uri in _LABEL_TO_CLASS.items():
        _add_owl_class_declarations(g, label, class_uri)
    _add_owl_property_declarations(g)
    _add_owl_disjoint_axioms(g)

    with driver.session() as session:
        # Nodes
        node_records = session.run(_FETCH_NODES).data()
        for rec in node_records:
            node_id    = rec.get("id")
            node_label = rec.get("label", "Entity")
            props      = rec.get("props", {})
            if not node_id:
                continue

            subj = _node_uri(node_id)
            class_uri = _LABEL_TO_CLASS.get(node_label, HER2["Entity"])
            g.add((subj, RDF.type, class_uri))

            # NCIt URI as owl:sameAs
            ncit_uri = props.get("ncit_uri")
            if ncit_uri and ncit_uri.startswith("NCIt:"):
                ncit_ref = NCIT[ncit_uri.split(":", 1)[1]]
                g.add((subj, OWL.sameAs, ncit_ref))

            # SNOMED URI
            snomed_uri = props.get("snomed_uri")
            if snomed_uri and snomed_uri.startswith("snomed:"):
                sn_ref = SNOMED[snomed_uri.split(":", 1)[1]]
                g.add((subj, OWL.sameAs, sn_ref))

            # Datatype properties
            for key, value in props.items():
                if key in {"id", "ncit_uri", "snomed_uri", "embedding",
                            "created_at", "type"}:
                    continue
                if value is None:
                    continue
                prop_uri = HER2[key]
                if isinstance(value, bool):
                    g.add((subj, prop_uri, Literal(value, datatype=XSD.boolean)))
                elif isinstance(value, int):
                    g.add((subj, prop_uri, Literal(value, datatype=XSD.integer)))
                elif isinstance(value, float):
                    g.add((subj, prop_uri, Literal(value, datatype=XSD.decimal)))
                else:
                    g.add((subj, prop_uri, Literal(str(value))))

        # Relationships
        rel_records = session.run(_FETCH_RELS).data()
        for rec in rel_records:
            s_id  = rec.get("subject_id")
            pred  = rec.get("predicate", "")
            o_id  = rec.get("object_id")
            rprops = rec.get("props", {})

            if not s_id or not o_id:
                continue

            subj = _node_uri(s_id)
            obj  = _node_uri(o_id)
            pred_uri = PREDICATE_URIS.get(pred, HER2[pred])
            g.add((subj, pred_uri, obj))

            # Reify edge properties as annotation axioms if meaningful
            conf = rprops.get("confidence")
            if conf is not None:
                # Store confidence on a blank-node reification
                stmt = URIRef(f"{subj}_{pred}_{obj}_stmt")
                g.add((stmt, RDF.type, OWL.Axiom))
                g.add((stmt, OWL.annotatedSource,   subj))
                g.add((stmt, OWL.annotatedProperty, pred_uri))
                g.add((stmt, OWL.annotatedTarget,   obj))
                g.add((stmt, HER2["confidence"],
                       Literal(float(conf), datatype=XSD.decimal)))

    return g


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_rdf(
    driver: Driver,
    output_dir: str | Path = "./output",
    verbose: bool = True,
    timestamp: str | None = None,
) -> dict[str, str]:
    """
    Export the Neo4j KG to RDF/Turtle and JSON-LD files.

    Args:
        timestamp: Optional suffix for the filename (e.g. '20260417_143022').
                   When provided the output is saved as
                   her2_knowledge_graph_<timestamp>.ttl to avoid overwriting
                   previous runs.

    Returns:
        dict with keys 'ttl' and 'jsonld' pointing to output file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[RDFExporter] Building RDF graph from Neo4j...")

    g = _build_rdf_graph(driver)

    suffix = f"_{timestamp}" if timestamp else ""
    ttl_path    = output_dir / f"her2_knowledge_graph{suffix}.ttl"
    jsonld_path = output_dir / f"her2_knowledge_graph{suffix}.jsonld"

    g.serialize(destination=str(ttl_path),    format="turtle")
    g.serialize(destination=str(jsonld_path), format="json-ld")

    triple_count = len(g)

    if verbose:
        print(f"  Exported {triple_count} triples")
        print(f"  Turtle:  {ttl_path}")
        print(f"  JSON-LD: {jsonld_path}")

    return {"ttl": str(ttl_path), "jsonld": str(jsonld_path), "triples": triple_count}
