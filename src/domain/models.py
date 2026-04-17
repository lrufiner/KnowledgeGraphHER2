"""
Domain models — pure dataclasses/Pydantic models.
No external dependencies beyond pydantic and stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    ALGORITHM          = "algorithm"
    CRITERIA           = "criteria"
    RECOMMENDATION     = "recommendation"
    FRACTAL_MAPPING    = "fractal_mapping"
    QA                 = "qa"
    TABLE              = "table"
    ONTOLOGY           = "ontology"
    GENERAL            = "general"
    TECHNICAL_APPENDIX = "technical_appendix"  # software/tutorial docs — skip extraction


class NodeType(str, Enum):
    CLINICAL_CATEGORY   = "ClinicalCategory"
    IHC_SCORE           = "IHCScore"
    ISH_GROUP           = "ISHGroup"
    STAINING_PATTERN    = "StainingPattern"
    THERAPEUTIC_AGENT   = "TherapeuticAgent"
    CLINICAL_TRIAL      = "ClinicalTrial"
    BIOMARKER           = "Biomarker"
    GUIDELINE           = "Guideline"
    QUALITY_MEASURE     = "QualityMeasure"
    FRACTAL_METRIC      = "FractalMetric"
    PATHOLOGICAL_FEATURE = "PathologicalFeature"
    ASSAY               = "Assay"
    DIAGNOSTIC_DECISION = "DiagnosticDecision"
    THRESHOLD           = "Threshold"


class EdgeType(str, Enum):
    IMPLIES                  = "implies"
    REQUIRES_REFLEX_TEST     = "requiresReflexTest"
    ELIGIBLE_FOR             = "eligibleFor"
    NOT_ELIGIBLE_FOR         = "notEligibleFor"
    DEFINED_IN               = "definedIn"
    HAS_QUALITY_REQUIREMENT  = "hasQualityRequirement"
    ASSOCIATED_WITH          = "associatedWith"
    PROPOSED_EQUIVALENCE     = "proposedEquivalence"
    INCONSISTENT_WITH        = "inconsistentWith"
    HAS_THRESHOLD            = "hasThreshold"
    CONTRADICTS_IF_CONCURRENT = "contradictsIfConcurrent"
    LEADS_TO                 = "leadsTo"
    CONDITIONED_ON           = "conditionedOn"
    OVERRIDES                = "overrides"
    SUPPORTED_BY_EVIDENCE    = "supportedByEvidence"
    HAS_STAINING_PATTERN     = "hasStainingPattern"
    REFINES_CATEGORY         = "refinesCategory"
    SUPERSEDED_BY            = "SUPERSEDED_BY"


class ValidationSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


# ---------------------------------------------------------------------------
# Chunk (output of ingestion)
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """Atomic unit of processing in the pipeline."""
    chunk_id:      str
    source_doc:    str
    section:       str
    content:       str
    content_type:  ContentType
    page_ref:      Optional[int] = None
    metadata:      dict          = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id":    self.chunk_id,
            "source_doc":  self.source_doc,
            "section":     self.section,
            "content":     self.content,
            "content_type": self.content_type.value,
            "page_ref":    self.page_ref,
            "metadata":    self.metadata,
        }


# ---------------------------------------------------------------------------
# Extraction output models (LLM response schema)
# ---------------------------------------------------------------------------

class EntityModel(BaseModel):
    """Schema for a single extracted entity."""
    id:             str = Field(..., description="Local ID within extraction (e.g. e001)")
    label:          str = Field(..., description="Human-readable name")
    type:           NodeType
    definition:     Optional[str] = None
    source_quote:   Optional[str] = Field(None, description="Verbatim text from source")
    candidate_uri:  Optional[str] = Field(None, description="Suggested canonical URI")
    confidence:     float = Field(default=1.0, ge=0.0, le=1.0)
    interobserver_variability: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence_level: Optional[str] = None


class RelationModel(BaseModel):
    """Schema for a single extracted relation."""
    subject_id:  str
    predicate:   EdgeType
    object_id:   str
    confidence:  float = Field(default=1.0, ge=0.0, le=1.0)
    evidence:    Optional[str] = None
    source_chunk: Optional[str] = None
    guideline_version: Optional[str] = None
    conditions:  Optional[str] = None


class ExtractionResult(BaseModel):
    """Full output of the EXTRACT phase for one chunk."""
    chunk_id:   str
    section:    str
    entities:   list[EntityModel] = Field(default_factory=list)
    relations:  list[RelationModel] = Field(default_factory=list)
    error:      Optional[str] = None
    raw_response: Optional[str] = None


# ---------------------------------------------------------------------------
# Resolved entity (after RESOLVE phase)
# ---------------------------------------------------------------------------

class ResolvedEntity(BaseModel):
    """Entity with canonical URI resolved."""
    id:           str
    label:        str
    type:         NodeType
    definition:   Optional[str] = None
    source_quote: Optional[str] = None
    resolved_uri: str            # canonical URI (NCIt, SNOMED, or local her2:)
    ncit_uri:     Optional[str] = None
    snomed_uri:   Optional[str] = None
    loinc_code:   Optional[str] = None
    source_doc:   Optional[str] = None
    confidence:   float = 1.0
    interobserver_variability: Optional[float] = None
    evidence_level: Optional[str] = None
    created_at:   datetime = Field(default_factory=datetime.utcnow)
    extra_props:  dict = Field(default_factory=dict)

    def to_neo4j_dict(self) -> dict:
        return {
            "id":           self.id,
            "label":        self.label,
            "definition":   self.definition or "",
            "source_quote": self.source_quote or "",
            "ncit_uri":     self.ncit_uri or "",
            "snomed_uri":   self.snomed_uri or "",
            "loinc_code":   self.loinc_code or "",
            "source_doc":   self.source_doc or "",
            "confidence":   self.confidence,
            "interobserver_variability": self.interobserver_variability,
            "evidence_level": self.evidence_level or "",
            "created_at":   self.created_at.isoformat(),
            **self.extra_props,
        }


# ---------------------------------------------------------------------------
# Resolved relation
# ---------------------------------------------------------------------------

class ResolvedRelation(BaseModel):
    """Relation with subject/object IDs pointing to ResolvedEntity IDs."""
    subject_id:   str
    predicate:    EdgeType
    object_id:    str
    confidence:   float = 1.0
    evidence:     Optional[str] = None
    source_chunk: Optional[str] = None
    guideline_version: Optional[str] = None
    conditions:   Optional[str] = None
    is_hypothesis: bool = False   # True for fractal proposedEquivalence


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    rule_id:   str
    valid:     bool
    severity:  ValidationSeverity
    message:   str
    source:    str


class ValidationReport(BaseModel):
    timestamp:   datetime = Field(default_factory=datetime.utcnow)
    results:     list[ValidationResult] = Field(default_factory=list)
    is_consistent: bool = True

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)
        if not result.valid and result.severity in (
            ValidationSeverity.CRITICAL, ValidationSeverity.HIGH
        ):
            self.is_consistent = False

    def summary(self) -> dict[str, Any]:
        failures = [r for r in self.results if not r.valid]
        return {
            "total":    len(self.results),
            "passed":   len(self.results) - len(failures),
            "failed":   len(failures),
            "critical": sum(1 for r in failures if r.severity == ValidationSeverity.CRITICAL),
            "is_consistent": self.is_consistent,
        }
