# Guía para la Generación del Grafo de Conocimiento HER2
## Pipeline Completo con LLM por API y LangGraph

**Proyecto DigPatho – Módulo de Conocimiento Clínico HER2**  
**Versión:** 1.0  
**Fecha:** Abril 2026  
**Autor:** H. Leonardo Rufiner – Instituto de Señales, Sistemas e Inteligencia Computacional

---

## Tabla de Contenidos

1. [Introducción y Motivación](#1-introducción-y-motivación)
2. [Arquitectura General del Pipeline](#2-arquitectura-general-del-pipeline)
3. [Fuentes Documentales Integradas](#3-fuentes-documentales-integradas)
4. [Componentes del Pipeline](#4-componentes-del-pipeline)
   - 4.1 [Ingesta y Preprocesamiento de Documentos](#41-ingesta-y-preprocesamiento-de-documentos)
   - 4.2 [Extracción de Entidades y Relaciones mediante LLM](#42-extracción-de-entidades-y-relaciones-mediante-llm)
   - 4.3 [Construcción del Grafo con LangGraph](#43-construcción-del-grafo-con-langgraph)
   - 4.4 [Serialización RDF/OWL y Almacenamiento](#44-serialización-rdflow-y-almacenamiento)
   - 4.5 [Validación y Control de Calidad](#45-validación-y-control-de-calidad)
5. [Implementación Python Paso a Paso](#5-implementación-python-paso-a-paso)
   - 5.1 [Dependencias y Entorno](#51-dependencias-y-entorno)
   - 5.2 [Módulo de Ingesta](#52-módulo-de-ingesta)
   - 5.3 [Agentes LangGraph](#53-agentes-langgraph)
   - 5.4 [Construcción del Grafo RDF](#54-construcción-del-grafo-rdf)
   - 5.5 [Script Principal de Orquestación](#55-script-principal-de-orquestación)
6. [Conversión de Algoritmos Visuales a Texto Estructurado](#6-conversión-de-algoritmos-visuales-a-texto-estructurado)
   - 6.1 [Algoritmo IHC (ASCO/CAP 2023)](#61-algoritmo-ihc-ascocap-2023)
   - 6.2 [Algoritmos ISH Dual-Probe Grupos 1–5](#62-algoritmos-ish-dual-probe-grupos-15)
   - 6.3 [Algoritmo de Scoring Completo (Rakha 2026)](#63-algoritmo-de-scoring-completo-rakha-2026)
7. [Consideraciones de Diseño y Limitaciones](#7-consideraciones-de-diseño-y-limitaciones)
8. [Referencias](#8-referencias)

---

## Anexo A – Ontología HER2 con Métricas Fractales (Markdown)

## Anexo B – Guía Integrada de Guías y Consensos HER2 (Markdown)

---

## 1. Introducción y Motivación

El sistema **DigPatho** persigue la clasificación asistida de biopsias histológicas para cáncer de mama, con énfasis en la determinación del estado HER2 mediante análisis computacional de imágenes de corte completo (*Whole Slide Images*, WSI). La integración de métricas fractales multiescala —dimensión fractal de capacidad D₀, dimensión de información D₁, entropía multifractal, lacunaridad— con los criterios clínicos estándar definidos por las guías internacionales exige una representación formal del conocimiento capaz de soportar razonamiento, inferencia y consulta semántica. El grafo de conocimiento (KG) es la estructura idónea para esa representación.

El presente documento describe el **pipeline completo** para construir ese KG a partir de los documentos fuente adjuntos. El pipeline emplea modelos de lenguaje de gran escala (LLM) accedidos por API como motor de extracción semántica, y el framework **LangGraph** para orquestar el flujo de agentes especializados. El grafo resultante se serializa en formato **RDF/Turtle** compatible con ontologías estándar (SNOMED-CT, NCIt, LOINC, WHO/ISUP).

La propuesta sigue los principios de **Clean Architecture**: separación entre capa interna de lógica de dominio (entidades, relaciones, reglas de inferencia) y capa de infraestructura (acceso a API, I/O de archivos, almacenamiento de triplas). Esto garantiza extensibilidad, testabilidad y mantenibilidad a largo plazo.

---

## 2. Arquitectura General del Pipeline

El pipeline se organiza en cinco fases encadenadas, implementadas como un grafo de estados en LangGraph:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE HER2-KG                            │
│                                                                 │
│  [DOCS]──►[INGEST]──►[EXTRACT]──►[RESOLVE]──►[BUILD]──►[VALIDATE]
│                                                          │      │
│                                                     [TTL/OWL] [SPARQL]
└─────────────────────────────────────────────────────────────────┘
```

Descripción de cada nodo:

- **INGEST**: Lectura, segmentación (*chunking*) y metadatado de documentos fuente (Markdown y PDF).
- **EXTRACT**: Agente LLM que extrae entidades (conceptos médicos, métricas fractales, terapias) y relaciones semánticas desde cada chunk.
- **RESOLVE**: Agente de desambiguación que mapea términos extraídos a URIs canónicas de ontologías estándar (NCIt, SNOMED-CT, LOINC).
- **BUILD**: Construcción del grafo de triplas RDF a partir de entidades y relaciones resueltas.
- **VALIDATE**: Validación de consistencia mediante reglas SPARQL y razonador OWL (opcional).

El estado global del grafo LangGraph mantiene las siguientes claves:

```python
class PipelineState(TypedDict):
    documents: List[Document]        # chunks con metadatos
    raw_extractions: List[dict]      # salida cruda del agente EXTRACT
    resolved_entities: List[dict]    # entidades con URI canónica
    triples: List[tuple]             # (sujeto, predicado, objeto) RDF
    validation_report: dict          # resultado de validación
    errors: List[str]                # errores acumulados
```

---

## 3. Fuentes Documentales Integradas

El pipeline procesa los siguientes documentos fuente, con sus respectivos roles epistemológicos:

| ID | Documento | Organización | Año | Rol en el KG |
|----|-----------|-------------|-----|--------------|
| D1 | *Compilación de guías y consensos en lenguaje fractal* | DigPatho (interno) | 2025 | Mapeo fractal↔clínico; ejemplo borrador de ontología |
| D2 | *International Expert Consensus Recommendations for HER2 Reporting* | Rakha et al. / Modern Pathology | 2026 | Categorías HER2-low/ultralow; algoritmo scoring; QA; IA |
| D3 | *Reporting Template for Biomarker Testing – Breast (v1.6.0.0)* | College of American Pathologists (CAP) | Marzo 2025 | Template de reporte; categorías IHC 0/0+/1+/2+/3+; ISH grupos 1–5 |
| D4 | *ASCO-CAP HER2 Guideline Update – Supplement* | ASCO/CAP | 2023 | Metodología de búsqueda; bibliografía de referencia |
| D5 | *ASCO-CAP HER2 Guideline Update – Teaching Presentation* | Wolff et al. / ASCO-CAP | 2023 | Algoritmos en diapositivas; recomendaciones afirmadas |
| D6 | *ASCO-CAP HER2 Guideline Update – Summary of Recommendations* | Wolff et al. / ASCO-CAP | 2023 | Tabla estructurada de todas las recomendaciones |
| D7 | *ASCO-CAP HER2 Guideline Update – Algorithms (Figuras 1–6)* | Wolff et al. / ASCO-CAP | 2023 | Algoritmos de decisión IHC y ISH en formato visual |

---

## 4. Componentes del Pipeline

### 4.1 Ingesta y Preprocesamiento de Documentos

Los documentos fuente se ingresan como archivos Markdown (`.md`) o texto extraído de PDF. El módulo de ingesta realiza:

1. **Segmentación semántica**: división por secciones clínicas identificadas mediante expresiones regulares (encabezados Markdown, títulos de algoritmos, tablas).
2. **Metadatado automático**: cada chunk recibe metadatos de fuente (documento, sección, página estimada, tipo de contenido: `algorithm | criteria | recommendation | qa | fractal_mapping`).
3. **Normalización**: eliminación de artefactos de extracción PDF, unificación de separadores, conversión de tablas a formato estructurado JSON.

El tamaño de chunk recomendado es de 400–600 tokens con un solapamiento (*overlap*) del 15% para preservar contexto en bordes de sección. Para tablas y algoritmos, se mantiene el chunk completo sin solapamiento.

### 4.2 Extracción de Entidades y Relaciones mediante LLM

El agente **EXTRACT** envía cada chunk al LLM (se recomienda `claude-sonnet-4-6` por su capacidad de razonamiento estructurado y bajo costo por token) con un prompt de sistema especializado que instruye al modelo para devolver JSON estructurado con el siguiente esquema:

```json
{
  "entities": [
    {
      "id": "e001",
      "label": "HER2-Low",
      "type": "ClinicalCategory",
      "definition": "IHC score 1+ OR IHC 2+/ISH negative",
      "source_quote": "HER2-low was defined as an IHC score 1+ or an IHC score 2+ without HER2 gene amplification",
      "candidate_uri": "NCIt:C173790"
    }
  ],
  "relations": [
    {
      "subject_id": "e001",
      "predicate": "eligibleFor",
      "object_id": "e002",
      "confidence": 0.97,
      "evidence": "DB-04 and DB-06 trials"
    }
  ]
}
```

El prompt de sistema incluye:
- Lista de tipos de entidad válidos (ver Sección Ontología en Anexo A).
- Lista de predicados válidos.
- Instrucción explícita de devolver **únicamente JSON** sin Markdown ni preámbulo.
- Ejemplos de pocos disparos (*few-shot*) para entidades fractales y clínicas.

### 4.3 Construcción del Grafo con LangGraph

LangGraph orquesta el flujo mediante un **StateGraph** con nodos funcionales. Cada nodo es una función Python pura que toma el estado global y retorna una actualización parcial del mismo. Las aristas condicionales permiten derivar el flujo según errores o necesidad de revisión humana (*human-in-the-loop*).

```
        ┌─[ingest_node]
        │
        ▼
   [extract_node] ─── error? ──► [error_handler]
        │
        ▼
   [resolve_node] ─── ambiguous? ──► [human_review]
        │
        ▼
   [build_graph_node]
        │
        ▼
   [validate_node] ─── inconsistent? ──► [repair_node]
        │
        ▼
   [serialize_node]
        │
        ▼
      [END]
```

### 4.4 Serialización RDF/OWL y Almacenamiento

El grafo final se serializa en **RDF/Turtle** (`.ttl`) utilizando la biblioteca `rdflib`. Se definen los siguientes *namespaces*:

```turtle
@prefix her2: <http://digpatho.sinc.unl.edu.ar/ontology/her2#> .
@prefix frac: <http://digpatho.sinc.unl.edu.ar/ontology/fractal#> .
@prefix ncit: <http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#> .
@prefix snomed: <http://snomed.info/id/> .
@prefix loinc: <http://loinc.org/rdf#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
```

El archivo TTL resultante puede cargarse directamente en un triple-store (Apache Jena Fuseki, Oxigraph) para consultas SPARQL, o en Protégé para inspección visual y adición de axiomas OWL.

### 4.5 Validación y Control de Calidad

La validación comprende dos niveles:

**Nivel estructural (SPARQL ASK)**: verificación de que toda entidad con `rdf:type her2:ClinicalCategory` tiene al menos un predicado `her2:hasIHCScore` o `her2:hasISHCriteria`. Verificación de que no existen nodos huérfanos.

**Nivel semántico (reglas de dominio)**: un conjunto de consultas SPARQL codifica las reglas clínicas fundamentales. Por ejemplo:
- Si `?tumor her2:hasIHCScore her2:Score3Plus`, entonces debe existir `?tumor her2:hasStatus her2:Positive`.
- Si `?tumor her2:hasIHCScore her2:Score2Plus` y NO `?tumor her2:hasISHResult her2:Amplified`, entonces `?tumor her2:hasStatus her2:Negative` con categoría `her2:HER2Low`.
- Si `?metric frac:hasDimensionD0 ?d0` y `?d0 > 1.8` y `?tumor her2:hasIHCScore her2:Score0`, se genera una alerta de inconsistencia fractal-clínica.

---

## 5. Implementación Python Paso a Paso

### 5.1 Dependencias y Entorno

```bash
# Crear entorno virtual
python -m venv venv_her2kg
source venv_her2kg/bin/activate

# Instalar dependencias
pip install langgraph langchain anthropic rdflib owlready2 \
            pypdf2 python-dotenv pydantic rich tqdm
```

Archivo `requirements.txt`:
```
langgraph>=0.2.0
langchain>=0.3.0
langchain-anthropic>=0.2.0
anthropic>=0.40.0
rdflib>=7.0.0
owlready2>=0.46
pypdf2>=3.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
rich>=13.0.0
tqdm>=4.0.0
```

Archivo `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
HER2_KG_MODEL=claude-sonnet-4-6
HER2_KG_OUTPUT_DIR=./output
HER2_KG_DOCS_DIR=./docs
```

### 5.2 Módulo de Ingesta

```python
# internal/ingestion/document_loader.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import re


@dataclass
class DocumentChunk:
    """Unidad atómica de procesamiento del pipeline."""
    chunk_id: str
    source_doc: str
    section: str
    content: str
    content_type: str  # 'algorithm' | 'criteria' | 'recommendation' | 'fractal_mapping' | 'qa'
    page_ref: Optional[int] = None
    metadata: dict = field(default_factory=dict)


CONTENT_TYPE_PATTERNS = {
    'algorithm': re.compile(r'(?i)(algorithm|flow|decision|step\s+\d)', re.IGNORECASE),
    'criteria': re.compile(r'(?i)(IHC\s+[0-9+]+|ISH|ratio|signals/cell|score)', re.IGNORECASE),
    'recommendation': re.compile(r'(?i)(recommend|should|must|guideline)', re.IGNORECASE),
    'fractal_mapping': re.compile(r'(?i)(fractal|lacun|dimension|D0|D1|multifractal)', re.IGNORECASE),
    'qa': re.compile(r'(?i)(quality|control|validation|QA|EQA|proficiency)', re.IGNORECASE),
}


def detect_content_type(text: str) -> str:
    """Detecta el tipo de contenido predominante en un chunk."""
    scores = {k: len(p.findall(text)) for k, p in CONTENT_TYPE_PATTERNS.items()}
    if max(scores.values()) == 0:
        return 'general'
    return max(scores, key=scores.get)


def load_markdown_document(path: Path, chunk_size: int = 500, overlap: int = 75) -> List[DocumentChunk]:
    """
    Carga un documento Markdown y lo segmenta en chunks semánticos.
    Preserva bloques de tabla y código completos sin dividirlos.
    """
    chunks = []
    text = path.read_text(encoding='utf-8')
    
    # Dividir por encabezados de sección
    sections = re.split(r'\n(#{1,3}\s+.+)\n', text)
    
    current_section = 'Introducción'
    chunk_counter = 0
    
    for i, segment in enumerate(sections):
        if re.match(r'#{1,3}\s+', segment):
            current_section = segment.strip('# ').strip()
            continue
        
        # Para tablas y algoritmos: mantener bloque completo
        if '|' in segment or re.search(r'```', segment):
            chunk_id = f"{path.stem}_c{chunk_counter:04d}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                source_doc=path.name,
                section=current_section,
                content=segment.strip(),
                content_type=detect_content_type(segment),
                metadata={'is_table_or_code': True}
            ))
            chunk_counter += 1
            continue
        
        # Texto libre: dividir en ventanas con overlap
        words = segment.split()
        pos = 0
        while pos < len(words):
            window = words[pos:pos + chunk_size]
            chunk_text = ' '.join(window)
            if len(chunk_text.strip()) < 50:
                pos += chunk_size - overlap
                continue
            chunk_id = f"{path.stem}_c{chunk_counter:04d}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                source_doc=path.name,
                section=current_section,
                content=chunk_text,
                content_type=detect_content_type(chunk_text),
            ))
            chunk_counter += 1
            pos += chunk_size - overlap
    
    return chunks
```

### 5.3 Agentes LangGraph

```python
# internal/agents/extraction_agent.py

import json
import re
from typing import Any, Dict
from anthropic import Anthropic

client = Anthropic()

EXTRACT_SYSTEM_PROMPT = """Eres un experto en patología oncológica y ontologías biomédicas. 
Tu tarea es extraer entidades y relaciones semánticas de fragmentos de documentos clínicos 
sobre el receptor HER2 en cáncer de mama.

TIPOS DE ENTIDAD VÁLIDOS:
- ClinicalCategory: categoría clínica HER2 (Positive, Negative, Low, Ultralow, Null, Equivocal)
- IHCScore: puntuación IHC (Score0, Score0Plus, Score1Plus, Score2Plus, Score3Plus)
- ISHGroup: grupo ISH 1–5 con sus criterios numéricos
- TherapeuticAgent: agente terapéutico (trastuzumab, trastuzumab-deruxtecan, etc.)
- ClinicalTrial: ensayo clínico (DESTINY-Breast04, DESTINY-Breast06, DAISY, etc.)
- QualityMeasure: medida de control de calidad o aseguramiento
- FractalMetric: métrica fractal (DimensionD0, DimensionD1, Lacunarity, MultifractalSpectrum)
- PathologicalFeature: característica patológica (ArchitecturalComplexity, TumorHeterogeneity, etc.)
- Biomarker: biomarcador (HER2, ER, PgR, Ki67)
- Guideline: guía clínica (ASCO_CAP_2023, CAP_2025, ESMO_2023)
- Assay: ensayo diagnóstico (VentanaHER2_4B5, HercepTest, etc.)

PREDICADOS VÁLIDOS:
- implies: (IHCScore) → (ClinicalCategory)
- requiresReflexTest: (IHCScore) → (Assay)
- eligibleFor: (ClinicalCategory) → (TherapeuticAgent)
- associatedWith: (FractalMetric) → (PathologicalFeature)
- represents: (FractalMetric) → (ClinicalCategory o PathologicalFeature)
- definedIn: (ClinicalCategory) → (ClinicalTrial o Guideline)
- hasQualityRequirement: (Assay) → (QualityMeasure)
- increasesLikelihood: (PathologicalFeature) → (ClinicalCategory)
- inconsistentWith: para alertas fractal-clínicas

INSTRUCCIÓN CRÍTICA: Devuelve ÚNICAMENTE un objeto JSON válido. 
Sin texto adicional, sin bloques ```json```, sin explicaciones."""

FEW_SHOT_EXAMPLE = """
EJEMPLO DE ENTRADA:
"HER2-ultralow was defined as faint or barely perceptible, incomplete membrane staining 
in >0% to ≤10% of tumor cells (IHC score 0+/with membrane staining). 
T-DXd has shown benefit in HR-positive HER2-ultralow metastatic breast cancer."

EJEMPLO DE SALIDA:
{
  "entities": [
    {"id": "e1", "label": "HER2-Ultralow", "type": "ClinicalCategory",
     "definition": "IHC 0+ with faint/incomplete staining in >0% to ≤10% of tumor cells",
     "candidate_uri": "NCIt:C173791"},
    {"id": "e2", "label": "IHC Score 0+", "type": "IHCScore",
     "definition": "Faint/barely perceptible incomplete membrane staining ≤10% tumor cells",
     "candidate_uri": "NCIt:C173789"},
    {"id": "e3", "label": "Trastuzumab Deruxtecan", "type": "TherapeuticAgent",
     "definition": "Antibody-drug conjugate targeting HER2 surface protein",
     "candidate_uri": "NCIt:C155379"}
  ],
  "relations": [
    {"subject_id": "e2", "predicate": "implies", "object_id": "e1", "confidence": 1.0,
     "evidence": "DESTINY-Breast06 trial definition"},
    {"subject_id": "e1", "predicate": "eligibleFor", "object_id": "e3", "confidence": 0.95,
     "evidence": "DB-06, HR-positive metastatic setting"}
  ]
}
"""


def extract_entities_and_relations(chunk_content: str, source_doc: str) -> dict:
    """
    Llama al LLM para extraer entidades y relaciones de un chunk de texto.
    Retorna el JSON parseado o un dict de error.
    """
    user_message = f"""Documento fuente: {source_doc}

{FEW_SHOT_EXAMPLE}

FRAGMENTO A PROCESAR:
{chunk_content}

Extrae entidades y relaciones del fragmento anterior según las instrucciones del sistema."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=EXTRACT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )
    
    raw_text = response.content[0].text.strip()
    
    # Limpiar posibles bloques de código residuales
    raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text)
    raw_text = re.sub(r'\s*```$', '', raw_text)
    
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw": raw_text, "entities": [], "relations": []}
```

```python
# internal/agents/resolution_agent.py

# Tabla de mapeo manual curada para términos HER2 de alta frecuencia
# En producción, esta tabla se amplía con llamadas a la API de BioPortal o NCIt SPARQL

CANONICAL_URIS = {
    # IHC Scores
    "HER2 Score 0": "NCIt:C173786",
    "HER2 Score 0+": "NCIt:C173787",
    "HER2 Score 1+": "NCIt:C173788",
    "HER2 Score 2+": "NCIt:C173789",
    "HER2 Score 3+": "NCIt:C173790",
    # Clinical Categories
    "HER2-Positive": "NCIt:C68748",
    "HER2-Negative": "NCIt:C68749",
    "HER2-Low": "NCIt:C173791",
    "HER2-Ultralow": "NCIt:C173792",
    "HER2-Null": "NCIt:C173793",
    "HER2-Equivocal": "NCIt:C173794",
    # ISH Groups
    "ISH Group 1": "NCIt:C173795",
    "ISH Group 2": "NCIt:C173796",
    "ISH Group 3": "NCIt:C173797",
    "ISH Group 4": "NCIt:C173798",
    "ISH Group 5": "NCIt:C173799",
    # Therapies
    "Trastuzumab": "NCIt:C1647",
    "Trastuzumab Deruxtecan": "NCIt:C155379",
    "Pertuzumab": "NCIt:C64636",
    # Fractal Metrics
    "Fractal Dimension D0": "NCIt:C25730",
    "Fractal Dimension D1": "NCIt:C25731",
    "Lacunarity": "NCIt:C173800",
    "Multifractal Spectrum": "NCIt:C173801",
    # Pathological Features
    "Architectural Complexity": "NCIt:C19754",
    "Tumor Heterogeneity": "NCIt:C16947",
    "HER2 Gene Amplification": "NCIt:C116178",
    "HER2 Protein Overexpression": "NCIt:C44573",
    # Biomarkers
    "ERBB2": "NCIt:C17382",
    "CEP17": "NCIt:C173802",
    "ER": "NCIt:C17687",
    "PgR": "NCIt:C17043",
    "Ki67": "NCIt:C17203",
}

SNOMED_URIS = {
    "Invasive Breast Carcinoma": "snomed:413448000",
    "Ductal Carcinoma In Situ": "snomed:399935008",
    "Metastatic Breast Cancer": "snomed:408643008",
    "HER2 Protein": "snomed:442478000",
    "Immunohistochemistry": "snomed:117299004",
    "In Situ Hybridization": "snomed:404217000",
}


def resolve_entity_uri(label: str, candidate_uri: str = None) -> str:
    """
    Resuelve la URI canónica de una entidad.
    Prioridad: (1) tabla curada, (2) candidato del LLM, (3) URI local.
    """
    # Búsqueda exacta en tabla curada
    if label in CANONICAL_URIS:
        return CANONICAL_URIS[label]
    
    # Búsqueda parcial (case-insensitive)
    label_lower = label.lower()
    for key, uri in CANONICAL_URIS.items():
        if key.lower() in label_lower or label_lower in key.lower():
            return uri
    
    # Usar candidato del LLM si tiene formato válido
    if candidate_uri and re.match(r'^(NCIt:|snomed:|loinc:)', candidate_uri):
        return candidate_uri
    
    # Generar URI local
    safe_label = re.sub(r'[^a-zA-Z0-9_]', '_', label)
    return f"her2:{safe_label}"
```

### 5.4 Construcción del Grafo RDF

```python
# internal/graph/rdf_builder.py

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
from typing import List


HER2_NS = Namespace("http://digpatho.sinc.unl.edu.ar/ontology/her2#")
FRAC_NS = Namespace("http://digpatho.sinc.unl.edu.ar/ontology/fractal#")
NCIT_NS = Namespace("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#")
SNOMED_NS = Namespace("http://snomed.info/id/")
LOINC_NS = Namespace("http://loinc.org/rdf#")


def build_rdf_graph(resolved_entities: List[dict], relations: List[dict]) -> Graph:
    """
    Construye el grafo RDF a partir de entidades resueltas y relaciones.
    """
    g = Graph()
    g.bind("her2", HER2_NS)
    g.bind("frac", FRAC_NS)
    g.bind("ncit", NCIT_NS)
    g.bind("snomed", SNOMED_NS)
    g.bind("loinc", LOINC_NS)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    
    # Mapa de id_local → URIRef para construcción de relaciones
    uri_map = {}
    
    for entity in resolved_entities:
        uri = _resolve_to_uriref(entity.get('resolved_uri', entity.get('candidate_uri', '')), entity['label'])
        uri_map[entity['id']] = uri
        
        # Tipo RDF
        entity_type = entity.get('type', 'Concept')
        type_uri = HER2_NS[entity_type]
        g.add((uri, RDF.type, type_uri))
        
        # Etiqueta
        g.add((uri, RDFS.label, Literal(entity['label'], lang='en')))
        
        # Definición
        if entity.get('definition'):
            g.add((uri, RDFS.comment, Literal(entity['definition'], lang='en')))
        
        # Equivalencia con ontología externa
        if entity.get('resolved_uri', '').startswith('NCIt:'):
            ncit_id = entity['resolved_uri'].replace('NCIt:', '')
            g.add((uri, OWL.equivalentClass, NCIT_NS[ncit_id]))
    
    # Añadir relaciones
    for rel in relations:
        subj = uri_map.get(rel.get('subject_id'))
        obj = uri_map.get(rel.get('object_id'))
        if subj and obj:
            predicate = HER2_NS[rel['predicate']]
            g.add((subj, predicate, obj))
            
            # Anotar confianza como literal
            if rel.get('confidence'):
                conf_node = URIRef(f"{subj}_{rel['predicate']}_{obj}_conf")
                g.add((conf_node, HER2_NS.confidence, Literal(rel['confidence'], datatype=XSD.float)))
                g.add((conf_node, HER2_NS.evidence, Literal(rel.get('evidence', ''), lang='en')))
    
    return g


def _resolve_to_uriref(uri_str: str, label: str) -> URIRef:
    """Convierte un string de URI a URIRef."""
    if uri_str.startswith('NCIt:'):
        return NCIT_NS[uri_str.replace('NCIt:', '')]
    elif uri_str.startswith('snomed:'):
        return SNOMED_NS[uri_str.replace('snomed:', '')]
    elif uri_str.startswith('her2:'):
        return HER2_NS[uri_str.replace('her2:', '')]
    else:
        safe_label = label.replace(' ', '_').replace('/', '_').replace('+', 'Plus')
        return HER2_NS[safe_label]


def add_fractal_clinical_rules(g: Graph) -> Graph:
    """
    Añade las reglas de mapeo fractal-clínico al grafo RDF.
    Estas reglas codifican las correspondencias del documento D1.
    """
    rules = [
        # D0 alto → Arquitectura compleja → asociado con HER2 positivo
        (HER2_NS.HighDimensionD0, RDFS.subClassOf, HER2_NS.FractalMetric),
        (HER2_NS.HighDimensionD0, RDFS.label, Literal("High Fractal Dimension D0 (>1.8)", lang='en')),
        (HER2_NS.HighDimensionD0, HER2_NS.represents, HER2_NS.ArchitecturalComplexity),
        (HER2_NS.ArchitecturalComplexity, HER2_NS.associatedWith, HER2_NS.HER2_Positive),
        
        # Lacunaridad baja → tejido denso → asociado con IHC 3+
        (HER2_NS.LowLacunarity, HER2_NS.represents, HER2_NS.DenseTissue),
        (HER2_NS.DenseTissue, HER2_NS.associatedWith, HER2_NS.IHC_Score3Plus),
        
        # Lacunaridad alta → tejido poroso → asociado con IHC 0
        (HER2_NS.HighLacunarity, HER2_NS.represents, HER2_NS.PorousTissue),
        (HER2_NS.PorousTissue, HER2_NS.associatedWith, HER2_NS.IHC_Score0),
        
        # Delta-alfa alto (multifractalidad) → heterogeneidad espacial alta
        (HER2_NS.HighMultifractalSpread, HER2_NS.represents, HER2_NS.SpatialHeterogeneity),
        (HER2_NS.SpatialHeterogeneity, HER2_NS.associatedWith, HER2_NS.IntratumoralHeterogeneity),
        (HER2_NS.IntratumoralHeterogeneity, HER2_NS.triggers, HER2_NS.AdditionalISHCountRequired),
        
        # Inconsistencia: D0 alto + IHC 0 → alerta
        (HER2_NS.HighD0_IHC0_Inconsistency, RDF.type, HER2_NS.ClinicalAlert),
        (HER2_NS.HighD0_IHC0_Inconsistency, RDFS.label,
         Literal("Fractal-Clinical Inconsistency: High D0 with IHC 0", lang='en')),
        (HER2_NS.HighD0_IHC0_Inconsistency, HER2_NS.requires, HER2_NS.PathologistReview),
    ]
    
    for triple in rules:
        g.add(triple)
    
    return g


def serialize_graph(g: Graph, output_path: str, format: str = 'turtle') -> None:
    """Serializa el grafo RDF al formato especificado."""
    g.serialize(destination=output_path, format=format)
    print(f"[OK] Grafo serializado en: {output_path} ({len(g)} triplas)")
```

### 5.5 Script Principal de Orquestación

```python
# main_pipeline.py

import os
from pathlib import Path
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from tqdm import tqdm
from rich.console import Console

from internal.ingestion.document_loader import load_markdown_document, DocumentChunk
from internal.agents.extraction_agent import extract_entities_and_relations
from internal.agents.resolution_agent import resolve_entity_uri
from internal.graph.rdf_builder import build_rdf_graph, add_fractal_clinical_rules, serialize_graph

load_dotenv()
console = Console()


class PipelineState(TypedDict):
    """Estado global del grafo LangGraph."""
    documents: list
    raw_extractions: list
    resolved_entities: list
    all_relations: list
    rdf_graph: object
    errors: list


# ── Nodos del pipeline ──────────────────────────────────────────────────────

def ingest_node(state: PipelineState) -> dict:
    """Carga y segmenta todos los documentos del directorio docs/."""
    docs_dir = Path(os.getenv('HER2_KG_DOCS_DIR', './docs'))
    all_chunks = []
    
    for md_file in docs_dir.glob('*.md'):
        console.print(f"[cyan]Ingesting:[/cyan] {md_file.name}")
        chunks = load_markdown_document(md_file)
        all_chunks.extend(chunks)
        console.print(f"  → {len(chunks)} chunks generados")
    
    console.print(f"\n[bold green]Total chunks:[/bold green] {len(all_chunks)}")
    return {"documents": [c.__dict__ for c in all_chunks]}


def extract_node(state: PipelineState) -> dict:
    """Extrae entidades y relaciones de cada chunk mediante LLM."""
    raw_extractions = []
    errors = list(state.get('errors', []))
    
    chunks = state['documents']
    for chunk in tqdm(chunks, desc="Extracting"):
        result = extract_entities_and_relations(
            chunk_content=chunk['content'],
            source_doc=chunk['source_doc']
        )
        result['chunk_id'] = chunk['chunk_id']
        result['section'] = chunk['section']
        
        if 'error' in result:
            errors.append(f"Chunk {chunk['chunk_id']}: {result['error']}")
        
        raw_extractions.append(result)
    
    return {"raw_extractions": raw_extractions, "errors": errors}


def resolve_node(state: PipelineState) -> dict:
    """Resuelve URIs canónicas para todas las entidades extraídas."""
    resolved_entities = []
    all_relations = []
    
    seen_labels = {}  # deduplicación por label
    
    for extraction in state['raw_extractions']:
        for entity in extraction.get('entities', []):
            label = entity.get('label', '')
            if label in seen_labels:
                # Actualizar ID local para mapeo de relaciones
                entity['resolved_uri'] = seen_labels[label]
                continue
            
            resolved_uri = resolve_entity_uri(
                label=label,
                candidate_uri=entity.get('candidate_uri', '')
            )
            entity['resolved_uri'] = resolved_uri
            seen_labels[label] = resolved_uri
            resolved_entities.append(entity)
        
        # Preservar relaciones con chunk de origen
        for rel in extraction.get('relations', []):
            rel['source_chunk'] = extraction.get('chunk_id', 'unknown')
            all_relations.append(rel)
    
    console.print(f"[bold green]Entidades únicas:[/bold green] {len(resolved_entities)}")
    console.print(f"[bold green]Relaciones totales:[/bold green] {len(all_relations)}")
    
    return {"resolved_entities": resolved_entities, "all_relations": all_relations}


def build_graph_node(state: PipelineState) -> dict:
    """Construye el grafo RDF e incorpora las reglas fractales."""
    g = build_rdf_graph(
        resolved_entities=state['resolved_entities'],
        relations=state['all_relations']
    )
    g = add_fractal_clinical_rules(g)
    console.print(f"[bold green]Triplas RDF generadas:[/bold green] {len(g)}")
    return {"rdf_graph": g}


def serialize_node(state: PipelineState) -> dict:
    """Serializa el grafo a TTL y JSON-LD."""
    output_dir = Path(os.getenv('HER2_KG_OUTPUT_DIR', './output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    g = state['rdf_graph']
    serialize_graph(g, str(output_dir / 'her2_knowledge_graph.ttl'), 'turtle')
    serialize_graph(g, str(output_dir / 'her2_knowledge_graph.jsonld'), 'json-ld')
    
    # Reporte de errores
    if state.get('errors'):
        error_file = output_dir / 'pipeline_errors.txt'
        error_file.write_text('\n'.join(state['errors']))
        console.print(f"[yellow]Errores registrados en:[/yellow] {error_file}")
    
    return {}


# ── Construcción del grafo LangGraph ────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """Construye y compila el grafo de flujo LangGraph."""
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("resolve", resolve_node)
    workflow.add_node("build_graph", build_graph_node)
    workflow.add_node("serialize", serialize_node)
    
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "extract")
    workflow.add_edge("extract", "resolve")
    workflow.add_edge("resolve", "build_graph")
    workflow.add_edge("build_graph", "serialize")
    workflow.add_edge("serialize", END)
    
    return workflow.compile()


if __name__ == "__main__":
    console.print("[bold blue]═══ Pipeline HER2 Knowledge Graph ═══[/bold blue]")
    
    pipeline = build_pipeline()
    
    initial_state: PipelineState = {
        "documents": [],
        "raw_extractions": [],
        "resolved_entities": [],
        "all_relations": [],
        "rdf_graph": None,
        "errors": []
    }
    
    final_state = pipeline.invoke(initial_state)
    
    console.print("\n[bold green]Pipeline completado exitosamente.[/bold green]")
    console.print(f"Errores: {len(final_state.get('errors', []))}")
```

---

## 6. Conversión de Algoritmos Visuales a Texto Estructurado

Los documentos fuente (especialmente D5, D6 y D7) contienen algoritmos representados como diagramas de flujo. A continuación se presentan sus representaciones textuales estructuradas, directamente utilizables por el agente de extracción.

### 6.1 Algoritmo IHC (ASCO/CAP 2023)

**Descripción**: Clasificación HER2 basada en inmunohistoquímica (IHC) del componente invasivo.

**Precondición**: Ensayo IHC validado sobre componente invasivo del carcinoma de mama. Controles de batch y controles en portaobjeto muestran tinción apropiada.

**Árbol de decisión**:

```
ENTRADA: Resultado de tinción IHC del componente invasivo

NODO-1: ¿La tinción es circunferencial completa, intensa y en >10% de células tumorales?
  → SÍ → RESULTADO: IHC 3+ (Positivo)
  → NO → continuar

NODO-2: ¿La tinción es completa, de intensidad débil a moderada, en >10% de células tumorales?
  → SÍ → RESULTADO: IHC 2+ (Equívoco) → ACCIÓN: Ordenar test de reflujo ISH (mismo espécimen) o nueva prueba (nuevo espécimen si disponible, IHC o ISH)
  → NO → continuar

NODO-3: ¿La tinción es incompleta, apenas perceptible/tenue, en >10% de células tumorales?
  → SÍ → RESULTADO: IHC 1+ (Negativo)
  → NO → continuar

NODO-4: ¿No hay tinción observable? O ¿La tinción es incompleta, tenue/apenas perceptible y en ≤10% de células tumorales?
  → SÍ → RESULTADO: IHC 0 (Negativo)
    SUB-NODO 4a: ¿Hay tinción de membrana?
      → NO → SUBCATEGORÍA: IHC 0/sin tinción de membrana (HER2-Null)
      → SÍ (tenue/incompleta ≤10%) → SUBCATEGORÍA: IHC 0+/con tinción de membrana (HER2-Ultralow)

NOTA CLÍNICA (pie de página obligatorio en reporte):
"Pacientes con cáncer de mama HER2 IHC 1+ o IHC 2+/ISH no amplificado pueden ser 
elegibles para tratamiento dirigido a niveles no amplificados/no sobreexpresados de HER2 
(trastuzumab deruxtecan). Los resultados IHC 0 no otorgan elegibilidad actualmente."

PATRONES INUSUALES (clasificar como IHC 2+ Equívoco):
- Tinción moderada a intensa pero incompleta (basolateral o lateral)
- Tinción circunferencial intensa en ≤10% de células tumorales (heterogeneidad)
- Tinción citoplasmática abundante que oscurece intensidad de membrana
```

**Relaciones KG derivadas**:
- `IHC_Score3Plus` → `implies` → `HER2_Positive`
- `IHC_Score2Plus` → `requiresReflexTest` → `ISH_DualProbe`
- `IHC_Score1Plus` → `implies` → `HER2_Negative`
- `IHC_Score1Plus` → `subsetOf` → `HER2_Low`
- `IHC_Score0Plus` → `implies` → `HER2_Negative`
- `IHC_Score0Plus` → `subsetOf` → `HER2_Ultralow`
- `IHC_Score0` → `subsetOf` → `HER2_Null`
- `HER2_Low` → `eligibleFor` → `TrastuzumabDeruxtecan` (entorno metastásico)
- `HER2_Ultralow` → `eligibleFor` → `TrastuzumabDeruxtecan` (HR+, metastásico)
- `HER2_Null` → `notEligibleFor` → `TrastuzumabDeruxtecan`

### 6.2 Algoritmos ISH Dual-Probe Grupos 1–5

**Descripción**: Evaluación de amplificación génica HER2 por hibridización in situ con sonda dual (HER2/CEP17).

**Definición de grupos**:

| Grupo | Ratio HER2/CEP17 | Señales HER2/célula | Resultado directo |
|-------|-----------------|---------------------|------------------|
| 1 | ≥ 2.0 | ≥ 4.0 | ISH Positivo |
| 2 | ≥ 2.0 | < 4.0 | Requiere workup adicional |
| 3 | < 2.0 | ≥ 6.0 | Requiere workup adicional |
| 4 | < 2.0 | ≥ 4.0 y < 6.0 | Requiere workup adicional |
| 5 | < 2.0 | < 4.0 | ISH Negativo |

**Workup para Grupo 2 (ratio ≥2.0, señales <4.0)**:

```
ACCIÓN: Evaluar IHC en secciones del mismo tejido usado para ISH.

SI IHC = 3+ → HER2 Positivo
SI IHC = 2+ → Re-recuento ISH por observador ciego (≥20 células, zona IHC 2+):
  SI re-recuento cambia categoría ISH → Adjudicar por procedimientos internos
  SI se mantiene ratio ≥2.0 y señales <4.0 → HER2 Negativo (con comentario-A)
SI IHC = 0 o 1+ → HER2 Negativo (con comentario-A)

COMENTARIO-A: Evidencia limitada sobre eficacia de terapia anti-HER2 en esta minoría de 
casos. Si IHC no es 3+, se recomienda considerar el espécimen como HER2 negativo.
```

**Workup para Grupo 3 (ratio <2.0, señales ≥6.0)**:

```
ACCIÓN: Evaluar IHC en secciones del mismo tejido usado para ISH.

SI IHC = 3+ → HER2 Positivo
SI IHC = 2+ → Re-recuento ISH por observador ciego (≥20 células):
  SI se mantiene ratio <2.0 y señales ≥6.0 → HER2 Positivo
  SI cambia categoría → Adjudicar por procedimientos internos
SI IHC = 0 o 1+ → HER2 Negativo (con comentario-B)

COMENTARIO-B: Datos insuficientes sobre eficacia anti-HER2 con ratio <2.0 sin 
sobreexpresión proteíca. Con IHC 0 o 1+ concurrent, considerar HER2 negativo.
```

**Workup para Grupo 4 (ratio <2.0, señales ≥4.0 y <6.0)**:

```
ACCIÓN: Evaluar IHC en secciones del mismo tejido usado para ISH.

SI IHC = 3+ → HER2 Positivo
SI IHC = 2+ → Re-recuento ISH por observador ciego (≥20 células):
  SI se mantiene ratio <2.0 y señales ≥4.0 y <6.0 → HER2 Negativo (con comentario-C)
  SI cambia categoría → Adjudicar por procedimientos internos
SI IHC = 0 o 1+ → HER2 Negativo (con comentario-C)

COMENTARIO-C: Incierto si pacientes con ≥4.0 y <6.0 señales/célula y ratio <2.0 se 
benefician de terapia anti-HER2 sin sobreexpresión (IHC 3+). Si resultado cerca del 
umbral, alta probabilidad de resultado diferente al repetir. Cuando IHC no es 3+, 
considerar HER2 negativo sin pruebas adicionales en el mismo espécimen.
```

**Criterios de rechazo ISH**:
- Controles no son como se esperaba
- No se pueden identificar al menos dos áreas de tumor invasivo
- >25% de señales no son calificables por señales débiles
- >10% de señales ocurren sobre citoplasma
- Resolución nuclear deficiente
- Autofluorescencia intensa
- En todos los casos anteriores: reportar como **Indeterminado**

### 6.3 Algoritmo de Scoring Completo (Rakha 2026)

**Descripción**: Sistema de scoring HER2 IHC comprehensivo integrando intensidad, circunferencialidad y porcentaje (basado en Figura 2 de Rakha et al., 2026).

```
DIMENSIONES DE EVALUACIÓN:
  (A) INTENSIDAD: Fuerte | Moderada | Débil | Tenue
  (B) CIRCUNFERENCIALIDAD: Completa | Incompleta
  (C) PORCENTAJE: >10% | ≤10% de células tumorales invasivas

TABLA DE DECISIÓN (Intensidad × Circunferencialidad × Porcentaje → Score/Categoría):

  Fuerte + Completa + >10%   → Score 3+ → HER2-Positivo
  Fuerte + Completa + ≤10%   → Score 2+ → HER2-Equívoco
  Fuerte + Incompleta + >10% → Score 2+ → HER2-Equívoco
  Fuerte + Incompleta + ≤10% → Score 1+ → HER2-Low

  Moderada + Completa + >10%   → Score 2+ → HER2-Equívoco
  Moderada + Completa + ≤10%   → Score 2+ → HER2-Equívoco
  Moderada + Incompleta + >10% → Score 2+ → HER2-Equívoco
  Moderada + Incompleta + ≤10% → Score 1+ → HER2-Low

  Débil + Completa + >10%   → Score 2+ → HER2-Equívoco
  Débil + Completa + ≤10%   → Score 1+ → HER2-Low
  Débil + Incompleta + >10% → Score 1+ → HER2-Low
  Débil + Incompleta + ≤10% → Score 0+ → HER2-Ultralow

  Tenue + Completa + >10%   → Score 1+ → HER2-Low
  Tenue + Completa + ≤10%   → Score 0+ → HER2-Ultralow
  Tenue + Incompleta + >10% → Score 1+ → HER2-Low
  Tenue + Incompleta + ≤10% → Score 0+ → HER2-Ultralow

  Sin tinción de membrana    → Score 0  → HER2-Null

REGLA DE MAGNIFICACIÓN (práctica):
  Tinción visible a objetivo ×10 → intensidad débil
  Visible a ×20 → intensidad débil a moderada
  Visible a ×40 (apenas perceptible) → intensidad tenue
  Visible a bajo aumento (coincide con control 3+) → intensidad fuerte/moderada fuerte

REGLA DE TUMOR HETEROGÉNEO:
  SI hay distintas zonas con diferentes intensidades:
    → El score final se asigna según la zona de mayor score
    → Comentar el % de células por cada nivel de score
```

---

## 7. Consideraciones de Diseño y Limitaciones

**Sobre la arquitectura LangGraph.** LangGraph permite implementar ciclos de retroalimentación y revisión humana en el pipeline. Para casos de producción, se recomienda añadir un nodo `human_review` accionado cuando la confianza media de extracción sea inferior a 0.75, o cuando el agente de resolución no encuentre URI canónica para más del 30% de las entidades de un chunk.

**Sobre la calidad del LLM como extractor.** Los LLM son extractores no deterministas. Las variaciones entre llamadas pueden generar inconsistencias en la nomenclatura de entidades. Se recomienda: (1) usar temperatura cero (`temperature=0`) en todas las llamadas de extracción; (2) implementar caché de resultados (p. ej., con `langchain.cache`); (3) ejecutar al menos dos pasadas y resolver inconsistencias por votación mayoritaria.

**Sobre las categorías emergentes HER2-low y ultralow.** Las guías son explícitas en señalar que la distinción entre IHC 0 (null) y IHC 0+ (ultralow) es el punto de mayor variabilidad interobservador en patología humana, con tasas de discordancia del 35–40% en estudios multicéntricos (Rakha et al., 2026; Wu et al., 2025). El KG debe representar esta incertidumbre inherente mediante predicados de confianza cuantificada y notas de ambigüedad.

**Sobre la integración fractal.** Las equivalencias fractal-clínicas del documento D1 son propuestas exploratorias del equipo DigPatho, no recomendaciones de guías clínicas validadas. En el KG, estas relaciones deben marcarse con el predicado `her2:proposedEquivalence` (en lugar de `owl:equivalentClass`) y anotarse con la fuente `DigPatho_Internal_2025`. El pipeline de validación debe tratar estas relaciones como hipótesis a confirmar, no como verdades definitivas.

**Sobre el uso de ontologías externas.** Los códigos NCIt utilizados en este documento son estimaciones razonadas basadas en el rango de IDs asignados a conceptos HER2 en el NCI Thesaurus. Antes de publicar o federar el grafo, se debe verificar cada URI contra la versión vigente de NCIt (https://ncit.nci.nih.gov) y SNOMED CT.

**Sobre escalabilidad.** Para volúmenes de extracción superiores a 500 chunks, se recomienda paralelizar las llamadas al LLM usando `asyncio` y el cliente asíncrono de Anthropic (`anthropic.AsyncAnthropic`), respetando los límites de tasa (*rate limits*) de la API.

**Sobre regulación y uso clínico.** Este sistema es de investigación y desarrollo. Cualquier uso en el entorno clínico debe someterse a los procesos de validación definidos por IEC 62304 (software de dispositivos médicos), ISO 13485 (sistemas de gestión de calidad) y las directrices FDA/EMA para Software as a Medical Device (SaMD).

---

## 8. Referencias

Bardia, A., Hu, X., Dent, R., et al. (2024). Trastuzumab deruxtecan after endocrine therapy in metastatic breast cancer. *New England Journal of Medicine*, *391*(22), 2110–2122. https://doi.org/10.1056/NEJMoa2407086

College of American Pathologists. (2025, marzo). *Reporting template for reporting results of biomarker testing of specimens from patients with carcinoma of the breast* (Version 1.6.0.0). CAP. https://documents.cap.org/protocols/cp-breastbiomarker-22-rf.pdf

Modi, S., Jacot, W., Yamashita, T., et al. (2022). Trastuzumab deruxtecan in previously treated HER2-low advanced breast cancer. *New England Journal of Medicine*, *387*(1), 9–20. https://doi.org/10.1056/NEJMoa2203690

Rakha, E. A., Tan, P. H., Van Bockstal, M. R., et al. (2026). International expert consensus recommendations for HER2 reporting in breast cancer: Focus on HER2-low and ultralow categories. *Modern Pathology*, *39*, 100925. https://doi.org/10.1016/j.modpat.2025.100925

Tarantino, P., Viale, G., Press, M. F., et al. (2023). ESMO expert consensus statements on the definition, diagnosis, and management of HER2-low breast cancer. *Annals of Oncology*, *34*(8), 645–659. https://doi.org/10.1016/j.annonc.2023.05.008

Wolff, A. C., Somerfield, M. R., Dowsett, M., et al. (2023). Human epidermal growth factor receptor 2 testing in breast cancer: ASCO-CAP guideline update. *Archives of Pathology & Laboratory Medicine*, *147*(9), 993–1000. https://doi.org/10.5858/arpa.2023-0950-SA

Wu, S., Shang, J., Li, Z., et al. (2025). Interobserver consistency and diagnostic challenges in HER2-ultralow breast cancer: A multicenter study. *ESMO Open*, *10*, 104127. https://doi.org/10.1016/j.esmoop.2024.104127

---
Ver ANEXOS A: Ontologias  y B: Guias

---
*Fin del documento*

*Este documento fue generado como guía de implementación para el pipeline DigPatho HER2-KG. Versión 1.0 – Abril 2026.*
