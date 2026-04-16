# Anexo A – Ontología HER2 con Métricas Fractales

## A.1 Descripción General

La presente ontología integra el vocabulario médico estándar para el estado HER2 en cáncer de mama con un vocabulario fractal propuesto por el equipo DigPatho. Está diseñada para ser serializada en OWL/RDF y consultada mediante SPARQL.

**Namespaces**:

- `her2:` → `http://digpatho.com`
- `frac:` → `http://digpatho.com`
- Externas: `ncit:`, `snomed:`, `loinc:`

## A.2 Jerarquía de Clases

```
owl:Thing
├── her2:BreastCancerSpecimen
│   ├── her2:InvasiveComponent
│   └── her2:InSituComponent
│
├── her2:HER2Status
│   ├── her2:HER2_Positive          (NCIt:C68748)
│   ├── her2:HER2_Negative
│   │   ├── her2:HER2_Low           (NCIt:C173791)  [IHC 1+ OR 2+/ISH-]
│   │   ├── her2:HER2_Ultralow      [IHC 0+]
│   │   └── her2:HER2_Null          [IHC 0, sin tinción]
│   └── her2:HER2_Equivocal         [IHC 2+, pendiente ISH]
│
├── her2:IHCScore
│   ├── her2:Score3Plus             → her2:HER2_Positive
│   ├── her2:Score2Plus             → her2:HER2_Equivocal
│   ├── her2:Score1Plus             → her2:HER2_Low
│   ├── her2:Score0Plus             → her2:HER2_Ultralow
│   └── her2:Score0                 → her2:HER2_Null
│
├── her2:ISHResult
│   ├── her2:ISH_Group1_Amplified
│   ├── her2:ISH_Group2_LowCopy
│   ├── her2:ISH_Group3_HighCopyLowRatio
│   ├── her2:ISH_Group4_IntermediateCopy
│   └── her2:ISH_Group5_NotAmplified
│
├── her2:StainingPattern
│   ├── her2:Intensity
│   │   ├── her2:StrongIntensity
│   │   ├── her2:ModerateIntensity
│   │   ├── her2:WeakIntensity
│   │   └── her2:FaintIntensity
│   ├── her2:Circumferentiality
│   │   ├── her2:CompleteCircumferential
│   │   └── her2:IncompleteCircumferential
│   └── her2:PercentageStained
│       ├── her2:MoreThan10Percent
│       └── her2:AtMost10Percent
│
├── her2:TherapeuticAgent
│   ├── her2:Trastuzumab            (NCIt:C1647)
│   ├── her2:TrastuzumabDeruxtecan  (NCIt:C155379)
│   ├── her2:Pertuzumab             (NCIt:C64636)
│   └── her2:TDM1                   (NCIt:C82492)
│
├── her2:ClinicalTrial
│   ├── her2:DESTINY_Breast04
│   ├── her2:DESTINY_Breast06
│   └── her2:DAISY_Trial
│
├── her2:Guideline
│   ├── her2:ASCO_CAP_2023
│   ├── her2:CAP_Biomarker_2025
│   ├── her2:ESMO_2023
│   └── her2:Rakha_International_2026
│
├── her2:QualityMeasure
│   ├── her2:FixationRequirement    [NBF 10%, 6-72h]
│   ├── her2:SectionAgeRequirement  [<6 semanas]
│   ├── her2:ControlRequirement     [rango completo 0, 1+, 2+, 3+]
│   └── her2:EQAParticipation
│
└── frac:FractalMetric              (NCIt:C25730)
    ├── frac:CapacityDimension_D0   [alta → arquitectura compleja]
    ├── frac:InformationDimension_D1[alta → heterogeneidad multifractal]
    ├── frac:Lacunarity             [baja → tejido denso; alta → poroso]
    ├── frac:MultifractalSpread_Da  [alta → variabilidad espacial]
    └── frac:MultiscaleEntropy      [alta → complejidad temporal]
```

## A.3 Propiedades de Objeto (Object Properties)

| Propiedad                      | Dominio                  | Rango                    | Descripción                        |
| ------------------------------ | ------------------------ | ------------------------ | ----------------------------------- |
| `her2:hasIHCScore`           | Specimen                 | IHCScore                 | Score IHC asignado                  |
| `her2:hasISHResult`          | Specimen                 | ISHResult                | Resultado ISH                       |
| `her2:hasStatus`             | Specimen                 | HER2Status               | Estado HER2 final                   |
| `her2:implies`               | IHCScore                 | HER2Status               | Implicación diagnóstica directa   |
| `her2:requiresReflexTest`    | IHCScore                 | her2:Assay               | Test de reflujo requerido           |
| `her2:eligibleFor`           | HER2Status               | TherapeuticAgent         | Elegibilidad terapéutica           |
| `her2:notEligibleFor`        | HER2Status               | TherapeuticAgent         | Inelegibilidad terapéutica         |
| `her2:definedIn`             | HER2Status               | ClinicalTrial            | Definición de ensayo               |
| `her2:regulatedBy`           | TherapeuticAgent         | her2:RegulatoryBody      | Aprobación regulatoria             |
| `her2:hasQualityRequirement` | her2:Assay               | QualityMeasure           | Requisito de calidad                |
| `frac:represents`            | FractalMetric            | her2:PathologicalFeature | Representación fractal→clínico   |
| `frac:associatedWith`        | FractalMetric            | HER2Status               | Asociación estadística            |
| `frac:proposedEquivalence`   | FractalMetric            | HER2Status               | Equivalencia propuesta (hipótesis) |
| `frac:inconsistentWith`      | FractalMetric + IHCScore | her2:ClinicalAlert       | Alerta de inconsistencia            |

## A.4 Propiedades de Datos (Data Properties)

| Propiedad                      | Tipo      | Descripción                    |
| ------------------------------ | --------- | ------------------------------- |
| `her2:ratioHER2CEP17`        | xsd:float | Ratio HER2/CEP17 medido         |
| `her2:avgHER2SignalsPerCell` | xsd:float | Señales HER2 promedio/célula  |
| `her2:percentageStained`     | xsd:float | % células con tinción         |
| `her2:confidence`            | xsd:float | Confianza de extracción (0–1) |
| `frac:valueD0`               | xsd:float | Valor numérico de D0           |
| `frac:valueD1`               | xsd:float | Valor numérico de D1           |
| `frac:valueLacunarity`       | xsd:float | Valor numérico de lacunaridad  |
| `frac:valueDeltaAlpha`       | xsd:float | Rango del espectro multifractal |

## A.5 Mapeo Fractal ↔ Clínico (tabla de equivalencias propuestas)

| Métrica Fractal                | Valor Umbral | Hallazgo Clínico Asociado                 | Correlato IHC/ISH           | Referencia   |
| ------------------------------- | ------------ | ------------------------------------------ | --------------------------- | ------------ |
| D0 (dimensión de capacidad)    | > 1.85       | Arquitectura glandular compleja            | IHC 2+ o 3+ (probable)      | D1 (interno) |
| D0                              | < 1.55       | Arquitectura simple, tejido homogéneo     | IHC 0 o 1+ (probable)       | D1 (interno) |
| Lacunaridad                     | < 0.15       | Tejido denso, gránulos fusionados         | IHC 3+ (tinción compacta)  | D1 (interno) |
| Lacunaridad                     | > 0.60       | Tejido poroso, espacios vacuolares         | IHC 0 o artefactos          | D1 (interno) |
| Δα (spread multifractal)      | > 0.80       | Alta variabilidad espacial multiescala     | Heterogeneidad intratumoral | D1 (interno) |
| Entropía multiscala            | > 1.50       | Complejidad temporal alta                  | Tumor de alto grado         | D1 (interno) |
| D1 (dimensión de información) | > 1.70       | Distribución heterogénea de información | ISH grupos 2–4 (posible)   | D1 (interno) |

**NOTA IMPORTANTE**: Todas las equivalencias de esta tabla son **propuestas de investigación** del equipo DigPatho, no recomendaciones clínicas validadas. En el grafo RDF se representan con `frac:proposedEquivalence` y la fuente `DigPatho_Internal_2025`.

## A.6 Axiomas OWL Clave (Turtle)

```turtle
# Clase IHC Score 3+ implica HER2 Positivo (OWL restriction)
her2:Score3Plus rdfs:subClassOf [
    owl:onProperty her2:implies ;
    owl:hasValue her2:HER2_Positive
] .

# HER2-Low es subclase de HER2-Negative
her2:HER2_Low rdfs:subClassOf her2:HER2_Negative .
her2:HER2_Ultralow rdfs:subClassOf her2:HER2_Negative .
her2:HER2_Null rdfs:subClassOf her2:HER2_Negative .

# Disyunción: Positive y Negative son disjuntos
her2:HER2_Positive owl:disjointWith her2:HER2_Negative .

# Elegibilidad T-DXd: Low y Ultralow son elegibles
her2:HER2_Low rdfs:subClassOf [
    owl:onProperty her2:eligibleFor ;
    owl:hasValue her2:TrastuzumabDeruxtecan
] .
her2:HER2_Ultralow rdfs:subClassOf [
    owl:onProperty her2:eligibleFor ;
    owl:hasValue her2:TrastuzumabDeruxtecan
] .

# HER2-Null NO es elegible
her2:HER2_Null rdfs:subClassOf [
    owl:onProperty her2:notEligibleFor ;
    owl:hasValue her2:TrastuzumabDeruxtecan
] .
```
