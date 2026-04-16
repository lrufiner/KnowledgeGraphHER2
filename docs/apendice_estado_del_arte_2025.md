# Apéndice: Estado del Arte 2024–2025
## Grafos de Conocimiento + LLMs — Lo que la literatura reciente agrega

> **Nota de contexto.** Este apéndice complementa el tutorial principal con cuatro líneas de investigación activas que no estaban cubiertas o quedaban superficialmente tratadas. Las referencias corresponden a papers publicados entre octubre de 2024 y principios de 2026, provenientes de arXiv, NeurIPS, ACL, ICLR y EMNLP.

---

## A.1 Grafos de conocimiento temporales (Temporal Knowledge Graphs)

### El problema que los KGs estáticos no pueden resolver

El tutorial original modela los grafos de conocimiento como estructuras **estáticas**: un hecho, una vez almacenado, permanece válido indefinidamente. Pero el mundo cambia. El hecho `(Obama, presidente_de, USA)` fue verdadero entre 2009 y 2017; ya no lo es. El hecho `(Barcelona FC, entrenador, Flick)` tiene una fecha de inicio. Los grafos estáticos no pueden representar ni razonar sobre esta dimensión temporal.

Los **Temporal Knowledge Graphs (TKGs)** resuelven esto extendiendo el triple clásico a una **cuádrupla**:

```
Triple estático:   (sujeto,   relación,        objeto)
Cuádrupla TKG:     (sujeto,   relación,        objeto,    timestamp)

Ejemplos:
  (Obama,    presidente_de,   USA,        [2009-01-20, 2017-01-20])
  (Biden,    presidente_de,   USA,        [2021-01-20, presente])
  (Messi,    juega_en,        Inter Miami, [2023-07-21, presente])
```

Esto permite razonamiento **predictivo** (¿quién será presidente en 2026?) y **retrospectivo** (¿quién era el CEO de Apple en 2010?), y es crítico para evitar un problema que los sistemas RAG clásicos no manejan bien: **dos hechos semánticamente similares pero temporalmente excluyentes** pueden coexistir en la base vectorial y ser recuperados juntos, generando contradicciones en la respuesta del LLM.

### Sistemas recientes

**EvoReasoner** (arXiv 2025) combina un grafo evolutivo (`EvoKG`) con un módulo de razonamiento temporal. El resultado más llamativo del paper: un modelo pequeño `LLaMA 3.1-8B` entrenado con datos hasta diciembre de 2023 mejoró su accuracy de **18.6% → 37.0%** simplemente al ser fundamentado en un TKG actualizado —igualando prácticamente al `DeepSeek-V3` de 671B parámetros (38.3%). Esto ilustra que la **actualización del grafo puede sustituir parcialmente el re-entrenamiento del modelo**.

**MemoTime** (arXiv 2025) introduce un framework plug-and-play que funciona como memoria externa temporal para cualquier LLM. Su pipeline `plan → retrieve → answer` descompone preguntas complejas en sub-tareas, recupera hechos temporalmente relevantes del TKG, y genera la respuesta. Logra mejoras de hasta **+24%** sobre baselines fuertes, y permite que modelos pequeños como `Qwen3-4B` alcancen rendimiento comparable a `GPT-4-Turbo`.

**TKG-Thinker** (arXiv feb. 2026) introduce un agente basado en reinforcement learning que interactúa dinámicamente con el TKG mediante un loop `think → action → observation`, con verificación interna de consistencia temporal. Resuelve un problema crítico de los pipelines estáticos: la falta de mecanismos para detectar y corregir evidencia que viola restricciones temporales.

### Representación en Cypher (Neo4j)

Neo4j no tiene soporte nativo para TKGs, pero el patrón estándar es agregar propiedades temporales a los nodos de relación:

```cypher
// Relación con validez temporal
MATCH (obama:Persona {nombre: 'Obama'}),
      (usa:Pais {nombre: 'USA'})
CREATE (obama)-[:PRESIDENTE_DE {
  desde: date('2009-01-20'),
  hasta:  date('2017-01-20'),
  activo: false
}]->(usa)

// Consulta temporal: ¿quién era presidente en 2012?
MATCH (p:Persona)-[r:PRESIDENTE_DE]->(pais:Pais {nombre: 'USA'})
WHERE r.desde <= date('2012-06-01') AND r.hasta >= date('2012-06-01')
RETURN p.nombre, r.desde, r.hasta
```

---

## A.2 Grafos de conocimiento multimodales (MMKGs)

### Más allá del texto

El tutorial describe la extracción de conocimiento **exclusivamente desde texto**. Pero una proporción enorme del conocimiento humano está codificado en imágenes, diagramas, audio y video. Los **Multimodal Knowledge Graphs (MMKGs)** integran estas modalidades en una estructura unificada.

```
MMKG típico:

  [Persona: "Frida Kahlo"]
       |
       |---[:RETRATADA_EN]---> [Imagen: autorretrato_1940.jpg]
       |                         {modalidad: "visual", confianza: 0.97}
       |---[:NACIDA_EN]------> [Lugar: "Coyoacán, México"]
       |---[:PINTÓ]----------> [Obra: "Las dos Fridas"]
                                   |
                                   |---[:TIENE_IMAGEN]---> [imagen.jpg]
```

### Sistemas recientes

**MR-MKG** (ACL 2024) propone un modelo de atención multimodal sobre grafos de conocimiento que conecta entidades textuales con representaciones visuales. El resultado clave: logra rendimiento superior entrenando solo **~2.25% de los parámetros del LLM** (el resto se congela), lo que lo hace muy eficiente para fine-tuning en dominios específicos.

**GraphVis** (NeurIPS 2024) identifica un problema fundamental en cómo los sistemas anteriores conectan grafos con modelos de visión-lenguaje: serializar el grafo a texto plano (como hace Graph-RAG de Microsoft) **destruye la estructura topológica**. GraphVis preserva la topología del grafo convirtiéndola en una representación visual y pasándola directamente a un Vision-Language Model (VLM), manteniendo las relaciones espaciales y jerárquicas.

**VaLiK** (2025) construye MMKGs sin anotación manual. Usa Vision-Language Models preentrenados en cascada para traducir características visuales a forma textual, seguido de un módulo de verificación cross-modal que filtra ruido, logrando enlace entidad-imagen completamente automático.

### Diagrama arquitectural de un MMKG pipeline

```
  Texto:  "Frida Kahlo pintó Las dos Fridas en 1939"
  Imagen:  [foto de la obra]
  Audio:   [entrevista con la artista]
                    │
                    ▼
         ┌─────────────────────┐
         │  Encoders           │
         │  - BERT/RoBERTa     │  ← texto
         │  - ViT/CLIP         │  ← imágenes
         │  - Wav2Vec          │  ← audio
         └────────┬────────────┘
                  │  embeddings multi-modales
                  ▼
         ┌─────────────────────┐
         │  Alineación         │
         │  cross-modal        │  ← contrastive learning
         │  (verificación)     │
         └────────┬────────────┘
                  │
                  ▼
         ┌─────────────────────┐
         │  MMKG               │
         │  nodos + relaciones │
         │  con modalidad      │
         └─────────────────────┘
```

---

## A.3 El ecosistema Graph-RAG: más allá de Microsoft

### El tutorial describe una sola técnica; hay toda una familia

El tutorial cubre **GraphRAG de Microsoft** (Edge et al., 2024) como si fuera la única variante de Graph-RAG. En realidad, en los 18 meses posteriores a ese paper surgió un ecosistema completo de variantes con trade-offs distintos. La siguiente tabla resume las principales:

| Sistema | Año | Idea clave | Ventaja principal |
|---|---|---|---|
| **Microsoft GraphRAG** | abr. 2024 | Comunidades jerárquicas (Leiden) + resúmenes | Consultas globales sobre corpus grandes |
| **LightRAG** | oct. 2024 | Índice dual (entidades + temas globales) + actualización incremental | 10x menos tokens, actualizable sin reconstruir |
| **HippoRAG** | 2024 | Personalized PageRank inspirado en memoria episódica hipocampal | Multi-hop en un solo paso de búsqueda |
| **PathRAG** | 2025 | Recuperar solo caminos relacionales, no subgrafos completos | Mayor precisión en preguntas de razonamiento |
| **GRAG** | 2024 | Poda suave de entidades irrelevantes + graph-aware prompt tuning | Reduce ruido en subgrafos recuperados |
| **HiRAG** | 2025 | Jerarquía de conocimiento con clustering semántico | Puente entre conocimiento local y global |

### LightRAG en detalle

**LightRAG** (Guo et al., HKUST, oct. 2024) es probablemente la variante más adoptada en producción. Sus dos innovaciones centrales:

**1. Recuperación de doble nivel (dual-level retrieval):**

```
Consulta del usuario
       │
       ├──► Nivel bajo (específico)
       │         Busca entidades concretas en el grafo
       │         Ejemplo: "¿Quién escribió Orgullo y Prejuicio?"
       │         → nodo :Persona {nombre: "Jane Austen"}
       │
       └──► Nivel alto (conceptual)
                 Busca temas y relaciones globales
                 Ejemplo: "tendencias en literatura romántica"
                 → clusters temáticos, relaciones entre géneros
```

**2. Actualización incremental** sin reconstruir el índice completo. Microsoft GraphRAG reconstruye toda la estructura de comunidades al agregar datos nuevos (costo ≈ `1399 comunidades × 2 × 5000 tokens`). LightRAG integra nuevas entidades en el grafo existente sin reconstrucción, reduciendo el overhead a una fracción.

```python
# Uso básico de LightRAG (API simplificada)
from lightrag import LightRAG, QueryParam

rag = LightRAG(working_dir="./mi_rag")

# Indexar documentos (construye el grafo automáticamente con el LLM)
with open("documentos.txt") as f:
    rag.insert(f.read())

# Consulta con recuperación híbrida
respuesta = rag.query(
    "¿Cuál es la relación entre Marcelo y la casa verde?",
    param=QueryParam(mode="hybrid")  # local + global
)

# Agregar nuevos datos sin reconstruir
rag.insert("Marcelo vendió la casa en 2025.")
```

### HippoRAG: memoria inspirada en el hipocampo

**HippoRAG** modela la recuperación sobre el grafo análogamente a cómo el hipocampo humano consolida la memoria episódica. Usa **Personalized PageRank (PPR)** para propagar relevancia desde las entidades identificadas en la pregunta hacia el resto del grafo, lo que permite multi-hop en un solo paso (sin necesidad de múltiples rondas de recuperación).

```
Pregunta: "¿Quién es el jefe del hermano de Marcelo?"

PPR desde nodo "Marcelo":
  Marcelo (seed, peso 1.0)
    → hermano_de → Carlos (peso 0.6)
       → trabaja_en → EmpresaX (peso 0.4)
          → CEO_de → Roberto (peso 0.3)   ← respuesta

Un solo recorrido PPR alcanza la respuesta multi-hop.
```

---

## A.4 Construcción automática de KGs con LLMs: el ciclo que se cierra

### El pipeline NLP clásico tiene sus límites

El tutorial describe un pipeline tradicional para construir grafos desde texto:

```
texto → tokenización → NER → dependency parsing → relation extraction → grafo
```

Este pipeline depende de modelos especializados por tarea (uno para NER, otro para parsing, otro para extracción de relaciones) entrenados sobre corpus anotados. Los errores se propagan en cascada y el sistema no generaliza bien a dominios nuevos sin datos de entrenamiento específicos.

La literatura de 2024–2025 documenta un cambio de paradigma: los **propios LLMs se convirtieron en el extractor principal**, gracias a su capacidad de zero-shot y few-shot generalization.

### AutoSchemaKG: escala industrial sin anotación humana

**AutoSchemaKG** (HKUST, 2025) construyó un grafo con **900 millones de nodos y 5.9 mil millones de aristas** procesando 50 millones de documentos, con **95% de alineación semántica** respecto a esquemas creados manualmente — sin intervención humana. El truco: usar el LLM para descubrir el esquema inductivamente desde los datos, en lugar de definirlo a priori.

### KGGEN: extracción iterativa con LLMs

```
Texto de entrada
      │
      ▼
  ┌─────────────────────────────────┐
  │  Invocación 1 al LLM            │
  │  Prompt: "Extrae todas las      │
  │  entidades de este texto"       │
  │  Output: [Marcelo, Casa1,       │
  │           Familia, verde, 7]    │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  Invocación 2 al LLM            │
  │  Prompt: "Dado este par de      │
  │  entidades (Marcelo, Casa1),    │
  │  ¿cuál es la relación?"         │
  │  Output: ES_DUENO_DE            │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  Agrupamiento semántico         │
  │  (deduplicación de entidades)   │
  │  "Casa" ≈ "casa1" ≈ "vivienda"  │
  └─────────────────────────────────┘
               │
               ▼
          GRAFO FINAL
```

### El ciclo virtuoso: razonamiento mejora construcción

El survey más completo sobre el tema (arXiv oct. 2025) identifica un patrón emergente que no existía en la era del NLP clásico: un **ciclo de mejora mutua** entre construcción del grafo y razonamiento sobre él.

```
  ┌─────────────────────────────────────────────────────┐
  │                   CICLO VIRTUOSO                    │
  │                                                     │
  │   Texto / datos del mundo                           │
  │          │                                          │
  │          ▼                                          │
  │   LLM extrae entidades y relaciones                 │
  │          │                                          │
  │          ▼                                          │
  │   Grafo de conocimiento crece                       │
  │          │                                          │
  │          ▼                                          │
  │   Graph-RAG mejora las respuestas del LLM           │
  │          │                                          │
  │          ▼                                          │
  │   LLM con mejor razonamiento extrae KG más preciso  │
  │          │                                          │
  │          └────────────────────► (vuelve al inicio)  │
  └─────────────────────────────────────────────────────┘
```

El grafo deja de ser una estructura construida una sola vez para convertirse en una **memoria dinámica que crece y se refina** con cada ciclo de inferencia del LLM.

### KG como memoria persistente de agentes

Una implicación práctica de este ciclo es el uso de grafos como **memoria persistente para agentes LLM**. Sistemas como **A-MEM** modelan la memoria del agente como notas interconectadas en un grafo, con metadatos contextuales que permiten reorganización continua. **Zep** usa específicamente un TKG para gestionar la validez temporal de los hechos recordados por el agente — algo que las memorias basadas en embeddings vectoriales no pueden hacer de forma estructurada.

```python
# Patrón conceptual: agente con memoria en grafo
class AgenteConMemoria:
    def __init__(self, neo4j_driver):
        self.grafo = neo4j_driver
        self.llm = LLM()

    def recordar(self, hecho, timestamp=None):
        """Guarda un hecho nuevo en el grafo de memoria"""
        triple = self.llm.extraer_triple(hecho)
        self.grafo.merge_triple(*triple, timestamp=timestamp)

    def responder(self, pregunta):
        """Recupera contexto del grafo antes de responder"""
        entidades = self.llm.extraer_entidades(pregunta)
        subgrafo = self.grafo.subgrafo_local(entidades, depth=2)
        contexto = self.serializar(subgrafo)
        return self.llm.generar(pregunta, contexto=contexto)
```

---

## Mapa de la literatura: dónde encaja cada novedad

```
                    GRAFOS DE CONOCIMIENTO + LLMs
                              │
          ┌───────────────────┼────────────────────┐
          │                   │                    │
     Temporales          Multimodales         Construcción
      (TKGs)               (MMKGs)            automática
          │                   │                    │
   EvoReasoner           MR-MKG (ACL'24)     AutoSchemaKG
   MemoTime (2025)       GraphVis (NeurIPS)   KGGEN (2025)
   TKG-Thinker (2026)    VaLiK (2025)         Ciclo virtuoso
   Cuádruplas (s,r,o,t)  CLIP + KG            A-MEM / Zep
          │
          └─────────────────────────────────────────────┐
                                                        │
                          Graph-RAG ecosystem
                                │
               ┌────────────────┼────────────────────┐
               │                │                    │
       Microsoft          LightRAG             HippoRAG
       GraphRAG            (oct 2024)           (2024/25)
       (abr 2024)          Dual-level           PageRank
       Comunidades         incremental          hipocampal
       Leiden              10x eficiente        multi-hop 1-shot
               │
       PathRAG, GRAG, HiRAG (2025)
```

---

## Referencias de esta sección

- Guo, Z. et al. (2024). **LightRAG: Simple and Fast Retrieval-Augmented Generation**. arXiv:2410.05779.
- Gutiérrez, B.J. et al. (2024). **HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models**. NeurIPS 2024.
- Edge, D. et al. (2024). **From Local to Global: A Graph RAG Approach to Query-Focused Summarization**. arXiv:2404.16130.
- [EvoReasoner] (2025). **Temporal Reasoning over Evolving Knowledge Graphs**. arXiv:2509.15464.
- [MemoTime] (2025). **Memory-Augmented Temporal Knowledge Graph Enhanced LLM Reasoning**. arXiv:2510.13614.
- [TKG-Thinker] (2026). **Towards Dynamic Reasoning over Temporal KGs via Agentic RL**. arXiv:2602.05818.
- [LLM-empowered KG construction survey] (2025). arXiv:2510.20345.
- Besta, M. et al. (2025). **When to use Graphs in RAG**. arXiv:2506.05690.
- [MR-MKG] (2024). ACL 2024.
- Su, Y. et al. (2024). **Temporal Knowledge Graph Question Answering: A Survey**. arXiv:2406.14191.
- [Large Language Models Meet KGs for QA] (2025). arXiv:2505.20099.

---

*Última actualización: marzo 2026. Este apéndice cubre el estado del arte hasta principios de 2026; el campo evoluciona rápidamente y se recomienda consultar arXiv (cs.AI, cs.IR) para novedades posteriores.*
