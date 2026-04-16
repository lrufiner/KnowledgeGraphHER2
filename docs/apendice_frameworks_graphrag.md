# Apéndice: Frameworks de Graph-RAG y Construcción de KGs — Guía de código completa
## Microsoft GraphRAG · LightRAG · HippoRAG · PathRAG · neo4j-graphrag · Integración con LangChain/LangGraph

> **Alcance.** Este apéndice cubre los cinco frameworks más relevantes para Graph-RAG y construcción automática de KGs con LLMs, con ejemplos de código real basados en sus repositorios oficiales. La sección final muestra cómo integrar cada uno con el ecosistema LangChain/LangGraph descrito en el apéndice anterior.

---

## Mapa de decisión: ¿qué framework usar?

```
¿Cuál es tu necesidad principal?
              │
    ┌─────────┼──────────────────┬──────────────────┐
    │         │                  │                  │
    ▼         ▼                  ▼                  ▼
Construir   Consultas         Multi-hop         Rutas entre
KG desde    globales sobre    semántico         entidades
texto       corpus grande     (1 paso)          (razonamiento)
    │         │                  │                  │
    ▼         ▼                  ▼                  ▼
neo4j-    Microsoft          HippoRAG           PathRAG
graphrag   GraphRAG          (NeurIPS'24)       (Feb 2025)
(oficial)  (Microsoft)       PageRank           Flow-pruning
    │         │
    ▼         ▼
LightRAG  → Si además necesitas actualización incremental
            y consultas locales + globales en un solo sistema
```

---

## B.1 Microsoft GraphRAG

### Qué es

GraphRAG es un pipeline de transformación de datos diseñado para extraer información significativa y estructurada desde texto no estructurado usando el poder de los LLMs. Su característica distintiva es la **detección de comunidades** sobre el grafo construido (algoritmo de Leiden), que permite responder preguntas globales sobre un corpus entero — algo imposible para RAG vectorial.

### Arquitectura interna

```
Corpus de texto
      │
      ▼
[Text chunking]
      │
      ▼
[Entity & Relation Extraction] ← LLM (GPT-4o-mini por defecto)
      │
      ▼
[Graph construction]
      │
      ▼
[Community detection] ← algoritmo de Leiden
      │
      ▼
[Community summaries] ← LLM resume cada comunidad
      │
      ├──► Local search:  responde sobre entidades específicas
      └──► Global search: responde sobre temas del corpus entero
```

### Instalación

```bash
pip install graphrag

# Crear workspace
mkdir mi_proyecto && cd mi_proyecto
graphrag init        # crea .env y settings.yaml
# Editar .env: agregar GRAPHRAG_API_KEY=tu_openai_key

mkdir -p input
# Copiar tus archivos .txt en input/
```

### settings.yaml mínimo

```yaml
# settings.yaml
models:
  default_chat_model:
    type: openai_chat
    model: gpt-4o-mini
    api_key: ${GRAPHRAG_API_KEY}
  default_embedding_model:
    type: openai_embedding
    model: text-embedding-3-small
    api_key: ${GRAPHRAG_API_KEY}

chunks:
  size: 1200
  overlap: 100

entity_extraction:
  entity_types: [organization, person, geo, event, concept]
```

### Indexación

```bash
# Indexar el corpus (ATENCIÓN: puede costar varios dólares con corpus grandes)
graphrag index --root .

# El índice se guarda en: output/<timestamp>/artifacts/*.parquet
# Incluye: entities, relationships, communities, community_reports, text_units
```

### Consultas desde CLI

```bash
# Global search: preguntas sobre temas del corpus entero
graphrag query \
  --root . \
  --method global \
  "¿Cuáles son los temas principales en este corpus?"

# Local search: preguntas sobre entidades específicas
graphrag query \
  --root . \
  --method local \
  "¿Qué relaciones tiene Marcelo con TechCorp?"
```

### Consultas desde Python API

```python
import asyncio
import pandas as pd
from pathlib import Path

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_relationships,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch

# ── Configuración ─────────────────────────────────────────────────
INPUT_DIR  = "./output/latest/artifacts"
COMMUNITY_LEVEL = 2   # nivel de granularidad de las comunidades

# ── Cargar artefactos del índice ──────────────────────────────────
entity_df      = pd.read_parquet(f"{INPUT_DIR}/create_final_nodes.parquet")
entity_emb_df  = pd.read_parquet(f"{INPUT_DIR}/create_final_entities.parquet")
reports_df     = pd.read_parquet(f"{INPUT_DIR}/create_final_community_reports.parquet")
text_units_df  = pd.read_parquet(f"{INPUT_DIR}/create_final_text_units.parquet")
relationships_df = pd.read_parquet(f"{INPUT_DIR}/create_final_relationships.parquet")
communities_df = pd.read_parquet(f"{INPUT_DIR}/create_final_communities.parquet")

entities      = read_indexer_entities(entity_df, entity_emb_df, COMMUNITY_LEVEL)
reports       = read_indexer_reports(reports_df, entity_df, COMMUNITY_LEVEL)
text_units    = read_indexer_text_units(text_units_df)
relationships = read_indexer_relationships(relationships_df)
communities   = read_indexer_communities(communities_df, entity_df, COMMUNITY_LEVEL)

# ── LLM ──────────────────────────────────────────────────────────
llm = ChatOpenAI(
    api_key="tu_openai_key",
    model="gpt-4o-mini",
    api_type=OpenaiApiType.OpenAI,
    max_retries=20,
)

# ── Global Search ─────────────────────────────────────────────────
context_builder = GlobalCommunityContext(
    community_reports=reports,
    communities=communities,
    entities=entities,
    token_encoder=None,
)

search_engine = GlobalSearch(
    llm=llm,
    context_builder=context_builder,
    token_encoder=None,
    max_data_tokens=12_000,
    map_llm_params={"max_tokens": 1000, "temperature": 0.0},
    reduce_llm_params={"max_tokens": 2000, "temperature": 0.0},
    allow_general_knowledge=False,
    response_type="multiple paragraphs",
)

async def buscar_global(pregunta: str):
    result = await search_engine.asearch(pregunta)
    print(f"\n=== RESPUESTA ===\n{result.response}")
    print(f"\n=== COMUNIDADES USADAS: {len(result.context_data['reports'])} ===")
    return result

asyncio.run(buscar_global("¿Cuáles son los temas principales del corpus?"))
```

### Neo4j como backend de GraphRAG (ms-graphrag-neo4j)

```bash
pip install ms-graphrag-neo4j
```

```python
import os
from ms_graphrag_neo4j import MsGraphRAG
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "tu_password")
)

ms_graph = MsGraphRAG(driver=driver, model="gpt-4o-mini")

# 1. Extraer entidades y relaciones
textos = [
    "Marcelo trabaja en TechCorp como ingeniero.",
    "TechCorp fue fundada en Buenos Aires en 2010.",
    "Ana García es la CEO de TechCorp desde 2015.",
]
result = ms_graph.extract_nodes_and_rels(
    textos,
    allowed_entities=["Person", "Organization", "Location"]
)

# 2. Generar resúmenes de nodos y relaciones
ms_graph.summarize_nodes_and_rels()

# 3. Detectar y resumir comunidades (usa GDS de Neo4j)
ms_graph.identify_and_summarize_communities()

# 4. Consulta global (busca sobre resúmenes de comunidades)
respuesta = ms_graph.global_query("¿Qué relaciones laborales existen?")
print(respuesta)
```

---

## B.2 LightRAG (HKUDS)

### Qué es

LightRAG incorpora estructuras de grafos en los procesos de indexación y recuperación de texto. Este framework emplea un sistema de recuperación de doble nivel que mejora la recuperación de información comprensiva tanto a nivel bajo como a nivel alto.

### Ventaja clave sobre GraphRAG

LightRAG permite **actualización incremental**: agregar nuevos documentos sin reconstruir el índice completo. GraphRAG reconstruye todas las comunidades al añadir datos nuevos; LightRAG integra las nuevas entidades en el grafo existente.

### Instalación

```bash
pip install lightrag-hku

# Para soporte de múltiples formatos (PDF, DOCX, etc.)
pip install lightrag-hku[api]
```

### Uso básico con OpenAI

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

os.environ["OPENAI_API_KEY"] = "tu_api_key"

WORKING_DIR = "./mi_lightrag"

async def main():
    # ── Inicializar LightRAG ──────────────────────────────────────
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,      # función de embedding
        llm_model_func=gpt_4o_mini_complete,  # LLM para extracción
    )

    await rag.initialize_storages()
    await initialize_pipeline_status(rag)

    # ── Insertar documentos ───────────────────────────────────────
    texto = """
    La casa de Marcelo está pintada de verde y tiene siete pisos.
    Marcelo trabaja como ingeniero en TechCorp desde 2018.
    TechCorp fue fundada en Buenos Aires por Ana García en 2010.
    Ana García también es mentora de Marcelo.
    """
    await rag.ainsert(texto)  # versión asíncrona

    # ── Cuatro modos de consulta ──────────────────────────────────

    # 1. Naive: búsqueda por similitud vectorial simple
    r1 = await rag.aquery(
        "¿Dónde vive Marcelo?",
        param=QueryParam(mode="naive")
    )

    # 2. Local: busca entidades y sus vecinos directos
    r2 = await rag.aquery(
        "¿Qué hace Marcelo en TechCorp?",
        param=QueryParam(mode="local")
    )

    # 3. Global: busca temas y conceptos globales del corpus
    r3 = await rag.aquery(
        "¿Cuáles son los temas principales del corpus?",
        param=QueryParam(mode="global")
    )

    # 4. Hybrid: combina local + global (recomendado para producción)
    r4 = await rag.aquery(
        "¿Cómo se relacionan Marcelo y Ana García?",
        param=QueryParam(mode="hybrid")
    )

    for nombre, r in [("naive", r1), ("local", r2), ("global", r3), ("hybrid", r4)]:
        print(f"\n=== {nombre.upper()} ===\n{r}")

asyncio.run(main())
```

### LightRAG con Neo4j como backend de grafos

```python
# LightRAG puede usar Neo4j para almacenar el grafo en lugar de archivos locales
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.neo4j_impl import Neo4JStorage

os.environ["OPENAI_API_KEY"]  = "tu_openai_key"
os.environ["NEO4J_URI"]       = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"]  = "neo4j"
os.environ["NEO4J_PASSWORD"]  = "tu_password"

rag = LightRAG(
    working_dir="./lightrag_neo4j",
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_mini_complete,
    graph_storage="Neo4JStorage",   # ← usa Neo4j en lugar de networkx local
    vector_storage="Neo4JStorage",  # ← almacena vectores también en Neo4j
)
```

### LightRAG con Ollama (local, sin costo de API)

```python
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

rag = LightRAG(
    working_dir="./lightrag_local",
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.2:latest",       # modelo local
    embedding_func=ollama_embed,
    embedding_model_name="nomic-embed-text", # embedding local
)

# Insertar y consultar igual que con OpenAI
with open("mi_documento.txt") as f:
    rag.insert(f.read())

respuesta = rag.query(
    "¿Cuál es el tema principal?",
    param=QueryParam(mode="hybrid")
)
print(respuesta)
```

### Actualización incremental (ventaja clave de LightRAG)

```python
# Agregar nuevos documentos sin reconstruir el índice completo
nuevos_textos = [
    "Marcelo fue promovido a Tech Lead en TechCorp en 2024.",
    "TechCorp abrió una oficina en Santiago de Chile en 2025.",
]

for texto in nuevos_textos:
    rag.insert(texto)
    # Solo se actualizan las entidades y relaciones nuevas
    # El grafo existente no se reconstruye

# Verificar el grafo actualizado en Neo4j:
# MATCH (n:__Entity__) RETURN n.entity_id, n.description LIMIT 10
```

---

## B.3 HippoRAG (OSU-NLP)

### Qué es

HippoRAG orquesta sinérgicamente LLMs, grafos de conocimiento y el algoritmo de Personalized PageRank para imitar los diferentes roles del neocórtex y el hipocampo en la memoria humana. Su ventaja principal es el **multi-hop en un solo paso de recuperación**, gracias a PPR.

### Cómo funciona el Personalized PageRank

```
Pregunta: "¿En qué país vive Marcelo?"

1. Identificar nodos semilla en el grafo:
   query → embedding → nodos más similares: [Marcelo, casa, Buenos Aires]

2. PPR desde los nodos semilla:
   Marcelo (1.0)
     → VIVE_EN → Casa (0.6)
     → TRABAJA_EN → TechCorp (0.5)
                      → UBICADA_EN → Buenos Aires (0.4)
                                       → CAPITAL_DE → Argentina (0.3)

3. Los nodos con mayor PageRank personalizado
   son recuperados como contexto → respuesta: "Argentina"

Sin iteraciones. Un solo recorrido PPR alcanza la respuesta multi-hop.
```

### Instalación

```bash
conda create -n hipporag python=3.10
conda activate hipporag
pip install hipporag

export OPENAI_API_KEY="tu_openai_key"
```

### Uso básico

```python
from hipporag import HippoRAG

# ── Inicializar ───────────────────────────────────────────────────
hipporag = HippoRAG(
    save_dir="./hipporag_output",
    llm_model_name="gpt-4o-mini",             # para extracción de triples
    embedding_model_name="nvidia/NV-Embed-v2", # para embeddings de entidades
)

# ── Indexar documentos ────────────────────────────────────────────
# HippoRAG extrae triples (sujeto, predicado, objeto) de cada documento
# y construye un grafo de conocimiento esquema-libre
docs = [
    "Marcelo es un ingeniero de software.",
    "Marcelo trabaja en TechCorp, que está ubicada en Buenos Aires.",
    "Buenos Aires es la capital de Argentina.",
    "Ana García fundó TechCorp en 2010.",
    "Ana García es mentora de Marcelo.",
]

hipporag.index(docs=docs)

# ── Consultas multi-hop ───────────────────────────────────────────

# Pregunta de 1 hop (directa)
q1 = "¿Dónde trabaja Marcelo?"

# Pregunta de 2 hops: Marcelo → TechCorp → Buenos Aires
q2 = "¿En qué ciudad trabaja Marcelo?"

# Pregunta de 3 hops: Marcelo → TechCorp → Buenos Aires → Argentina
q3 = "¿En qué país trabaja Marcelo?"

# Pregunta sobre red de relaciones
q4 = "¿Quién fundó la empresa donde trabaja el mentoreado de Ana García?"

queries = [q1, q2, q3, q4]
resultados = hipporag.rag_qa(queries=queries)

for q, res in zip(queries, resultados):
    print(f"\nPregunta: {q}")
    print(f"Respuesta: {res['answer']}")
    print(f"Documentos recuperados: {[d['passage'] for d in res['retrieved_passages']]}")
```

### Inspeccionar el grafo construido

```python
import networkx as nx

# HippoRAG almacena el grafo como NetworkX
grafo = hipporag.get_knowledge_graph()  # retorna nx.Graph

print(f"Nodos: {grafo.number_of_nodes()}")
print(f"Aristas: {grafo.number_of_edges()}")

# Explorar vecinos de Marcelo
vecinos = list(grafo.neighbors("Marcelo"))
print(f"Vecinos de Marcelo: {vecinos}")

# Ver el camino entre dos nodos
camino = nx.shortest_path(grafo, "Marcelo", "Argentina")
print(f"Camino Marcelo → Argentina: {camino}")
# ['Marcelo', 'TechCorp', 'Buenos Aires', 'Argentina']
```

### HippoRAG con modelos locales (vLLM)

```python
# Para usar modelos locales sin costo de API
# Primero levantar vLLM: vllm serve meta-llama/Llama-3.1-8B-Instruct --port 6578

hipporag_local = HippoRAG(
    save_dir="./hipporag_local",
    llm_model_name="meta-llama/Llama-3.1-8B-Instruct",
    embedding_model_name="nvidia/NV-Embed-v2",
    llm_base_url="http://localhost:6578/v1",  # vLLM local
)
```

---

## B.4 PathRAG (BUPT-GAMMA)

### Qué es

PathRAG recupera caminos relacionales clave del grafo de indexación y los convierte a forma textual para el prompting de LLMs. PathRAG reduce efectivamente la información redundante con poda basada en flujo, mientras guía a los LLMs para generar respuestas más lógicas y coherentes con prompting basado en caminos.

### Por qué caminos y no subgrafos

```
LightRAG recupera:                PathRAG recupera:
  ego-network completo              solo caminos relevantes

  Marcelo ── TechCorp               Marcelo
    │           │                     └──[TRABAJA_EN]──► TechCorp
    │         Ana                           └──[FUNDADA_POR]──► Ana
    │           │
    └── casa ── Buenos Aires     Sin ruido. Solo la ruta necesaria
         │                       para responder la pregunta.
         └── 7 pisos
```

### Instalación

```bash
pip install git+https://github.com/BUPT-GAMMA/PathRAG.git
# O clonar el repositorio
git clone https://github.com/BUPT-GAMMA/PathRAG.git
cd PathRAG
pip install -e .
```

### Uso básico

```python
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete

os.environ["OPENAI_API_KEY"]  = "tu_api_key"
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

WORKING_DIR = "./pathrag_output"
os.makedirs(WORKING_DIR, exist_ok=True)

# ── Inicializar ───────────────────────────────────────────────────
rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
)

# ── Indexar ───────────────────────────────────────────────────────
with open("mi_corpus.txt") as f:
    rag.insert(f.read())

# ── Consultar ─────────────────────────────────────────────────────
# Modo hybrid: combina recuperación de caminos locales y globales
respuesta = rag.query(
    "¿Cómo se relacionan Marcelo y Ana García?",
    param=QueryParam(mode="hybrid")
)
print(respuesta)
```

### PathRAG con modelos Hugging Face

```python
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import hf_model_complete, hf_embed
from transformers import AutoModel, AutoTokenizer

# Cargar modelos locales
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model     = AutoModel.from_pretrained(LLM_MODEL)

rag = PathRAG(
    working_dir="./pathrag_local",
    llm_model_func=hf_model_complete,
    llm_model_name=LLM_MODEL,
    embedding_func=hf_embed,
    embedding_model_name=EMBEDDING_MODEL,
)
```

### Entender la poda por flujo (flow-based pruning)

PathRAG asigna un **score de confiabilidad** a cada camino basado en:
1. Longitud del camino (caminos más cortos = más confiables)
2. Fuerza semántica de las relaciones en el camino
3. Relevancia de los nodos terminales con la consulta

```python
# PathRAG construye un grafo de flujo donde el "caudal"
# representa la confiabilidad de cada camino

# Camino A: Marcelo → TechCorp → Ana (score: 0.85)
# Camino B: Marcelo → casa → Buenos Aires → Ana (score: 0.32, más largo, menos relevante)

# Solo el camino A se incluye en el prompt → menos ruido
# El prompt enviado al LLM se ve así:
"""
Relational paths:
Path 1 (reliability: 0.85):
  Marcelo -[TRABAJA_EN]-> TechCorp -[FUNDADA_POR]-> Ana García

Please answer the question based on these paths: 
"¿Cómo se relacionan Marcelo y Ana García?"
"""
```

---

## B.5 neo4j-graphrag (paquete oficial de Neo4j)

### Qué es

El paquete Python de GraphRAG de Neo4j permite crear grafos de conocimiento e implementar fácilmente patrones de recuperación que usan combinaciones de recorridos de grafo, generación de consultas con text2Cypher, búsqueda vectorial y full-text.

Este paquete es la solución de **primera parte** de Neo4j para Graph-RAG — con mantenimiento garantizado a largo plazo y soporte para múltiples LLMs (OpenAI, Anthropic, Google VertexAI, Ollama, Azure OpenAI).

### Instalación

```bash
# Base
pip install neo4j-graphrag

# Con soporte específico de modelo
pip install "neo4j-graphrag[openai]"      # OpenAI
pip install "neo4j-graphrag[anthropic]"   # Anthropic Claude
pip install "neo4j-graphrag[google-vertexai]"  # Google
pip install "neo4j-graphrag[ollama]"      # Ollama local

# Con NLP (spaCy para resolución de entidades)
pip install "neo4j-graphrag[nlp]"
```

### Pipeline completo: texto → KG → RAG

```python
import asyncio
import neo4j
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter
)
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG

# ── Credenciales ──────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "tu_password"

driver  = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
llm     = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
embedder = OpenAIEmbeddings(model="text-embedding-3-small")

# ── PASO 1: Definir el schema del KG ─────────────────────────────
node_types = ["Persona", "Empresa", "Ciudad", "Concepto"]
rel_types  = ["TRABAJA_EN", "FUNDADA_EN", "UBICADA_EN", "MENTOREADO_POR"]
patterns   = [
    ("Persona", "TRABAJA_EN",    "Empresa"),
    ("Persona", "MENTOREADO_POR","Persona"),
    ("Empresa", "FUNDADA_EN",    "Ciudad"),
    ("Empresa", "UBICADA_EN",    "Ciudad"),
]

# ── PASO 2: Construir el KG desde texto ───────────────────────────
kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    text_splitter=FixedSizeSplitter(chunk_size=500, chunk_overlap=100),
    schema={
        "node_types": node_types,
        "relationship_types": rel_types,
        "patterns": patterns,
    },
    from_pdf=False,  # True si quieres procesar PDFs directamente
)

# Textos de ejemplo
textos = [
    "Marcelo es un ingeniero de software que trabaja en TechCorp desde 2018.",
    "TechCorp fue fundada en Buenos Aires por Ana García en 2010.",
    "Ana García es mentora de Marcelo y CEO de TechCorp.",
    "Buenos Aires es la capital de Argentina y tiene 15 millones de habitantes.",
]

async def construir_kg():
    for texto in textos:
        result = await kg_builder.run_async(text=texto)
        print(f"Procesado: {result}")

asyncio.run(construir_kg())

# ── PASO 3: Crear índice vectorial ────────────────────────────────
create_vector_index(
    driver,
    name="text_embeddings",
    label="Chunk",
    embedding_property="embedding",
    dimensions=1536,   # dimensiones de text-embedding-3-small
    similarity_fn="cosine",
)

# ── PASO 4a: RAG con VectorRetriever (búsqueda semántica) ─────────
vector_retriever = VectorRetriever(
    driver=driver,
    index_name="text_embeddings",
    embedder=embedder,
    return_properties=["text", "id"],
)

rag_vectorial = GraphRAG(
    retriever=vector_retriever,
    llm=llm,
)

respuesta = rag_vectorial.search(
    query_text="¿Dónde trabaja Marcelo?",
    retriever_config={"top_k": 3}
)
print(f"\nVector RAG: {respuesta.answer}")

# ── PASO 4b: RAG con VectorCypherRetriever (vector + grafo) ───────
# Este retriever primero encuentra chunks similares, luego expande
# el contexto recorriendo el grafo desde las entidades en esos chunks
cypher_query = """
MATCH (chunk:Chunk)-[:FROM_CHUNK]->(entity)
OPTIONAL MATCH (entity)-[r]->(related)
RETURN chunk.text AS chunk_text, 
       entity.id  AS entidad,
       type(r)    AS relacion, 
       related.id AS entidad_relacionada
"""

cypher_retriever = VectorCypherRetriever(
    driver=driver,
    index_name="text_embeddings",
    embedder=embedder,
    retrieval_query=cypher_query,
)

rag_hibrido = GraphRAG(
    retriever=cypher_retriever,
    llm=llm,
)

respuesta_hibrida = rag_hibrido.search(
    query_text="¿Cuál es la relación entre Marcelo y Ana García?",
    retriever_config={"top_k": 5}
)
print(f"\nHybrid RAG: {respuesta_hibrida.answer}")
```

### Usar Claude (Anthropic) como LLM

```python
from neo4j_graphrag.llm import AnthropicLLM

llm_claude = AnthropicLLM(
    model_name="claude-sonnet-4-20250514",
    model_params={"temperature": 0, "max_tokens": 2000},
)

rag_claude = GraphRAG(retriever=vector_retriever, llm=llm_claude)
respuesta = rag_claude.search("¿Quién fundó TechCorp?")
print(respuesta.answer)
```

### Usar Ollama (100% local)

```python
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.embeddings import OllamaEmbeddings

llm_local     = OllamaLLM(model_name="llama3.2")
embed_local   = OllamaEmbeddings(model="nomic-embed-text")

rag_local = GraphRAG(retriever=VectorRetriever(driver, "text_embeddings", embed_local),
                     llm=llm_local)
```

---

## B.6 Integración con LangChain y LangGraph

Esta sección muestra cómo los cinco frameworks anteriores se conectan con el ecosistema LangChain/LangGraph descrito en el apéndice anterior.

### Mapa de integración

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PUNTO DE INTEGRACIÓN                              │
│                                                                      │
│  Framework externo          Capa de integración      LangChain/LangGraph
│                                                                      │
│  Microsoft GraphRAG    ──►  GraphRAG como @tool  ──►  LangGraph agent│
│  LightRAG              ──►  LightRAG como @tool  ──►  LangGraph agent│
│  HippoRAG              ──►  HippoRAG como retriever ► LCEL chain     │
│  PathRAG               ──►  PathRAG como @tool   ──►  LangGraph agent│
│  neo4j-graphrag        ──►  Custom retriever     ──►  LangChain RAG  │
│                                                                      │
│  Patrón universal: envolver en @tool o BaseRetriever                 │
└──────────────────────────────────────────────────────────────────────┘
```

### Patrón 1: LightRAG como tool en LangGraph

```python
import asyncio
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from typing import TypedDict, Annotated
from operator import add

# ── Inicializar LightRAG ──────────────────────────────────────────
rag_instance = LightRAG(
    working_dir="./lightrag_langraph",
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embed,
)

# ── Envolver LightRAG como herramienta de LangChain ───────────────
@tool
def lightrag_local(query: str) -> str:
    """
    Busca información específica sobre entidades y sus relaciones directas
    en el grafo de conocimiento. Úsala para preguntas factuales concretas.
    Args:
        query: Pregunta sobre una entidad o relación específica
    """
    return rag_instance.query(query, param=QueryParam(mode="local"))

@tool
def lightrag_global(query: str) -> str:
    """
    Busca información sobre temas generales o patrones del corpus completo.
    Úsala para preguntas de alto nivel o resúmenes temáticos.
    Args:
        query: Pregunta sobre temas generales del corpus
    """
    return rag_instance.query(query, param=QueryParam(mode="global"))

@tool
def lightrag_hybrid(query: str) -> str:
    """
    Combinación de búsqueda local y global. Úsala cuando no estés seguro
    de qué tipo de información necesitas, o para preguntas complejas.
    Args:
        query: Pregunta que puede requerir tanto contexto local como global
    """
    return rag_instance.query(query, param=QueryParam(mode="hybrid"))

# ── Crear agente ReAct con las tres herramientas ──────────────────
llm   = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [lightrag_local, lightrag_global, lightrag_hybrid]

agente_lightrag = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier="""Eres un asistente que responde preguntas usando un grafo de conocimiento
    con LightRAG. Elige la herramienta apropiada según el tipo de pregunta:
    - lightrag_local para preguntas sobre entidades específicas
    - lightrag_global para preguntas sobre temas generales
    - lightrag_hybrid cuando no estés seguro
    Siempre explica qué herramienta usaste y por qué."""
)

# ── Invocar el agente ─────────────────────────────────────────────
from langchain_core.messages import HumanMessage

resultado = agente_lightrag.invoke({
    "messages": [HumanMessage(content="¿Cómo se relacionan Marcelo y Ana García?")]
})

for msg in resultado["messages"]:
    print(f"[{msg.type}] {msg.content}\n")
```

### Patrón 2: HippoRAG como retriever de LangChain

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from hipporag import HippoRAG
from typing import List

# ── Wrapper: HippoRAG como BaseRetriever de LangChain ────────────
class HippoRAGRetriever(BaseRetriever):
    """
    Wrapper que convierte HippoRAG en un retriever compatible con LangChain.
    Permite usar HippoRAG en cadenas LCEL y agentes de LangGraph.
    """
    hipporag: HippoRAG
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # HippoRAG retorna una lista de passages con scores de PageRank
        results = self.hipporag.retrieve(queries=[query], top_k=self.top_k)

        docs = []
        for passage_info in results[0]:
            docs.append(Document(
                page_content=passage_info["passage"],
                metadata={
                    "score":  passage_info.get("score", 0),
                    "source": passage_info.get("source", "unknown"),
                }
            ))
        return docs

# ── Instanciar el retriever ───────────────────────────────────────
hippo = HippoRAG(
    save_dir="./hipporag_output",
    llm_model_name="gpt-4o-mini",
    embedding_model_name="nvidia/NV-Embed-v2",
)

retriever = HippoRAGRetriever(hipporag=hippo, top_k=4)

# ── Construir cadena RAG con LCEL ─────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Responde la pregunta basándote únicamente en el siguiente contexto recuperado
por HippoRAG (con razonamiento multi-hop via Personalized PageRank):

{context}

Pregunta: {question}

Respuesta:
""")

def formatear_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Score: {d.metadata['score']:.3f}]\n{d.page_content}"
        for d in docs
    )

# Cadena LCEL completa
cadena_hipporag = (
    {"context": retriever | formatear_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

respuesta = cadena_hipporag.invoke("¿En qué país trabaja Marcelo?")
print(respuesta)
```

### Patrón 3: Agente LangGraph que elige entre múltiples frameworks

Este es el patrón más potente: un agente LangGraph que selecciona dinámicamente el framework Graph-RAG más apropiado para cada pregunta.

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ── Herramientas de cada framework ────────────────────────────────

@tool
def buscar_graphrag_global(query: str) -> str:
    """
    Usa Microsoft GraphRAG para responder preguntas sobre TEMAS GLOBALES
    del corpus. Ideal para: resúmenes temáticos, análisis de patrones,
    preguntas sobre el corpus entero. NO usar para preguntas específicas
    sobre entidades individuales.
    """
    # En producción: invocar search_engine.asearch(query)
    return f"[GraphRAG Global] Resultado para: {query}"

@tool
def buscar_lightrag_hibrido(query: str) -> str:
    """
    Usa LightRAG con recuperación híbrida para preguntas que combinan
    contexto local y global. Ideal para: preguntas sobre relaciones entre
    entidades que también tienen contexto temático más amplio.
    También permite actualización incremental del índice.
    """
    return rag_instance.query(query, param=QueryParam(mode="hybrid"))

@tool
def buscar_hipporag_multihop(query: str) -> str:
    """
    Usa HippoRAG con Personalized PageRank para RAZONAMIENTO MULTI-HOP.
    Ideal cuando la respuesta requiere encadenar 2 o más relaciones:
    'A trabaja en B, B está en C, entonces A está en C'.
    Más eficiente que GraphRAG para multi-hop (10-30x más rápido).
    """
    results = hippo.rag_qa(queries=[query])
    return results[0]["answer"] if results else "Sin resultado"

@tool
def buscar_pathrag_relacional(query: str) -> str:
    """
    Usa PathRAG para recuperar CAMINOS RELACIONALES entre entidades.
    Ideal para: preguntas de razonamiento lógico, trazabilidad de
    relaciones, preguntas tipo '¿cómo llegamos de A a B?'.
    Produce respuestas más precisas con menos ruido que otros métodos.
    """
    return path_rag.query(query, param=QueryParam(mode="hybrid"))

@tool
def buscar_neo4j_cypher(query: str) -> str:
    """
    Usa neo4j-graphrag con VectorCypherRetriever para búsqueda híbrida
    (vectorial + recorrido de grafo). Ideal para: preguntas sobre
    propiedades específicas, filtros, agregaciones o cuando necesitas
    trazabilidad completa a los documentos fuente.
    """
    result = rag_hibrido.search(query, retriever_config={"top_k": 5})
    return result.answer

# ── Estado del agente ─────────────────────────────────────────────
class MultiFrameworkState(TypedDict):
    pregunta:     str
    herramienta:  str   # cuál framework eligió el agente
    respuestas:   Annotated[list[str], add]
    respuesta_final: str

# ── Nodos del grafo ───────────────────────────────────────────────

def clasificar_pregunta(state: MultiFrameworkState) -> dict:
    """El LLM elige el framework más apropiado para la pregunta."""
    llm_clasificador = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
    Clasifica la siguiente pregunta y elige el framework RAG más apropiado:

    - graphrag_global:   preguntas sobre temas globales, resúmenes del corpus
    - lightrag_hibrido:  preguntas relacionales con contexto mixto
    - hipporag_multihop: preguntas que requieren 2+ saltos de razonamiento
    - pathrag_relacional: preguntas sobre cómo se conectan dos entidades
    - neo4j_cypher:      preguntas con filtros, propiedades, o que necesitan trazabilidad

    Pregunta: {state['pregunta']}
    Responde SOLO con el nombre del framework (una de las 5 opciones).
    """
    herramienta = llm_clasificador.invoke(prompt).content.strip()
    return {"herramienta": herramienta}

def ejecutar_framework(state: MultiFrameworkState) -> dict:
    """Ejecuta el framework seleccionado."""
    herramienta = state["herramienta"]
    pregunta    = state["pregunta"]

    dispatch = {
        "graphrag_global":    buscar_graphrag_global,
        "lightrag_hibrido":   buscar_lightrag_hibrido,
        "hipporag_multihop":  buscar_hipporag_multihop,
        "pathrag_relacional": buscar_pathrag_relacional,
        "neo4j_cypher":       buscar_neo4j_cypher,
    }

    fn = dispatch.get(herramienta, buscar_lightrag_hibrido)
    resultado = fn.invoke({"query": pregunta})
    return {"respuestas": [f"[{herramienta}]: {resultado}"]}

def sintetizar(state: MultiFrameworkState) -> dict:
    """Genera la respuesta final."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = f"""
    Pregunta original: {state['pregunta']}
    Framework usado: {state['herramienta']}
    Resultado: {state['respuestas'][-1]}

    Genera una respuesta clara y concisa para el usuario.
    """
    respuesta = llm.invoke(prompt).content
    return {"respuesta_final": respuesta}

# ── Construir el grafo ────────────────────────────────────────────
workflow = StateGraph(MultiFrameworkState)

workflow.add_node("clasificar",  clasificar_pregunta)
workflow.add_node("ejecutar",    ejecutar_framework)
workflow.add_node("sintetizar",  sintetizar)

workflow.add_edge(START,         "clasificar")
workflow.add_edge("clasificar",  "ejecutar")
workflow.add_edge("ejecutar",    "sintetizar")
workflow.add_edge("sintetizar",  END)

app_multi_framework = workflow.compile()

# ── Prueba con diferentes tipos de preguntas ──────────────────────
preguntas = [
    "¿Cuáles son los temas principales del corpus?",        # → graphrag_global
    "¿En qué país trabaja Marcelo?",                         # → hipporag_multihop
    "¿Cómo se conectan Marcelo y Buenos Aires?",             # → pathrag_relacional
    "¿Cuántas empresas hay en Buenos Aires?",                # → neo4j_cypher
]

for pregunta in preguntas:
    result = app_multi_framework.invoke({"pregunta": pregunta, "respuestas": []})
    print(f"\n[{result['herramienta']}] {pregunta}")
    print(f"→ {result['respuesta_final']}")
```

### Patrón 4: Ingesta unificada con LangGraph + neo4j-graphrag

Un pipeline LangGraph que orquesta la construcción del KG y los índices vectoriales desde un conjunto de documentos, con validación en cada paso.

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_vector_index
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio

class IngestState(TypedDict):
    archivos:       list[str]          # rutas a los PDFs
    chunks:         Annotated[list, add]
    kg_results:     Annotated[list, add]
    indices_creados: list[str]
    errores:        Annotated[list, add]
    completado:     bool

def cargar_documentos(state: IngestState) -> dict:
    """Carga y divide los documentos en chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150
    )
    todos_los_chunks = []
    errores = []

    for archivo in state["archivos"]:
        try:
            loader = PyMuPDFLoader(archivo)
            docs   = loader.load()
            chunks = splitter.split_documents(docs)
            todos_los_chunks.extend(chunks)
            print(f"✓ Cargado: {archivo} ({len(chunks)} chunks)")
        except Exception as e:
            errores.append(f"Error en {archivo}: {str(e)}")

    return {"chunks": todos_los_chunks, "errores": errores}

async def _construir_kg_async(chunks, kg_builder):
    """Función auxiliar async para construir el KG."""
    results = []
    for chunk in chunks:
        result = await kg_builder.run_async(text=chunk.page_content)
        results.append(str(result))
    return results

def construir_kg(state: IngestState) -> dict:
    """Construye el grafo de conocimiento desde los chunks."""
    results = asyncio.run(
        _construir_kg_async(state["chunks"], kg_builder)
    )
    print(f"✓ KG construido: {len(results)} chunks procesados")
    return {"kg_results": results}

def crear_indices(state: IngestState) -> dict:
    """Crea los índices vectoriales necesarios."""
    indices = []
    try:
        create_vector_index(
            driver,
            name="text_embeddings",
            label="Chunk",
            embedding_property="embedding",
            dimensions=1536,
            similarity_fn="cosine",
        )
        indices.append("text_embeddings")
        print("✓ Índice vectorial creado: text_embeddings")
    except Exception as e:
        # El índice ya puede existir
        print(f"ℹ Índice ya existente o error: {e}")
        indices.append("text_embeddings (existente)")

    return {"indices_creados": indices, "completado": True}

# Construir el workflow de ingesta
ingest_workflow = StateGraph(IngestState)

ingest_workflow.add_node("cargar",  cargar_documentos)
ingest_workflow.add_node("kg",      construir_kg)
ingest_workflow.add_node("indices", crear_indices)

ingest_workflow.add_edge(START,    "cargar")
ingest_workflow.add_edge("cargar", "kg")
ingest_workflow.add_edge("kg",     "indices")
ingest_workflow.add_edge("indices", END)

pipeline_ingesta = ingest_workflow.compile()

# Ejecutar
resultado = pipeline_ingesta.invoke({
    "archivos":   ["documento1.pdf", "documento2.pdf"],
    "chunks":     [],
    "kg_results": [],
    "errores":    [],
    "indices_creados": [],
    "completado": False,
})

print(f"\n=== RESUMEN DE INGESTA ===")
print(f"Chunks procesados: {len(resultado['chunks'])}")
print(f"Índices creados:   {resultado['indices_creados']}")
print(f"Errores:           {resultado['errores']}")
```

---

## Tabla comparativa final

| Característica | MS GraphRAG | LightRAG | HippoRAG | PathRAG | neo4j-graphrag |
|---|---|---|---|---|---|
| **PyPI** | `graphrag` | `lightrag-hku` | `hipporag` | GitHub | `neo4j-graphrag` |
| **Backend de grafo** | Parquet / Neo4j | NetworkX / Neo4j / PG | NetworkX | NetworkX | Neo4j (nativo) |
| **Actualización incremental** | ❌ reconstruye | ✅ nativo | ✅ nativo | ✅ nativo | ✅ MERGE en Neo4j |
| **Multi-hop** | ✅ (via comunidades) | ✅ parcial | ✅ nativo (PPR) | ✅ (via caminos) | ✅ (via Cypher) |
| **Consultas globales** | ✅ excelente | ✅ bueno | ❌ no nativo | ❌ no nativo | ✅ (via Cypher) |
| **Costo computacional** | Alto (reconstruye todo) | Bajo (incremental) | Medio | Bajo-Medio | Según Cypher |
| **Integración LangChain** | via `@tool` | via `@tool` nativo | via `BaseRetriever` | via `@tool` | nativo (`langchain-neo4j`) |
| **Integración LangGraph** | `@tool` en nodo | `@tool` en nodo | retriever en nodo | `@tool` en nodo | `GraphCypherQAChain` |
| **Modelos locales** | Sí (Ollama) | Sí (Ollama/HuggingFace) | Sí (vLLM/HuggingFace) | Sí (HuggingFace) | Sí (Ollama) |
| **UI incluida** | ❌ | ✅ Web UI | ❌ | ❌ | ❌ |
| **Repo oficial** | microsoft/graphrag | HKUDS/LightRAG | OSU-NLP-Group/HippoRAG | BUPT-GAMMA/PathRAG | neo4j/neo4j-graphrag-python |

---

## Referencias

- Microsoft GraphRAG: https://github.com/microsoft/graphrag · https://microsoft.github.io/graphrag/
- LightRAG (HKUDS): https://github.com/HKUDS/LightRAG · arXiv:2410.05779
- HippoRAG (OSU-NLP): https://github.com/OSU-NLP-Group/HippoRAG · NeurIPS 2024
- PathRAG (BUPT-GAMMA): https://github.com/BUPT-GAMMA/PathRAG · arXiv:2502.14902
- Neo4j GraphRAG Python: https://github.com/neo4j/neo4j-graphrag-python · https://neo4j.com/docs/neo4j-graphrag-python/
- ms-graphrag-neo4j: https://github.com/neo4j-contrib/ms-graphrag-neo4j
- LangGraph: https://github.com/langchain-ai/langgraph
- LangChain Neo4j integration: https://neo4j.com/labs/genai-ecosystem/langchain/

---

*Última actualización: marzo 2026. Versiones de referencia: graphrag≥0.5, lightrag-hku≥1.0, hipporag≥latest, neo4j-graphrag≥1.8, langgraph≥0.2.*
