# Apéndice: LangChain y LangGraph en el contexto de Grafos de Conocimiento
## Una guía técnica completa con ejemplos de código

> **Alcance.** Este apéndice cubre las dos librerías del ecosistema LangChain más relevantes para construir sistemas RAG y agentes sobre grafos de conocimiento: **LangChain** (orquestación de cadenas y herramientas) y **LangGraph** (flujos de trabajo agénticos con estado). Todos los ejemplos de código son Python y asumen familiaridad con los conceptos del tutorial principal.

---

## Índice

1. [El ecosistema LangChain: mapa conceptual](#1-el-ecosistema-langchain-mapa-conceptual)
2. [LangChain + Neo4j: bloques fundamentales](#2-langchain--neo4j-bloques-fundamentales)
3. [LLMGraphTransformer: texto → grafo con un LLM](#3-llmgraphtransformer-texto--grafo-con-un-llm)
4. [GraphCypherQAChain: lenguaje natural → Cypher → respuesta](#4-graphcypherqachain-lenguaje-natural--cypher--respuesta)
5. [Neo4jVector: búsqueda vectorial sobre el grafo](#5-neo4jvector-búsqueda-vectorial-sobre-el-grafo)
6. [LangGraph: fundamentos del framework agéntico](#6-langgraph-fundamentos-del-framework-agéntico)
7. [Agente Graph-RAG con LangGraph](#7-agente-graph-rag-con-langgraph)
8. [GraphReader: exploración autónoma del grafo](#8-graphreader-exploración-autónoma-del-grafo)
9. [Sistema multi-agente con LangGraph](#9-sistema-multi-agente-con-langgraph)
10. [Persistencia y memoria con Neo4j como checkpointer](#10-persistencia-y-memoria-con-neo4j-como-checkpointer)
11. [Stack de producción completo](#11-stack-de-producción-completo)
12. [Errores frecuentes y buenas prácticas](#12-errores-frecuentes-y-buenas-prácticas)

---

## 1. El ecosistema LangChain: mapa conceptual

LangChain y LangGraph son dos capas distintas dentro del mismo ecosistema, con responsabilidades claramente separadas:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ECOSISTEMA LANGCHAIN                         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  LangGraph (orquestación agéntica)                       │   │
│  │  - Grafos de estado con ciclos                           │   │
│  │  - Nodos = funciones/agentes                             │   │
│  │  - Aristas condicionales (routing dinámico)              │   │
│  │  - Persistencia de estado (checkpointers)                │   │
│  │  - Human-in-the-loop                                     │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │ usa                                 │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │  LangChain (componentes y cadenas)                       │   │
│  │  - LLMs / Chat Models                                    │   │
│  │  - Prompts / PromptTemplates                             │   │
│  │  - Chains (LCEL: LangChain Expression Language)          │   │
│  │  - Document Loaders / Text Splitters                     │   │
│  │  - Vector Stores / Retrievers                            │   │
│  │  - Tools / Toolkits                                      │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │ se conecta con                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │  Integraciones externas                                  │   │
│  │  - Neo4j (grafo + vector)                                │   │
│  │  - OpenAI / Anthropic / Ollama / Groq                    │   │
│  │  - Pinecone / Weaviate / Qdrant                          │   │
│  │  - LangSmith (observabilidad)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Cuándo usar cada uno:**

| Necesidad | Herramienta |
|---|---|
| Extraer un grafo desde texto con un LLM | `LangChain: LLMGraphTransformer` |
| Preguntas en lenguaje natural → Cypher | `LangChain: GraphCypherQAChain` |
| Búsqueda semántica sobre nodos del grafo | `LangChain: Neo4jVector` |
| Flujos con decisiones dinámicas / ciclos | `LangGraph: StateGraph` |
| Agente que traversa el grafo autónomamente | `LangGraph + Neo4j tools` |
| Memoria persistente entre sesiones | `LangGraph + Neo4jSaver` |
| Multi-agente con especialización | `LangGraph: Supervisor pattern` |

---

## 2. LangChain + Neo4j: bloques fundamentales

### Instalación

```bash
# Paquetes core
pip install langchain langchain-community langchain-neo4j
pip install langchain-openai   # o langchain-anthropic, langchain-ollama
pip install langchain-experimental  # para LLMGraphTransformer

# Driver de Neo4j
pip install neo4j
```

### Conexión básica a Neo4j

```python
import os
from langchain_neo4j import Neo4jGraph

# Configurar credenciales (usar variables de entorno en producción)
os.environ["NEO4J_URI"]      = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "tu_password"

# Conexión básica
graph = Neo4jGraph()

# Conexión con schema enriquecido (incluye rangos y valores posibles)
# Muy útil para que el LLM genere Cypher más preciso
graph_enhanced = Neo4jGraph(enhanced_schema=True)

# Consulta directa
result = graph.query("MATCH (n) RETURN count(n) AS total")
print(result)  # [{'total': 0}]

# Ver el schema actual (el LLM lo usará para generar Cypher)
print(graph.schema)
# Node properties:
# - Persona: nombre STRING, edad INTEGER
# - Edificio: pisos INTEGER, color STRING
# Relationships:
# (:Persona)-[:VIVE_EN]->(:Edificio)
```

### Insertar datos manualmente

```python
# Cargar datos de ejemplo: Marcelo y su casa
graph.query("""
MERGE (m:Persona {nombre: 'Marcelo', edad: 34})
MERGE (c:Edificio {nombre: 'Casa de Marcelo', pisos: 7, color: 'verde'})
MERGE (f:Familia {descripcion: 'Familia de Marcelo'})
MERGE (m)-[:VIVE_EN]->(c)
MERGE (m)-[:VIVE_CON]->(f)
MERGE (c)-[:PERTENECE_A]->(m)
""")

# Refrescar el schema para que refleje los nuevos datos
graph.refresh_schema()
```

---

## 3. LLMGraphTransformer: texto → grafo con un LLM

`LLMGraphTransformer` es el componente de LangChain que reemplaza el pipeline NLP clásico (tokenización → NER → dependency parsing) por un LLM que extrae entidades y relaciones directamente. Funciona en dos modos: **tool-based** (cuando el LLM soporta function calling, como GPT-4 o Claude) y **prompt-based** (fallback para modelos sin soporte de herramientas).

```
Texto en lenguaje natural
         │
         ▼
┌─────────────────────────────────┐
│     LLMGraphTransformer         │
│                                 │
│  Modo 1: Tool-based (default)   │
│    LLM.with_structured_output() │
│    → extrae JSON tipado         │
│                                 │
│  Modo 2: Prompt-based (fallback)│
│    Few-shot prompting           │
│    → parsea salida de texto     │
└────────────────┬────────────────┘
                 │
                 ▼
         GraphDocument
         ├── nodes: [Node(id, type, properties)]
         └── relationships: [Relationship(source, target, type)]
```

### Ejemplo básico

```python
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_transformer = LLMGraphTransformer(llm=llm)

# Texto de ejemplo (el del tutorial principal)
texto = """
La casa de Marcelo está pintada de verde y tiene siete pisos.
Marcelo vive con su familia. Trabaja como ingeniero en TechCorp,
empresa fundada en Buenos Aires en 2010.
"""

documents = [Document(page_content=texto)]

# Extracción (puede tardar 5-15s según el modelo)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Inspeccionar resultado
for gd in graph_documents:
    print("=== NODOS ===")
    for node in gd.nodes:
        print(f"  [{node.id}] tipo={node.type} props={node.properties}")

    print("=== RELACIONES ===")
    for rel in gd.relationships:
        print(f"  ({rel.source.id})-[:{rel.type}]->({rel.target.id})")

# Output esperado:
# === NODOS ===
#   [Marcelo] tipo=Person props={}
#   [Casa de Marcelo] tipo=Building props={'color': 'verde', 'pisos': 7}
#   [TechCorp] tipo=Organization props={'ciudad': 'Buenos Aires', 'fundacion': 2010}
#   [Familia de Marcelo] tipo=Family props={}
# === RELACIONES ===
#   (Marcelo)-[:LIVES_IN]->(Casa de Marcelo)
#   (Marcelo)-[:LIVES_WITH]->(Familia de Marcelo)
#   (Marcelo)-[:WORKS_AT]->(TechCorp)
```

### Control fino: schema restringido

En producción, conviene definir explícitamente qué tipos de nodos y relaciones debe extraer el LLM, para evitar variabilidad en los nombres:

```python
llm_transformer_restringido = LLMGraphTransformer(
    llm=llm,
    # Solo estos tipos de nodos
    allowed_nodes=["Persona", "Edificio", "Empresa", "Familia", "Ciudad"],
    # Solo estas relaciones
    allowed_relationships=[
        "VIVE_EN", "VIVE_CON", "TRABAJA_EN", "FUNDADA_EN", "UBICADA_EN"
    ],
    # Extraer también propiedades de nodos
    node_properties=["color", "pisos", "edad", "año_fundacion"],
    # Extraer propiedades de relaciones
    relationship_properties=["desde", "cargo"],
)

graph_docs = llm_transformer_restringido.convert_to_graph_documents(documents)
```

### Persistir en Neo4j

```python
# Agregar los documentos de grafo a Neo4j
graph.add_graph_documents(
    graph_docs,
    baseEntityLabel=True,   # agrega etiqueta __Entity__ a todos los nodos
    include_source=True      # vincula cada nodo al Document original
)

# Con include_source=True, el grafo queda así:
# (:Document {text: "La casa de Marcelo..."})-[:MENTIONS]->(:Persona {id: "Marcelo"})
# Esto permite trazar cada entidad a su fuente textual original
```

---

## 4. GraphCypherQAChain: lenguaje natural → Cypher → respuesta

`GraphCypherQAChain` implementa el ciclo completo de Text-to-Cypher: el LLM recibe el schema del grafo, la pregunta del usuario, genera una consulta Cypher, la ejecuta contra Neo4j, y usa los resultados para construir una respuesta en lenguaje natural.

```
Pregunta del usuario
       │
       ▼
┌──────────────────────┐
│  LLM (Cypher gen.)   │◄── schema del grafo
│  "Genera Cypher para │
│   responder esto"    │
└────────┬─────────────┘
         │ Cypher query
         ▼
┌──────────────────────┐
│  Neo4j               │
│  Ejecuta la query    │
└────────┬─────────────┘
         │ resultados estructurados
         ▼
┌──────────────────────┐
│  LLM (síntesis)      │◄── pregunta original + resultados
│  "Responde en        │
│   lenguaje natural"  │
└────────┬─────────────┘
         │
         ▼
    Respuesta final
```

### Uso básico

```python
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
graph = Neo4jGraph()

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,              # muestra el Cypher generado
    validate_cypher=True,      # valida sintaxis antes de ejecutar
    allow_dangerous_requests=True,  # requerido en v0.3+
)

# Consulta en lenguaje natural
response = chain.invoke({"query": "¿En qué casa vive Marcelo y cuántos pisos tiene?"})
print(response["result"])

# Con verbose=True se imprime:
# > Entering new GraphCypherQAChain chain...
# Generated Cypher:
#   MATCH (m:Persona {nombre: 'Marcelo'})-[:VIVE_EN]->(c:Edificio)
#   RETURN c.nombre, c.pisos
# Full Context: [{'c.nombre': 'Casa de Marcelo', 'c.pisos': 7}]
# > Finished chain.
# "Marcelo vive en 'Casa de Marcelo', que tiene 7 pisos."
```

### Prompt personalizado con Few-Shot examples

Uno de los problemas más comunes con GraphCypherQAChain es que el LLM genera Cypher sintácticamente incorrecto o usa propiedades que no existen. La solución más efectiva es proveer ejemplos pregunta→Cypher relevantes:

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_neo4j import GraphCypherQAChain

# Ejemplos de preguntas y sus Cypher correctos
ejemplos = [
    {
        "question": "¿Dónde vive Marcelo?",
        "query": "MATCH (p:Persona {nombre: 'Marcelo'})-[:VIVE_EN]->(e:Edificio) RETURN e.nombre, e.pisos, e.color"
    },
    {
        "question": "¿Con quién vive Marcelo?",
        "query": "MATCH (p:Persona {nombre: 'Marcelo'})-[:VIVE_CON]->(f) RETURN f"
    },
    {
        "question": "¿Qué empresas hay en Buenos Aires?",
        "query": "MATCH (e:Empresa)-[:UBICADA_EN]->(:Ciudad {nombre: 'Buenos Aires'}) RETURN e.nombre"
    },
]

example_prompt = PromptTemplate.from_template(
    "Pregunta: {question}\nCypher: {query}"
)

cypher_prompt = FewShotPromptTemplate(
    examples=ejemplos,
    example_prompt=example_prompt,
    prefix="""Eres un experto en Neo4j. Genera una consulta Cypher sintácticamente 
correcta para responder la pregunta del usuario.
Usa SOLO los tipos de nodos y relaciones del schema.
No incluyas texto adicional, solo el Cypher.

Schema: {schema}

Ejemplos:""",
    suffix="Pregunta: {question}\nCypher:",
    input_variables=["schema", "question"],
)

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    allow_dangerous_requests=True,
)
```

### Obtener pasos intermedios y trazabilidad

```python
chain_con_pasos = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)

result = chain_con_pasos.invoke({"query": "¿Cuántos pisos tiene la casa de Marcelo?"})

# Acceder a los pasos intermedios
cypher_generado   = result["intermediate_steps"][0]["query"]
contexto_neo4j    = result["intermediate_steps"][1]["context"]
respuesta_final   = result["result"]

print(f"Cypher:   {cypher_generado}")
print(f"Contexto: {contexto_neo4j}")
print(f"Respuesta: {respuesta_final}")
```

---

## 5. Neo4jVector: búsqueda vectorial sobre el grafo

Neo4j soporta índices vectoriales nativos desde la versión 5.11. `Neo4jVector` de LangChain permite almacenar embeddings directamente en nodos del grafo y realizar búsqueda híbrida (vectorial + keyword).

```
Documentos
    │
    ├── Chunks de texto
    │        │
    │        ▼
    │   OpenAIEmbeddings  →  vector [0.12, -0.34, ..., 0.87]
    │        │
    │        ▼
    │   Nodo Neo4j: (:Chunk {text: "...", embedding: [...]})
    │
    └── Búsqueda
             │
             ├── Vector similarity (coseno) 
             ├── Keyword (BM25 full-text index)
             └── Hybrid: combina ambas con RRF
```

### Crear índice vectorial desde documentos

```python
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Crear vector store desde documentos nuevos
docs = [
    Document(page_content="La casa de Marcelo tiene 7 pisos y está pintada de verde."),
    Document(page_content="Marcelo trabaja como ingeniero en TechCorp desde 2018."),
    Document(page_content="TechCorp fue fundada en Buenos Aires por Ana García en 2010."),
]

vector_store = Neo4jVector.from_documents(
    docs,
    embeddings,
    url="bolt://localhost:7687",
    username="neo4j",
    password="tu_password",
    index_name="documento_vector",      # nombre del índice en Neo4j
    node_label="Chunk",                 # etiqueta de los nodos
    text_node_property="texto",         # propiedad que guarda el texto
    embedding_node_property="embedding",# propiedad que guarda el vector
)

# Búsqueda por similitud
resultados = vector_store.similarity_search(
    "¿Cuántos pisos tiene la casa?", k=2
)
for doc in resultados:
    print(doc.page_content)
```

### Conectarse a un índice existente

```python
# Cuando el índice ya existe en Neo4j (caso más frecuente en producción)
vector_store_existente = Neo4jVector.from_existing_index(
    embeddings,
    url="bolt://localhost:7687",
    username="neo4j",
    password="tu_password",
    index_name="documento_vector",
    text_node_property="texto",
)
```

### Búsqueda híbrida (vector + keyword)

```python
# Requiere índice full-text además del vectorial
vector_store_hibrido = Neo4jVector.from_existing_graph(
    embeddings,
    url="bolt://localhost:7687",
    username="neo4j",
    password="tu_password",
    index_name="hybrid_index",
    keyword_index_name="keyword_index",
    search_type="hybrid",           # "vector", "keyword", o "hybrid"
    node_label="Chunk",
    text_node_properties=["texto"], # propiedades a incluir en el texto
    embedding_node_property="embedding",
)

resultados = vector_store_hibrido.similarity_search(
    "ingeniero TechCorp Buenos Aires", k=3
)
```

### Usar como retriever en una cadena RAG

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
Responde la pregunta basándote únicamente en el siguiente contexto:

{context}

Pregunta: {question}
""")

def formatear_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Cadena RAG con LCEL (LangChain Expression Language)
cadena_rag = (
    {"context": retriever | formatear_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

respuesta = cadena_rag.invoke("¿Quién fundó TechCorp?")
print(respuesta)
```

---

## 6. LangGraph: fundamentos del framework agéntico

LangGraph modela flujos de trabajo como **grafos de estado dirigidos**, donde cada nodo es una función Python y cada arista determina a qué nodo pasar el control a continuación. A diferencia de las cadenas LangChain (DAGs lineales), LangGraph soporta **ciclos**, lo que permite que un agente itere, reintente, y tome decisiones dinámicas.

```
Conceptos clave:

  StateGraph  = el grafo completo, con su estado compartido
  State       = TypedDict que define qué información se propaga
  Node        = función Python (state: State) → dict con updates
  Edge        = conexión directa entre dos nodos
  Conditional Edge = función que elige el próximo nodo en runtime
  Checkpointer = persistencia del estado entre invocaciones
```

### Anatomía de un LangGraph mínimo

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 1. Definir el estado compartido
class GraphState(TypedDict):
    pregunta:  str
    cypher:    str
    resultado: list
    respuesta: str

# 2. Definir nodos (funciones que reciben y retornan estado)
def generar_cypher(state: GraphState) -> dict:
    """Nodo 1: Pregunta → Cypher"""
    prompt = f"Genera Cypher para Neo4j para responder: {state['pregunta']}"
    cypher = llm.invoke(prompt).content
    return {"cypher": cypher}

def ejecutar_cypher(state: GraphState) -> dict:
    """Nodo 2: Cypher → resultados de Neo4j"""
    resultado = graph.query(state["cypher"])
    return {"resultado": resultado}

def sintetizar_respuesta(state: GraphState) -> dict:
    """Nodo 3: resultados → respuesta en lenguaje natural"""
    prompt = f"""
    Pregunta: {state['pregunta']}
    Datos del grafo: {state['resultado']}
    Responde en lenguaje natural basándote en los datos.
    """
    respuesta = llm.invoke(prompt).content
    return {"respuesta": respuesta}

# 3. Construir el grafo
workflow = StateGraph(GraphState)

workflow.add_node("generar_cypher",     generar_cypher)
workflow.add_node("ejecutar_cypher",    ejecutar_cypher)
workflow.add_node("sintetizar_respuesta", sintetizar_respuesta)

# 4. Conectar nodos con aristas
workflow.add_edge(START,               "generar_cypher")
workflow.add_edge("generar_cypher",    "ejecutar_cypher")
workflow.add_edge("ejecutar_cypher",   "sintetizar_respuesta")
workflow.add_edge("sintetizar_respuesta", END)

# 5. Compilar
app = workflow.compile()

# 6. Invocar
resultado = app.invoke({"pregunta": "¿Cuántos pisos tiene la casa de Marcelo?"})
print(resultado["respuesta"])
```

### Aristas condicionales: el corazón del routing dinámico

```python
from typing import Literal

def router(state: GraphState) -> Literal["busqueda_grafo", "busqueda_vectorial"]:
    """
    Decide qué rama del grafo seguir según la pregunta.
    Esta función es ejecutada como arista condicional.
    """
    pregunta = state["pregunta"].lower()

    # Preguntas sobre relaciones → grafo estructurado
    palabras_grafo = ["quién", "cuántos", "dónde vive", "trabaja en", "relacionado"]
    if any(p in pregunta for p in palabras_grafo):
        return "busqueda_grafo"

    # Preguntas semánticas → búsqueda vectorial
    return "busqueda_vectorial"

# En el grafo:
workflow.add_conditional_edges(
    "clasificar_pregunta",          # nodo de origen
    router,                          # función que retorna el nombre del próximo nodo
    {
        "busqueda_grafo":     "ejecutar_cypher",
        "busqueda_vectorial": "busqueda_semantica",
    }
)
```

---

## 7. Agente Graph-RAG con LangGraph

Este es el patrón más completo y útil: un agente que combina búsqueda vectorial y consulta al grafo, con routing dinámico y ciclos de corrección. La arquitectura está basada en el workflow documentado por Neo4j (julio 2025).

```
                    [START]
                       │
                       ▼
              [clasificar_pregunta]
                /              \
    "grafo simple"          "semántica"
              │                  │
              ▼                  ▼
    [generar_cypher]    [descomponer_query]
              │                  │
              ▼                  ▼
    [ejecutar_neo4j]    [busqueda_vectorial]
              │                  │
              ▼                  ▼
    [validar_resultado] ◄────────┘
              │
    ¿resultado vacío?
         /        \
       "sí"       "no"
        │           │
        ▼           ▼
  [reescribir]  [sintetizar]
        │           │
        └──────┐    ▼
          (loop)  [END]
```

```python
import os
from typing import TypedDict, Annotated, Literal
from operator import add

from langgraph.graph import StateGraph, START, END
from langchain_neo4j import Neo4jGraph, Neo4jVector, GraphCypherQAChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

# ── Configuración ────────────────────────────────────────────────
llm    = ChatOpenAI(model="gpt-4o", temperature=0)
graph  = Neo4jGraph()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Neo4jVector.from_existing_index(
    embeddings,
    index_name="documento_vector",
    text_node_property="texto",
)

# ── Estado del agente ─────────────────────────────────────────────
class AgentState(TypedDict):
    pregunta:      str
    subqueries:    Annotated[list[str], add]  # se acumula (operator.add)
    cypher:        str
    neo4j_result:  list
    vector_docs:   list[Document]
    respuesta:     str
    intentos:      int

# ── Nodos ────────────────────────────────────────────────────────

def clasificar(state: AgentState) -> dict:
    """Clasifica si la pregunta requiere Cypher o búsqueda semántica."""
    prompt = f"""
    Clasifica la siguiente pregunta en una de estas categorías:
    - "grafo": preguntas sobre relaciones, conteos, propiedades específicas
    - "semantica": preguntas conceptuales, resúmenes, preguntas abiertas
    - "hibrida": requiere ambas

    Pregunta: {state['pregunta']}
    Responde SOLO con una de las tres palabras.
    """
    tipo = llm.invoke(prompt).content.strip().lower()
    return {"subqueries": [tipo]}  # guardamos el tipo como primer subquery

def descomponer_query(state: AgentState) -> dict:
    """Descompone una pregunta compleja en sub-consultas."""
    prompt = f"""
    Descompone la siguiente pregunta en 2-3 sub-preguntas más simples
    que se puedan responder independientemente con una base de datos de grafos.
    Devuelve una lista Python de strings, sin ningún otro texto.

    Pregunta: {state['pregunta']}
    """
    import ast
    raw = llm.invoke(prompt).content.strip()
    try:
        subqueries = ast.literal_eval(raw)
    except Exception:
        subqueries = [state["pregunta"]]
    return {"subqueries": subqueries}

def generar_cypher(state: AgentState) -> dict:
    """Genera una consulta Cypher a partir de la pregunta."""
    schema = graph.schema
    prompt = f"""
    Eres un experto en Neo4j. Genera SOLO la consulta Cypher, sin explicaciones.

    Schema del grafo:
    {schema}

    Pregunta: {state['pregunta']}

    Reglas:
    - Usa solo los nodos y relaciones del schema
    - Usa LIMIT 10 para evitar resultados masivos
    - Si no puedes responder con el schema, devuelve: MATCH (n) RETURN null LIMIT 0
    """
    cypher = llm.invoke(prompt).content.strip()
    # Limpiar posibles markdown fences
    cypher = cypher.replace("```cypher", "").replace("```", "").strip()
    return {"cypher": cypher}

def ejecutar_neo4j(state: AgentState) -> dict:
    """Ejecuta el Cypher en Neo4j."""
    try:
        resultado = graph.query(state["cypher"])
    except Exception as e:
        resultado = [{"error": str(e)}]
    return {"neo4j_result": resultado, "intentos": state.get("intentos", 0) + 1}

def busqueda_vectorial(state: AgentState) -> dict:
    """Recupera documentos semánticamente relevantes."""
    docs = vector_store.similarity_search(state["pregunta"], k=4)
    return {"vector_docs": docs}

def reescribir_cypher(state: AgentState) -> dict:
    """Si el Cypher falló, intenta corregirlo."""
    prompt = f"""
    El siguiente Cypher falló o devolvió resultados vacíos.
    Error/resultado: {state['neo4j_result']}
    Schema: {graph.schema}
    Cypher original: {state['cypher']}
    Pregunta: {state['pregunta']}

    Genera una consulta Cypher alternativa y corregida.
    Devuelve SOLO el Cypher.
    """
    cypher_nuevo = llm.invoke(prompt).content.strip()
    cypher_nuevo = cypher_nuevo.replace("```cypher", "").replace("```", "").strip()
    return {"cypher": cypher_nuevo}

def sintetizar(state: AgentState) -> dict:
    """Combina toda la evidencia y genera la respuesta final."""
    contexto_grafo   = state.get("neo4j_result", [])
    contexto_vector  = "\n".join(d.page_content for d in state.get("vector_docs", []))

    prompt = f"""
    Responde la siguiente pregunta usando la evidencia disponible.
    Si la evidencia es insuficiente, dilo explícitamente.

    Pregunta: {state['pregunta']}

    Evidencia del grafo de conocimiento:
    {contexto_grafo}

    Evidencia de búsqueda semántica:
    {contexto_vector}

    Responde de forma clara y concisa.
    """
    respuesta = llm.invoke(prompt).content
    return {"respuesta": respuesta}

# ── Funciones de routing ──────────────────────────────────────────

def routing_inicial(state: AgentState) -> Literal["grafo", "semantica", "hibrida"]:
    tipo = state["subqueries"][0] if state["subqueries"] else "hibrida"
    if "grafo" in tipo:
        return "grafo"
    elif "semantica" in tipo:
        return "semantica"
    return "hibrida"

def validar_resultado(
    state: AgentState
) -> Literal["sintetizar", "reescribir", "vectorial_fallback"]:
    resultado = state.get("neo4j_result", [])
    intentos  = state.get("intentos", 0)

    if resultado and "error" not in str(resultado[0]):
        return "sintetizar"
    elif intentos < 2:
        return "reescribir"
    else:
        return "vectorial_fallback"  # fallback a vector si Cypher falla 2 veces

# ── Construcción del grafo ────────────────────────────────────────

workflow = StateGraph(AgentState)

# Agregar todos los nodos
workflow.add_node("clasificar",          clasificar)
workflow.add_node("descomponer_query",   descomponer_query)
workflow.add_node("generar_cypher",      generar_cypher)
workflow.add_node("ejecutar_neo4j",      ejecutar_neo4j)
workflow.add_node("busqueda_vectorial",  busqueda_vectorial)
workflow.add_node("reescribir_cypher",   reescribir_cypher)
workflow.add_node("sintetizar",          sintetizar)

# Flujo principal
workflow.add_edge(START, "clasificar")

# Routing desde clasificar
workflow.add_conditional_edges(
    "clasificar",
    routing_inicial,
    {
        "grafo":     "generar_cypher",
        "semantica": "busqueda_vectorial",
        "hibrida":   "descomponer_query",
    }
)

# Rama del grafo
workflow.add_edge("generar_cypher",    "ejecutar_neo4j")
workflow.add_conditional_edges(
    "ejecutar_neo4j",
    validar_resultado,
    {
        "sintetizar":        "sintetizar",
        "reescribir":        "reescribir_cypher",
        "vectorial_fallback": "busqueda_vectorial",
    }
)
workflow.add_edge("reescribir_cypher", "ejecutar_neo4j")  # ciclo de corrección

# Rama semántica y convergencia
workflow.add_edge("descomponer_query",  "busqueda_vectorial")
workflow.add_edge("busqueda_vectorial", "sintetizar")
workflow.add_edge("sintetizar",         END)

# Compilar
app = workflow.compile()

# ── Uso ──────────────────────────────────────────────────────────

resultado = app.invoke({
    "pregunta": "¿Qué relación tiene Marcelo con TechCorp y dónde está ubicada?",
    "subqueries": [],
    "intentos": 0,
})

print(resultado["respuesta"])
```

---

## 8. GraphReader: exploración autónoma del grafo

**GraphReader** (implementado con Neo4j + LangGraph por Tomaz Bratanic, 2024) es un agente que traversa el grafo de forma autónoma, manteniendo un "cuaderno de notas" donde acumula evidencia antes de responder. Es el patrón más sofisticado para preguntas multi-hop.

```
[plan_racional]           ← descompone la pregunta en pasos
       │
       ▼
[seleccion_inicial]       ← elige los nodos de entrada al grafo
       │
       ▼
[verificar_hechos]        ← lee hechos atómicos de los nodos seleccionados
       │
  ¿suficiente?
     /     \
   sí       no
   │         │
   │    [leer_chunks]     ← lee texto asociado a nodos relevantes
   │         │
   │    [explorar_vecinos]← expande a nodos vecinos del grafo
   │         │
   │    (loop ──────────► [verificar_hechos])
   │
   ▼
[respuesta_final]         ← sintetiza el cuaderno de notas
```

```python
from typing import TypedDict, Annotated
from operator import add

class GraphReaderState(TypedDict):
    pregunta:        str
    plan:            str                       # plan racional inicial
    nodos_actuales:  list[str]                 # IDs de nodos en exploración
    cuaderno:        Annotated[list[str], add] # hechos acumulados (append)
    acciones_prev:   Annotated[list[str], add] # historial de acciones
    respuesta:       str

def plan_racional(state: GraphReaderState) -> dict:
    """El agente crea un plan de exploración antes de comenzar."""
    prompt = f"""
    Crea un plan racional para responder la siguiente pregunta
    usando un grafo de conocimiento. Identifica:
    1. Qué entidades necesitas encontrar
    2. Qué relaciones necesitas recorrer
    3. Qué hechos debes verificar

    Pregunta: {state['pregunta']}
    """
    plan = llm.invoke(prompt).content
    return {"plan": plan}

def verificar_hechos_atomicos(state: GraphReaderState) -> dict:
    """Lee propiedades de los nodos actuales."""
    hechos = []
    for nodo_id in state["nodos_actuales"]:
        resultado = graph.query(
            "MATCH (n {id: $id}) RETURN n, [(n)-[r]->(m) | [type(r), m.id]] AS relaciones",
            params={"id": nodo_id}
        )
        if resultado:
            hecho = f"Nodo {nodo_id}: {resultado[0]}"
            hechos.append(hecho)

    return {
        "cuaderno": hechos,
        "acciones_prev": [f"verificar_hechos({state['nodos_actuales']})"]
    }

def explorar_vecinos(state: GraphReaderState) -> dict:
    """Expande la exploración a nodos vecinos relevantes."""
    vecinos_relevantes = []
    for nodo_id in state["nodos_actuales"]:
        resultado = graph.query("""
            MATCH (n {id: $id})-[r]->(vecino)
            RETURN vecino.id AS vecino_id, type(r) AS relacion, vecino AS datos
            LIMIT 5
        """, params={"id": nodo_id})

        for row in resultado:
            vecinos_relevantes.append(row["vecino_id"])

    # El LLM decide cuáles vecinos explorar basándose en el plan
    prompt = f"""
    Plan original: {state['plan']}
    Cuaderno actual: {state['cuaderno'][-5:]}  
    Vecinos disponibles: {vecinos_relevantes}
    
    ¿Cuáles de estos vecinos son más relevantes para responder la pregunta?
    Devuelve una lista Python de IDs. Máximo 3.
    """
    import ast
    raw = llm.invoke(prompt).content.strip()
    try:
        seleccionados = ast.literal_eval(raw)
    except Exception:
        seleccionados = vecinos_relevantes[:2]

    return {
        "nodos_actuales": seleccionados,
        "acciones_prev": [f"explorar_vecinos → {seleccionados}"]
    }

def decidir_siguiente_accion(
    state: GraphReaderState
) -> Literal["explorar_vecinos", "respuesta_final"]:
    """El LLM decide si tiene suficiente información o necesita más exploración."""
    prompt = f"""
    Pregunta: {state['pregunta']}
    Plan: {state['plan']}
    Hechos recopilados: {state['cuaderno']}
    Acciones realizadas: {state['acciones_prev']}

    ¿Tienes suficiente información para responder la pregunta?
    Responde SOLO: "suficiente" o "explorar"
    """
    decision = llm.invoke(prompt).content.strip().lower()

    # Forzar terminación después de 5 iteraciones
    if len(state.get("acciones_prev", [])) >= 5:
        return "respuesta_final"

    return "respuesta_final" if "suficiente" in decision else "explorar_vecinos"

def generar_respuesta_final(state: GraphReaderState) -> dict:
    prompt = f"""
    Basándote en los siguientes hechos recopilados del grafo de conocimiento,
    responde la pregunta de forma completa y precisa.

    Pregunta: {state['pregunta']}
    Hechos: {chr(10).join(state['cuaderno'])}
    """
    respuesta = llm.invoke(prompt).content
    return {"respuesta": respuesta}
```

---

## 9. Sistema multi-agente con LangGraph

Para dominios complejos, el patrón **Supervisor + Workers** permite dividir el trabajo entre agentes especializados:

```
                 [Supervisor]
                 (decide qué agente invocar)
                /      |        \
               ▼       ▼         ▼
    [Agente Grafo] [Agente Vector] [Agente Cypher]
    (relaciones)   (semántica)    (text-to-cypher)
               \       |        /
                ▼       ▼       ▼
               [Supervisor] (¿terminado?)
                       │
                       ▼
                 [Respuesta Final]
```

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated
from operator import add

# ── Herramientas especializadas ───────────────────────────────────

@tool
def consultar_grafo(cypher: str) -> str:
    """
    Ejecuta una consulta Cypher en el grafo de conocimiento Neo4j.
    Úsala para preguntas sobre relaciones entre entidades.
    Args:
        cypher: Consulta Cypher válida para Neo4j
    """
    try:
        resultado = graph.query(cypher)
        return str(resultado)
    except Exception as e:
        return f"Error al ejecutar Cypher: {str(e)}"

@tool
def buscar_semanticamente(query: str, k: int = 3) -> str:
    """
    Busca documentos semánticamente relacionados con la query.
    Úsala para preguntas conceptuales o cuando necesites contexto textual.
    Args:
        query: Pregunta o tema a buscar
        k: Número de documentos a recuperar
    """
    docs = vector_store.similarity_search(query, k=k)
    return "\n---\n".join(d.page_content for d in docs)

@tool
def obtener_schema_grafo() -> str:
    """
    Retorna el schema actual del grafo de conocimiento.
    Úsala antes de generar Cypher para conocer los tipos de nodos y relaciones disponibles.
    """
    return graph.schema

# ── Agente individual con herramientas (ReAct pattern) ───────────

herramientas = [consultar_grafo, buscar_semanticamente, obtener_schema_grafo]

# create_react_agent construye automáticamente el loop: LLM → Tool → LLM → ...
agente_grafo = create_react_agent(
    model=llm,
    tools=herramientas,
    state_modifier="""Eres un especialista en grafos de conocimiento. 
    Tu objetivo es responder preguntas usando el grafo Neo4j.
    Siempre obtén el schema primero, luego genera el Cypher apropiado."""
)

# ── Estado del supervisor ─────────────────────────────────────────

class SupervisorState(TypedDict):
    messages:     Annotated[list, add]
    next_agent:   str
    final_answer: str

AGENTES = ["agente_grafo", "FINISH"]

def supervisor(state: SupervisorState) -> dict:
    """El supervisor decide qué agente invocar a continuación."""
    system_prompt = f"""
    Eres el supervisor de un sistema multi-agente para consulta de grafos de conocimiento.
    Los agentes disponibles son: {AGENTES}

    Dado el historial de la conversación, decide qué agente debe actuar a continuación.
    Si la pregunta ya fue respondida satisfactoriamente, responde "FINISH".

    Responde SOLO con el nombre del agente o "FINISH".
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    decision = llm.invoke(messages).content.strip()
    return {"next_agent": decision}

def routing_supervisor(state: SupervisorState) -> str:
    return state["next_agent"]

# ── Construcción del grafo multi-agente ──────────────────────────

multi_agent = StateGraph(SupervisorState)

multi_agent.add_node("supervisor",   supervisor)
multi_agent.add_node("agente_grafo", agente_grafo)

multi_agent.add_edge(START, "supervisor")
multi_agent.add_conditional_edges(
    "supervisor",
    routing_supervisor,
    {"agente_grafo": "agente_grafo", "FINISH": END}
)
multi_agent.add_edge("agente_grafo", "supervisor")

app_multi = multi_agent.compile()
```

---

## 10. Persistencia y memoria con Neo4j como checkpointer

Una de las funcionalidades más poderosas de LangGraph es la **persistencia de estado** entre sesiones. El paquete `langchain-neo4j` incluye `Neo4jSaver`, un checkpointer que almacena el estado del agente directamente en el grafo:

```python
from langchain_neo4j import Neo4jSaver
from langgraph.checkpoint.memory import MemorySaver
from neo4j import GraphDatabase

# ── Opción 1: Persistencia en memoria (desarrollo) ────────────────
memory_saver = MemorySaver()

app_con_memoria = workflow.compile(checkpointer=memory_saver)

# Cada thread_id es una sesión independiente
config_sesion1 = {"configurable": {"thread_id": "usuario_123_sesion_1"}}
config_sesion2 = {"configurable": {"thread_id": "usuario_456_sesion_1"}}

# Primera invocación
resultado = app_con_memoria.invoke(
    {"pregunta": "¿Dónde vive Marcelo?"},
    config=config_sesion1
)

# Segunda invocación (mismo thread_id → recuerda el contexto anterior)
resultado2 = app_con_memoria.invoke(
    {"pregunta": "¿Y con quién vive?"},
    config=config_sesion1  # mismo thread → continuación de la conversación
)

# ── Opción 2: Persistencia en Neo4j (producción) ──────────────────
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "tu_password")
)

neo4j_saver = Neo4jSaver(driver)

app_persistente = workflow.compile(checkpointer=neo4j_saver)

# El estado del agente se guarda como nodos y relaciones en Neo4j
# Esto permite inspeccionar el historial con Cypher:
# MATCH (checkpoint:Checkpoint {thread_id: "usuario_123_sesion_1"})
# RETURN checkpoint ORDER BY checkpoint.created_at DESC

# ── Inspección del estado ─────────────────────────────────────────

# Ver el estado actual de una sesión
estado_actual = app_persistente.get_state(config_sesion1)
print(estado_actual.values)  # todos los valores del TypedDict

# Ver el historial completo de la sesión
historial = list(app_persistente.get_state_history(config_sesion1))
for checkpoint in historial:
    print(f"Step {checkpoint.metadata['step']}: {checkpoint.values.get('respuesta', 'en proceso...')}")

# Volver a un estado anterior (time-travel debugging)
checkpoint_anterior = historial[2]
resultado_desde_pasado = app_persistente.invoke(
    None,  # None = continuar desde el checkpoint
    config={**config_sesion1, "checkpoint_id": checkpoint_anterior.config["configurable"]["checkpoint_id"]}
)
```

### Human-in-the-loop: pausar y solicitar aprobación

```python
from langgraph.graph import interrupt

def ejecutar_cypher_con_aprobacion(state: AgentState) -> dict:
    """
    Pausa el grafo y espera aprobación humana antes de ejecutar el Cypher.
    """
    cypher = state["cypher"]

    # interrupt() detiene la ejecución y retorna el control al humano
    aprobado = interrupt({
        "mensaje": "¿Aprobás esta consulta Cypher?",
        "cypher": cypher,
        "schema": graph.schema,
    })

    if aprobado:
        resultado = graph.query(cypher)
        return {"neo4j_result": resultado}
    else:
        return {"neo4j_result": [{"error": "Consulta rechazada por el usuario"}]}

# Para reanudar después de la aprobación:
# app.invoke(Command(resume=True), config=config_sesion)
# app.invoke(Command(resume=False), config=config_sesion)  # para rechazar
```

---

## 11. Stack de producción completo

Juntando todos los componentes, este es el stack recomendado para un sistema de consulta Graph-RAG en producción:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STACK DE PRODUCCIÓN                              │
│                                                                     │
│  ┌────────────────────┐    ┌──────────────────────────────────────┐ │
│  │  Ingesta           │    │  Query                               │ │
│  │                    │    │                                      │ │
│  │  Docling/PyMuPDF   │    │  FastAPI / LangServe                 │ │
│  │       ↓            │    │         ↓                            │ │
│  │  LangChain         │    │  LangGraph App (compilado)           │ │
│  │  TextSplitter      │    │    ├── Clasificador                  │ │
│  │       ↓            │    │    ├── GraphCypherQAChain            │ │
│  │  LLMGraph          │    │    ├── Neo4jVector retriever         │ │
│  │  Transformer       │    │    └── Sintetizador                  │ │
│  │       ↓            │    │         ↓                            │ │
│  │  Neo4j             │◄───┤  Neo4j (KG + Vector Index)          │ │
│  │  (grafo + vector)  │    │         ↓                            │ │
│  └────────────────────┘    │  Neo4jSaver (checkpoints)           │ │
│                            │         ↓                            │ │
│                            │  LangSmith (observabilidad)         │ │
│                            └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# Ejemplo: pipeline de ingesta completo
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def ingestar_documento(ruta_pdf: str):
    """Pipeline completo: PDF → KG + Vector Index en Neo4j"""

    # 1. Cargar documento
    loader = PyMuPDFLoader(ruta_pdf)
    docs   = loader.load()
    print(f"[1/4] Cargados {len(docs)} páginas")

    # 2. Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"[2/4] Generados {len(chunks)} chunks")

    # 3. Extraer grafo de conocimiento
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Persona", "Empresa", "Lugar", "Concepto", "Evento"],
        allowed_relationships=["TRABAJA_EN", "UBICADA_EN", "PARTE_DE", "RELACIONADO_CON"],
        node_properties=True,
    )

    graph = Neo4jGraph()
    graph_docs = transformer.convert_to_graph_documents(chunks)
    graph.add_graph_documents(graph_docs, include_source=True)
    print(f"[3/4] Grafo construido con {len(graph_docs)} documentos")

    # 4. Crear índice vectorial
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Neo4jVector.from_documents(
        chunks,
        embeddings,
        index_name="doc_index",
        node_label="Chunk",
        text_node_property="texto",
        embedding_node_property="embedding",
    )
    print("[4/4] Índice vectorial creado")
    return f"Ingesta completada: {len(chunks)} chunks, {len(graph_docs)} graph docs"
```

---

## 12. Errores frecuentes y buenas prácticas

### Errores comunes

```python
# ❌ MAL: No refrescar el schema después de modificar el grafo
graph.query("CREATE (n:NuevoTipo {x: 1})")
chain = GraphCypherQAChain.from_llm(llm, graph=graph)
# El schema no incluye NuevoTipo → el LLM no puede generar Cypher correcto

# ✅ BIEN: Siempre refrescar el schema
graph.query("CREATE (n:NuevoTipo {x: 1})")
graph.refresh_schema()  # ← imprescindible
chain = GraphCypherQAChain.from_llm(llm, graph=graph)

# ❌ MAL: Estado TypedDict sin Annotated para listas acumulables
class State(TypedDict):
    mensajes: list  # add_edge sobrescribe la lista completa

# ✅ BIEN: Usar Annotated + operator.add para listas que crecen
from typing import Annotated
from operator import add
class State(TypedDict):
    mensajes: Annotated[list, add]  # los updates se añaden al final

# ❌ MAL: Cypher con propiedades que no existen en el schema
# LLM genera: MATCH (p:Persona {email: 'x@y.com'}) RETURN p
# El schema no tiene 'email' → resultado vacío sin error

# ✅ BIEN: Usar enhanced_schema + validación
graph = Neo4jGraph(enhanced_schema=True)
chain = GraphCypherQAChain.from_llm(
    llm, graph=graph,
    validate_cypher=True  # valida antes de ejecutar
)

# ❌ MAL: LLMGraphTransformer sin tipos definidos
# Genera tipos inconsistentes: "Person", "Persona", "PERSON", "Human"...
transformer = LLMGraphTransformer(llm=llm)

# ✅ BIEN: Definir schema estricto
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Persona", "Empresa", "Lugar"],
    allowed_relationships=["TRABAJA_EN", "VIVE_EN", "UBICADA_EN"],
)
```

### Buenas prácticas consolidadas

| Práctica | Razón |
|---|---|
| Usar `Neo4jGraph(enhanced_schema=True)` | El LLM genera Cypher más preciso al conocer los valores posibles |
| Proveer ejemplos few-shot en el prompt de Cypher | Reduce errores de sintaxis y uso de propiedades inexistentes en un 40-60% |
| Definir `allowed_nodes` y `allowed_relationships` | Evita variabilidad en nombres durante la extracción |
| Usar `include_source=True` en `add_graph_documents` | Permite trazar cada entidad a su fuente textual original |
| Combinar búsqueda vectorial + Cypher (híbrida) | Cubre tanto preguntas relacionales como semánticas |
| Usar `validate_cypher=True` | Evita errores en producción por Cypher malformado |
| `thread_id` por usuario + sesión | Mantiene contexto sin mezclar conversaciones |
| LangSmith para observabilidad | Permite debuggear qué Cypher generó el LLM y por qué falló |

### Comando de referencia rápida

```bash
# Verificar versiones instaladas
pip show langchain langchain-neo4j langchain-experimental langgraph

# Versiones mínimas recomendadas (marzo 2026)
# langchain           >= 0.3.0
# langchain-neo4j     >= 0.8.0
# langchain-experimental >= 0.3.0
# langgraph           >= 0.2.0
# neo4j (driver)      >= 5.0.0
```

---

## Referencias

- LangChain Neo4j Integration. Neo4j Labs. https://neo4j.com/labs/genai-ecosystem/langchain/
- LangGraph official repository. GitHub / langchain-ai. https://github.com/langchain-ai/langgraph
- Bratanic, T. (2024). *Building Knowledge Graphs with LLM Graph Transformer*. Towards Data Science.
- Bratanic, T. (2024). *Implementing GraphReader with Neo4j and LangGraph*. Towards Data Science.
- Neo4j. (2025). *Create a Neo4j GraphRAG Workflow Using LangChain and LangGraph*. Neo4j Developer Blog.
- langchain-neo4j PyPI package (v0.8.0). https://pypi.org/project/langchain-neo4j/
- LangGraph documentation: https://langchain-ai.github.io/langgraph/

---

*Última actualización: marzo 2026. Los ejemplos fueron validados contra langchain-neo4j v0.8.0 y langgraph v0.2.x. Verificar la documentación oficial ante cambios de API.*
