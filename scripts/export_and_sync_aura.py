"""
Export local Neo4j graph to output/ and seed AuraDB with the corrected ontology.

Usage:
    python scripts/export_and_sync_aura.py

Steps:
  1. Connect to local Neo4j (bolt://localhost:7687, neo4jpassword)
  2. Export corrected graph → output/her2_knowledge_graph_<timestamp>.ttl/.jsonld
  3. Connect to AuraDB (from .env)
  4. Initialize schema + load seed data into AuraDB
"""
from __future__ import annotations

import sys
import os
from datetime import datetime
from pathlib import Path

# Project root on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv, dotenv_values
load_dotenv(ROOT / ".env", override=False)

# Read AuraDB credentials directly from .env file (bypassing shell env overrides)
_env_file_vals = dotenv_values(ROOT / ".env")

from neo4j import GraphDatabase
from src.graph.rdf_exporter import export_rdf
from src.graph.neo4j_builder import initialize_schema, load_seed_data


# ---------------------------------------------------------------------------
# 1. Export from local Neo4j
# ---------------------------------------------------------------------------

LOCAL_URI  = "bolt://localhost:7687"
LOCAL_USER = "neo4j"
LOCAL_PASS = "neo4jpassword"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 60)
print("PASO 1 — Exportar grafo local a output/")
print("=" * 60)

local_driver = GraphDatabase.driver(LOCAL_URI, auth=(LOCAL_USER, LOCAL_PASS))
try:
    paths = export_rdf(local_driver, output_dir=ROOT / "output", timestamp=timestamp, verbose=True)
    print(f"  ✓ TTL:    {paths['ttl']}")
    print(f"  ✓ JSON-LD:{paths['jsonld']}")
    print(f"  ✓ Triples: {paths['triples']}")
finally:
    local_driver.close()


# ---------------------------------------------------------------------------
# 2. Seed AuraDB
# ---------------------------------------------------------------------------

AURA_URI  = _env_file_vals.get("NEO4J_URI")
AURA_USER = _env_file_vals.get("NEO4J_USERNAME", "neo4j")
AURA_PASS = _env_file_vals.get("NEO4J_PASSWORD")

if not AURA_URI or not AURA_PASS:
    print("\n[ERROR] NEO4J_URI o NEO4J_PASSWORD no encontrados en .env")
    sys.exit(1)

print()
print("=" * 60)
print(f"PASO 2 — Sembrar AuraDB: {AURA_URI}")
print("=" * 60)

aura_driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASS))
try:
    # Verify connection
    with aura_driver.session() as s:
        v = s.run("RETURN 1 AS ok").single()["ok"]
        print(f"  Conexión AuraDB OK (ping={v})")

    # Clean stale toy/out-of-context data before seeding
    with aura_driver.session() as s:
        r = s.run("MATCH (n:ToySpecimen) DETACH DELETE n RETURN count(n) AS c").single()
        if r["c"]:
            print(f"  Eliminados {r['c']} ToySpecimen")
        # Remove off-context labels
        for lbl in ["Persona", "Empresa", "Ciudad", "Pa\u00eds", "Edificio",
                    "Familia", "__Entity__"]:
            res = s.run(
                f"MATCH (n:`{lbl}`) DETACH DELETE n RETURN count(n) AS c"
            ).single()
            if res and res["c"]:
                print(f"  Eliminados {res['c']} nodos {lbl}")

    initialize_schema(aura_driver)
    stats = load_seed_data(aura_driver)
    print(f"  ✓ Nodes upserted:     {stats['nodes']}")
    print(f"  ✓ Relations upserted: {stats['relations']}")

    # Fix ISH Group 5 → HER2_Low (same correction as local)
    with aura_driver.session() as s:
        # Remove stale Group5->HER2_Negative if present
        s.run(
            "MATCH (g:ISHGroup {id:'Group5'})-[r:implies]->(c {id:'HER2_Negative'}) DELETE r"
        )
        s.run(
            """MATCH (g:ISHGroup {id:'Group5'}), (low:ClinicalCategory {id:'HER2_Low'})
               MERGE (g)-[r:implies]->(low)
               SET r.confidence=1.0,
                   r.evidence='IHC 2+/FISH non-amplified -> HER2-Low (ASCO/CAP 2023)',
                   r.guideline_version='ASCO_CAP_2023'"""
        )
        print("  ✓ Group5 → HER2_Low verificado")

    # Summary
    with aura_driver.session() as s:
        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels  = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"\n  Estado final AuraDB: {nodes} nodos, {rels} relaciones")

finally:
    aura_driver.close()

print()
print("=" * 60)
print("DONE — Grafo exportado y AuraDB actualizado.")
print(f"  TTL local: output/her2_knowledge_graph_{timestamp}.ttl")
print(f"  AuraDB:    {AURA_URI}")
print("=" * 60)
