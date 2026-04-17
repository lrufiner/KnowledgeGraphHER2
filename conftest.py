"""Pytest configuration — makes project root importable without pip install."""
import sys
import os
from pathlib import Path

import pytest

# Ensure project root is on the path so `from src.xxx import ...` works
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _neo4j_available() -> bool:
    """Return True if a Neo4j instance is reachable on the configured URL."""
    try:
        from neo4j import GraphDatabase
        uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
        user     = os.getenv("NEO4J_USER",     "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def neo4j_driver():
    """
    Session-scoped fixture that provides a Neo4j driver.
    Tests using this fixture are automatically skipped when Neo4j is unavailable.
    """
    if not _neo4j_available():
        pytest.skip("Neo4j not available — skipping integration test")

    from neo4j import GraphDatabase
    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    yield driver
    driver.close()
