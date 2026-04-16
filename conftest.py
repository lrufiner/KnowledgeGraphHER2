"""Pytest configuration — makes project root importable without pip install."""
import sys
from pathlib import Path

# Ensure project root is on the path so `from src.xxx import ...` works
sys.path.insert(0, str(Path(__file__).parent))
