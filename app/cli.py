"""
CLI for the HER2 Knowledge Graph pipeline.

Usage:
    python -m app.cli run-pipeline
    python -m app.cli run-pipeline --llm-mode ollama
    python -m app.cli run-pipeline --llm-mode openai
    python -m app.cli run-pipeline --llm-mode claude
    python -m app.cli seed-only
    python -m app.cli validate
    python -m app.cli stats
"""
from __future__ import annotations

import argparse
import sys


def cmd_run_pipeline(args):
    from src.pipeline.kg_pipeline import run_pipeline
    final = run_pipeline(llm_mode=args.llm_mode)
    return 0 if final.get("is_consistent") else 1


def cmd_seed_only(args):
    """Load only seed data (no LLM extraction). Fast and free."""
    from src.pipeline.config import PipelineConfig
    from src.graph.neo4j_builder import (
        initialize_schema, load_seed_data
    )
    cfg = PipelineConfig.from_env()
    driver = cfg.get_neo4j_driver()
    try:
        initialize_schema(driver)
        load_seed_data(driver)
        print("\n[Seed] Done! Explore at http://localhost:7474")
        print("  Query: MATCH (n) RETURN n LIMIT 80")
    finally:
        driver.close()
    return 0


def cmd_validate(args):
    """Run validation rules against the current KG state."""
    from src.pipeline.config import PipelineConfig
    from src.graph.validator import run_validation
    cfg = PipelineConfig.from_env()
    driver = cfg.get_neo4j_driver()
    try:
        report = run_validation(driver, verbose=True)
    finally:
        driver.close()
    return 0 if report.is_consistent else 1


def cmd_stats(args):
    """Print current graph statistics."""
    from src.pipeline.config import PipelineConfig
    from src.graph.neo4j_builder import get_graph_stats
    cfg = PipelineConfig.from_env()
    driver = cfg.get_neo4j_driver()
    try:
        stats = get_graph_stats(driver)
    finally:
        driver.close()
    print(f"\nGraph Statistics:")
    print(f"  Total nodes:     {stats['total_nodes']}")
    print(f"  Total relations: {stats['total_relations']}")
    for label, count in stats["node_counts"].items():
        if count > 0:
            print(f"  {label:30s}: {count}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="HER2 Knowledge Graph CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # run-pipeline
    p_run = subparsers.add_parser("run-pipeline", help="Run the full KG construction pipeline")
    p_run.add_argument(
        "--llm-mode", default="ollama",
        choices=["ollama", "openai", "claude"],
        help="LLM provider (default: ollama for local/free runs)"
    )

    # seed-only
    subparsers.add_parser("seed-only", help="Load only seed data into Neo4j (no LLM, free)")

    # validate
    subparsers.add_parser("validate", help="Run clinical validation rules against the KG")

    # stats
    subparsers.add_parser("stats", help="Print current KG statistics")

    args = parser.parse_args()

    if args.command == "run-pipeline":
        sys.exit(cmd_run_pipeline(args))
    elif args.command == "seed-only":
        sys.exit(cmd_seed_only(args))
    elif args.command == "validate":
        sys.exit(cmd_validate(args))
    elif args.command == "stats":
        sys.exit(cmd_stats(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
