"""
Clinical validation runner — executes all Cypher-based validation rules
against the Neo4j KG and produces a ValidationReport.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Driver

from src.domain.models import ValidationReport, ValidationResult, ValidationSeverity
from src.domain.validation_rules import VALIDATION_RULES


def run_validation(driver: Driver, verbose: bool = True) -> ValidationReport:
    """
    Run all clinical validation rules against the Neo4j KG.

    Returns a ValidationReport with pass/fail status for each rule.
    """
    report = ValidationReport()

    if verbose:
        print(f"\n[Validation] Running {len(VALIDATION_RULES)} clinical rules...")

    with driver.session() as session:
        for rule in VALIDATION_RULES:
            rule_id = rule["rule_id"]
            severity = rule["severity"]
            source = rule["source"]
            cypher = rule["cypher"].strip()
            msg_pass = rule["message_pass"]
            msg_fail = rule["message_fail"]

            try:
                result = session.run(cypher)
                record = result.single()
                valid = bool(record["valid"]) if record and "valid" in record else False
            except Exception as e:
                valid = False
                msg_fail = f"{msg_fail} | Error: {str(e)[:120]}"

            vr = ValidationResult(
                rule_id=rule_id,
                valid=valid,
                severity=severity,
                message=msg_pass if valid else msg_fail,
                source=source,
            )
            report.add(vr)

            if verbose:
                status = "[PASS]" if valid else ("[FAIL]" if severity == ValidationSeverity.CRITICAL else "[WARN]")
                print(f"  {status} [{severity.value}] {rule_id}: {vr.message[:80]}")

    summary = report.summary()
    if verbose:
        print(f"\n[Validation] Summary: {summary['passed']}/{summary['total']} passed | "
              f"{summary['critical']} critical failures | "
              f"Consistent: {report.is_consistent}")

    return report
