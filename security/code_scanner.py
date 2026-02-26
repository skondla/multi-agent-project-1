# ─────────────────────────────────────────────────────────────────────────────
# security/code_scanner.py
#
# Programmatic wrappers for:
#   - Bandit  : Static Application Security Testing (SAST) for Python source
#   - pip-audit: Dependency vulnerability scanning (CVE / GHSA database lookup)
#
# Both wrappers:
#   1. Invoke the CLI tool via subprocess.run() with --format json
#   2. Parse the JSON output into a structured dict
#   3. Compute severity counts and a pass/fail boolean
#   4. Return a ScanResult TypedDict
#
# Usage:
#   from security.code_scanner import run_bandit, run_pip_audit
#   result = run_bandit(source_path=".")
#   result = run_pip_audit(requirements_path="requirements.txt")
#
# CLI:
#   python -m security.code_scanner [--source .] [--requirements requirements.txt]
# ─────────────────────────────────────────────────────────────────────────────
"""SAST and dependency CVE scanning via Bandit and pip-audit."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNDEFINED"]


# ─── Return type ─────────────────────────────────────────────────────────────

class ScanResult(TypedDict):
    tool: str                        # "bandit" | "pip_audit"
    success: bool                    # True if tool ran without internal error
    passed: bool                     # True if no CRITICAL/HIGH findings
    findings: list[dict]             # Normalized finding dicts
    severity_counts: dict[str, int]  # {"CRITICAL": 0, "HIGH": 2, ...}
    total_findings: int
    error: str | None                # Set if subprocess failed to run at all


# ─── Bandit ──────────────────────────────────────────────────────────────────

def run_bandit(
    source_path: str = ".",
    confidence_level: str = "LOW",
    severity_level: str = "LOW",
    timeout: int = 120,
    extra_args: list[str] | None = None,
) -> ScanResult:
    """
    Run Bandit SAST analysis on Python source code.

    Bandit exit codes:
      0 — no issues found
      1 — issues found (still valid JSON output)
      2 — internal error (e.g. no Python files found)

    Args:
        source_path: Directory or file to scan (default: current directory).
        confidence_level: Minimum confidence to report: LOW | MEDIUM | HIGH.
        severity_level: Minimum severity to report: LOW | MEDIUM | HIGH.
        timeout: Subprocess timeout in seconds.
        extra_args: Additional CLI args forwarded to Bandit.

    Returns:
        ScanResult with findings, severity_counts, passed, and success.
    """
    cmd = [
        sys.executable, "-m", "bandit",
        "-r", str(source_path),
        "-f", "json",
        "--confidence-level", confidence_level.lower(),
        "--severity-level", severity_level.lower(),
        "--quiet",
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running Bandit on %s", source_path)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        msg = "bandit not found — install with: pip install bandit"
        logger.error(msg)
        return _error_result("bandit", msg)
    except subprocess.TimeoutExpired:
        msg = f"Bandit timed out after {timeout}s"
        logger.error(msg)
        return _error_result("bandit", msg)

    # Exit code 2 = internal error; 0 or 1 = valid JSON output
    if proc.returncode == 2:
        msg = f"Bandit internal error: {proc.stderr.strip()[:300]}"
        logger.error(msg)
        return _error_result("bandit", msg)

    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        msg = f"Failed to parse Bandit JSON output: {exc}"
        logger.error(msg)
        return _error_result("bandit", msg)

    findings = _normalize_bandit_findings(raw.get("results", []))
    severity_counts = _count_severities(findings, severity_key="severity")
    passed = severity_counts.get("CRITICAL", 0) == 0 and severity_counts.get("HIGH", 0) == 0

    logger.info(
        "Bandit: %d findings — CRITICAL=%d HIGH=%d MEDIUM=%d LOW=%d",
        len(findings),
        severity_counts.get("CRITICAL", 0),
        severity_counts.get("HIGH", 0),
        severity_counts.get("MEDIUM", 0),
        severity_counts.get("LOW", 0),
    )
    return ScanResult(
        tool="bandit",
        success=True,
        passed=passed,
        findings=findings,
        severity_counts=severity_counts,
        total_findings=len(findings),
        error=None,
    )


def _normalize_bandit_findings(raw_results: list[dict]) -> list[dict]:
    """Normalize Bandit raw result dicts into a consistent schema."""
    normalized = []
    for r in raw_results:
        normalized.append({
            "id": r.get("test_id", ""),
            "name": r.get("test_name", ""),
            "severity": r.get("issue_severity", "UNDEFINED").upper(),
            "confidence": r.get("issue_confidence", "UNDEFINED").upper(),
            "description": r.get("issue_text", ""),
            "file": r.get("filename", ""),
            "line": r.get("line_number", 0),
            "snippet": r.get("code", "").strip(),
            "fix_versions": [],
            "tool": "bandit",
        })
    return normalized


# ─── pip-audit ───────────────────────────────────────────────────────────────

def run_pip_audit(
    requirements_path: str = "requirements.txt",
    timeout: int = 180,
    extra_args: list[str] | None = None,
) -> ScanResult:
    """
    Run pip-audit to check Python dependencies for known CVEs.

    pip-audit exit codes:
      0 — no vulnerabilities found
      1 — vulnerabilities found (valid JSON output)

    Args:
        requirements_path: Path to requirements.txt.
        timeout: Subprocess timeout in seconds (network-bound for CVE lookup).
        extra_args: Additional CLI args forwarded to pip-audit.

    Returns:
        ScanResult with findings, severity_counts, passed, and success.
    """
    req_path = Path(requirements_path)
    if not req_path.exists():
        msg = f"requirements file not found: {requirements_path}"
        logger.error(msg)
        return _error_result("pip_audit", msg)

    cmd = [
        sys.executable, "-m", "pip_audit",
        "--format", "json",
        "--requirement", str(req_path),
        "--progress-spinner", "off",
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running pip-audit on %s", requirements_path)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        msg = "pip-audit not found — install with: pip install pip-audit"
        logger.error(msg)
        return _error_result("pip_audit", msg)
    except subprocess.TimeoutExpired:
        msg = f"pip-audit timed out after {timeout}s"
        logger.error(msg)
        return _error_result("pip_audit", msg)

    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        msg = f"Failed to parse pip-audit JSON output: {exc}\nstdout={proc.stdout[:300]}"
        logger.error(msg)
        return _error_result("pip_audit", msg)

    findings = _normalize_pip_audit_findings(raw.get("dependencies", []))
    severity_counts = _count_severities(findings, severity_key="severity")
    passed = severity_counts.get("CRITICAL", 0) == 0 and severity_counts.get("HIGH", 0) == 0

    logger.info(
        "pip-audit: %d vulnerabilities found across %d packages",
        len(findings),
        len(raw.get("dependencies", [])),
    )
    return ScanResult(
        tool="pip_audit",
        success=True,
        passed=passed,
        findings=findings,
        severity_counts=severity_counts,
        total_findings=len(findings),
        error=None,
    )


def _normalize_pip_audit_findings(dependencies: list[dict]) -> list[dict]:
    """Normalize pip-audit dependency list into flat finding records."""
    findings = []
    for dep in dependencies:
        for vuln in dep.get("vulns", []):
            severity = _infer_pip_audit_severity(vuln)
            findings.append({
                "id": vuln.get("id", ""),
                "name": f"{dep.get('name', '')} {dep.get('version', '')}",
                "severity": severity,
                "confidence": "HIGH",
                "description": vuln.get("description", "")[:300],
                "file": "requirements.txt",
                "line": 0,
                "snippet": "",
                "fix_versions": vuln.get("fix_versions", []),
                "aliases": vuln.get("aliases", []),
                "tool": "pip_audit",
            })
    return findings


def _infer_pip_audit_severity(vuln: dict) -> str:
    """Map a pip-audit vulnerability record to a severity string."""
    # pip-audit >= 2.7 includes severity field directly
    if "severity" in vuln and vuln["severity"]:
        return str(vuln["severity"]).upper()
    # Conservatively mark as HIGH when CVE is known but severity is absent
    for alias in vuln.get("aliases", []):
        if alias.startswith("CVE-"):
            return "HIGH"
    return "HIGH"


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _count_severities(findings: list[dict], severity_key: str) -> dict[str, int]:
    """Count findings per severity level."""
    counts: dict[str, int] = {s: 0 for s in SEVERITY_ORDER}
    for finding in findings:
        sev = finding.get(severity_key, "UNDEFINED").upper()
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def _error_result(tool: str, error_message: str) -> ScanResult:
    """Return a failed ScanResult when the tool cannot be invoked."""
    return ScanResult(
        tool=tool,
        success=False,
        passed=False,
        findings=[],
        severity_counts={s: 0 for s in SEVERITY_ORDER},
        total_findings=0,
        error=error_message,
    )


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main() -> int:
    """
    CLI: python -m security.code_scanner

    Runs Bandit and pip-audit and prints a summary table.
    Exit codes:
      0 — all passed (no CRITICAL/HIGH)
      1 — findings detected
      2 — tool error
    """
    import argparse
    from rich.console import Console
    from rich.table import Table

    parser = argparse.ArgumentParser(description="Run SAST and dependency vulnerability scanning")
    parser.add_argument("--source", default=".", help="Source directory for Bandit (default: .)")
    parser.add_argument(
        "--requirements", default="requirements.txt",
        help="Path to requirements.txt for pip-audit",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    args = parser.parse_args()

    console = Console(stderr=False)
    exit_code = 0

    console.print("\n[bold cyan]Security Code Scanner[/bold cyan]")
    console.print("─" * 50)

    console.print("\n[bold]Running Bandit SAST...[/bold]")
    bandit_result = run_bandit(source_path=args.source)

    console.print("[bold]Running pip-audit dependency scan...[/bold]")
    audit_result = run_pip_audit(requirements_path=args.requirements)

    if args.json:
        print(json.dumps({"bandit": bandit_result, "pip_audit": audit_result}, indent=2))
        return 0

    for result in [bandit_result, audit_result]:
        tool_name = result["tool"].replace("_", "-").upper()
        if not result["success"]:
            console.print(f"\n[bold red]{tool_name}[/bold red]: ERROR — {result['error']}")
            exit_code = 2
            continue

        table = Table(title=f"{tool_name} Findings", show_header=True)
        table.add_column("Severity", style="bold", min_width=12)
        table.add_column("Count", justify="right", min_width=8)
        for sev in SEVERITY_ORDER:
            count = result["severity_counts"].get(sev, 0)
            style = {"CRITICAL": "red", "HIGH": "red", "MEDIUM": "yellow",
                     "LOW": "green", "UNDEFINED": "dim"}.get(sev, "white")
            table.add_row(f"[{style}]{sev}[/{style}]", str(count))
        console.print(table)

        status = "[green]PASSED[/green]" if result["passed"] else "[red]FAILED[/red]"
        console.print(f"  Status: {status}  |  Total: {result['total_findings']} findings\n")

        if not result["passed"]:
            exit_code = max(exit_code, 1)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
