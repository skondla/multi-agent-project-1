# ─────────────────────────────────────────────────────────────────────────────
# security/report.py
#
# Unified security report aggregator for the Customer Intelligence Platform.
#
# Responsibilities:
#   1. Aggregate findings from code_scanner and image_scanner results
#   2. Generate a human-readable Markdown summary
#   3. Generate SARIF 2.1.0 output for GitHub Security tab
#   4. Enforce a configurable severity gate (block on CRITICAL/HIGH > threshold)
#
# Environment variables (configurable):
#   CRITICAL_THRESHOLD  — Max allowed CRITICAL findings (default: 0)
#   HIGH_THRESHOLD      — Max allowed HIGH findings (default: 5)
#
# Usage:
#   from security.report import SecurityReport
#   from security.code_scanner import run_bandit, run_pip_audit
#   from security.image_scanner import scan_filesystem
#
#   report = SecurityReport([run_bandit("."), run_pip_audit(), scan_filesystem(".")])
#   report.print_markdown()
#   report.write_sarif("security-report.sarif")
#   gate_passed = report.check_gate()
#
# CLI:
#   python -m security.report [--source .] [--requirements requirements.txt]
#                              [--image-tag TAG] [--sarif-out FILE]
#                              [--critical-threshold N] [--high-threshold N]
# ─────────────────────────────────────────────────────────────────────────────
"""Unified security report: Markdown, SARIF 2.1.0, and severity gate."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Union

from security.code_scanner import ScanResult, run_bandit, run_pip_audit
from security.image_scanner import ImageScanResult, scan_filesystem, scan_image

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Configurable thresholds ─────────────────────────────────────────────────
CRITICAL_THRESHOLD: int = int(os.environ.get("CRITICAL_THRESHOLD", "0"))
HIGH_THRESHOLD: int = int(os.environ.get("HIGH_THRESHOLD", "5"))

# ─── SARIF constants ──────────────────────────────────────────────────────────
SARIF_SCHEMA = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/"
    "Schemata/sarif-schema-2.1.0.json"
)
SARIF_VERSION = "2.1.0"

SARIF_LEVEL_MAP: dict[str, str] = {
    "CRITICAL": "error",
    "HIGH": "error",
    "MEDIUM": "warning",
    "LOW": "note",
    "UNDEFINED": "note",
    "UNKNOWN": "note",
}

AnyResult = Union[ScanResult, ImageScanResult]


class SecurityReport:
    """
    Aggregates scan results from all security tools and produces reports.

    Accepts a list of ScanResult (from code_scanner) and/or ImageScanResult
    (from image_scanner) objects and provides:
      - Markdown summary (for humans and GitHub Step Summary)
      - SARIF 2.1.0 document (for GitHub Security tab via upload-sarif action)
      - Severity gate check (pass/fail with configurable thresholds)
    """

    def __init__(
        self,
        results: list[AnyResult],
        critical_threshold: int = CRITICAL_THRESHOLD,
        high_threshold: int = HIGH_THRESHOLD,
    ) -> None:
        self.results = results
        self.critical_threshold = critical_threshold
        self.high_threshold = high_threshold
        self._all_findings: list[dict] = []
        self._severity_totals: dict[str, int] = {}
        self._aggregate()

    def _aggregate(self) -> None:
        """Combine all findings from all scanners into a flat list."""
        totals: dict[str, int] = {}
        all_findings: list[dict] = []

        for result in self.results:
            if not result["success"]:
                continue
            all_findings.extend(result["findings"])
            for sev, count in result["severity_counts"].items():
                totals[sev] = totals.get(sev, 0) + count

        self._all_findings = all_findings
        self._severity_totals = totals

    # ─── Severity gate ───────────────────────────────────────────────────────

    def check_gate(self) -> bool:
        """
        Evaluate the severity gate.

        Returns True if CRITICAL and HIGH counts are within configured thresholds.
        This is the authoritative pass/fail signal for the security-gate CI job.
        """
        critical = self._severity_totals.get("CRITICAL", 0)
        high = self._severity_totals.get("HIGH", 0)

        if critical > self.critical_threshold:
            logger.warning(
                "Gate FAILED: %d CRITICAL findings exceed threshold %d",
                critical, self.critical_threshold,
            )
            return False
        if high > self.high_threshold:
            logger.warning(
                "Gate FAILED: %d HIGH findings exceed threshold %d",
                high, self.high_threshold,
            )
            return False

        logger.info(
            "Gate PASSED: CRITICAL=%d (limit=%d), HIGH=%d (limit=%d)",
            critical, self.critical_threshold, high, self.high_threshold,
        )
        return True

    # ─── Markdown report ─────────────────────────────────────────────────────

    def to_markdown(self) -> str:
        """
        Generate a Markdown security summary.

        Output is suitable for:
          - GitHub Actions Step Summary ($GITHUB_STEP_SUMMARY)
          - PR sticky comments
          - Standalone Markdown artifacts
        """
        gate_passed = self.check_gate()
        gate_icon = "PASSED" if gate_passed else "FAILED"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "# Security Scan Report",
            "",
            f"**Generated**: {timestamp}  ",
            f"**Gate Status**: `{gate_icon}`  ",
            f"**CRITICAL threshold**: {self.critical_threshold}  ",
            f"**HIGH threshold**: {self.high_threshold}",
            "",
            "## Summary by Severity",
            "",
            "| Severity | Count |",
            "|----------|-------|",
        ]

        sev_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN", "UNDEFINED"]
        for sev in sev_order:
            count = self._severity_totals.get(sev, 0)
            bold = sev in ("CRITICAL", "HIGH") and count > 0
            prefix = "**" if bold else ""
            lines.append(f"| {prefix}{sev}{prefix} | {prefix}{count}{prefix} |")

        lines += ["", "## Results by Tool", ""]

        for result in self.results:
            tool_name = result["tool"].upper().replace("_", " ")
            if not result["success"]:
                status_icon = "ERROR"
            elif result["passed"]:
                status_icon = "PASS"
            else:
                status_icon = "FAIL"

            lines.append(f"### {tool_name}  `[{status_icon}]`")
            lines.append("")

            if not result["success"]:
                lines.append(f"> **Error**: {result['error']}")
                lines.append("")
                continue

            counts = result["severity_counts"]
            lines.append(
                f"Total findings: **{result['total_findings']}** — "
                f"CRITICAL: {counts.get('CRITICAL', 0)}, "
                f"HIGH: {counts.get('HIGH', 0)}, "
                f"MEDIUM: {counts.get('MEDIUM', 0)}, "
                f"LOW: {counts.get('LOW', 0)}"
            )
            lines.append("")

            # Show top 10 CRITICAL/HIGH findings for this tool
            top_findings = [
                f for f in result["findings"]
                if f.get("severity") in ("CRITICAL", "HIGH")
            ][:10]

            if top_findings:
                lines += [
                    "**Top CRITICAL/HIGH findings:**",
                    "",
                    "| ID | Package / File | Severity | Description |",
                    "|----|---------------|----------|-------------|",
                ]
                for f in top_findings:
                    desc = (f.get("description") or "")[:80].replace("|", "\\|")
                    lines.append(
                        f"| `{f.get('id', 'N/A')}` "
                        f"| `{f.get('name', 'N/A')}` "
                        f"| **{f.get('severity')}** "
                        f"| {desc} |"
                    )
                lines.append("")

        # Gate decision table
        critical = self._severity_totals.get("CRITICAL", 0)
        high = self._severity_totals.get("HIGH", 0)
        lines += [
            "## Gate Decision",
            "",
            "| Metric | Actual | Threshold | Status |",
            "|--------|--------|-----------|--------|",
            (
                f"| CRITICAL | {critical} | ≤ {self.critical_threshold} | "
                f"{'PASS' if critical <= self.critical_threshold else 'FAIL'} |"
            ),
            (
                f"| HIGH | {high} | ≤ {self.high_threshold} | "
                f"{'PASS' if high <= self.high_threshold else 'FAIL'} |"
            ),
            "",
        ]

        return "\n".join(lines)

    def print_markdown(self) -> None:
        """Print Markdown report to stdout and append to GitHub Step Summary."""
        md = self.to_markdown()
        print(md)

        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            try:
                with open(summary_file, "a", encoding="utf-8") as fh:
                    fh.write(md)
                    fh.write("\n")
            except OSError as exc:
                logger.warning("Could not write to GITHUB_STEP_SUMMARY: %s", exc)

    # ─── SARIF 2.1.0 ─────────────────────────────────────────────────────────

    def generate_sarif(self) -> dict:
        """
        Generate a SARIF 2.1.0 document for all scan results.

        Each scanner produces one SARIF `run` with:
          - `tool.driver` describing the scanner
          - `rules[]` — one rule per unique finding ID
          - `results[]` — one result per finding with location information

        Upload to GitHub Code Scanning via:
          github/codeql-action/upload-sarif@v3

        Returns:
            SARIF document as a Python dict (serialize with json.dumps).
        """
        runs = []
        for result in self.results:
            if not result["success"]:
                continue
            run = _build_sarif_run(result)
            runs.append(run)

        return {
            "$schema": SARIF_SCHEMA,
            "version": SARIF_VERSION,
            "runs": runs,
        }

    def write_sarif(self, output_path: str = "security-report.sarif") -> None:
        """Serialize SARIF document to a JSON file."""
        sarif = self.generate_sarif()
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(sarif, fh, indent=2)
        logger.info("SARIF report written to %s", output_path)


# ─── SARIF builder helpers ────────────────────────────────────────────────────

def _build_sarif_run(result: AnyResult) -> dict:
    """Build a single SARIF run from one scanner result."""
    tool_name = result["tool"]
    findings = result["findings"]

    # Deduplicate rules by finding ID
    rules_seen: dict[str, dict] = {}
    for f in findings:
        rule_id = f.get("id") or f.get("name") or "UNKNOWN"
        if rule_id not in rules_seen:
            rules_seen[rule_id] = {
                "id": rule_id,
                "name": _to_pascal_case(f.get("name", rule_id)),
                "shortDescription": {"text": (f.get("description", rule_id) or rule_id)[:200]},
                "fullDescription": {"text": f.get("description", rule_id) or rule_id},
                "defaultConfiguration": {
                    "level": SARIF_LEVEL_MAP.get(f.get("severity", "LOW"), "note")
                },
                "helpUri": (f.get("references") or [""])[0] if f.get("references") else "",
            }

    sarif_results = []
    for f in findings:
        rule_id = f.get("id") or f.get("name") or "UNKNOWN"
        level = SARIF_LEVEL_MAP.get(f.get("severity", "LOW"), "note")
        message_text = (
            f"{f.get('description', 'No description')[:300]} "
            f"[Severity: {f.get('severity', 'UNKNOWN')}]"
        )

        sarif_result: dict = {
            "ruleId": rule_id,
            "level": level,
            "message": {"text": message_text},
        }

        location = _build_sarif_location(f)
        if location:
            sarif_result["locations"] = [location]

        sarif_results.append(sarif_result)

    return {
        "tool": {
            "driver": {
                "name": _tool_display_name(tool_name),
                "version": "latest",
                "informationUri": _tool_info_uri(tool_name),
                "rules": list(rules_seen.values()),
            }
        },
        "results": sarif_results,
    }


def _build_sarif_location(finding: dict) -> dict | None:
    """Build a SARIF physicalLocation from a finding dict."""
    file_path = finding.get("file", "")
    if not file_path:
        return None

    location: dict = {
        "physicalLocation": {
            "artifactLocation": {
                "uri": file_path.lstrip("/"),
                "uriBaseId": "%SRCROOT%",
            }
        }
    }
    line_number = finding.get("line", 0)
    if line_number and line_number > 0:
        location["physicalLocation"]["region"] = {
            "startLine": line_number,
            "snippet": {"text": finding.get("snippet", "")[:200]},
        }
    return location


def _to_pascal_case(s: str) -> str:
    return "".join(
        word.capitalize()
        for word in s.replace("-", " ").replace("_", " ").split()
    )


def _tool_display_name(tool: str) -> str:
    names = {
        "bandit": "Bandit Python SAST",
        "pip_audit": "pip-audit Dependency CVE",
        "trivy_fs": "Trivy Filesystem Scanner",
        "trivy_image": "Trivy Container Image Scanner",
    }
    return names.get(tool, tool)


def _tool_info_uri(tool: str) -> str:
    uris = {
        "bandit": "https://bandit.readthedocs.io/",
        "pip_audit": "https://pypi.org/project/pip-audit/",
        "trivy_fs": "https://aquasecurity.github.io/trivy/",
        "trivy_image": "https://aquasecurity.github.io/trivy/",
    }
    return uris.get(tool, "https://github.com/")


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main() -> int:
    """
    CLI: python -m security.report

    Runs all configured scanners and generates a unified report.
    Exit codes:
      0 — gate passed (CRITICAL/HIGH within thresholds)
      1 — gate failed
      2 — tool error
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Run full security scan and generate unified report"
    )
    parser.add_argument("--source", default=".", help="Source directory for Bandit + Trivy fs")
    parser.add_argument(
        "--requirements", default="requirements.txt",
        help="Path to requirements.txt for pip-audit",
    )
    parser.add_argument("--image-tag", default="", help="Docker image tag for Trivy image scan")
    parser.add_argument(
        "--sarif-out", default="security-report.sarif",
        help="SARIF output file path (default: security-report.sarif)",
    )
    parser.add_argument("--no-sarif", action="store_true", help="Skip SARIF file generation")
    parser.add_argument(
        "--critical-threshold", type=int, default=CRITICAL_THRESHOLD,
        help=f"Max CRITICAL findings allowed (default: {CRITICAL_THRESHOLD})",
    )
    parser.add_argument(
        "--high-threshold", type=int, default=HIGH_THRESHOLD,
        help=f"Max HIGH findings allowed (default: {HIGH_THRESHOLD})",
    )
    parser.add_argument("--json", action="store_true", help="Emit SARIF as JSON to stdout")
    args = parser.parse_args()

    results: list[AnyResult] = []

    print("─" * 60)
    print("  Customer Intelligence Platform — Security Report")
    print("─" * 60)

    print("\n[1/3] Running Bandit SAST...")
    results.append(run_bandit(source_path=args.source))

    print("[2/3] Running pip-audit dependency scan...")
    results.append(run_pip_audit(requirements_path=args.requirements))

    print("[3/3] Running Trivy filesystem scan...")
    results.append(scan_filesystem(path=args.source))

    if args.image_tag:
        print(f"[4/4] Running Trivy image scan on {args.image_tag}...")
        results.append(scan_image(image_tag=args.image_tag))

    report = SecurityReport(
        results,
        critical_threshold=args.critical_threshold,
        high_threshold=args.high_threshold,
    )

    if args.json:
        print(json.dumps(report.generate_sarif(), indent=2))
    else:
        report.print_markdown()

    if not args.no_sarif:
        report.write_sarif(args.sarif_out)
        print(f"\nSARIF written to: {args.sarif_out}")

    gate_passed = report.check_gate()
    status = "PASSED" if gate_passed else "FAILED"
    print(f"\n{'─' * 60}")
    print(f"  Security Gate: {status}")
    print(f"{'─' * 60}\n")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    sys.exit(main())
