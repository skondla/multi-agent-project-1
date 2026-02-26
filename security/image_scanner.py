# ─────────────────────────────────────────────────────────────────────────────
# security/image_scanner.py
#
# Programmatic wrapper for the Trivy CLI vulnerability scanner.
# Supports two scan modes:
#   scan_filesystem(path)     — Scan repo filesystem for dependency manifests
#                               and embedded secrets. Fast, no Docker required.
#   scan_image(image_tag)     — Scan a full Docker image (OS packages + app deps).
#                               Requires the image to be loaded in the local daemon.
#
# Both return an ImageScanResult TypedDict with normalized findings,
# severity counts, and a pass/fail status based on CRITICAL/HIGH thresholds.
#
# Usage:
#   from security.image_scanner import scan_filesystem, scan_image
#   result = scan_filesystem(path=".")
#   result = scan_image(image_tag="customer-intelligence:abc123")
#
# CLI:
#   python -m security.image_scanner --mode fs --path .
#   python -m security.image_scanner --mode image --tag customer-intelligence:latest
# ─────────────────────────────────────────────────────────────────────────────
"""Container and filesystem vulnerability scanning via Trivy CLI."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from typing import TypedDict

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]


# ─── Return type ─────────────────────────────────────────────────────────────

class ImageScanResult(TypedDict):
    tool: str                         # "trivy_fs" | "trivy_image"
    target: str                       # path or image tag scanned
    success: bool
    passed: bool                      # True if CRITICAL==0 and HIGH==0
    findings: list[dict]              # Normalized vulnerability records
    severity_counts: dict[str, int]
    total_findings: int
    results_by_target: list[dict]     # Raw Trivy Results[] for reference
    error: str | None


# ─── Filesystem scan ─────────────────────────────────────────────────────────

def scan_filesystem(
    path: str = ".",
    severity: str = "UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL",
    scanners: str = "vuln,secret",
    timeout: int = 120,
    trivy_bin: str = "trivy",
    extra_args: list[str] | None = None,
) -> ImageScanResult:
    """
    Run Trivy filesystem scan on a local directory.

    Scans dependency manifests (requirements.txt, package.json, etc.) and
    embedded secrets without requiring Docker. Fast enough for every PR.

    Args:
        path: Directory to scan (default: current directory).
        severity: Comma-separated severity levels to include.
        scanners: Comma-separated Trivy scanner types ("vuln", "secret", "config").
        timeout: Subprocess timeout in seconds.
        trivy_bin: Path to the trivy binary (default: "trivy" on PATH).
        extra_args: Additional CLI args forwarded to Trivy.

    Returns:
        ImageScanResult with findings and severity counts.
    """
    cmd = [
        trivy_bin, "fs",
        "--format", "json",
        "--severity", severity,
        "--scanners", scanners,
        "--exit-code", "0",   # We gate separately in report.py
        path,
    ]
    if extra_args:
        cmd.extend(extra_args)

    return _run_trivy(cmd, tool="trivy_fs", target=path, timeout=timeout)


# ─── Image scan ──────────────────────────────────────────────────────────────

def scan_image(
    image_tag: str,
    severity: str = "UNKNOWN,LOW,MEDIUM,HIGH,CRITICAL",
    scanners: str = "vuln,secret",
    timeout: int = 300,
    trivy_bin: str = "trivy",
    extra_args: list[str] | None = None,
) -> ImageScanResult:
    """
    Run Trivy container image scan.

    Scans all layers of a Docker image for OS package vulnerabilities,
    application dependency CVEs, and embedded secrets.

    The image must be available in the local Docker daemon. In CI, build
    with docker/build-push-action using `load: true` and `push: false`.

    Args:
        image_tag: Full image reference, e.g. "customer-intelligence:abc123".
        severity: Comma-separated severity levels to include.
        scanners: Comma-separated Trivy scanner types.
        timeout: Subprocess timeout (Trivy downloads DB on first run: ~60s).
        trivy_bin: Path to the trivy binary.
        extra_args: Additional CLI args forwarded to Trivy.

    Returns:
        ImageScanResult with findings and severity counts.
    """
    cmd = [
        trivy_bin, "image",
        "--format", "json",
        "--severity", severity,
        "--scanners", scanners,
        "--exit-code", "0",
        image_tag,
    ]
    if extra_args:
        cmd.extend(extra_args)

    return _run_trivy(cmd, tool="trivy_image", target=image_tag, timeout=timeout)


# ─── Shared runner ───────────────────────────────────────────────────────────

def _run_trivy(
    cmd: list[str],
    tool: str,
    target: str,
    timeout: int,
) -> ImageScanResult:
    """Execute a Trivy command and parse its JSON output."""
    logger.info("Running: %s", " ".join(cmd))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        msg = (
            "trivy not found. Install from https://github.com/aquasecurity/trivy/releases "
            "or use aquasecurity/trivy-action in GitHub Actions."
        )
        logger.error(msg)
        return _error_result(tool, target, msg)
    except subprocess.TimeoutExpired:
        msg = f"Trivy timed out after {timeout}s scanning {target}"
        logger.error(msg)
        return _error_result(tool, target, msg)

    # Exit code 0 = clean, 1 = findings found (we use --exit-code 0 so both are 0)
    # Non-zero from Trivy itself (e.g., 2) means a tool error
    if proc.returncode not in (0, 1):
        msg = f"Trivy exited with code {proc.returncode}: {proc.stderr.strip()[:500]}"
        logger.error(msg)
        return _error_result(tool, target, msg)

    if not proc.stdout.strip():
        logger.warning("Trivy produced no output for target: %s (treating as clean)", target)
        return _clean_result(tool, target)

    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        msg = f"Failed to parse Trivy JSON output: {exc}"
        logger.error(msg)
        return _error_result(tool, target, msg)

    results = raw.get("Results", [])
    findings = _normalize_trivy_findings(results)
    severity_counts = _count_severities(findings)
    passed = (
        severity_counts.get("CRITICAL", 0) == 0
        and severity_counts.get("HIGH", 0) == 0
    )

    logger.info(
        "Trivy %s scan of %s: %d findings — CRITICAL=%d HIGH=%d MEDIUM=%d LOW=%d",
        tool.split("_")[-1],
        target,
        len(findings),
        severity_counts.get("CRITICAL", 0),
        severity_counts.get("HIGH", 0),
        severity_counts.get("MEDIUM", 0),
        severity_counts.get("LOW", 0),
    )
    return ImageScanResult(
        tool=tool,
        target=target,
        success=True,
        passed=passed,
        findings=findings,
        severity_counts=severity_counts,
        total_findings=len(findings),
        results_by_target=results,
        error=None,
    )


def _normalize_trivy_findings(results: list[dict]) -> list[dict]:
    """
    Flatten Trivy Results[].Vulnerabilities[] into normalized finding records.

    Trivy JSON v2 schema per vulnerability:
      VulnerabilityID, PkgName, InstalledVersion, FixedVersion,
      Severity, Title, Description, PrimaryURL
    """
    findings = []
    for result in results:
        target_name = result.get("Target", "")
        target_type = result.get("Type", "")
        for vuln in result.get("Vulnerabilities") or []:
            findings.append({
                "id": vuln.get("VulnerabilityID", ""),
                "name": f"{vuln.get('PkgName', '')} {vuln.get('InstalledVersion', '')}",
                "severity": vuln.get("Severity", "UNKNOWN").upper(),
                "confidence": "HIGH",
                "description": (vuln.get("Title") or vuln.get("Description", ""))[:200],
                "file": target_name,
                "line": 0,
                "snippet": "",
                "fix_versions": [vuln.get("FixedVersion", "")] if vuln.get("FixedVersion") else [],
                "references": [vuln.get("PrimaryURL", "")] if vuln.get("PrimaryURL") else [],
                "pkg_name": vuln.get("PkgName", ""),
                "installed_version": vuln.get("InstalledVersion", ""),
                "fixed_version": vuln.get("FixedVersion", ""),
                "target_type": target_type,
                "tool": "trivy",
            })
    return findings


def _count_severities(findings: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {s: 0 for s in SEVERITY_ORDER}
    for f in findings:
        sev = f.get("severity", "UNKNOWN").upper()
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def _clean_result(tool: str, target: str) -> ImageScanResult:
    return ImageScanResult(
        tool=tool, target=target, success=True, passed=True,
        findings=[], severity_counts={s: 0 for s in SEVERITY_ORDER},
        total_findings=0, results_by_target=[], error=None,
    )


def _error_result(tool: str, target: str, error_message: str) -> ImageScanResult:
    return ImageScanResult(
        tool=tool, target=target, success=False, passed=False,
        findings=[], severity_counts={s: 0 for s in SEVERITY_ORDER},
        total_findings=0, results_by_target=[], error=error_message,
    )


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main() -> int:
    """
    CLI: python -m security.image_scanner

    Exit codes:
      0 — no CRITICAL/HIGH findings (gate passed)
      1 — findings above threshold
      2 — tool error (Trivy not installed or scan failed)
    """
    import argparse
    from rich.console import Console
    from rich.table import Table

    parser = argparse.ArgumentParser(description="Run Trivy vulnerability scanner")
    parser.add_argument(
        "--mode", choices=["fs", "image"], default="fs",
        help="Scan mode: 'fs' for filesystem (default), 'image' for container image",
    )
    parser.add_argument("--path", default=".", help="Path to scan (--mode fs)")
    parser.add_argument("--tag", default="", help="Image tag to scan (--mode image)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    args = parser.parse_args()

    console = Console(stderr=False)
    console.print("\n[bold cyan]Trivy Image Scanner[/bold cyan]")
    console.print("─" * 50)

    if args.mode == "fs":
        console.print(f"\n[bold]Running Trivy filesystem scan on {args.path!r}...[/bold]")
        result = scan_filesystem(path=args.path)
    else:
        if not args.tag:
            console.print("[bold red]--tag is required for --mode image[/bold red]")
            return 2
        console.print(f"\n[bold]Running Trivy image scan on {args.tag!r}...[/bold]")
        result = scan_image(image_tag=args.tag)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1

    if not result["success"]:
        console.print(f"\n[bold red]ERROR:[/bold red] {result['error']}")
        return 2

    table = Table(
        title=f"Trivy {args.mode.upper()} Scan — {result['target']}", show_header=True
    )
    table.add_column("Severity", style="bold", min_width=12)
    table.add_column("Count", justify="right", min_width=8)
    for sev in SEVERITY_ORDER:
        count = result["severity_counts"].get(sev, 0)
        style = {"CRITICAL": "red", "HIGH": "red", "MEDIUM": "yellow",
                 "LOW": "green", "UNKNOWN": "dim"}.get(sev, "white")
        table.add_row(f"[{style}]{sev}[/{style}]", str(count))
    console.print(table)

    status = "[green]PASSED[/green]" if result["passed"] else "[red]FAILED[/red]"
    console.print(f"  Status: {status}  |  Total: {result['total_findings']} findings\n")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
