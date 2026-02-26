# ─────────────────────────────────────────────────────────────────────────────
# security/__init__.py
#
# Security scanning package for the Customer Intelligence Platform.
# Provides programmatic wrappers for Bandit, pip-audit, and Trivy.
#
# Sub-modules:
#   code_scanner   — Bandit (SAST) + pip-audit (dependency CVE)
#   image_scanner  — Trivy CLI (filesystem + container image)
#   report         — Unified aggregator, Markdown/SARIF output, severity gate
#
# Quick start:
#   python -m security.code_scanner --source .
#   python -m security.image_scanner --mode fs --path .
#   python -m security.report --source . --requirements requirements.txt
# ─────────────────────────────────────────────────────────────────────────────
"""Security scanning utilities: SAST, dependency CVE, and container image scanning."""

__version__ = "1.0.0"

__all__ = ["code_scanner", "image_scanner", "owasp_ai_scanner", "report"]
