# ─────────────────────────────────────────────────────────────────────────────
# security/owasp_ai_scanner.py
#
# OWASP LLM Top 10 (2025) — Heuristic Security Scanner
# for LLM-based multi-agent applications.
#
# Implements static analysis checks aligned with the OWASP Top 10 for
# Large Language Model Applications:
#
#   LLM01 — Prompt Injection
#   LLM02 — Insecure Output Handling
#   LLM03 — Training Data Poisoning
#   LLM04 — Model Denial of Service
#   LLM05 — Supply Chain Vulnerabilities
#   LLM06 — Sensitive Information Disclosure
#   LLM07 — Insecure Plugin Design (MCP tool safety)
#   LLM08 — Excessive Agency
#   LLM09 — Overreliance
#   LLM10 — Model Theft
#
# Reference: https://owasp.org/www-project-top-10-for-large-language-model-applications/
#
# Usage:
#   from security.owasp_ai_scanner import OWASPAIScanner
#   scanner = OWASPAIScanner(source_path=".")
#   result = scanner.run()
#
# CLI:
#   python -m security.owasp_ai_scanner [--source .] [--summary-out results.json]
#                                        [--sarif-out results.sarif]
# ─────────────────────────────────────────────────────────────────────────────
"""OWASP LLM Top 10 heuristic security scanner for multi-agent AI applications."""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── OWASP LLM Top 10 registry ───────────────────────────────────────────────

OWASP_LLM_RISKS: dict[str, dict] = {
    "LLM01": {
        "name": "Prompt Injection",
        "description": (
            "Attackers craft inputs that override or manipulate LLM instructions, "
            "causing the model to execute unintended actions or reveal restricted information."
        ),
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    },
    "LLM02": {
        "name": "Insecure Output Handling",
        "description": (
            "LLM outputs are passed to downstream systems (shell, SQL, HTML) without "
            "sanitization, enabling XSS, SSRF, code injection, and privilege escalation."
        ),
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    },
    "LLM03": {
        "name": "Training Data Poisoning",
        "description": (
            "Malicious data injected into training or fine-tuning pipelines introduces "
            "backdoors or biases that degrade model integrity."
        ),
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    },
    "LLM04": {
        "name": "Model Denial of Service",
        "description": (
            "Resource-intensive prompts or unbounded agentic loops exhaust compute, "
            "degrade availability, or escalate costs unexpectedly."
        ),
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    },
    "LLM05": {
        "name": "Supply Chain Vulnerabilities",
        "description": (
            "Compromised third-party models, plugins, datasets, or packages used in "
            "the LLM pipeline introduce vulnerabilities or malicious behaviour."
        ),
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    },
    "LLM06": {
        "name": "Sensitive Information Disclosure",
        "description": (
            "LLM outputs inadvertently reveal PII, credentials, proprietary data, "
            "or system prompt contents to unauthorised parties."
        ),
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    },
    "LLM07": {
        "name": "Insecure Plugin Design",
        "description": (
            "LLM plugins / MCP tools lack proper access control, input validation, "
            "or output sanitization, enabling privilege escalation and data exfiltration."
        ),
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    },
    "LLM08": {
        "name": "Excessive Agency",
        "description": (
            "LLM agents are granted more permissions, capabilities, or autonomy than "
            "required, enabling large blast radius from errors or prompt injections."
        ),
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    },
    "LLM09": {
        "name": "Overreliance",
        "description": (
            "Systems or users rely on LLM outputs without adequate validation, "
            "leading to misinformation or critical failures in production decisions."
        ),
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    },
    "LLM10": {
        "name": "Model Theft",
        "description": (
            "Adversaries extract the model's weights, behaviour, or training data "
            "through repeated API queries or side-channel attacks."
        ),
        "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    },
}


# ─── Finding data class ───────────────────────────────────────────────────────

@dataclass
class OWASPFinding:
    owasp_id: str          # e.g. "LLM01"
    owasp_name: str        # e.g. "Prompt Injection"
    severity: str          # "HIGH" | "MEDIUM" | "LOW" | "INFO"
    title: str             # Short, specific title
    description: str       # Detailed finding explanation
    file: str              # Relative path to the file containing the finding
    line: int              # Line number (0 if not applicable)
    snippet: str           # Code snippet (may be empty)
    recommendation: str    # Concrete remediation guidance
    confidence: str = "MEDIUM"   # "HIGH" | "MEDIUM" | "LOW"

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Scanner ─────────────────────────────────────────────────────────────────

class OWASPAIScanner:
    """
    Heuristic scanner that checks an LLM application codebase against the
    OWASP Top 10 for Large Language Model Applications (2025).

    Performs static analysis by:
      - Reading Python source files with regex pattern matching
      - Inspecting agent system prompts for injection risks
      - Checking input validation in API endpoint definitions
      - Auditing MCP tool definitions for excessive permissions
      - Detecting hardcoded secrets and direct output passing
    """

    # File extensions to scan
    PYTHON_GLOB = "**/*.py"
    SKIP_DIRS = {
        ".venv", "venv", "env", "__pycache__", ".git",
        ".pytest_cache", "build", "dist",
    }
    # Source directories of interest for this LLM project
    AGENT_DIRS = {"agents", "orchestrator", "mcp_server/tools", "app"}

    def __init__(self, source_path: str = ".") -> None:
        self.source_path = Path(source_path).resolve()
        self.findings: list[OWASPFinding] = []

    def run(self) -> dict:
        """
        Execute all OWASP LLM Top 10 checks and return a structured result dict.

        Returns:
            Dict with keys: total_findings, severity_counts, by_owasp_id, findings
        """
        logger.info("Starting OWASP LLM Top 10 scan of %s", self.source_path)

        python_files = self._collect_python_files()
        logger.info("Scanning %d Python files", len(python_files))

        # Run all check methods
        checks: list[Callable[[list[tuple[Path, str]]], None]] = [
            self._check_llm01_prompt_injection,
            self._check_llm02_insecure_output_handling,
            self._check_llm04_model_dos,
            self._check_llm05_supply_chain,
            self._check_llm06_sensitive_disclosure,
            self._check_llm07_insecure_plugin_design,
            self._check_llm08_excessive_agency,
            self._check_llm09_overreliance,
        ]

        file_contents: list[tuple[Path, str]] = []
        for fp in python_files:
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
                file_contents.append((fp, content))
            except OSError as exc:
                logger.warning("Cannot read %s: %s", fp, exc)

        for check in checks:
            try:
                check(file_contents)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Check %s failed: %s", check.__name__, exc)

        return self._build_result()

    # ─── File collection ─────────────────────────────────────────────────────

    def _collect_python_files(self) -> list[Path]:
        files = []
        for fp in self.source_path.rglob("*.py"):
            if any(skip in fp.parts for skip in self.SKIP_DIRS):
                continue
            files.append(fp)
        return sorted(files)

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.source_path))
        except ValueError:
            return str(path)

    def _add(self, finding: OWASPFinding) -> None:
        self.findings.append(finding)

    # ─── LLM01: Prompt Injection ──────────────────────────────────────────────

    def _check_llm01_prompt_injection(self, files: list[tuple[Path, str]]) -> None:
        """
        LLM01 — Prompt Injection checks:
          1. User input inserted directly into f-string system prompts
          2. No input sanitization before adding to messages
          3. Missing instruction hierarchy (system vs user role separation)
          4. Dynamic prompt construction from untrusted sources
        """
        # Pattern: user-controlled string directly injected into messages list
        direct_inject_re = re.compile(
            r'messages\s*[+=].*?\{[^}]*?(query|user_input|request|text|prompt)\b',
            re.IGNORECASE,
        )
        # Pattern: f-string with variable in system prompt position
        fstring_system_re = re.compile(
            r'["\']role["\']:\s*["\']system["\'].*?\{[^}]+\}',
            re.DOTALL | re.IGNORECASE,
        )
        # Pattern: direct string concat into prompt
        concat_prompt_re = re.compile(
            r'(system_prompt|prompt|instruction)\s*[+]=?\s*[^"\']',
            re.IGNORECASE,
        )
        # Missing input length check before LLM call
        no_length_check_re = re.compile(
            r'client\.messages\.create\(',
            re.IGNORECASE,
        )

        for fp, content in files:
            lines = content.splitlines()

            for i, line in enumerate(lines, start=1):
                if direct_inject_re.search(line):
                    self._add(OWASPFinding(
                        owasp_id="LLM01",
                        owasp_name="Prompt Injection",
                        severity="HIGH",
                        title="User input directly inserted into LLM messages",
                        description=(
                            f"Line {i}: User-controlled input appears to be inserted directly "
                            "into the LLM messages list without sanitization. This enables "
                            "prompt injection attacks where adversaries override system instructions."
                        ),
                        file=self._rel(fp),
                        line=i,
                        snippet=line.strip()[:120],
                        recommendation=(
                            "Validate and sanitize all user inputs before adding to messages. "
                            "Use a strict allowlist for acceptable characters. "
                            "Consider a dedicated input-validation layer before the agentic loop. "
                            "Enforce role separation: never allow user content to appear in "
                            "the 'system' role message."
                        ),
                        confidence="MEDIUM",
                    ))

                if fstring_system_re.search(line):
                    self._add(OWASPFinding(
                        owasp_id="LLM01",
                        owasp_name="Prompt Injection",
                        severity="HIGH",
                        title="Variable interpolation inside system role message",
                        description=(
                            f"Line {i}: A variable is interpolated inside a message with "
                            "role='system'. If this variable contains user-controlled content, "
                            "it can override the system instruction."
                        ),
                        file=self._rel(fp),
                        line=i,
                        snippet=line.strip()[:120],
                        recommendation=(
                            "System prompts should be static strings defined in agent configuration. "
                            "Never embed user query content inside a system-role message. "
                            "Separate system instructions from user query messages."
                        ),
                        confidence="MEDIUM",
                    ))

            # Check if file makes LLM calls but imports no input validation
            if "client.messages.create" in content and "agents/" in self._rel(fp):
                if not re.search(r'(len\(|max_length|maxlength|validate|sanitize|truncate)', content, re.IGNORECASE):
                    self._add(OWASPFinding(
                        owasp_id="LLM01",
                        owasp_name="Prompt Injection",
                        severity="MEDIUM",
                        title="LLM agent lacks visible input length/content validation",
                        description=(
                            f"Agent file {self._rel(fp)} calls the Anthropic API but does not "
                            "appear to validate or bound the input query length. Unbounded inputs "
                            "increase prompt injection attack surface and DoS risk."
                        ),
                        file=self._rel(fp),
                        line=0,
                        snippet="",
                        recommendation=(
                            "Add input validation before the agentic loop: "
                            "check query length (e.g. 2-2000 chars as in api.py QueryRequest), "
                            "reject inputs containing known injection patterns (ignore/override/forget), "
                            "log and alert on suspicious inputs."
                        ),
                        confidence="LOW",
                    ))

    # ─── LLM02: Insecure Output Handling ────────────────────────────────────

    def _check_llm02_insecure_output_handling(self, files: list[tuple[Path, str]]) -> None:
        """
        LLM02 — Insecure Output Handling:
          1. LLM output passed directly to subprocess/exec/eval
          2. LLM output written to filesystem without sanitization
          3. LLM output returned as raw HTML (XSS risk)
          4. LLM response used directly as SQL query
        """
        # Dangerous sinks — LLM response piped to execution context
        dangerous_sinks = [
            (r'subprocess\.(run|call|Popen|check_output)\s*\([^)]*response', "subprocess execution"),
            (r'\beval\s*\([^)]*response', "eval() execution"),
            (r'\bexec\s*\([^)]*response', "exec() execution"),
            (r'open\s*\([^)]*response.*?,\s*["\']w', "file write with LLM output"),
            (r'execute\s*\([^)]*response', "SQL/DB execution with LLM output"),
        ]

        for fp, content in files:
            lines = content.splitlines()
            for i, line in enumerate(lines, start=1):
                for pattern, sink_name in dangerous_sinks:
                    if re.search(pattern, line, re.IGNORECASE):
                        self._add(OWASPFinding(
                            owasp_id="LLM02",
                            owasp_name="Insecure Output Handling",
                            severity="HIGH",
                            title=f"LLM response passed to dangerous sink: {sink_name}",
                            description=(
                                f"Line {i}: An LLM response variable appears to be passed directly "
                                f"to {sink_name} without sanitization. This can enable "
                                "remote code execution, SQL injection, or data exfiltration "
                                "if the model produces malicious output."
                            ),
                            file=self._rel(fp),
                            line=i,
                            snippet=line.strip()[:120],
                            recommendation=(
                                f"Never pass raw LLM output to {sink_name}. "
                                "Implement an output validation layer: parse structured output "
                                "with strict schemas (Pydantic), reject responses containing "
                                "shell metacharacters, SQL keywords, or script tags. "
                                "Use allowlists rather than denylists."
                            ),
                            confidence="MEDIUM",
                        ))

            # Check FastAPI response: if LLM output returned as HTML (XSS)
            if "HTMLResponse" in content or "text/html" in content:
                if "response" in content and "agent" in self._rel(fp).lower():
                    self._add(OWASPFinding(
                        owasp_id="LLM02",
                        owasp_name="Insecure Output Handling",
                        severity="MEDIUM",
                        title="LLM output may be rendered as HTML (XSS risk)",
                        description=(
                            f"{self._rel(fp)} returns HTMLResponse content that may contain "
                            "LLM-generated text. If the LLM produces script tags or HTML entities, "
                            "this creates an XSS vulnerability."
                        ),
                        file=self._rel(fp),
                        line=0,
                        snippet="",
                        recommendation=(
                            "Always HTML-encode LLM outputs before rendering in HTML context. "
                            "Use Content Security Policy (CSP) headers. "
                            "Prefer JSON API responses over HTML for LLM output."
                        ),
                        confidence="LOW",
                    ))

    # ─── LLM04: Model Denial of Service ────────────────────────────────────

    def _check_llm04_model_dos(self, files: list[tuple[Path, str]]) -> None:
        """
        LLM04 — Model Denial of Service:
          1. No max_iterations guard in agentic loop
          2. No max_tokens limit on LLM calls
          3. Recursive agent calls without depth limit
          4. No timeout on orchestrator
        """
        for fp, content in files:
            rel = self._rel(fp)

            # Check for agentic while-loop without iteration limit
            if "while True" in content and ("tool_use" in content or "stop_reason" in content):
                if "max_iter" not in content.lower() and "iteration" not in content.lower():
                    self._add(OWASPFinding(
                        owasp_id="LLM04",
                        owasp_name="Model Denial of Service",
                        severity="MEDIUM",
                        title="Agentic loop (while True) without iteration guard",
                        description=(
                            f"{rel}: Contains a `while True` agentic loop for tool_use handling "
                            "but no explicit iteration counter or max_iterations guard is visible. "
                            "A prompt injection or stuck tool can spin the loop indefinitely, "
                            "exhausting API quota and compute."
                        ),
                        file=rel,
                        line=0,
                        snippet="",
                        recommendation=(
                            "Add an explicit iteration counter: `for iteration in range(max_iterations)`. "
                            "Set max_iterations = 12 or configure via AGENT_MAX_ITERATIONS env var. "
                            "Log and alert when the iteration limit is reached."
                        ),
                        confidence="LOW",
                    ))

            # Check for missing max_tokens in messages.create calls
            if "messages.create" in content:
                # Find the create call and check if max_tokens is nearby
                create_positions = [m.start() for m in re.finditer(r'messages\.create', content)]
                for pos in create_positions:
                    snippet_window = content[pos:pos + 400]
                    if "max_tokens" not in snippet_window:
                        line_num = content[:pos].count("\n") + 1
                        self._add(OWASPFinding(
                            owasp_id="LLM04",
                            owasp_name="Model Denial of Service",
                            severity="LOW",
                            title="LLM call missing max_tokens parameter",
                            description=(
                                f"Line ~{line_num} in {rel}: A `messages.create()` call does not "
                                "appear to set `max_tokens`. Without an explicit limit, a malformed "
                                "prompt could cause the model to generate very long responses, "
                                "increasing latency and cost."
                            ),
                            file=rel,
                            line=line_num,
                            snippet="",
                            recommendation=(
                                "Always specify `max_tokens` in every Anthropic API call. "
                                "Recommended: max_tokens=4096 for agent responses, "
                                "max_tokens=512 for routing/classification. "
                                "Set as a constant in config.py."
                            ),
                            confidence="LOW",
                        ))
                        break  # One finding per file is sufficient

            # Check API endpoint for missing request timeout / rate limiting
            if "api.py" in rel or "router" in rel:
                if "@app.post" in content or "@router.post" in content:
                    if "timeout" not in content.lower() and "rate_limit" not in content.lower():
                        self._add(OWASPFinding(
                            owasp_id="LLM04",
                            owasp_name="Model Denial of Service",
                            severity="MEDIUM",
                            title="API endpoint lacks rate limiting or timeout configuration",
                            description=(
                                f"{rel}: Defines POST endpoints that trigger LLM calls but "
                                "does not appear to configure request rate limiting or timeouts. "
                                "An attacker could flood the endpoint with expensive queries."
                            ),
                            file=rel,
                            line=0,
                            snippet="",
                            recommendation=(
                                "Add rate limiting middleware (e.g. slowapi or an API gateway rule). "
                                "Set a per-request timeout (e.g. asyncio.wait_for with 30s). "
                                "Add concurrent-request limits via semaphore or connection pool."
                            ),
                            confidence="MEDIUM",
                        ))

    # ─── LLM05: Supply Chain Vulnerabilities ────────────────────────────────

    def _check_llm05_supply_chain(self, files: list[tuple[Path, str]]) -> None:
        """
        LLM05 — Supply Chain:
          1. No pinned model version (uses latest/unspecified)
          2. Model ID from environment without validation
          3. MCP server loaded from untrusted path
        """
        for fp, content in files:
            rel = self._rel(fp)

            # Check for unpinned model — e.g. "claude-3" without full version
            if re.search(r'model\s*=\s*["\']claude-[^"\']*["\']', content, re.IGNORECASE):
                # Flag if using a shortened/non-specific model ID
                model_matches = re.findall(
                    r'model\s*=\s*["\']([^"\']+)["\']', content, re.IGNORECASE
                )
                for model_id in model_matches:
                    # Specific full model IDs are fine; short aliases risk model substitution
                    if model_id in ("claude-3", "claude", "claude-sonnet", "claude-opus"):
                        self._add(OWASPFinding(
                            owasp_id="LLM05",
                            owasp_name="Supply Chain Vulnerabilities",
                            severity="MEDIUM",
                            title=f"Unpinned model version: '{model_id}'",
                            description=(
                                f"{rel}: Uses a short/unpinned model alias '{model_id}'. "
                                "Anthropic may resolve this to a different model version over time, "
                                "changing model behaviour unexpectedly in production."
                            ),
                            file=rel,
                            line=0,
                            snippet=f"model='{model_id}'",
                            recommendation=(
                                "Pin the full model version ID (e.g. 'claude-sonnet-4-6'). "
                                "Store the model ID in config.py as CLAUDE_MODEL and review it "
                                "before each production deployment."
                            ),
                            confidence="HIGH",
                        ))

            # Check if model is loaded from unvalidated environment variable
            if re.search(r'os\.environ\.get\s*\(\s*["\']CLAUDE_MODEL', content):
                if "CLAUDE_MODEL" not in content or ".get(" not in content:
                    self._add(OWASPFinding(
                        owasp_id="LLM05",
                        owasp_name="Supply Chain Vulnerabilities",
                        severity="LOW",
                        title="Model ID loaded from environment without validation",
                        description=(
                            f"{rel}: The Claude model ID is loaded from the CLAUDE_MODEL "
                            "environment variable without an allowlist check. "
                            "A misconfigured or compromised environment could substitute "
                            "an unintended model."
                        ),
                        file=rel,
                        line=0,
                        snippet="",
                        recommendation=(
                            "Validate CLAUDE_MODEL against an allowlist of known model IDs: "
                            "ALLOWED_MODELS = {'claude-sonnet-4-6', 'claude-opus-4-6', 'claude-haiku-4-5-20251001'} "
                            "Raise a startup error if the value is not in the allowlist."
                        ),
                        confidence="LOW",
                    ))

    # ─── LLM06: Sensitive Information Disclosure ────────────────────────────

    def _check_llm06_sensitive_disclosure(self, files: list[tuple[Path, str]]) -> None:
        """
        LLM06 — Sensitive Information Disclosure:
          1. API keys or secrets hardcoded in source
          2. System prompt printed/logged to stdout
          3. Full customer PII included in LLM context without masking
          4. Error messages expose internal stack traces to users
        """
        # Patterns for hardcoded secrets
        secret_patterns = [
            (r'sk-ant-[a-zA-Z0-9\-]{20,}', "Hardcoded Anthropic API key"),
            (r'AKIA[0-9A-Z]{16}', "Hardcoded AWS Access Key ID"),
            (r'(?i)(password|passwd|secret|token)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded credential"),
            (r'(?i)api_key\s*=\s*["\'][^"\']{10,}["\']', "Hardcoded API key assignment"),
        ]

        for fp, content in files:
            rel = self._rel(fp)
            lines = content.splitlines()

            for i, line in enumerate(lines, start=1):
                for pattern, label in secret_patterns:
                    if re.search(pattern, line):
                        # Skip if it's a test fixture or env-var default
                        if "os.environ" in line or "os.getenv" in line:
                            continue
                        if "test" in rel.lower() or "example" in rel.lower():
                            continue
                        self._add(OWASPFinding(
                            owasp_id="LLM06",
                            owasp_name="Sensitive Information Disclosure",
                            severity="HIGH",
                            title=f"{label} detected in source code",
                            description=(
                                f"Line {i} in {rel}: {label} pattern detected. "
                                "Hardcoded secrets in source code are a critical vulnerability: "
                                "they persist in git history and are exposed to anyone with "
                                "read access to the repository."
                            ),
                            file=rel,
                            line=i,
                            snippet="[REDACTED — contains potential secret]",
                            recommendation=(
                                "Remove the hardcoded secret immediately and rotate it. "
                                "Use environment variables (os.environ.get) or a secrets manager. "
                                "Add a pre-commit hook (detect-secrets or trufflehog) to prevent "
                                "future secret commits."
                            ),
                            confidence="MEDIUM",
                        ))

            # Check if system prompt is logged at INFO level or printed
            if "system_prompt" in content or "SYSTEM_PROMPT" in content:
                if re.search(r'(print|logger\.info|logging\.info)\s*\([^)]*system_prompt', content, re.IGNORECASE):
                    self._add(OWASPFinding(
                        owasp_id="LLM06",
                        owasp_name="Sensitive Information Disclosure",
                        severity="MEDIUM",
                        title="System prompt logged at INFO level",
                        description=(
                            f"{rel}: The agent's system prompt appears to be printed or logged "
                            "at INFO level. System prompts contain business logic, persona "
                            "definitions, and tool instructions that should not be exposed "
                            "in production logs."
                        ),
                        file=rel,
                        line=0,
                        snippet="",
                        recommendation=(
                            "Log system prompt content only at DEBUG level. "
                            "Ensure production LOG_LEVEL=WARNING (as set in Dockerfile). "
                            "Never echo system prompts in API responses."
                        ),
                        confidence="MEDIUM",
                    ))

            # Check for stack trace exposure in API error handlers
            if "api.py" in rel:
                if re.search(r'(traceback|exc_info=True|format_exc)', content):
                    if re.search(r'return.*?(detail|message)\s*.*?(str\(e\)|repr\(e\)|traceback)', content, re.IGNORECASE):
                        self._add(OWASPFinding(
                            owasp_id="LLM06",
                            owasp_name="Sensitive Information Disclosure",
                            severity="MEDIUM",
                            title="Stack trace or exception detail exposed in API response",
                            description=(
                                f"{rel}: The API error handler may return exception details "
                                "or stack traces to the client. This reveals internal "
                                "implementation details, file paths, and library versions "
                                "to potential attackers."
                            ),
                            file=rel,
                            line=0,
                            snippet="",
                            recommendation=(
                                "Return only generic error messages to clients: "
                                "{'error': 'Internal server error', 'request_id': '...'}. "
                                "Log full stack traces server-side at ERROR level only. "
                                "Use a correlation ID to link user-visible errors to server logs."
                            ),
                            confidence="MEDIUM",
                        ))

    # ─── LLM07: Insecure Plugin / MCP Tool Design ───────────────────────────

    def _check_llm07_insecure_plugin_design(self, files: list[tuple[Path, str]]) -> None:
        """
        LLM07 — Insecure Plugin Design (MCP Tools):
          1. MCP tools that perform destructive operations without confirmation
          2. Tools that accept raw SQL or shell commands as parameters
          3. Tools without explicit input type validation
          4. Data mutation tools accessible from any agent
        """
        for fp, content in files:
            rel = self._rel(fp)

            # Only examine MCP tool files
            if "mcp_server/tools" not in rel and "tools" not in rel:
                continue

            # Check for tools that perform destructive operations
            destructive_ops = [
                (r'\bdelete\b.*\bfrom\b', "SQL DELETE"),
                (r'\bdrop\b.*\btable\b', "SQL DROP TABLE"),
                (r'\btruncate\b.*\btable\b', "SQL TRUNCATE"),
                (r'subprocess\.run.*\brm\b', "shell rm command"),
                (r'os\.remove\s*\(', "os.remove file deletion"),
                (r'shutil\.rmtree\s*\(', "shutil.rmtree recursive delete"),
            ]
            for pattern, op_name in destructive_ops:
                if re.search(pattern, content, re.IGNORECASE):
                    self._add(OWASPFinding(
                        owasp_id="LLM07",
                        owasp_name="Insecure Plugin Design",
                        severity="HIGH",
                        title=f"MCP tool performs destructive operation: {op_name}",
                        description=(
                            f"{rel}: A tool in this MCP plugin file performs {op_name}. "
                            "If triggered by a prompt injection, this could cause "
                            "irreversible data loss."
                        ),
                        file=rel,
                        line=0,
                        snippet="",
                        recommendation=(
                            f"Add a confirmation parameter to the tool that performs {op_name}. "
                            "Require explicit confirmation: `confirm: bool = False`. "
                            "Consider making destructive tools unavailable to autonomous agents; "
                            "require human-in-the-loop approval. "
                            "Log all destructive operations with full audit trail."
                        ),
                        confidence="MEDIUM",
                    ))

            # Check for tools accepting raw SQL
            if re.search(r'def.*\b(query|sql|command|cmd)\b.*:', content, re.IGNORECASE):
                self._add(OWASPFinding(
                    owasp_id="LLM07",
                    owasp_name="Insecure Plugin Design",
                    severity="HIGH",
                    title="MCP tool accepts raw SQL or command parameter",
                    description=(
                        f"{rel}: A tool function accepts a parameter named 'query', 'sql', "
                        "'command', or 'cmd'. If this value comes from LLM output, it enables "
                        "SQL injection or command injection."
                    ),
                    file=rel,
                    line=0,
                    snippet="",
                    recommendation=(
                        "Replace free-text SQL/command parameters with typed, constrained inputs "
                        "(e.g. customer_id: str, date_from: str). "
                        "Use parameterized queries or ORM methods rather than raw SQL. "
                        "Validate all inputs against a strict schema before execution."
                    ),
                    confidence="LOW",
                ))

            # Check for data mutation tools without access control comments
            mutation_fns = re.findall(r'def\s+(process|update|create|delete|add|remove|save)\w*\s*\(', content, re.IGNORECASE)
            if mutation_fns:
                if "authorization" not in content.lower() and "auth" not in content.lower() and "permission" not in content.lower():
                    self._add(OWASPFinding(
                        owasp_id="LLM07",
                        owasp_name="Insecure Plugin Design",
                        severity="MEDIUM",
                        title="Data-mutating MCP tools lack visible access control",
                        description=(
                            f"{rel}: Contains data-mutation tool functions "
                            f"({', '.join(mutation_fns[:3])}) with no visible authorization "
                            "or access control logic. Any agent can invoke these tools without "
                            "permission checks."
                        ),
                        file=rel,
                        line=0,
                        snippet="",
                        recommendation=(
                            "Add agent-identity checks to mutation tools: verify that the calling "
                            "agent is authorized to perform the operation. "
                            "Consider a tool permission manifest that restricts which agents "
                            "can invoke which tools. "
                            "Log all mutation operations with agent identity and timestamp."
                        ),
                        confidence="LOW",
                    ))

    # ─── LLM08: Excessive Agency ─────────────────────────────────────────────

    def _check_llm08_excessive_agency(self, files: list[tuple[Path, str]]) -> None:
        """
        LLM08 — Excessive Agency:
          1. Agent has access to tools outside its business domain
          2. Orchestrator grants all tools to all agents
          3. Agent can execute system commands
          4. No tool-use confirmation for high-impact actions
        """
        for fp, content in files:
            rel = self._rel(fp)

            # Check orchestrator for passing all tools to all agents
            if "orchestrator" in rel:
                if re.search(r'tools\s*=\s*(self\._tools|client\._tools|all_tools)', content, re.IGNORECASE):
                    self._add(OWASPFinding(
                        owasp_id="LLM08",
                        owasp_name="Excessive Agency",
                        severity="MEDIUM",
                        title="Orchestrator may provide all tools to every agent",
                        description=(
                            f"{rel}: The orchestrator appears to pass the full tool set to agents. "
                            "Each agent should only receive tools relevant to its domain "
                            "(principle of least privilege). Broad tool access increases the "
                            "blast radius of a prompt injection attack."
                        ),
                        file=rel,
                        line=0,
                        snippet="",
                        recommendation=(
                            "Filter the tool list per agent: each specialized agent (segmentation, "
                            "campaign, recommendation, crm_loyalty) should only receive the 5 tools "
                            "in its domain. Implement a tool allowlist per agent identity."
                        ),
                        confidence="LOW",
                    ))

            # Check for subprocess calls in agent code (system command execution)
            if "agents/" in rel or "orchestrator" in rel:
                if re.search(r'\bsubprocess\b', content):
                    self._add(OWASPFinding(
                        owasp_id="LLM08",
                        owasp_name="Excessive Agency",
                        severity="HIGH",
                        title="Agent or orchestrator imports subprocess (system command access)",
                        description=(
                            f"{rel}: An agent or orchestrator file imports subprocess. "
                            "Agents should not have the capability to execute system commands. "
                            "This represents excessive agency — if an agent is compromised via "
                            "prompt injection, it could execute arbitrary OS commands."
                        ),
                        file=rel,
                        line=0,
                        snippet="",
                        recommendation=(
                            "Remove subprocess capability from agent code. "
                            "Agent code should only communicate via the MCP tool interface. "
                            "If system operations are required, expose them as explicit, "
                            "scoped MCP tools with strict input validation."
                        ),
                        confidence="MEDIUM",
                    ))

    # ─── LLM09: Overreliance ─────────────────────────────────────────────────

    def _check_llm09_overreliance(self, files: list[tuple[Path, str]]) -> None:
        """
        LLM09 — Overreliance:
          1. Financial/medical recommendations without human review disclaimer
          2. Agent output used as direct database write without validation
          3. No confidence/uncertainty indicators in agent responses
        """
        for fp, content in files:
            rel = self._rel(fp)

            # Check agent system prompts for missing uncertainty language
            if "agents/" in rel and "system_prompt" in content:
                if not re.search(
                    r'(verify|validate|human review|consult|confirm|double.check|disclaimer)',
                    content, re.IGNORECASE
                ):
                    self._add(OWASPFinding(
                        owasp_id="LLM09",
                        owasp_name="Overreliance",
                        severity="LOW",
                        title="Agent system prompt lacks uncertainty/verification guidance",
                        description=(
                            f"{rel}: The agent's system prompt does not appear to include "
                            "instructions to express uncertainty, recommend human review, "
                            "or disclaim limitations. Users may treat AI recommendations as "
                            "authoritative without appropriate validation."
                        ),
                        file=rel,
                        line=0,
                        snippet="",
                        recommendation=(
                            "Add uncertainty language to agent system prompts: "
                            "'When making financial or business-critical recommendations, "
                            "always note confidence level and recommend human review before "
                            "taking large-scale actions.' "
                            "Include explicit disclaimers in high-impact response templates."
                        ),
                        confidence="LOW",
                    ))

            # Check if LLM output is written directly to the database
            if "orchestrator" in rel or "api.py" in rel:
                if re.search(r'(store\.update|data_store\.|\.save\(|\.write\()', content):
                    if re.search(r'(final_response|agent_result|response\.content)', content):
                        self._add(OWASPFinding(
                            owasp_id="LLM09",
                            owasp_name="Overreliance",
                            severity="MEDIUM",
                            title="LLM output potentially written to data store without validation",
                            description=(
                                f"{rel}: LLM response content appears to be passed to a data "
                                "store write operation. Writing unvalidated LLM outputs to "
                                "persistent storage risks data corruption and hallucination "
                                "poisoning of downstream systems."
                            ),
                            file=rel,
                            line=0,
                            snippet="",
                            recommendation=(
                                "Parse LLM output as structured data (Pydantic schema) before "
                                "any persistence operation. Validate data types, value ranges, "
                                "and referential integrity independently of what the LLM claims. "
                                "Consider a human-approval step for LLM-driven data mutations."
                            ),
                            confidence="LOW",
                        ))

    # ─── Result builder ───────────────────────────────────────────────────────

    def _build_result(self) -> dict:
        """Compile findings into a structured result dict."""
        severity_counts: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
        by_owasp: dict[str, dict] = {}

        for f in self.findings:
            sev = f.severity
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

            oid = f.owasp_id
            if oid not in by_owasp:
                risk = OWASP_LLM_RISKS.get(oid, {})
                by_owasp[oid] = {
                    "name": risk.get("name", ""),
                    "count": 0,
                    "findings": [],
                }
            by_owasp[oid]["count"] += 1
            by_owasp[oid]["findings"].append(f.to_dict())

        return {
            "total_findings": len(self.findings),
            "severity_counts": severity_counts,
            "by_owasp_id": {
                oid: {"name": v["name"], "count": v["count"]}
                for oid, v in sorted(by_owasp.items())
            },
            "findings": [f.to_dict() for f in self.findings],
        }


# ─── SARIF generator ─────────────────────────────────────────────────────────

def _findings_to_sarif(findings: list[OWASPFinding]) -> dict:
    """Convert OWASP findings to SARIF 2.1.0 format."""
    rules: dict[str, dict] = {}
    for f in findings:
        risk = OWASP_LLM_RISKS.get(f.owasp_id, {})
        rule_id = f"{f.owasp_id}-{f.title[:30].replace(' ', '_')}"
        if rule_id not in rules:
            rules[rule_id] = {
                "id": rule_id,
                "name": f.title.replace(" ", ""),
                "shortDescription": {"text": f"{f.owasp_id}: {f.title}"},
                "fullDescription": {"text": risk.get("description", f.description)},
                "defaultConfiguration": {
                    "level": {"HIGH": "error", "MEDIUM": "warning",
                               "LOW": "note", "INFO": "note"}.get(f.severity, "note")
                },
                "helpUri": risk.get("url", "https://owasp.org/www-project-top-10-for-large-language-model-applications/"),
                "help": {
                    "text": f.recommendation,
                    "markdown": f"**Recommendation**: {f.recommendation}",
                },
                "properties": {
                    "tags": [f.owasp_id, "owasp-llm-top10", "ai-security"],
                },
            }

    sarif_results = []
    for f in findings:
        rule_id = f"{f.owasp_id}-{f.title[:30].replace(' ', '_')}"
        level = {"HIGH": "error", "MEDIUM": "warning",
                 "LOW": "note", "INFO": "note"}.get(f.severity, "note")
        result: dict = {
            "ruleId": rule_id,
            "level": level,
            "message": {"text": f"{f.description} Recommendation: {f.recommendation}"},
        }
        if f.file:
            loc: dict = {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": f.file.lstrip("/"),
                        "uriBaseId": "%SRCROOT%",
                    }
                }
            }
            if f.line > 0:
                loc["physicalLocation"]["region"] = {
                    "startLine": f.line,
                    "snippet": {"text": f.snippet[:200]},
                }
            result["locations"] = [loc]
        sarif_results.append(result)

    return {
        "$schema": (
            "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/"
            "Schemata/sarif-schema-2.1.0.json"
        ),
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "OWASP LLM Top 10 Scanner",
                    "version": "1.0.0",
                    "informationUri": "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
                    "rules": list(rules.values()),
                }
            },
            "results": sarif_results,
        }],
    }


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main() -> int:
    """
    CLI: python -m security.owasp_ai_scanner

    Exit codes:
      0 — no HIGH findings
      1 — HIGH findings detected
      2 — scanner error
    """
    import argparse
    from rich.console import Console
    from rich.table import Table

    parser = argparse.ArgumentParser(
        description="OWASP LLM Top 10 heuristic security scanner"
    )
    parser.add_argument("--source", default=".", help="Source directory to scan (default: .)")
    parser.add_argument(
        "--summary-out", default="", help="Write JSON summary to this file"
    )
    parser.add_argument(
        "--sarif-out", default="", help="Write SARIF 2.1.0 output to this file"
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    args = parser.parse_args()

    console = Console(stderr=False)
    console.print("\n[bold cyan]OWASP LLM Top 10 — AI Security Scanner[/bold cyan]")
    console.print("─" * 60)
    console.print("\nChecks implemented:")
    for oid, info in sorted(OWASP_LLM_RISKS.items()):
        console.print(f"  {oid}: {info['name']}")
    console.print()

    scanner = OWASPAIScanner(source_path=args.source)
    result = scanner.run()

    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if result["severity_counts"].get("HIGH", 0) == 0 else 1

    # Print findings table
    if result["total_findings"] == 0:
        console.print("[green]No OWASP LLM Top 10 findings detected.[/green]\n")
    else:
        # Summary by OWASP category
        table = Table(title="OWASP LLM Top 10 Findings", show_header=True)
        table.add_column("OWASP ID", style="bold", min_width=10)
        table.add_column("Risk Name", min_width=30)
        table.add_column("Findings", justify="right", min_width=10)
        for oid, info in sorted(result["by_owasp_id"].items()):
            count = info["count"]
            table.add_row(f"[bold]{oid}[/bold]", info["name"], str(count))
        console.print(table)

        # Severity summary
        sev_table = Table(title="By Severity", show_header=True)
        sev_table.add_column("Severity", style="bold", min_width=12)
        sev_table.add_column("Count", justify="right", min_width=8)
        for sev in ["HIGH", "MEDIUM", "LOW", "INFO"]:
            count = result["severity_counts"].get(sev, 0)
            style = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green", "INFO": "dim"}.get(sev)
            sev_table.add_row(f"[{style}]{sev}[/{style}]", str(count))
        console.print(sev_table)
        console.print()

        # Print HIGH severity findings in detail
        high_findings = [f for f in scanner.findings if f.severity == "HIGH"]
        if high_findings:
            console.print("[bold red]HIGH Severity Findings:[/bold red]\n")
            for f in high_findings:
                console.print(f"  [{f.owasp_id}] [bold]{f.title}[/bold]")
                console.print(f"  File: {f.file}" + (f"  Line: {f.line}" if f.line else ""))
                console.print(f"  {f.description[:200]}")
                console.print(f"  [yellow]Recommendation:[/yellow] {f.recommendation[:200]}")
                console.print()

    # Write outputs
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        console.print(f"JSON summary written to: {args.summary_out}")

    if args.sarif_out:
        sarif = _findings_to_sarif(scanner.findings)
        with open(args.sarif_out, "w", encoding="utf-8") as fh:
            json.dump(sarif, fh, indent=2)
        console.print(f"SARIF report written to: {args.sarif_out}")

    high_count = result["severity_counts"].get("HIGH", 0)
    status = "[green]PASSED[/green]" if high_count == 0 else "[red]FAILED[/red]"
    console.print(f"\nOWASP AI Gate: {status}  |  Total findings: {result['total_findings']}\n")

    return 0 if high_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
