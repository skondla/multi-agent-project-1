"""
LLMOps — Observability & Distributed Tracing
Provides structured logging and span-based tracing for multi-agent pipelines.
Compatible with OpenTelemetry semantic conventions (exportable to OTLP/Jaeger/Zipkin).

Key concepts:
  - Trace  : A single end-to-end request through the orchestrator
  - Span   : One unit of work (agent run, MCP tool call, LLM call)
  - Events : Key moments within a span (tool result, routing decision, etc.)

Usage:
    from llmops.observability import Tracer, get_tracer

    tracer = get_tracer("orchestrator")

    with tracer.start_span("orchestrate", {"query": query}) as span:
        span.set_attribute("agents_selected", ["segmentation", "campaign"])

        with tracer.start_span("agent.segmentation", parent=span) as agent_span:
            # ... run agent ...
            agent_span.set_attribute("tool_calls", 3)
            agent_span.add_event("tool_result", {"tool": "compute_rfm_scores"})

    tracer.flush()  # writes to llmops/traces/

Report:
    python -m llmops.observability report [--trace-id <id>]
    python -m llmops.observability tail   [--n 20]
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
TRACES_DIR = PROJECT_ROOT / "llmops" / "traces"
TRACE_LOG = TRACES_DIR / "traces.jsonl"


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SpanEvent:
    name: str
    timestamp: str
    attributes: dict = field(default_factory=dict)


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    service: str
    start_time: str
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "ok"            # ok | error | unset
    attributes: dict = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    error_message: Optional[str] = None

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        self.events.append(SpanEvent(
            name=name,
            timestamp=datetime.now().isoformat(),
            attributes=attributes or {},
        ))

    def end(self, error: str | None = None) -> None:
        now = time.monotonic()
        self.end_time = datetime.now().isoformat()
        # duration computed from wall clock stored at start
        if hasattr(self, "_start_monotonic"):
            self.duration_ms = round((now - self._start_monotonic) * 1000, 2)
        if error:
            self.status = "error"
            self.error_message = error

    def to_dict(self) -> dict:
        d = asdict(self)
        d["events"] = [asdict(e) for e in self.events]
        return d


@dataclass
class Trace:
    trace_id: str
    root_span_id: str
    service: str
    started_at: str
    spans: list[Span] = field(default_factory=list)
    completed: bool = False

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "root_span_id": self.root_span_id,
            "service": self.service,
            "started_at": self.started_at,
            "span_count": len(self.spans),
            "duration_ms": self._total_duration(),
            "completed": self.completed,
            "spans": [s.to_dict() for s in self.spans],
        }

    def _total_duration(self) -> float:
        durations = [s.duration_ms for s in self.spans if s.duration_ms is not None]
        root = next((s for s in self.spans if s.parent_span_id is None), None)
        return root.duration_ms or 0.0 if root and root.duration_ms else sum(durations)


# ─── Span context manager ─────────────────────────────────────────────────────

class SpanContext:
    """Context manager returned by tracer.start_span()."""

    def __init__(self, span: Span, tracer: "Tracer"):
        self._span = span
        self._tracer = tracer

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        self._span.add_event(name, attributes)

    @property
    def span_id(self) -> str:
        return self._span.span_id

    @property
    def trace_id(self) -> str:
        return self._span.trace_id

    def __enter__(self) -> "SpanContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        error_msg = str(exc_val) if exc_type else None
        self._span.end(error=error_msg)
        return False  # don't suppress exceptions


# ─── Tracer ───────────────────────────────────────────────────────────────────

class Tracer:
    """
    Lightweight multi-agent tracer. One Tracer instance per request/trace.
    Thread-safety: designed for single-thread asyncio use.
    """

    def __init__(self, service: str = "customer-intelligence", trace_id: str | None = None):
        self.service = service
        self.trace_id = trace_id or str(uuid.uuid4())[:16]
        self._trace = Trace(
            trace_id=self.trace_id,
            root_span_id="",
            service=service,
            started_at=datetime.now().isoformat(),
        )
        self._active_spans: dict[str, Span] = {}

    def start_span(
        self,
        name: str,
        attributes: dict | None = None,
        parent: "SpanContext | None" = None,
    ) -> SpanContext:
        """Start a new span. Use as context manager."""
        span_id = str(uuid.uuid4())[:12]
        parent_id = parent.span_id if parent else None

        if not self._trace.root_span_id:
            self._trace.root_span_id = span_id

        span = Span(
            span_id=span_id,
            trace_id=self.trace_id,
            parent_span_id=parent_id,
            name=name,
            service=self.service,
            start_time=datetime.now().isoformat(),
            attributes=attributes or {},
        )
        span._start_monotonic = time.monotonic()  # type: ignore[attr-defined]

        self._trace.spans.append(span)
        self._active_spans[span_id] = span
        logger.debug(f"[TRACE {self.trace_id}] span start: {name} ({span_id})")
        return SpanContext(span, self)

    def flush(self) -> Path:
        """Write completed trace to JSONL log."""
        self._trace.completed = True
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        with open(TRACE_LOG, "a") as f:
            f.write(json.dumps(self._trace.to_dict()) + "\n")
        logger.info(f"[TRACE {self.trace_id}] flushed {len(self._trace.spans)} spans")
        return TRACE_LOG

    def summary(self) -> dict:
        t = self._trace.to_dict()
        spans = t["spans"]
        return {
            "trace_id": self.trace_id,
            "service": self.service,
            "span_count": len(spans),
            "total_duration_ms": t["duration_ms"],
            "spans": [
                {
                    "name": s["name"],
                    "duration_ms": s.get("duration_ms"),
                    "status": s["status"],
                    "attributes": s["attributes"],
                }
                for s in spans
            ],
        }


# ─── Global tracer factory ────────────────────────────────────────────────────

_active_tracer: Tracer | None = None


def get_tracer(service: str = "customer-intelligence") -> Tracer:
    """Get or create the active tracer for the current request."""
    global _active_tracer
    if _active_tracer is None:
        _active_tracer = Tracer(service=service)
    return _active_tracer


def new_tracer(service: str = "customer-intelligence") -> Tracer:
    """Create a fresh tracer (new trace_id) — call once per request."""
    global _active_tracer
    _active_tracer = Tracer(service=service)
    return _active_tracer


# ─── Convenience decorators ───────────────────────────────────────────────────

def trace_agent(agent_name: str):
    """Decorator: wrap an async agent .run() method in a trace span."""
    def decorator(fn):
        import functools
        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            tracer = get_tracer()
            with tracer.start_span(f"agent.{agent_name}", {"agent": agent_name}) as span:
                try:
                    result = await fn(self, *args, **kwargs)
                    span.set_attribute("response_length", len(str(result)))
                    return result
                except Exception as e:
                    span.add_event("error", {"message": str(e)})
                    raise
        return wrapper
    return decorator


def trace_tool_call(span_ctx: SpanContext, tool_name: str, tool_input: dict, result: str) -> None:
    """Record a single MCP tool call as a span event."""
    span_ctx.add_event("tool_call", {
        "tool": tool_name,
        "input_keys": list(tool_input.keys()),
        "result_length": len(result),
        "result_status": "error" if '"status": "error"' in result else "ok",
    })


# ─── Log reader ───────────────────────────────────────────────────────────────

def load_traces(since: str | None = None, n: int = 0) -> list[dict]:
    if not TRACE_LOG.exists():
        return []
    traces = []
    with open(TRACE_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if since and t.get("started_at", "") < since:
                    continue
                traces.append(t)
            except json.JSONDecodeError:
                pass
    if n:
        traces = traces[-n:]
    return traces


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Observability trace viewer")
    sub = parser.add_subparsers(dest="command")

    report_p = sub.add_parser("report", help="Summarize recent traces")
    report_p.add_argument("--since", default=None)
    report_p.add_argument("--trace-id", default=None)

    tail_p = sub.add_parser("tail", help="Show the last N traces")
    tail_p.add_argument("--n", type=int, default=10)

    sub.add_parser("demo", help="Run a demo trace")

    args = parser.parse_args()

    if args.command == "demo":
        tracer = new_tracer("demo-pipeline")
        with tracer.start_span("orchestrate", {"query": "Who are at-risk customers?"}) as root:
            root.set_attribute("parallel", True)
            root.add_event("routing_complete", {"agents": ["segmentation"]})

            with tracer.start_span("agent.segmentation", parent=root) as seg:
                seg.set_attribute("agent", "segmentation")
                time.sleep(0.05)
                seg.add_event("tool_call", {"tool": "identify_churn_risk"})
                time.sleep(0.02)
                seg.add_event("tool_call", {"tool": "compute_rfm_scores"})
                seg.set_attribute("tool_calls", 2)

            root.add_event("synthesis_start")

        path = tracer.flush()
        print(f"Demo trace written to {path}")
        print(json.dumps(tracer.summary(), indent=2))
        return

    if args.command == "tail":
        traces = load_traces(n=args.n)
        if not traces:
            print("No traces found.")
            return
        for t in traces:
            errors = sum(1 for s in t.get("spans", []) if s.get("status") == "error")
            print(f"  {t['trace_id']}  spans={t['span_count']:>3}  "
                  f"duration={t['duration_ms']:.0f}ms  "
                  f"errors={errors}  started={t['started_at'][:19]}")

    elif args.command == "report" or not args.command:
        since = getattr(args, "since", None)
        tid = getattr(args, "trace_id", None)
        traces = load_traces(since=since)

        if tid:
            traces = [t for t in traces if t["trace_id"] == tid]
            if not traces:
                print(f"Trace {tid} not found.")
                return
            for t in traces:
                print(json.dumps(t, indent=2))
            return

        if not traces:
            print("No traces found.")
            return

        total = len(traces)
        avg_dur = sum(t.get("duration_ms", 0) for t in traces) / total
        error_traces = sum(
            1 for t in traces
            if any(s.get("status") == "error" for s in t.get("spans", []))
        )
        avg_spans = sum(t.get("span_count", 0) for t in traces) / total

        print(f"\n=== Trace Report ===")
        if since:
            print(f"  Since: {since}")
        print(f"  Total traces:     {total}")
        print(f"  Avg duration:     {avg_dur:.0f} ms")
        print(f"  Avg spans/trace:  {avg_spans:.1f}")
        print(f"  Error traces:     {error_traces} ({100*error_traces/total:.1f}%)")

        # P50/P95 latency
        durations = sorted(t.get("duration_ms", 0) for t in traces)
        p50 = durations[int(len(durations) * 0.50)]
        p95 = durations[int(len(durations) * 0.95)]
        print(f"  P50 latency:      {p50:.0f} ms")
        print(f"  P95 latency:      {p95:.0f} ms")

        # Agent breakdown
        agent_counts: dict[str, int] = {}
        agent_durations: dict[str, list[float]] = {}
        for t in traces:
            for s in t.get("spans", []):
                if s["name"].startswith("agent."):
                    a = s["name"].split(".", 1)[1]
                    agent_counts[a] = agent_counts.get(a, 0) + 1
                    agent_durations.setdefault(a, []).append(s.get("duration_ms", 0))

        if agent_counts:
            print("\n  Agent invocations:")
            for agent, count in sorted(agent_counts.items(), key=lambda x: -x[1]):
                avg = sum(agent_durations[agent]) / count
                print(f"    {agent}: {count}x, avg {avg:.0f}ms")


if __name__ == "__main__":
    main()
