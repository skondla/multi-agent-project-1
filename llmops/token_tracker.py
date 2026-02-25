"""
LLMOps — Token Usage & Cost Tracker
Wraps Anthropic API calls to capture token counts, latency, and estimated costs.

Usage — decorator pattern (drop-in for BaseAgent):
    from llmops.token_tracker import track_usage, UsageStore

    store = UsageStore()

    @track_usage(store, agent="segmentation")
    def call_claude(messages, tools, ...):
        return client.messages.create(...)

Usage — context manager:
    with TokenTracker(store, agent="orchestrator") as tracker:
        response = client.messages.create(...)
        tracker.record(response)

Report:
    python -m llmops.token_tracker [--report] [--since 2026-01-01]
"""
from __future__ import annotations

import functools
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LLMOPS_DIR = PROJECT_ROOT / "llmops"
USAGE_LOG = LLMOPS_DIR / "usage_log.jsonl"

# Anthropic pricing (per 1M tokens) — update as prices change
PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6":   {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5":  {"input": 0.80, "output": 4.00},
}
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class UsageRecord:
    timestamp: str
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    latency_ms: float
    stop_reason: str
    tool_calls: int = 0
    query_id: str = ""
    metadata: dict = field(default_factory=dict)


def _cost(tokens: int, price_per_million: float) -> float:
    return round(tokens / 1_000_000 * price_per_million, 6)


def _record_from_response(
    response: Any,
    agent: str,
    latency_ms: float,
    query_id: str = "",
    metadata: dict | None = None,
) -> UsageRecord:
    model = getattr(response, "model", "unknown")
    usage = getattr(response, "usage", None)
    pricing = PRICING.get(model, DEFAULT_PRICING)

    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    total_tokens = input_tokens + output_tokens

    tool_calls = sum(
        1 for block in (getattr(response, "content", []) or [])
        if getattr(block, "type", "") == "tool_use"
    )

    return UsageRecord(
        timestamp=datetime.now().isoformat(),
        agent=agent,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_cost_usd=_cost(input_tokens, pricing["input"]),
        output_cost_usd=_cost(output_tokens, pricing["output"]),
        total_cost_usd=_cost(input_tokens, pricing["input"]) + _cost(output_tokens, pricing["output"]),
        latency_ms=round(latency_ms, 2),
        stop_reason=getattr(response, "stop_reason", "unknown"),
        tool_calls=tool_calls,
        query_id=query_id,
        metadata=metadata or {},
    )


# ─── UsageStore ───────────────────────────────────────────────────────────────

class UsageStore:
    """Append-only JSONL store for usage records. Thread-safe for single-process use."""

    def __init__(self, log_path: Path = USAGE_LOG):
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: UsageRecord) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def load(self, since: str | None = None) -> list[UsageRecord]:
        if not self._path.exists():
            return []
        records = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if since and d.get("timestamp", "") < since:
                        continue
                    records.append(UsageRecord(**d))
                except (json.JSONDecodeError, TypeError):
                    pass
        return records

    def summary(self, since: str | None = None) -> dict:
        records = self.load(since=since)
        if not records:
            return {"total_records": 0}

        by_agent: dict[str, dict] = {}
        by_model: dict[str, dict] = {}

        for r in records:
            for group, key in [(by_agent, r.agent), (by_model, r.model)]:
                if key not in group:
                    group[key] = {
                        "calls": 0, "input_tokens": 0, "output_tokens": 0,
                        "total_cost_usd": 0.0, "total_latency_ms": 0.0, "tool_calls": 0,
                    }
                g = group[key]
                g["calls"] += 1
                g["input_tokens"] += r.input_tokens
                g["output_tokens"] += r.output_tokens
                g["total_cost_usd"] = round(g["total_cost_usd"] + r.total_cost_usd, 6)
                g["total_latency_ms"] += r.latency_ms
                g["tool_calls"] += r.tool_calls

        # Compute averages
        for g in {**by_agent, **by_model}.values():
            g["avg_latency_ms"] = round(g["total_latency_ms"] / max(g["calls"], 1), 2)
            g["avg_tokens_per_call"] = round(
                (g["input_tokens"] + g["output_tokens"]) / max(g["calls"], 1), 1
            )

        return {
            "total_records": len(records),
            "since": since,
            "total_input_tokens": sum(r.input_tokens for r in records),
            "total_output_tokens": sum(r.output_tokens for r in records),
            "total_cost_usd": round(sum(r.total_cost_usd for r in records), 4),
            "avg_latency_ms": round(sum(r.latency_ms for r in records) / len(records), 2),
            "by_agent": by_agent,
            "by_model": by_model,
        }


# ─── Decorator and context manager ────────────────────────────────────────────

def track_usage(
    store: UsageStore,
    agent: str,
    query_id: str = "",
    metadata: dict | None = None,
) -> Callable:
    """Decorator that wraps a function returning an Anthropic response object."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            response = fn(*args, **kwargs)
            latency_ms = (time.monotonic() - t0) * 1000
            record = _record_from_response(response, agent, latency_ms, query_id, metadata)
            store.append(record)
            logger.debug(
                f"[{agent}] tokens={record.total_tokens} "
                f"cost=${record.total_cost_usd:.4f} "
                f"latency={record.latency_ms:.0f}ms"
            )
            return response
        return wrapper
    return decorator


class TokenTracker:
    """Context manager for tracking a single API call."""

    def __init__(self, store: UsageStore, agent: str, query_id: str = "", metadata: dict | None = None):
        self._store = store
        self._agent = agent
        self._query_id = query_id
        self._metadata = metadata or {}
        self._t0: float = 0.0

    def __enter__(self) -> "TokenTracker":
        self._t0 = time.monotonic()
        return self

    def record(self, response: Any) -> UsageRecord:
        latency_ms = (time.monotonic() - self._t0) * 1000
        r = _record_from_response(response, self._agent, latency_ms, self._query_id, self._metadata)
        self._store.append(r)
        logger.debug(f"[{self._agent}] tokens={r.total_tokens} cost=${r.total_cost_usd:.4f}")
        return r

    def __exit__(self, *_):
        pass


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Token usage report")
    parser.add_argument("--since", default=None, help="ISO date filter (e.g. 2026-01-01)")
    parser.add_argument("--agent", default=None, help="Filter by agent name")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    store = UsageStore()
    summary = store.summary(since=args.since)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    if summary["total_records"] == 0:
        print("No usage records found.")
        return

    print(f"\n=== LLM Token Usage Report ===")
    if args.since:
        print(f"  Since: {args.since}")
    print(f"  Total API calls:      {summary['total_records']}")
    print(f"  Total input tokens:   {summary['total_input_tokens']:,}")
    print(f"  Total output tokens:  {summary['total_output_tokens']:,}")
    print(f"  Total cost (USD):     ${summary['total_cost_usd']:.4f}")
    print(f"  Avg latency (ms):     {summary['avg_latency_ms']:.0f}")

    if summary.get("by_agent"):
        print("\n  By Agent:")
        print(f"  {'Agent':>20}  {'Calls':>8}  {'Input Tok':>12}  {'Output Tok':>12}  {'Cost USD':>10}  {'Avg ms':>8}")
        print("  " + "-" * 80)
        for agent, g in sorted(summary["by_agent"].items(), key=lambda x: -x[1]["total_cost_usd"]):
            print(f"  {agent:>20}  {g['calls']:>8}  {g['input_tokens']:>12,}  "
                  f"{g['output_tokens']:>12,}  ${g['total_cost_usd']:>9.4f}  {g['avg_latency_ms']:>7.0f}")

    if summary.get("by_model"):
        print("\n  By Model:")
        for model, g in summary["by_model"].items():
            print(f"  {model}: {g['calls']} calls, ${g['total_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
