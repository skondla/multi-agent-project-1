"""
LLMOps — Response Quality Evaluator
Scores agent responses on multiple quality dimensions without requiring a second LLM call.

Dimensions evaluated:
  - completeness   : Does the response address all aspects of the query?
  - groundedness   : Does it reference tool results / data (not hallucinate)?
  - structure      : Is the response well-organised (sections, bullets, metrics)?
  - actionability  : Does it contain concrete recommendations or next steps?
  - conciseness    : Is it appropriately sized (not too sparse, not bloated)?

Each dimension is scored 0.0–1.0. An overall quality score is the weighted average.

Usage:
    from llmops.response_evaluator import ResponseEvaluator

    evaluator = ResponseEvaluator()
    result = evaluator.evaluate(
        query="Who are our at-risk customers?",
        response=agent_output_text,
        agent="segmentation",
        tool_results=["identify_churn_risk result..."],
    )
    print(result["overall_score"], result["flags"])

Report:
    python -m llmops.response_evaluator --log-file llmops/eval_log.jsonl
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_LOG = PROJECT_ROOT / "llmops" / "eval_log.jsonl"

# Dimension weights for overall score
WEIGHTS = {
    "completeness": 0.30,
    "groundedness": 0.25,
    "structure":    0.20,
    "actionability": 0.15,
    "conciseness":  0.10,
}

# Quality thresholds
THRESHOLDS = {"poor": 0.40, "acceptable": 0.65, "good": 0.80, "excellent": 0.90}


@dataclass
class EvalResult:
    timestamp: str
    agent: str
    query: str
    response_length: int
    scores: dict
    overall_score: float
    quality_label: str
    flags: list
    query_id: str = ""


# ─── Dimension scorers ────────────────────────────────────────────────────────

def _score_completeness(query: str, response: str) -> tuple[float, list[str]]:
    """Check that response touches on key concepts from the query."""
    flags = []
    if not response.strip():
        return 0.0, ["empty_response"]

    # Extract likely intent keywords from the query
    query_lower = query.lower()
    response_lower = response.lower()

    # Domain keyword groups
    keyword_groups = {
        "customer": ["customer", "user", "client", "segment"],
        "campaign": ["campaign", "budget", "channel", "roi", "email", "sms"],
        "recommendation": ["recommend", "suggest", "product", "next"],
        "crm": ["crm", "loyalty", "tier", "points", "clv", "lifetime"],
        "segmentation": ["segment", "rfm", "cluster", "churn", "at-risk", "at risk"],
    }

    matched_groups = 0
    total_relevant_groups = 0
    for group, kws in keyword_groups.items():
        if any(k in query_lower for k in kws):
            total_relevant_groups += 1
            if any(k in response_lower for k in kws):
                matched_groups += 1

    if total_relevant_groups == 0:
        return 0.75, []  # generic query — hard to judge

    score = matched_groups / total_relevant_groups

    # Penalty for very short responses to complex queries
    words_in_query = len(query.split())
    words_in_response = len(response.split())
    if words_in_query > 10 and words_in_response < 30:
        score *= 0.6
        flags.append("response_too_brief")

    return round(min(score, 1.0), 3), flags


def _score_groundedness(response: str, tool_results: list[str]) -> tuple[float, list[str]]:
    """Check that claims in the response appear to be grounded in tool data."""
    flags = []
    response_lower = response.lower()

    # Signals of data grounding
    grounding_patterns = [
        r"\b\d+\.?\d*%",                   # percentages
        r"\$[\d,]+\.?\d*",                  # dollar amounts
        r"\b\d+\s+customers?\b",            # numeric counts
        r"\bcluster\s+\d\b",               # cluster references
        r"\bscore\s+of\s+\d",             # score references
        r"\broi\s+of\s+[\d.]+",            # ROI references
        r"\bchurn\s+risk\b",               # churn risk mention
        r"\blifetime\s+value\b",           # CLV mention
        r"\bplatinum|gold|silver|bronze\b", # tier names
        r"\bsegment\b.{0,30}\bchampion|at.risk|loyal\b",  # segment names
    ]

    matched = sum(1 for pat in grounding_patterns if re.search(pat, response_lower))
    grounding_score = min(matched / 4, 1.0)  # 4+ signals = fully grounded

    # Bonus if response contains specific values from tool results
    if tool_results:
        all_tool_text = " ".join(tool_results).lower()
        # Find numbers in tool results and check if they appear in response
        tool_numbers = set(re.findall(r"\b\d{2,}\b", all_tool_text))
        response_numbers = set(re.findall(r"\b\d{2,}\b", response_lower))
        overlap = len(tool_numbers & response_numbers)
        if overlap >= 2:
            grounding_score = min(grounding_score + 0.2, 1.0)

    # Hallucination signals
    hallucination_patterns = [
        r"\bas of \d{4}\b",        # year references not in tools
        r"\baccording to \w+\b",   # unexplained sources
        r"\bi believe\b",
        r"\bi think\b",
        r"\bi am not sure\b",
        r"\bi cannot access\b",
    ]
    hallucination_hits = sum(1 for p in hallucination_patterns if re.search(p, response_lower))
    if hallucination_hits > 0:
        grounding_score = max(0, grounding_score - 0.15 * hallucination_hits)
        flags.append(f"possible_hallucination_signals_{hallucination_hits}")

    return round(grounding_score, 3), flags


def _score_structure(response: str) -> tuple[float, list[str]]:
    """Check for well-organised structure: headings, bullets, numeric data."""
    flags = []
    lines = response.split("\n")

    has_headers = any(line.strip().startswith("#") or
                      (line.strip().endswith(":") and len(line.strip()) < 60)
                      for line in lines)
    has_bullets = any(line.strip().startswith(("-", "*", "•", "–")) for line in lines)
    has_numbered = bool(re.search(r"^\s*\d+[.)]\s", response, re.MULTILINE))
    has_metrics = bool(re.search(r"\b(score|value|rate|roi|clv|pct|%|\$)\b", response.lower()))
    has_sections = len([l for l in lines if l.strip().startswith("#")]) >= 2

    score_parts = [
        (0.25, has_headers),
        (0.25, has_bullets or has_numbered),
        (0.25, has_metrics),
        (0.25, has_sections or (has_headers and has_bullets)),
    ]
    score = sum(w for w, v in score_parts if v)

    # All-prose responses are penalised unless very short
    word_count = len(response.split())
    if word_count > 100 and not has_bullets and not has_headers:
        score *= 0.6
        flags.append("unstructured_prose")

    return round(score, 3), flags


def _score_actionability(response: str) -> tuple[float, list[str]]:
    """Check for concrete recommendations, next steps, or action verbs."""
    flags = []
    response_lower = response.lower()

    action_patterns = [
        r"\b(recommend|suggest|should|consider|action|step|next|implement|launch|send|create)\b",
        r"\b(priority|immediate|urgently|first|then|finally)\b",
        r"\b(campaign|email|offer|discount|reward|message|contact)\b",
        r"\b(increase|decrease|allocate|target|focus|prioritize)\b",
    ]

    matched = sum(1 for p in action_patterns if re.search(p, response_lower))
    score = min(matched / 3, 1.0)  # 3+ action patterns = fully actionable

    if score < 0.3:
        flags.append("low_actionability")

    return round(score, 3), flags


def _score_conciseness(response: str, query: str) -> tuple[float, list[str]]:
    """Penalise both very short and excessively long responses."""
    flags = []
    words = len(response.split())
    query_complexity = len(query.split())

    # Expected response range based on query complexity
    expected_min = max(50, query_complexity * 5)
    expected_max = max(500, query_complexity * 30)

    if words < expected_min:
        score = words / expected_min
        flags.append("response_too_short")
    elif words > expected_max * 2:
        score = expected_max / words
        flags.append("response_too_long")
    elif words > expected_max:
        score = 0.85  # slightly long but acceptable
    else:
        score = 1.0

    return round(score, 3), flags


# ─── Evaluator ────────────────────────────────────────────────────────────────

class ResponseEvaluator:
    def __init__(self, eval_log: Path = EVAL_LOG, weights: dict = WEIGHTS):
        self._log = eval_log
        self._log.parent.mkdir(parents=True, exist_ok=True)
        self._weights = weights

    def evaluate(
        self,
        query: str,
        response: str,
        agent: str = "unknown",
        tool_results: list[str] | None = None,
        query_id: str = "",
    ) -> dict:
        """Score a single agent response. Returns EvalResult as dict."""
        tool_results = tool_results or []

        # Score each dimension
        completeness, c_flags = _score_completeness(query, response)
        groundedness, g_flags = _score_groundedness(response, tool_results)
        structure, s_flags = _score_structure(response)
        actionability, a_flags = _score_actionability(response)
        conciseness, con_flags = _score_conciseness(response, query)

        scores = {
            "completeness": completeness,
            "groundedness": groundedness,
            "structure": structure,
            "actionability": actionability,
            "conciseness": conciseness,
        }

        overall = round(
            sum(scores[k] * self._weights[k] for k in self._weights), 4
        )

        quality_label = "poor"
        for label, threshold in [
            ("excellent", THRESHOLDS["excellent"]),
            ("good", THRESHOLDS["good"]),
            ("acceptable", THRESHOLDS["acceptable"]),
        ]:
            if overall >= threshold:
                quality_label = label
                break

        all_flags = c_flags + g_flags + s_flags + a_flags + con_flags

        result = EvalResult(
            timestamp=datetime.now().isoformat(),
            agent=agent,
            query=query[:200],
            response_length=len(response.split()),
            scores=scores,
            overall_score=overall,
            quality_label=quality_label,
            flags=all_flags,
            query_id=query_id,
        )

        # Append to log
        with open(self._log, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

        return asdict(result)

    def load_log(self, since: str | None = None) -> list[dict]:
        if not self._log.exists():
            return []
        records = []
        with open(self._log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if since and d.get("timestamp", "") < since:
                        continue
                    records.append(d)
                except json.JSONDecodeError:
                    pass
        return records

    def summary_report(self, since: str | None = None) -> dict:
        records = self.load_log(since=since)
        if not records:
            return {"total_evaluations": 0}

        by_agent: dict[str, list[float]] = {}
        label_counts: dict[str, int] = {}
        flag_counts: dict[str, int] = {}
        dim_totals: dict[str, float] = {k: 0.0 for k in WEIGHTS}

        for r in records:
            agent = r.get("agent", "unknown")
            by_agent.setdefault(agent, []).append(r["overall_score"])
            label_counts[r["quality_label"]] = label_counts.get(r["quality_label"], 0) + 1
            for flag in r.get("flags", []):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
            for dim in dim_totals:
                dim_totals[dim] += r.get("scores", {}).get(dim, 0)

        n = len(records)
        return {
            "total_evaluations": n,
            "since": since,
            "avg_overall_score": round(sum(r["overall_score"] for r in records) / n, 4),
            "quality_distribution": label_counts,
            "avg_dimension_scores": {k: round(v / n, 4) for k, v in dim_totals.items()},
            "by_agent": {
                agent: {
                    "evaluations": len(scores),
                    "avg_score": round(sum(scores) / len(scores), 4),
                    "min_score": round(min(scores), 4),
                    "max_score": round(max(scores), 4),
                }
                for agent, scores in by_agent.items()
            },
            "top_flags": sorted(flag_counts.items(), key=lambda x: -x[1])[:10],
        }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Response quality evaluation report")
    parser.add_argument("--since", default=None, help="ISO date filter")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Run a demo evaluation")
    args = parser.parse_args()

    if args.demo:
        evaluator = ResponseEvaluator()
        demo_query = "Who are our at-risk customers and what should we do?"
        demo_response = """
# At-Risk Customer Analysis

## Key Findings
Based on RFM analysis, I identified **7 customers** with high churn risk scores (>0.6):

- **Carol Test** (C003): risk=0.92, last purchase 580 days ago, CLV=$150
- **Bob Test** (C002): risk=0.71, last purchase 175 days ago, CLV=$900

## Recommended Actions

1. **Immediate win-back campaign** for critical-risk customers (C003):
   - Offer 30% discount on their top category (Electronics)
   - Send via email within 48 hours
   - Expected response rate: ~12%

2. **High-risk nurture sequence** for C002:
   - 3-part email series over 2 weeks
   - Feature loyalty points reminder ($1,500 balance)
   - Include personalised product recommendations

3. **Budget allocation**: Allocate $2,000 from Q1 budget to reactivation campaigns
   targeting customers with CLV > $500 and risk_score > 0.6.
"""
        result = evaluator.evaluate(
            query=demo_query,
            response=demo_response,
            agent="segmentation",
            tool_results=["identify_churn_risk result: 7 at-risk, avg risk 0.75"],
        )
        print("\n=== Demo Evaluation ===")
        print(f"  Overall score:  {result['overall_score']:.4f} ({result['quality_label']})")
        print("  Dimension scores:")
        for dim, score in result["scores"].items():
            bar = "█" * int(score * 20)
            print(f"    {dim:>14}: {score:.3f} {bar}")
        print(f"  Flags: {result['flags'] or 'none'}")
        return

    evaluator = ResponseEvaluator()
    summary = evaluator.summary_report(since=args.since)

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    if summary["total_evaluations"] == 0:
        print("No evaluation records found.")
        return

    print("\n=== Response Quality Report ===")
    if args.since:
        print(f"  Since: {args.since}")
    print(f"  Total evaluations:   {summary['total_evaluations']}")
    print(f"  Avg overall score:   {summary['avg_overall_score']:.4f}")
    print(f"  Quality distribution: {summary['quality_distribution']}")

    print("\n  Avg dimension scores:")
    for dim, score in summary["avg_dimension_scores"].items():
        bar = "█" * int(score * 20)
        weight = WEIGHTS.get(dim, 0)
        print(f"    {dim:>14} (w={weight:.2f}): {score:.4f} {bar}")

    if summary.get("by_agent"):
        print("\n  By agent:")
        for agent, stats in summary["by_agent"].items():
            print(f"    {agent}: avg={stats['avg_score']:.4f}, "
                  f"min={stats['min_score']:.4f}, max={stats['max_score']:.4f}")

    if summary.get("top_flags"):
        print("\n  Top quality flags:")
        for flag, count in summary["top_flags"][:5]:
            print(f"    {flag}: {count}x")


if __name__ == "__main__":
    main()
