"""
Multi-Agent Orchestrator for the Customer Intelligence Platform.

Three-phase execution pipeline:
  Phase 1 - Route:     Claude analyzes the query and selects agents + execution mode
  Phase 2 - Execute:   Agents run in parallel (asyncio.gather) or sequentially
  Phase 3 - Synthesize: Claude integrates all agent responses into a final report

Routing returns a structured JSON plan:
  {
    "agents": ["segmentation", "crm"],
    "parallel": true,
    "reasoning": "explanation",
    "query_per_agent": {
      "segmentation": "specific sub-query for segmentation agent",
      "crm": "specific sub-query for CRM agent"
    }
  }
"""
import asyncio
import json
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"

ROUTING_SYSTEM_PROMPT = """You are the routing controller for a Customer Intelligence Platform.
Analyze the user query and determine which specialist agents to invoke.

Available agents and their capabilities:
- "segmentation": RFM analysis, customer clustering, churn risk identification, segment profiling, segment comparison
- "campaign": Campaign performance analysis, budget allocation, audience targeting, ROI forecasting, A/B testing
- "recommendation": Product recommendations, similar customer discovery, next-best-action, trending products, content personalization
- "crm": Customer profiles, CLV calculation, loyalty tier management, points transactions, tier upgrade candidates

Respond ONLY with valid JSON in exactly this format (no markdown, no extra text):
{
  "agents": ["agent1"],
  "parallel": true,
  "reasoning": "One sentence explaining the routing decision",
  "query_per_agent": {
    "agent1": "Specific focused sub-query for this agent"
  }
}

Rules:
- Select ONLY the agents whose domain is relevant to the query
- Use parallel=true when agents need independent data (they can run simultaneously)
- Use parallel=false when agent2 needs agent1's results as input (sequential dependency)
- Tailor each "query_per_agent" to that agent's specific domain — avoid generic queries
- Include 1-4 agents maximum; prefer fewer focused agents over many generic ones

Examples:
Query: "How are our campaigns performing?"
→ {"agents": ["campaign"], "parallel": true, "reasoning": "Pure campaign performance question.", "query_per_agent": {"campaign": "Analyze all campaign performance metrics including ROI, CTR, and revenue by channel"}}

Query: "Who are our best customers and what should we sell them?"
→ {"agents": ["segmentation", "recommendation"], "parallel": true, "reasoning": "Segmentation identifies Champions; Recommendations are computed independently per customer.", "query_per_agent": {"segmentation": "Identify and profile the Champions segment using RFM analysis", "recommendation": "Get top product recommendations for high-value Champion customers"}}

Query: "Find at-risk customers and create a personalized win-back campaign"
→ {"agents": ["segmentation", "campaign", "recommendation"], "parallel": false, "reasoning": "Sequential: segmentation identifies at-risk customers first, then campaign targets them, then recommendations personalize the offer.", "query_per_agent": {"segmentation": "Identify all at-risk customers with churn scores and profiles", "campaign": "Design a reactivation campaign targeting at-risk customers with budget and channel recommendations", "recommendation": "Recommend products for at-risk customers to include in win-back offers"}}"""


SYNTHESIS_SYSTEM_PROMPT = """You are the synthesis engine for a Customer Intelligence Platform.
You receive outputs from multiple specialized AI agents and combine them into a single,
comprehensive, actionable intelligence report for a business user.

Guidelines for synthesis:
1. Integrate insights across domains — connect the dots between segmentation, campaigns, recommendations, and CRM data
2. Eliminate redundancy — don't repeat the same finding from multiple agents
3. Organize with clear headers and bullet points for scannability
4. Lead with the most important/surprising finding
5. End with a prioritized "Key Actions" section (ranked by expected business impact)
6. Use concrete numbers ($, %, counts) from agent data — avoid vague language
7. If agents contradict each other, note the discrepancy and offer reconciliation
8. Maintain professional, executive-summary tone

Structure your response as:
## [Descriptive Title]

### Key Findings
[2-4 most important insights]

### [Domain 1 Insights]
[Specific findings from relevant agents]

### [Domain 2 Insights]
[Additional domain findings if multiple agents ran]

### Key Actions (Prioritized)
1. **[Action]** — [Expected impact in $] | [Urgency]
2. ...
3. ..."""


class Orchestrator:
    """
    Multi-agent orchestrator for the Customer Intelligence Platform.

    Coordinates up to 4 specialized agents:
    - SegmentationAgent (RFM, clustering, churn)
    - CampaignAgent (performance, budget, targeting)
    - RecommendationAgent (products, NBA, trending)
    - CRMLoyaltyAgent (profiles, CLV, loyalty)
    """

    def __init__(self, mcp_client):
        """
        Args:
            mcp_client: MCPClientWrapper instance (shared by all agents)
        """
        from agents.segmentation_agent import SegmentationAgent
        from agents.campaign_agent import CampaignAgent
        from agents.recommendation_agent import RecommendationAgent
        from agents.crm_loyalty_agent import CRMLoyaltyAgent

        self.mcp_client = mcp_client

        self.agents = {
            "segmentation": SegmentationAgent(mcp_client),
            "campaign": CampaignAgent(mcp_client),
            "recommendation": RecommendationAgent(mcp_client),
            "crm": CRMLoyaltyAgent(mcp_client),
        }

        try:
            import anthropic
            self._claude = anthropic.Anthropic()
        except ImportError:
            raise ImportError("anthropic package not found. Install with: pip install anthropic")

    async def route(self, query: str) -> dict:
        """
        Phase 1: Use Claude to determine which agents to invoke and execution mode.

        Returns a routing plan dict:
        {
          "agents": [...],
          "parallel": bool,
          "reasoning": "...",
          "query_per_agent": {...}
        }
        """
        logger.info(f"Routing query: {query[:100]}...")

        response = await asyncio.to_thread(
            self._claude.messages.create,
            model=MODEL,
            max_tokens=512,
            system=ROUTING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )

        raw = response.content[0].text.strip() if response.content else "{}"

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            plan = json.loads(raw)
            # Validate structure
            if "agents" not in plan:
                raise ValueError("Missing 'agents' key in routing plan")
            # Filter to only valid agent names
            plan["agents"] = [a for a in plan["agents"] if a in self.agents]
            if not plan["agents"]:
                plan["agents"] = ["segmentation"]
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Routing response parse error: {e}. Raw: {raw[:200]}")
            # Safe fallback: route to most relevant agent based on keywords
            plan = self._fallback_routing(query)

        logger.info(
            f"Routing plan: agents={plan['agents']}, "
            f"parallel={plan.get('parallel', True)}, "
            f"reason={plan.get('reasoning', '')[:80]}"
        )
        return plan

    def _fallback_routing(self, query: str) -> dict:
        """Keyword-based fallback routing when Claude routing fails."""
        q = query.lower()
        agents = []
        if any(w in q for w in ["segment", "rfm", "churn", "cluster", "at risk", "champions"]):
            agents.append("segmentation")
        if any(w in q for w in ["campaign", "budget", "roi", "marketing", "channel", "email", "sms"]):
            agents.append("campaign")
        if any(w in q for w in ["recommend", "product", "suggest", "trending", "similar", "next"]):
            agents.append("recommendation")
        if any(w in q for w in ["customer", "profile", "clv", "loyalty", "points", "tier", "crm"]):
            agents.append("crm")

        if not agents:
            agents = ["segmentation", "crm"]

        return {
            "agents": agents,
            "parallel": True,
            "reasoning": "Keyword-based fallback routing",
            "query_per_agent": {a: query for a in agents},
        }

    async def execute_agents(
        self,
        plan: dict,
        original_query: str,
        on_agent_start: Optional[Callable] = None,
        on_agent_complete: Optional[Callable] = None,
        on_tool_call: Optional[Callable] = None,
    ) -> dict[str, str]:
        """
        Phase 2: Execute selected agents per the routing plan.

        Parallel mode: all agents run concurrently via asyncio.gather()
        Sequential mode: each agent receives prior agents' output as context

        Returns: {agent_name: response_text}
        """
        selected = plan.get("agents", [])
        parallel = plan.get("parallel", True)
        query_map = plan.get("query_per_agent", {})

        agent_results: dict[str, str] = {}

        if parallel:
            async def run_one(name: str):
                agent = self.agents.get(name)
                if not agent:
                    return name, f"[Unknown agent: {name}]"

                sub_query = query_map.get(name, original_query)
                logger.info(f"Starting agent: {name} (parallel)")

                if on_agent_start:
                    try:
                        on_agent_start(name)
                    except Exception:
                        pass

                result = await agent.run(sub_query, on_tool_call=on_tool_call)

                if on_agent_complete:
                    try:
                        on_agent_complete(name, result)
                    except Exception:
                        pass

                logger.info(f"Agent {name} completed (parallel)")
                return name, result

            tasks = [run_one(name) for name in selected]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

            for outcome in outcomes:
                if isinstance(outcome, Exception):
                    logger.error(f"Agent task exception: {outcome}")
                    agent_results[f"error_{id(outcome)}"] = f"[Agent error: {outcome}]"
                else:
                    name, result = outcome
                    agent_results[name] = result

        else:
            # Sequential: each agent gets prior results as context
            accumulated_context = ""

            for name in selected:
                agent = self.agents.get(name)
                if not agent:
                    logger.warning(f"Unknown agent requested: {name}")
                    continue

                sub_query = query_map.get(name, original_query)
                logger.info(f"Starting agent: {name} (sequential, iteration {len(agent_results)+1}/{len(selected)})")

                if on_agent_start:
                    try:
                        on_agent_start(name)
                    except Exception:
                        pass

                context = accumulated_context if accumulated_context else None
                result = await agent.run(sub_query, context=context, on_tool_call=on_tool_call)
                agent_results[name] = result

                # Accumulate for next agent
                accumulated_context += f"\n\n=== {name.upper()} AGENT ANALYSIS ===\n{result}"

                if on_agent_complete:
                    try:
                        on_agent_complete(name, result)
                    except Exception:
                        pass

                logger.info(f"Agent {name} completed (sequential)")

        return agent_results

    async def synthesize(
        self,
        query: str,
        agent_results: dict[str, str],
    ) -> str:
        """
        Phase 3: Synthesize all agent outputs into a coherent final report.

        If only one agent ran, returns its response directly (no synthesis overhead).
        """
        if len(agent_results) == 1:
            return next(iter(agent_results.values()))

        if not agent_results:
            return "No agent results were produced. Please try again with a more specific query."

        # Build synthesis prompt
        synthesis_input = f"Original user query: {query}\n\n"
        for agent_name, result in agent_results.items():
            synthesis_input += f"{'='*50}\n"
            synthesis_input += f"## {agent_name.upper().replace('_', ' ')} AGENT ANALYSIS\n"
            synthesis_input += f"{'='*50}\n"
            synthesis_input += f"{result}\n\n"

        logger.info(f"Synthesizing {len(agent_results)} agent responses...")

        response = await asyncio.to_thread(
            self._claude.messages.create,
            model=MODEL,
            max_tokens=2048,
            system=SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": synthesis_input}],
        )

        return response.content[0].text if response.content else "[Synthesis failed]"

    async def process(
        self,
        query: str,
        on_routing_complete: Optional[Callable] = None,
        on_agent_start: Optional[Callable] = None,
        on_agent_complete: Optional[Callable] = None,
        on_tool_call: Optional[Callable] = None,
    ) -> dict:
        """
        Full three-phase pipeline: route → execute → synthesize.

        Args:
            query: The user's natural language query.
            on_routing_complete: Callback(plan) fired after routing.
            on_agent_start: Callback(agent_name) fired when an agent begins.
            on_agent_complete: Callback(agent_name, result) fired when an agent finishes.
            on_tool_call: Callback(tool_name, args, result) fired for each tool call.

        Returns:
            {
                "plan": routing plan dict,
                "agent_results": {agent_name: response_text},
                "final_response": synthesized response string,
            }
        """
        # Phase 1: Route
        plan = await self.route(query)

        if on_routing_complete:
            try:
                on_routing_complete(plan)
            except Exception:
                pass

        # Phase 2: Execute
        agent_results = await self.execute_agents(
            plan,
            query,
            on_agent_start=on_agent_start,
            on_agent_complete=on_agent_complete,
            on_tool_call=on_tool_call,
        )

        # Phase 3: Synthesize
        final_response = await self.synthesize(query, agent_results)

        return {
            "plan": plan,
            "agent_results": agent_results,
            "final_response": final_response,
        }
