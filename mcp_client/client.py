"""
MCP Client Wrapper for the Customer Intelligence Platform.

Manages the FastMCP server subprocess lifecycle via stdio transport.
Provides a clean async interface for agents to:
  - List available tools (converted to Anthropic tool_use format)
  - Call tools and get results

Key design: MCP uses camelCase 'inputSchema' but Anthropic requires
snake_case 'input_schema'. This wrapper handles the conversion.

Usage:
    # As async context manager (recommended)
    async with MCPClientWrapper() as client:
        tools = await client.get_tools_for_anthropic()
        result = await client.call_tool("compute_rfm_scores", {"recency_days": 90})

    # Manual lifecycle
    client = MCPClientWrapper()
    await client.connect()
    tools = await client.get_tools_for_anthropic()
    await client.disconnect()
"""
import sys
import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Path to the MCP server script
_PACKAGE_ROOT = Path(__file__).parent.parent
SERVER_SCRIPT = str(_PACKAGE_ROOT / "mcp_server" / "server.py")


class MCPClientWrapper:
    """
    Wraps the MCP ClientSession and manages the server subprocess.

    The server is started as a Python subprocess communicating via stdio.
    All tool calls go through JSON-RPC over stdin/stdout of the subprocess.

    Thread/coroutine safety: each MCPClientWrapper instance manages its own
    subprocess. For parallel agent execution, share a single instance across agents
    (the MCP protocol is multiplexed over a single connection).
    """

    def __init__(self, server_script: Optional[str] = None):
        self.server_script = server_script or SERVER_SCRIPT
        self._session = None
        self._exit_stack = AsyncExitStack()
        self._tools_cache: Optional[list[dict]] = None

    async def connect(self) -> None:
        """
        Start the MCP server subprocess and initialize the MCP session.
        Must be called before any tool operations.
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "MCP library not found. Install with: pip install mcp"
            )

        logger.info(f"Starting MCP server subprocess: {self.server_script}")

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
            env=None,  # Inherit environment (ANTHROPIC_API_KEY will be available)
        )

        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()
        logger.info("MCP client session initialized successfully.")

    async def disconnect(self) -> None:
        """Shut down the MCP session and server subprocess cleanly."""
        await self._exit_stack.aclose()
        self._session = None
        self._tools_cache = None
        logger.info("MCP client disconnected.")

    async def __aenter__(self) -> "MCPClientWrapper":
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()

    async def get_tools_for_anthropic(self) -> list[dict]:
        """
        List all MCP tools and convert to Anthropic tool_use format.

        MCP format:   tool.name, tool.description, tool.inputSchema (camelCase)
        Anthropic:    {"name": ..., "description": ..., "input_schema": ...}  (snake_case)

        Results are cached after first call.
        """
        if self._tools_cache is not None:
            return self._tools_cache

        if self._session is None:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        result = await self._session.list_tools()
        anthropic_tools = []

        for tool in result.tools:
            # Convert MCP inputSchema (camelCase) → Anthropic input_schema (snake_case)
            input_schema = tool.inputSchema if hasattr(tool, "inputSchema") else {}

            # Ensure it's a proper JSON Schema object
            if not isinstance(input_schema, dict):
                input_schema = {"type": "object", "properties": {}}

            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description or f"Tool: {tool.name}",
                "input_schema": input_schema,
            })

        self._tools_cache = anthropic_tools
        logger.info(f"Loaded {len(anthropic_tools)} tools from MCP server.")
        return anthropic_tools

    async def get_domain_tools(self, domain: str) -> list[dict]:
        """
        Get tools filtered to a specific domain.

        Domain prefix mapping:
          segmentation → compute_rfm, run_customer_clust, get_segment, identify_churn, compare_segments
          campaign     → get_campaign, recommend_budget, identify_target, forecast_campaign, ab_test
          recommendation → get_product_rec, get_similar_cust, get_next_best, get_trending, get_content_rec
          crm          → get_customer_profile, calculate_clv, get_loyalty, process_points, get_tier_upgrade
        """
        all_tools = await self.get_tools_for_anthropic()

        domain_prefixes: dict[str, list[str]] = {
            "segmentation": [
                "compute_rfm_scores",
                "run_customer_clustering",
                "get_segment_profile",
                "identify_churn_risk",
                "compare_segments",
            ],
            "campaign": [
                "get_campaign_performance",
                "recommend_budget_allocation",
                "identify_target_audience",
                "forecast_campaign_roi",
                "ab_test_analysis",
            ],
            "recommendation": [
                "get_product_recommendations",
                "get_similar_customers",
                "get_next_best_action",
                "get_trending_products",
                "get_content_recommendations",
            ],
            "crm": [
                "get_customer_profile",
                "calculate_clv",
                "get_loyalty_summary",
                "process_points_transaction",
                "get_tier_upgrade_candidates",
            ],
        }

        allowed_names = set(domain_prefixes.get(domain, []))
        if not allowed_names:
            # Unknown domain — return all tools
            return all_tools

        filtered = [t for t in all_tools if t["name"] in allowed_names]
        logger.debug(f"Domain '{domain}': {len(filtered)} tools available")
        return filtered

    async def call_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str:
        """
        Call a named tool on the MCP server and return the text result.

        Returns a JSON error string (not exception) if the call fails,
        so agents can reason about the error gracefully.
        """
        if self._session is None:
            return json.dumps({"error": "MCP client not connected"})

        try:
            result = await self._session.call_tool(tool_name, tool_args)

            if not result.content:
                return json.dumps({"status": "success", "result": None})

            # MCP content can be text or other types; extract text
            first_content = result.content[0]
            if hasattr(first_content, "text"):
                return first_content.text
            else:
                return str(first_content)

        except Exception as e:
            error_msg = f"Tool '{tool_name}' call failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({
                "error": error_msg,
                "tool_name": tool_name,
                "args": str(tool_args)[:200],
            })

    async def list_tool_names(self) -> list[str]:
        """Return just the names of all available tools."""
        tools = await self.get_tools_for_anthropic()
        return [t["name"] for t in tools]

    async def warmup(self) -> int:
        """Pre-load tool cache and return tool count."""
        tools = await self.get_tools_for_anthropic()
        return len(tools)
