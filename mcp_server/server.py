"""
Customer Intelligence Platform - FastMCP Server.

This server exposes 20 tools across 4 domains:
  - Customer Segmentation (5 tools)
  - Campaign Optimization (5 tools)
  - Recommendations (5 tools)
  - CRM/Loyalty (5 tools)

Transport: stdio (communicates over stdin/stdout with the MCP client).

CRITICAL: This process must NEVER write to stdout directly.
Any print() call will corrupt the JSON-RPC protocol stream.
All logging MUST go to stderr via the logging module.
"""
import sys
import logging

# Configure logging to stderr BEFORE any other imports that might log
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mcp_server")

# Import FastMCP
try:
    from fastmcp import FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.error("FastMCP not found. Install with: pip install fastmcp or pip install mcp")
        sys.exit(1)

# Import data store and tool modules
try:
    from mcp_server.data_store import DataStore
    from mcp_server.tools import segmentation_tools, campaign_tools, recommendation_tools, crm_loyalty_tools
except ImportError:
    # Allow running as __main__ from project root
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mcp_server.data_store import DataStore
    from mcp_server.tools import segmentation_tools, campaign_tools, recommendation_tools, crm_loyalty_tools

# Initialize FastMCP server
mcp = FastMCP(
    "customer-intelligence-mcp",
    instructions=(
        "Customer Intelligence Platform MCP Server. "
        "Provides 20 tools for customer segmentation (RFM analysis, clustering, churn risk), "
        "campaign optimization (performance, budget allocation, forecasting), "
        "product recommendations (collaborative filtering, trending, next-best-action), "
        "and CRM/loyalty management (CLV, tier management, points transactions). "
        "All tool results are returned as JSON strings."
    ),
)

# Initialize data store (loads all JSON files at startup)
logger.info("Initializing DataStore...")
data_store = DataStore()
logger.info("DataStore initialized successfully.")

# Register all domain tools
logger.info("Registering tools...")
segmentation_tools.register(mcp, data_store)
campaign_tools.register(mcp, data_store)
recommendation_tools.register(mcp, data_store)
crm_loyalty_tools.register(mcp, data_store)
logger.info("All tools registered. Starting MCP server on stdio transport.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
