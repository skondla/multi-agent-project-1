"""
Customer Intelligence Platform - FastAPI Web Service.

Wraps the multi-agent orchestrator as a REST API suitable for
Kubernetes/EKS deployment. The MCP server subprocess is managed
by a single long-lived MCPClientWrapper for the life of the service.

Architecture in K8s:
  Each pod: uvicorn worker → Orchestrator → MCPClientWrapper → MCP subprocess
  Scaling: horizontal pod autoscaler (stateless, no shared state between pods)

Endpoints:
  GET  /health              Liveness probe
  GET  /readiness           Readiness probe (waits for MCP connection)
  GET  /metrics             Basic operational metrics
  GET  /api/v1/agents       List available specialist agents
  GET  /api/v1/tools        List all MCP tools
  POST /api/v1/query        Process a natural language intelligence query

Usage (local):
  uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

Usage (production):
  uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 1
  (single worker: MCP subprocess is in-process; scale via K8s replicas)
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

APP_VERSION = "0.1.0"


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language query for the intelligence platform",
        examples=["Who are our at-risk customers?"],
    )
    show_agent_details: bool = Field(
        False,
        description="Include per-agent responses in the result (verbose mode)",
    )


class QueryResponse(BaseModel):
    query: str
    final_response: str
    agents_used: list[str]
    execution_mode: str  # "parallel" | "sequential"
    routing_reasoning: str
    elapsed_seconds: float
    agent_results: Optional[dict[str, str]] = None


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    mcp_connected: bool
    tools_available: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    uptime_seconds: float
    tools_available: int
    version: str


class AgentInfo(BaseModel):
    name: str
    domain: str
    description: str
    tools: list[str]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ─── Application State ────────────────────────────────────────────────────────

class _AppState:
    """Container for application-level state (managed by lifespan)."""
    mcp_client: Any = None
    orchestrator: Any = None
    tools_count: int = 0
    startup_time: Optional[float] = None


_state = _AppState()


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    - Startup: start MCP server subprocess, create orchestrator, warm up tool cache.
    - Shutdown: gracefully disconnect MCP subprocess.
    """
    # Lazy import to allow running tests without all deps
    from mcp_client.client import MCPClientWrapper
    from orchestrator.orchestrator import Orchestrator

    logger.info("Customer Intelligence Platform API starting up...")

    _state.mcp_client = MCPClientWrapper()
    await _state.mcp_client.connect()
    logger.info("MCP client connected.")

    _state.orchestrator = Orchestrator(_state.mcp_client)

    # Pre-warm tool cache
    _state.tools_count = await _state.mcp_client.warmup()
    _state.startup_time = time.monotonic()

    logger.info(
        f"API ready. {_state.tools_count} MCP tools loaded across 4 domains."
    )

    yield  # ← Application is running

    # Graceful shutdown
    logger.info("Shutting down MCP client...")
    if _state.mcp_client:
        await _state.mcp_client.disconnect()
    logger.info("Shutdown complete.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Intelligence Platform API",
    description=(
        "Multi-agent AI API for customer segmentation, campaign optimization, "
        "product recommendations, and CRM/loyalty management. "
        "Powered by Claude claude-sonnet-4-6 and Model Context Protocol (MCP)."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware (configure allowed origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Exception Handlers ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ─── Health & Readiness Probes ────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness probe",
)
async def health():
    """
    Kubernetes liveness probe.
    Returns 200 if the API process is alive (regardless of MCP status).
    """
    return {"status": "healthy", "version": APP_VERSION}


@app.get(
    "/readiness",
    response_model=ReadinessResponse,
    tags=["Health"],
    summary="Readiness probe",
)
async def readiness():
    """
    Kubernetes readiness probe.
    Returns 200 only when the MCP server is connected and tools are loaded.
    Returns 503 if the service is still initializing.
    """
    if _state.mcp_client is None or _state.tools_count == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP server not ready. Service is still initializing.",
        )
    uptime = time.monotonic() - (_state.startup_time or time.monotonic())
    return {
        "status": "ready",
        "mcp_connected": True,
        "tools_available": _state.tools_count,
        "uptime_seconds": round(uptime, 1),
    }


# ─── Metrics ─────────────────────────────────────────────────────────────────

@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Health"],
    summary="Basic operational metrics",
)
async def metrics():
    """Basic operational metrics for monitoring."""
    uptime = (
        time.monotonic() - _state.startup_time
        if _state.startup_time else 0
    )
    return {
        "uptime_seconds": round(uptime, 1),
        "tools_available": _state.tools_count,
        "version": APP_VERSION,
    }


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.get(
    "/api/v1/agents",
    tags=["Platform"],
    summary="List available specialist agents",
)
async def list_agents():
    """List all specialist agents, their domains, and available MCP tools."""
    return {
        "agents": [
            {
                "name": "segmentation",
                "domain": "Customer Segmentation",
                "description": "RFM analysis, K-Means clustering, churn risk scoring, segment profiling",
                "tools": ["compute_rfm_scores", "run_customer_clustering", "get_segment_profile",
                          "identify_churn_risk", "compare_segments"],
            },
            {
                "name": "campaign",
                "domain": "Campaign Optimization",
                "description": "Performance analytics, budget allocation, audience targeting, ROI forecasting",
                "tools": ["get_campaign_performance", "recommend_budget_allocation",
                          "identify_target_audience", "forecast_campaign_roi", "ab_test_analysis"],
            },
            {
                "name": "recommendation",
                "domain": "Personalization & Recommendations",
                "description": "Product recommendations, next-best-action, trending products, content personalization",
                "tools": ["get_product_recommendations", "get_similar_customers", "get_next_best_action",
                          "get_trending_products", "get_content_recommendations"],
            },
            {
                "name": "crm",
                "domain": "CRM / Loyalty",
                "description": "Customer profiles, CLV projection, loyalty tier management, points transactions",
                "tools": ["get_customer_profile", "calculate_clv", "get_loyalty_summary",
                          "process_points_transaction", "get_tier_upgrade_candidates"],
            },
        ],
        "total_agents": 4,
    }


@app.get(
    "/api/v1/tools",
    tags=["Platform"],
    summary="List all available MCP tools",
)
async def list_tools():
    """List all 20 MCP tools available on the server."""
    if not _state.mcp_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready.",
        )
    tools = await _state.mcp_client.get_tools_for_anthropic()
    return {
        "total": len(tools),
        "tools": [
            {"name": t["name"], "description": t["description"][:120]}
            for t in tools
        ],
    }


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["Intelligence"],
    summary="Process a natural language intelligence query",
    responses={
        200: {"description": "Query processed successfully"},
        503: {"description": "Service not ready", "model": ErrorResponse},
        500: {"description": "Query processing failed", "model": ErrorResponse},
    },
)
async def process_query(request: QueryRequest):
    """
    Process a natural language query through the multi-agent orchestrator.

    The orchestrator:
    1. **Routes** the query to the relevant specialist agents (Claude-powered routing)
    2. **Executes** agents in parallel or sequentially based on data dependencies
    3. **Synthesizes** all agent responses into a single actionable report

    Example queries:
    - "Who are our at-risk customers and what campaigns should we run?"
    - "Show product recommendations for customer C001"
    - "Analyze campaign performance and suggest budget allocation for next quarter"
    - "Give me a loyalty summary and CLV for our gold-tier customers"
    """
    if not _state.orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized. Service may be starting up.",
        )

    start_time = time.monotonic()

    try:
        result = await _state.orchestrator.process(request.query)
    except Exception as exc:
        logger.error(f"Query processing failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(exc)}",
        )

    elapsed = round(time.monotonic() - start_time, 2)
    plan = result.get("plan", {})

    return QueryResponse(
        query=request.query,
        final_response=result["final_response"],
        agents_used=list(result["agent_results"].keys()),
        execution_mode="parallel" if plan.get("parallel", True) else "sequential",
        routing_reasoning=plan.get("reasoning", ""),
        elapsed_seconds=elapsed,
        agent_results=result["agent_results"] if request.show_agent_details else None,
    )
