# Customer Intelligence Platform

A production-quality **multi-agent AI application** demonstrating orchestrated AI agents powered by Claude claude-sonnet-4-6 and the **Model Context Protocol (MCP)**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    main.py (Rich CLI)                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│              Orchestrator (Claude Routing + Synthesis)           │
│  Phase 1: Route → Phase 2: Execute → Phase 3: Synthesize        │
└────┬─────────┬──────────┬────────────────────────────────────────┘
     │         │          │           │
     ▼         ▼          ▼           ▼
┌─────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────┐
│Segment- │ │Campaign  │ │Recommend-  │ │CRM/Loyalty   │
│ation    │ │Agent     │ │ation Agent │ │Agent         │
│Agent    │ │          │ │            │ │              │
└────┬────┘ └─────┬────┘ └─────┬──────┘ └──────┬───────┘
     └─────────────┴────────────┴───────────────┘
                           │
                    MCPClientWrapper
                    (stdio transport)
                           │
                    FastMCP Server
               ┌───────────┴───────────┐
               │  20 Tools / 4 Domains  │
               │ Segmentation(5)        │
               │ Campaign(5)            │
               │ Recommendations(5)     │
               │ CRM/Loyalty(5)         │
               └───────────────────────┘
                    DataStore (JSON)
```

## Features

| Domain | Capabilities |
|--------|-------------|
| **Customer Segmentation** | RFM analysis, K-Means clustering, churn risk scoring, segment profiling, segment comparison |
| **Campaign Optimization** | Performance analytics (ROI/CTR/ROAS), budget allocation, audience targeting, ROI forecasting, A/B testing |
| **Recommendations** | Collaborative filtering, next-best-action, trending products, similar customers, content personalization |
| **CRM / Loyalty** | Full customer profiles, DCF-based CLV, loyalty tier management, points transactions, tier upgrade detection |

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Anthropic API key ([get one here](https://console.anthropic.com))

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 4. Run the platform

```bash
# Interactive mode (recommended)
python main.py

# Single query
python main.py --query "Who are our at-risk customers?"

# Run all demo queries
python main.py --demo
```

## Project Structure

```
multi-agent-project-1/
├── main.py                       # Interactive CLI entry point
├── config.py                     # Configuration management
├── requirements.txt
├── pyproject.toml
├── .env.example                  # Copy to .env with your API key
│
├── data/                         # Sample data (JSON)
│   ├── customers.json            # 20 customers with profiles
│   ├── products.json             # 30 products across 5 categories
│   ├── campaigns.json            # 10 campaigns with metrics
│   ├── transactions.json         # 95 transactions
│   └── loyalty_events.json       # Loyalty points events
│
├── mcp_server/                   # FastMCP server
│   ├── server.py                 # Server entry point (stdio transport)
│   ├── data_store.py             # In-memory data layer
│   └── tools/
│       ├── segmentation_tools.py # 5 segmentation tools
│       ├── campaign_tools.py     # 5 campaign tools
│       ├── recommendation_tools.py # 5 recommendation tools
│       └── crm_loyalty_tools.py  # 5 CRM/loyalty tools
│
├── mcp_client/
│   └── client.py                 # MCP client wrapper (subprocess management)
│
├── agents/
│   ├── base_agent.py             # Agentic loop implementation
│   ├── segmentation_agent.py     # RFM / segmentation specialist
│   ├── campaign_agent.py         # Campaign optimization specialist
│   ├── recommendation_agent.py   # Personalization specialist
│   └── crm_loyalty_agent.py      # CRM / loyalty specialist
│
├── orchestrator/
│   └── orchestrator.py           # 3-phase orchestration engine
│
└── ui/
    └── console.py                # Rich terminal UI utilities
```

## MCP Tools Reference

### Segmentation Domain
| Tool | Description |
|------|-------------|
| `compute_rfm_scores` | RFM quintile scoring with segment classification |
| `run_customer_clustering` | K-Means clustering on purchase behavior |
| `get_segment_profile` | Detailed stats for Champions, At Risk, Lost, etc. |
| `identify_churn_risk` | Decay-model churn risk scoring |
| `compare_segments` | Side-by-side segment comparison |

### Campaign Domain
| Tool | Description |
|------|-------------|
| `get_campaign_performance` | ROI, CTR, conversion, ROAS by campaign |
| `recommend_budget_allocation` | Historical ROI-weighted budget split |
| `identify_target_audience` | Best customers for campaign type |
| `forecast_campaign_roi` | Predicted ROI for proposed campaign |
| `ab_test_analysis` | Statistical significance testing |

### Recommendation Domain
| Tool | Description |
|------|-------------|
| `get_product_recommendations` | Collaborative filtering recommendations |
| `get_similar_customers` | Jaccard similarity on purchase behavior |
| `get_next_best_action` | Prioritized action list with expected value |
| `get_trending_products` | Purchase velocity trending |
| `get_content_recommendations` | Email/SMS/push content topics |

### CRM/Loyalty Domain
| Tool | Description |
|------|-------------|
| `get_customer_profile` | Full profile + transaction summary |
| `calculate_clv` | DCF-based CLV 1/3/5-year projection |
| `get_loyalty_summary` | Tier, points, rewards, tier progress |
| `process_points_transaction` | Earn/redeem/adjust points |
| `get_tier_upgrade_candidates` | Near-tier-upgrade customer list |

## How It Works

### 1. MCP Server (stdio transport)
The FastMCP server runs as a **subprocess** communicating over stdin/stdout using JSON-RPC. It loads all 5 JSON data files at startup and keeps them in memory. Tools are registered via the `@mcp.tool` decorator pattern.

> ⚠️ Critical: The server **must never write to stdout** — only to stderr. Any `print()` statement corrupts the JSON-RPC stream.

### 2. MCP Client (subprocess management)
`MCPClientWrapper` manages the server subprocess lifecycle using `asyncio` and `AsyncExitStack`. It converts MCP's camelCase `inputSchema` to Anthropic's snake_case `input_schema`.

### 3. Agentic Loop (BaseAgent)
Each agent runs a standard tool-use loop:
```
user_message → Claude API → tool_use blocks → MCP call → results → Claude API → ... → end_turn → response
```

### 4. Multi-Agent Orchestration
The orchestrator runs a 3-phase pipeline:
- **Phase 1 (Route)**: Claude analyzes the query and returns a JSON routing plan
- **Phase 2 (Execute)**: Agents run in parallel (`asyncio.gather`) or sequentially with context passing
- **Phase 3 (Synthesize)**: Claude combines all agent outputs into a final report

## Demo Queries

The interactive CLI includes 8 pre-built demo queries:

1. **Full Analysis** — Identify at-risk customers + win-back campaign + product recommendations
2. **Champions Profile** — RFM Champions segment profile + top recommendations
3. **Budget Allocation** — Campaign performance analysis + $50K budget optimization
4. **Customer Briefing** — Complete profile, CLV, loyalty, and NBA for a specific customer
5. **Loyalty Strategy** — Find tier-upgrade candidates + design targeted campaign
6. **Trending + Targeting** — Trending products + best customer segments to target
7. **CRM Deep Dive** — CLV projection, loyalty engagement, and next-best-actions
8. **Win-Back Campaign** — Personalized win-back for high-CLV dormant customers

## Tech Stack

| Component | Technology |
|-----------|------------|
| AI Model | Claude claude-sonnet-4-6 (Anthropic) |
| Tool Protocol | Model Context Protocol (MCP) |
| MCP Server | FastMCP (stdio transport) |
| Agent Framework | Custom agentic loop with `anthropic` SDK |
| ML (clustering) | scikit-learn (K-Means, StandardScaler) |
| Terminal UI | Rich |
| Python | 3.11+ with asyncio |
