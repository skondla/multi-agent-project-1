"""
Customer Intelligence Platform - Interactive CLI Entry Point.

Demonstrates multi-agent AI orchestration with MCP (Model Context Protocol):
  - FastMCP Server: 20 tools across 4 business domains
  - MCP Client: stdio transport to server subprocess
  - 4 Specialized Claude Agents: Segmentation, Campaign, Recommendation, CRM/Loyalty
  - Orchestrator: 3-phase routing → execution → synthesis

Usage:
    python main.py
    python main.py --query "Show me at-risk customers"
    python main.py --demo  (run all demo queries non-interactively)

Requirements:
    - ANTHROPIC_API_KEY environment variable must be set
    - pip install -r requirements.txt
"""
import asyncio
import logging
import os
import sys
import time
import argparse
from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box
from rich.columns import Columns
from rich.text import Text

from ui.console import console, AGENT_COLORS, AGENT_ICONS, AGENT_LABELS

# Suppress noisy library logs
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


BANNER = """
 ██████╗██╗   ██╗███████╗████████╗ ██████╗ ███╗   ███╗███████╗██████╗
██╔════╝██║   ██║██╔════╝╚══██╔══╝██╔═══██╗████╗ ████║██╔════╝██╔══██╗
██║     ██║   ██║███████╗   ██║   ██║   ██║██╔████╔██║█████╗  ██████╔╝
██║     ██║   ██║╚════██║   ██║   ██║   ██║██║╚██╔╝██║██╔══╝  ██╔══██╗
╚██████╗╚██████╔╝███████║   ██║   ╚██████╔╝██║ ╚═╝ ██║███████╗██║  ██║
 ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
██╗███╗   ██╗████████╗███████╗██╗     ██╗     ██╗ ██████╗ ███████╗███╗   ██╗ ██████╗███████╗
██║████╗  ██║╚══██╔══╝██╔════╝██║     ██║     ██║██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔════╝
██║██╔██╗ ██║   ██║   █████╗  ██║     ██║     ██║██║  ███╗█████╗  ██╔██╗ ██║██║     █████╗
██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██║     ██║██║   ██║██╔══╝  ██║╚██╗██║██║     ██╔══╝
██║██║ ╚████║   ██║   ███████╗███████╗███████╗██║╚██████╔╝███████╗██║ ╚████║╚██████╗███████╗
╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
"""

PLATFORM_SUBTITLE = "  [dim]Powered by Claude claude-sonnet-4-6 · MCP Protocol · Multi-Agent Orchestration[/dim]"

DEMO_QUERIES = [
    ("Full Analysis", "Identify our at-risk customers and recommend the best campaign and products to win them back"),
    ("Champions Profile", "Who are our Champion customers? Show their profile and top product recommendations"),
    ("Budget Allocation", "Analyze campaign performance and recommend how to allocate a $50,000 quarterly budget"),
    ("Customer Briefing", "Give me a complete intelligence briefing for customer C001: profile, CLV, loyalty status, and personalized recommendations"),
    ("Loyalty Strategy", "Find customers close to a Gold tier upgrade and design a targeted loyalty campaign for them"),
    ("Trending + Targeting", "What products are trending this month and which customer segments should we target?"),
    ("CRM Deep Dive", "Show me customer C018's lifetime value projection, loyalty program engagement, and next best actions"),
    ("Win-Back Campaign", "Design a personalized win-back campaign for dormant customers with the highest lifetime value"),
]


def print_banner() -> None:
    """Print the application banner."""
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")
    console.print(PLATFORM_SUBTITLE)
    console.print()

    # Capability grid
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="center")
    grid.add_column(justify="center")
    grid.add_column(justify="center")
    grid.add_column(justify="center")

    grid.add_row(
        Panel("[blue]◈ Segmentation[/blue]\nRFM · Clustering\nChurn Risk · Profiles", border_style="blue", padding=(0, 1)),
        Panel("[green]◉ Campaigns[/green]\nPerformance · ROI\nBudget · Targeting", border_style="green", padding=(0, 1)),
        Panel("[magenta]◎ Recommendations[/magenta]\nProducts · NBA\nTrending · Content", border_style="magenta", padding=(0, 1)),
        Panel("[yellow]◐ CRM/Loyalty[/yellow]\nCLV · Tiers · Points\nRewards · Upgrades", border_style="yellow", padding=(0, 1)),
    )
    console.print(grid)
    console.print()


def print_demo_menu() -> None:
    """Print the interactive demo query menu."""
    table = Table(
        title="[bold cyan]Demo Queries[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
        expand=False,
    )
    table.add_column("#", style="bold cyan", width=4, justify="center")
    table.add_column("Category", style="bold white", width=18)
    table.add_column("Query", style="white", max_width=70)

    for i, (category, query) in enumerate(DEMO_QUERIES, 1):
        table.add_row(str(i), category, query)

    table.add_row("[dim]C[/dim]", "[dim]Custom[/dim]", "[dim]Type your own query[/dim]")
    table.add_row("[dim]Q[/dim]", "[dim]Quit[/dim]", "[dim]Exit the platform[/dim]")
    console.print(table)


def on_routing_complete(plan: dict) -> None:
    """Display routing decision to user."""
    agents = plan.get("agents", [])
    parallel = plan.get("parallel", True)
    reasoning = plan.get("reasoning", "")
    mode = "[bold green]parallel[/bold green]" if parallel else "[bold yellow]sequential[/bold yellow]"

    agent_parts = []
    for a in agents:
        color = AGENT_COLORS.get(a, "white")
        icon = AGENT_ICONS.get(a, a.upper())
        agent_parts.append(f"[{color}]{icon}[/{color}]")

    agents_display = "  ·  ".join(agent_parts)

    content = Text()
    content.append("Agents: ", style="bold white")
    console.print()
    console.print(
        Panel(
            f"{agents_display}   │   Mode: {mode}\n[dim]{reasoning}[/dim]",
            title="[bold white]⚙  Orchestrator Routing[/bold white]",
            border_style="white",
            padding=(0, 1),
        )
    )


def on_agent_start(agent_name: str) -> None:
    """Display agent starting message."""
    color = AGENT_COLORS.get(agent_name, "white")
    icon = AGENT_ICONS.get(agent_name, agent_name.upper())
    label = AGENT_LABELS.get(agent_name, agent_name.capitalize())
    console.print(f"  [{color}]▶ {icon}[/{color}] [bold {color}]{label}[/bold {color}] [dim]starting...[/dim]")


def on_agent_complete(agent_name: str, result: str) -> None:
    """Display agent completion message."""
    color = AGENT_COLORS.get(agent_name, "white")
    icon = AGENT_ICONS.get(agent_name, agent_name.upper())
    label = AGENT_LABELS.get(agent_name, agent_name.capitalize())
    console.print(f"  [{color}]✓ {icon}[/{color}] [bold {color}]{label}[/bold {color}] [dim]complete.[/dim]")


def on_tool_call(tool_name: str, tool_args: dict, result: str) -> None:
    """Display tool call progress."""
    args_preview = ", ".join(f"{k}={repr(v)}" for k, v in list(tool_args.items())[:3])
    console.print(f"    [dim cyan]  ⚙ {tool_name}({args_preview[:80]})[/dim cyan]")


async def run_query(
    orchestrator,
    query: str,
    show_agent_details: bool = True,
) -> dict:
    """Execute a query through the multi-agent orchestrator and display results."""

    console.print()
    console.print(
        Panel(
            f"[bold white]{query}[/bold white]",
            title="[bold cyan]Query[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    console.print()

    start_time = time.time()

    result = await orchestrator.process(
        query,
        on_routing_complete=on_routing_complete if show_agent_details else None,
        on_agent_start=on_agent_start if show_agent_details else None,
        on_agent_complete=on_agent_complete if show_agent_details else None,
        on_tool_call=on_tool_call if show_agent_details else None,
    )

    elapsed = time.time() - start_time

    console.print()
    console.print(
        Panel(
            Markdown(result["final_response"]),
            title="[bold cyan]Intelligence Report[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Footer with metadata
    agents_used = list(result["agent_results"].keys())
    agent_tags = " · ".join(
        f"[{AGENT_COLORS.get(a, 'white')}]{AGENT_ICONS.get(a, a)}[/{AGENT_COLORS.get(a, 'white')}]"
        for a in agents_used
    )
    console.print(
        f"[dim]  Agents: {agent_tags}  │  Time: {elapsed:.1f}s  │  "
        f"Tools called via MCP server[/dim]"
    )

    return result


async def interactive_loop(orchestrator) -> None:
    """Run the interactive query loop."""
    while True:
        console.print()
        print_demo_menu()
        console.print()

        choice = Prompt.ask("[bold cyan]Select[/bold cyan]", default="1")
        choice = choice.strip().upper()

        if choice == "Q":
            console.print()
            console.print(Panel(
                "[bold cyan]Thank you for using Customer Intelligence Platform![/bold cyan]\n"
                "[dim]Multi-Agent AI with MCP · Powered by Claude claude-sonnet-4-6[/dim]",
                border_style="cyan",
            ))
            break

        elif choice == "C":
            console.print()
            query = Prompt.ask("[bold]Enter your query[/bold]")
            if query.strip():
                await run_query(orchestrator, query.strip())
            else:
                console.print("[error]Empty query. Please try again.[/error]")

        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(DEMO_QUERIES):
                    _, query = DEMO_QUERIES[idx]
                    await run_query(orchestrator, query)
                else:
                    console.print(f"[error]Invalid selection: {choice}. Enter 1-{len(DEMO_QUERIES)}, C, or Q.[/error]")
            except ValueError:
                console.print(f"[error]Invalid input: '{choice}'. Enter a number, C, or Q.[/error]")

        console.print("\n" + "─" * 80)


async def main(query: Optional[str] = None, demo_mode: bool = False) -> None:
    """Main entry point for the Customer Intelligence Platform."""

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            Panel(
                "[error]ANTHROPIC_API_KEY environment variable is not set.[/error]\n\n"
                "Set it with:\n"
                "  [bold]export ANTHROPIC_API_KEY=sk-ant-...[/bold]",
                title="[error]Configuration Error[/error]",
                border_style="red",
            )
        )
        sys.exit(1)

    print_banner()

    # Import here to avoid circular import issues
    from mcp_client.client import MCPClientWrapper
    from orchestrator.orchestrator import Orchestrator

    console.print("[dim]Connecting to MCP server and loading tools...[/dim]")

    async with MCPClientWrapper() as mcp_client:
        # Warmup: pre-load tool cache
        with console.status("[dim]Loading 20 MCP tools...[/dim]", spinner="dots"):
            tool_count = await mcp_client.warmup()

        console.print(
            f"[success]✓ MCP server connected · {tool_count} tools loaded across 4 domains[/success]"
        )
        console.print()

        orchestrator = Orchestrator(mcp_client)

        if query:
            # Single query mode (--query argument)
            await run_query(orchestrator, query)

        elif demo_mode:
            # Run all demo queries non-interactively
            console.print(Panel(
                "[bold]Running all demo queries in sequence...[/bold]",
                border_style="cyan",
            ))
            for i, (category, demo_query) in enumerate(DEMO_QUERIES, 1):
                console.print(f"\n[bold cyan]Demo {i}/{len(DEMO_QUERIES)}: {category}[/bold cyan]")
                await run_query(orchestrator, demo_query, show_agent_details=False)
                if i < len(DEMO_QUERIES):
                    await asyncio.sleep(1)  # Brief pause between queries

        else:
            # Full interactive mode
            await interactive_loop(orchestrator)


def main_sync():
    """Synchronous entry point for setuptools scripts."""
    parser = argparse.ArgumentParser(
        description="Customer Intelligence Platform - Multi-Agent AI with MCP"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and exit",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run all demo queries non-interactively",
    )
    args = parser.parse_args()

    asyncio.run(main(query=args.query, demo_mode=args.demo))


if __name__ == "__main__":
    main_sync()
