"""
Shared Rich console instance and display utilities for the Customer Intelligence Platform.
"""
from rich.console import Console
from rich.theme import Theme
from rich.text import Text

# Custom color theme
THEME = Theme({
    "header": "bold cyan",
    "agent.segmentation": "bold blue",
    "agent.campaign": "bold green",
    "agent.recommendation": "bold magenta",
    "agent.crm": "bold yellow",
    "agent.orchestrator": "bold white",
    "tool.call": "dim cyan",
    "tool.result": "dim green",
    "error": "bold red",
    "success": "bold green",
    "info": "dim white",
    "highlight": "bold cyan",
})

# Shared console instance (use this everywhere)
console = Console(theme=THEME, highlight=False)

# Agent color/icon mapping
AGENT_COLORS = {
    "segmentation": "blue",
    "campaign": "green",
    "recommendation": "magenta",
    "crm": "yellow",
}

AGENT_ICONS = {
    "segmentation": "◈ SEG",
    "campaign": "◉ CAM",
    "recommendation": "◎ REC",
    "crm": "◐ CRM",
}

AGENT_LABELS = {
    "segmentation": "Segmentation Agent",
    "campaign": "Campaign Agent",
    "recommendation": "Recommendation Agent",
    "crm": "CRM/Loyalty Agent",
}


def agent_tag(agent_name: str) -> Text:
    """Return a styled Rich Text tag for an agent name."""
    color = AGENT_COLORS.get(agent_name, "white")
    icon = AGENT_ICONS.get(agent_name, agent_name.upper())
    label = AGENT_LABELS.get(agent_name, agent_name.capitalize())
    t = Text()
    t.append(f"[{icon}] ", style=f"bold {color}")
    t.append(label, style=f"bold {color}")
    return t


def format_currency(value: float) -> str:
    """Format a number as currency string."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value:,.0f}"
    else:
        return f"${value:.2f}"
