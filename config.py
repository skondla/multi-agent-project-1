"""
Central configuration for the Customer Intelligence Platform.
Loads environment variables and provides typed config access.
"""
import os
from pathlib import Path

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

# ─── API Configuration ───────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MCP_SERVER_SCRIPT = PROJECT_ROOT / "mcp_server" / "server.py"

# ─── Agent Configuration ─────────────────────────────────────────────────────
AGENT_MAX_ITERATIONS = 12     # Max tool-use iterations per agent
AGENT_MAX_TOKENS = 4096       # Max tokens per Claude response

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "WARNING")
