"""
LLMOps — Prompt Registry
Versions, stores, and compares system prompts for all agents and the orchestrator.

Features:
  - Register prompt versions with metadata (author, model, description)
  - Diff two versions side-by-side
  - Promote a version to "active" for an agent
  - A/B test prompt variants by tagging evaluations with version ID
  - Load active prompt for an agent at runtime

Usage:
    from llmops.prompt_registry import PromptRegistry

    registry = PromptRegistry()
    vid = registry.register(
        agent="segmentation",
        content=SYSTEM_PROMPT_V2,
        description="Added churn context to segmentation instructions",
        author="ml-team",
    )
    registry.promote(agent="segmentation", version_id=vid)
    active = registry.get_active("segmentation")

CLI:
    python -m llmops.prompt_registry list --agent segmentation
    python -m llmops.prompt_registry diff --v1 seg_v1 --v2 seg_v2
    python -m llmops.prompt_registry promote --agent segmentation --version seg_v2
    python -m llmops.prompt_registry show --version seg_v1
"""
from __future__ import annotations

import difflib
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "llmops" / "prompts"
REGISTRY_FILE = PROMPTS_DIR / "registry.json"

KNOWN_AGENTS = [
    "segmentation", "campaign", "recommendation", "crm_loyalty", "orchestrator"
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _version_id(agent: str, n: int) -> str:
    return f"{agent}_v{n}"


# ─── Registry ────────────────────────────────────────────────────────────────

class PromptRegistry:
    """
    Registry structure (registry.json):
    {
      "agents": {
        "segmentation": {
          "active_version": "segmentation_v2",
          "versions": [
            {
              "version_id": "segmentation_v1",
              "registered_at": "...",
              "description": "...",
              "author": "...",
              "model": "claude-sonnet-4-6",
              "content_hash": "abc123",
              "stage": "active" | "archived" | "staging",
              "prompt_file": "segmentation_v1.txt"
            }
          ]
        }
      }
    }
    """

    def __init__(self, prompts_dir: Path = PROMPTS_DIR):
        self._dir = prompts_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._dir / "registry.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self._registry_path.exists():
            return json.loads(self._registry_path.read_text())
        return {"agents": {}}

    def _save(self) -> None:
        self._registry_path.write_text(json.dumps(self._data, indent=2))

    def _agent_data(self, agent: str) -> dict:
        if agent not in self._data["agents"]:
            self._data["agents"][agent] = {"active_version": None, "versions": []}
        return self._data["agents"][agent]

    # ── Core operations ────────────────────────────────────────────────────────

    def register(
        self,
        agent: str,
        content: str,
        description: str = "",
        author: str = "system",
        model: str = "claude-sonnet-4-6",
    ) -> str:
        """Register a new prompt version. Returns the version_id."""
        agent_data = self._agent_data(agent)
        n = len(agent_data["versions"]) + 1
        version_id = _version_id(agent, n)

        # Save prompt text to a file
        prompt_file = f"{version_id}.txt"
        (self._dir / prompt_file).write_text(content)

        entry = {
            "version_id": version_id,
            "registered_at": datetime.now().isoformat(),
            "description": description,
            "author": author,
            "model": model,
            "content_hash": _content_hash(content),
            "word_count": len(content.split()),
            "stage": "staging",
            "prompt_file": prompt_file,
        }
        agent_data["versions"].append(entry)
        self._save()
        logger.info(f"Registered {version_id} (hash={entry['content_hash']})")
        return version_id

    def promote(self, agent: str, version_id: str) -> bool:
        """Set a version as the active prompt for the agent."""
        agent_data = self._agent_data(agent)
        target = None
        for v in agent_data["versions"]:
            if v["version_id"] == version_id:
                target = v
            elif v["stage"] == "active":
                v["stage"] = "archived"

        if not target:
            logger.error(f"Version {version_id} not found for agent {agent}")
            return False

        target["stage"] = "active"
        target["promoted_at"] = datetime.now().isoformat()
        agent_data["active_version"] = version_id
        self._save()
        logger.info(f"Promoted {version_id} to active for agent '{agent}'")
        return True

    def get_active(self, agent: str) -> str | None:
        """Return the content of the active prompt for an agent, or None."""
        agent_data = self._agent_data(agent)
        vid = agent_data.get("active_version")
        if not vid:
            return None
        return self.get_content(agent, vid)

    def get_content(self, agent: str, version_id: str) -> str | None:
        agent_data = self._agent_data(agent)
        for v in agent_data["versions"]:
            if v["version_id"] == version_id:
                prompt_file = self._dir / v["prompt_file"]
                if prompt_file.exists():
                    return prompt_file.read_text()
        return None

    def list_versions(self, agent: str | None = None) -> list[dict]:
        if agent:
            return self._agent_data(agent).get("versions", [])
        all_versions = []
        for a, data in self._data["agents"].items():
            for v in data.get("versions", []):
                all_versions.append({**v, "agent": a})
        return all_versions

    # ── Diff ──────────────────────────────────────────────────────────────────

    def diff(self, agent: str, v1_id: str, v2_id: str) -> str:
        """Return a unified diff between two prompt versions."""
        c1 = self.get_content(agent, v1_id) or ""
        c2 = self.get_content(agent, v2_id) or ""

        if not c1 and not c2:
            return f"Neither {v1_id} nor {v2_id} have content."

        diff = difflib.unified_diff(
            c1.splitlines(keepends=True),
            c2.splitlines(keepends=True),
            fromfile=v1_id,
            tofile=v2_id,
            lineterm="",
        )
        return "".join(diff) or "(no differences)"

    # ── Snapshot current agent prompts ────────────────────────────────────────

    def snapshot_from_agents(self) -> dict[str, str]:
        """
        Read current system prompts from agent source files and register them.
        Returns {agent_name: version_id} for newly registered versions.
        """
        import importlib.util

        agent_modules = {
            "segmentation": "agents/segmentation_agent.py",
            "campaign": "agents/campaign_agent.py",
            "recommendation": "agents/recommendation_agent.py",
            "crm_loyalty": "agents/crm_loyalty_agent.py",
            "orchestrator": "orchestrator/orchestrator.py",
        }

        registered = {}
        for agent, module_path in agent_modules.items():
            full_path = PROJECT_ROOT / module_path
            if not full_path.exists():
                continue

            source = full_path.read_text()
            # Extract system_prompt string from source file
            match = None
            for pattern in [
                r'system_prompt\s*=\s*"""(.*?)"""',
                r"system_prompt\s*=\s*'''(.*?)'''",
                r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""',
            ]:
                import re
                m = re.search(pattern, source, re.DOTALL)
                if m:
                    match = m.group(1).strip()
                    break

            if not match:
                logger.debug(f"Could not extract system_prompt for {agent}")
                continue

            # Check if this content is already registered (by hash)
            content_hash = _content_hash(match)
            existing = self.list_versions(agent)
            if any(v.get("content_hash") == content_hash for v in existing):
                logger.debug(f"No change in {agent} prompt — skipping")
                continue

            vid = self.register(
                agent=agent,
                content=match,
                description=f"Snapshotted from {module_path}",
                author="auto-snapshot",
            )
            registered[agent] = vid

        return registered


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prompt registry management")
    sub = parser.add_subparsers(dest="command")

    list_p = sub.add_parser("list", help="List prompt versions")
    list_p.add_argument("--agent", default=None)

    diff_p = sub.add_parser("diff", help="Diff two versions")
    diff_p.add_argument("--agent", required=True)
    diff_p.add_argument("--v1", required=True)
    diff_p.add_argument("--v2", required=True)

    promo_p = sub.add_parser("promote", help="Set active version")
    promo_p.add_argument("--agent", required=True)
    promo_p.add_argument("--version", required=True)

    show_p = sub.add_parser("show", help="Print prompt content")
    show_p.add_argument("--agent", required=True)
    show_p.add_argument("--version", required=True)

    sub.add_parser("snapshot", help="Snapshot current agent system prompts")

    args = parser.parse_args()
    registry = PromptRegistry()

    if args.command == "list" or not args.command:
        agent = getattr(args, "agent", None)
        versions = registry.list_versions(agent=agent)
        if not versions:
            print("No prompt versions found.")
            return
        print(f"\n{'Version ID':>22}  {'Stage':>10}  {'Words':>7}  {'Author':>12}  Registered At")
        print("-" * 80)
        for v in versions:
            a_prefix = f"({v['agent']}) " if "agent" in v else ""
            print(f"  {a_prefix}{v['version_id']:>20}  {v['stage']:>10}  "
                  f"{v['word_count']:>7}  {v['author']:>12}  {v['registered_at'][:19]}")

    elif args.command == "diff":
        print(registry.diff(args.agent, args.v1, args.v2))

    elif args.command == "promote":
        ok = registry.promote(args.agent, args.version)
        print(f"{'✓' if ok else '✗'} Promoted {args.version} for {args.agent}")
        if not ok:
            sys.exit(1)

    elif args.command == "show":
        content = registry.get_content(args.agent, args.version)
        if content:
            print(content)
        else:
            print(f"Version {args.version} not found for agent {args.agent}")
            sys.exit(1)

    elif args.command == "snapshot":
        registered = registry.snapshot_from_agents()
        if registered:
            print(f"Snapshotted {len(registered)} agent prompts:")
            for agent, vid in registered.items():
                print(f"  {agent} → {vid}")
        else:
            print("No new prompts to snapshot (all hashes match existing versions).")


if __name__ == "__main__":
    main()
