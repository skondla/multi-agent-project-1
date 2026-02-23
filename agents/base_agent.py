"""
Base Agent for the Customer Intelligence Platform.

Implements the core agentic loop:
  1. Receive user message (+ optional context from other agents)
  2. Call Claude claude-sonnet-4-6 with domain-specific tools
  3. If Claude requests tool_use → call MCP server → append result → repeat
  4. When Claude returns end_turn → return final text response

All specialized agents inherit from BaseAgent and provide:
  - domain_name: identifies which MCP tools to load
  - system_prompt: domain expertise and behavioral guidelines
"""
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Callable

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 12  # Safety guard against infinite tool-use loops


class BaseAgent(ABC):
    """
    Abstract base class for all domain-specific agents.

    Each agent owns:
    - A domain name (used to filter MCP tools)
    - A system prompt (defines expertise and behavior)
    - A lazy-loaded tool cache
    - Access to the shared MCPClientWrapper

    The run() method executes the full agentic loop for a given task.
    """

    def __init__(self, mcp_client):
        """
        Args:
            mcp_client: MCPClientWrapper instance (shared across agents)
        """
        self.mcp_client = mcp_client
        self._tools: Optional[list[dict]] = None

        # Import Anthropic here to avoid issues if not installed
        try:
            import anthropic
            self._claude = anthropic.Anthropic()
        except ImportError:
            raise ImportError("anthropic package not found. Install with: pip install anthropic")

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """
        Domain identifier used to filter MCP tools.
        Must be one of: 'segmentation', 'campaign', 'recommendation', 'crm'
        """
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        System prompt defining this agent's expertise, personality, and behavior.
        Should be specific, actionable, and data-focused.
        """
        ...

    async def get_tools(self) -> list[dict]:
        """Lazy-load and cache domain-specific tools from MCP server."""
        if self._tools is None:
            self._tools = await self.mcp_client.get_domain_tools(self.domain_name)
            logger.debug(f"[{self.domain_name}] Loaded {len(self._tools)} tools")
        return self._tools

    async def run(
        self,
        user_message: str,
        context: Optional[str] = None,
        on_tool_call: Optional[Callable] = None,
    ) -> str:
        """
        Execute the agentic loop for a user request.

        The loop:
          1. Send user_message (+ optional context) to Claude
          2. Claude responds with either:
             a. end_turn → return the text response
             b. tool_use → call MCP tools, append results, go to 1

        Args:
            user_message: The user's query or task for this agent.
            context: Optional context from prior agents (for sequential execution).
            on_tool_call: Optional callback(tool_name, tool_args, result_text)
                          for UI progress updates.

        Returns:
            Final text response from Claude.
        """
        tools = await self.get_tools()

        # Build the initial user message (inject context if provided)
        if context:
            full_message = (
                f"<context_from_prior_analysis>\n{context}\n</context_from_prior_analysis>\n\n"
                f"User request: {user_message}"
            )
        else:
            full_message = user_message

        messages = [{"role": "user", "content": full_message}]

        for iteration in range(MAX_ITERATIONS):
            logger.debug(
                f"[{self.domain_name}] Iteration {iteration + 1}/{MAX_ITERATIONS}, "
                f"messages: {len(messages)}"
            )

            # Call Claude API (synchronous; wrap in thread for true parallelism)
            response = await asyncio.to_thread(
                self._claude.messages.create,
                model=MODEL,
                max_tokens=4096,
                system=self.system_prompt,
                tools=tools,
                messages=messages,
            )

            # Append assistant response to history
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract text blocks from response
                text_parts = []
                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                return "\n".join(text_parts) if text_parts else "[No response generated]"

            elif response.stop_reason == "tool_use":
                # Process all tool_use blocks in this response
                tool_results = []

                for block in response.content:
                    if not hasattr(block, "type") or block.type != "tool_use":
                        continue

                    tool_name = block.name
                    tool_input = block.input if isinstance(block.input, dict) else {}
                    tool_use_id = block.id

                    logger.info(
                        f"[{self.domain_name}] Calling tool: {tool_name}"
                        f"({json.dumps(tool_input)[:100]}...)"
                    )

                    # Call the tool via MCP
                    result_text = await self.mcp_client.call_tool(tool_name, tool_input)

                    # Fire UI callback if provided (non-blocking, best effort)
                    if on_tool_call:
                        try:
                            on_tool_call(tool_name, tool_input, result_text)
                        except Exception as cb_err:
                            logger.debug(f"on_tool_call callback error: {cb_err}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result_text,
                    })

                # Append tool results to message history
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason (max_tokens, etc.)
                logger.warning(
                    f"[{self.domain_name}] Unexpected stop_reason: {response.stop_reason}"
                )
                # Try to extract any text from the response
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return f"[Agent stopped: {response.stop_reason}]"

        return (
            f"[{self.domain_name.capitalize()} agent reached maximum iterations "
            f"({MAX_ITERATIONS}) without completing. Please try a more focused query.]"
        )
