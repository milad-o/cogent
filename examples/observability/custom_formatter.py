"""
Custom Formatter Example

Shows how to create a custom formatter that handles specific event types
with your own formatting logic.

Formatters transform Event objects into formatted strings for output.

Usage:
    uv run python examples/observability/custom_formatter.py
"""

from __future__ import annotations

from cogent import Agent
from cogent.observability import Observer
from cogent.observability.core.config import FormatConfig
from cogent.observability.core.event import Event
from cogent.observability.formatters.base import BaseFormatter
from cogent.tools import tool


# === Tools for the demo ===


@tool
def get_user(user_id: str) -> dict:
    """Look up user information."""
    users = {
        "u123": {"name": "Alice", "email": "alice@example.com", "plan": "pro"},
        "u456": {"name": "Bob", "email": "bob@example.com", "plan": "free"},
    }
    return users.get(user_id, {"error": "User not found"})


@tool
def place_order(user_id: str, product: str, quantity: int) -> dict:
    """Place an order for a user."""
    return {
        "order_id": "ORD-789",
        "user_id": user_id,
        "product": product,
        "quantity": quantity,
        "status": "confirmed",
    }


# === Example 1: Emoji Tool Formatter ===


class EmojiToolFormatter(BaseFormatter):
    """
    A formatter that uses emojis for tool events.

    Makes tool calls and results more visually distinctive.
    """

    patterns = ["tool.*"]

    TOOL_EMOJIS = {
        "get_user": "👤",
        "place_order": "🛒",
        "calculate": "🔢",
        "search": "🔍",
    }

    def format(self, event: Event, config: FormatConfig) -> str | None:
        tool_name = event.get("tool_name", "unknown")
        emoji = self.TOOL_EMOJIS.get(tool_name, "🔧")
        agent = event.get("agent_name", "Agent")

        if event.action == "called":
            args = event.get("args", {})
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
            return f"[{agent}] {emoji} CALLING {tool_name}({args_str})"

        elif event.action == "result":
            result = event.get("result", "")
            # Truncate long results
            if len(str(result)) > 100:
                result = str(result)[:97] + "..."
            return f"[{agent}] {emoji} RESULT: {result}"

        return None  # Let other formatters handle


# === Example 2: Compact Agent Formatter ===


class CompactAgentFormatter(BaseFormatter):
    """
    Minimal single-line agent status formatter.

    Great for dense logging or when screen space is limited.
    """

    patterns = ["agent.*"]

    STATUS_CHARS = {
        "invoked": "▶",
        "thinking": "💭",
        "responded": "✓",
        "error": "✗",
    }

    def format(self, event: Event, config: FormatConfig) -> str | None:
        agent = event.get("agent_name", "Agent")
        status = self.STATUS_CHARS.get(event.action, "?")

        if event.action == "responded":
            duration = event.get("duration_ms", 0)
            tokens = event.get("total_tokens", 0)
            return f"{status} {agent} done ({duration:.0f}ms, {tokens} tokens)"

        return f"{status} {agent}"


# === Example 3: Structured Log Formatter ===


class LogfmtFormatter(BaseFormatter):
    """
    Formats events as logfmt (key=value pairs).

    Great for log aggregation systems like Loki, Splunk, etc.
    """

    patterns = ["*"]  # Handles all events

    def format(self, event: Event, config: FormatConfig) -> str | None:
        parts = [
            f'time="{event.timestamp.isoformat()}"',
            f'type="{event.type}"',
            f'category="{event.category}"',
        ]

        # Add all data fields
        for key, value in event.data.items():
            if isinstance(value, str):
                value = value.replace('"', '\\"')
                parts.append(f'{key}="{value}"')
            elif isinstance(value, dict):
                parts.append(f'{key}="{value}"')
            else:
                parts.append(f"{key}={value}")

        return " ".join(parts)


# === Usage Demo ===


async def main() -> None:
    print("=" * 60)
    print("1. EMOJI TOOL FORMATTER - Visual tool tracking")
    print("=" * 60 + "\n")

    observer = Observer(level="verbose")
    observer.add_formatter(EmojiToolFormatter())

    agent = Agent(
        name="OrderBot",
        model="gpt-4o-mini",
        tools=[get_user, place_order],
        observer=observer,
    )

    await agent.run("Look up user u123 and place an order for 2 widgets")

    print("\n" + observer.summary())

    print("\n" + "=" * 60)
    print("2. COMPACT AGENT FORMATTER - Minimal output")
    print("=" * 60 + "\n")

    observer2 = Observer(level="progress")
    observer2.add_formatter(CompactAgentFormatter())

    agent2 = Agent(
        name="Helper",
        model="gpt-4o-mini",
        tools=[get_user],
        observer=observer2,
    )

    await agent2.run("Who is user u456?")

    print("\n" + observer2.summary())

    print("\n" + "=" * 60)
    print("3. LOGFMT FORMATTER - Structured logging")
    print("=" * 60 + "\n")

    observer3 = Observer(level="progress")
    observer3.add_formatter(LogfmtFormatter())

    agent3 = Agent(
        name="Logger",
        model="gpt-4o-mini",
        tools=[get_user],
        observer=observer3,
    )

    await agent3.run("Get info for user u123")

    print("\n" + observer3.summary())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
