"""
Custom Sink Example

Shows how to create custom sinks that capture agent events and send them
to different destinations (buffers, databases, webhooks, etc.).

Sinks receive pre-formatted strings and write them to their destination.

Usage:
    uv run python examples/observability/custom_sink.py
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from cogent import Agent
from cogent.observability import Observer
from cogent.observability.sinks.base import BaseSink
from cogent.tools import tool

if TYPE_CHECKING:
    from collections.abc import Callable


# === Tools for the demo ===


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Found 3 results for '{query}': Result A, Result B, Result C"


@tool
def summarize(text: str) -> str:
    """Summarize text."""
    return f"Summary: {text[:50]}..." if len(text) > 50 else f"Summary: {text}"


# === Example 1: Ring Buffer Sink ===


class RingBufferSink(BaseSink):
    """
    Keeps the last N events in memory.

    Useful for displaying recent events in a UI or post-run analysis.
    """

    def __init__(self, max_size: int = 100) -> None:
        self._buffer: deque[str] = deque(maxlen=max_size)

    def write(self, output: str) -> None:
        self._buffer.append(output)

    def flush(self) -> None:
        pass  # In-memory, nothing to flush

    def close(self) -> None:
        pass

    def get_events(self) -> list[str]:
        """Get all buffered events."""
        return list(self._buffer)

    def get_recent(self, n: int = 10) -> list[str]:
        """Get the N most recent events."""
        return list(self._buffer)[-n:]

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()


# === Example 2: Filter Sink ===


class FilterSink(BaseSink):
    """
    Wraps another sink with a filter function.

    Only passes through events matching the filter.
    """

    def __init__(
        self,
        sink: BaseSink,
        predicate: Callable[[str], bool],
    ) -> None:
        self._sink = sink
        self._predicate = predicate

    def write(self, output: str) -> None:
        if self._predicate(output):
            self._sink.write(output)

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()


# === Example 3: Webhook Sink ===


class WebhookSink(BaseSink):
    """
    Batches events and sends them to a webhook endpoint.

    In production, use httpx or aiohttp for the actual POST.
    """

    def __init__(
        self,
        url: str,
        batch_size: int = 5,
    ) -> None:
        self.url = url
        self.batch_size = batch_size
        self._buffer: list[str] = []

    def write(self, output: str) -> None:
        self._buffer.append(output)

        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        # In real code, use httpx.post(self.url, json=payload)
        print(f"\n  📤 [WebhookSink] POST to {self.url}")
        print(f"     Batch: {len(self._buffer)} events")

        self._buffer.clear()

    def close(self) -> None:
        self.flush()


# === Example 4: Metrics Sink ===


class MetricsSink(BaseSink):
    """
    Extracts metrics from events for analysis.

    Counts events by type and tracks timing.
    """

    def __init__(self) -> None:
        self.event_counts: dict[str, int] = {}
        self.tool_calls: list[str] = []

    def write(self, output: str) -> None:
        # Count event types
        if "[starting]" in output:
            self.event_counts["starts"] = self.event_counts.get("starts", 0) + 1
        elif "[thinking]" in output:
            self.event_counts["thinks"] = self.event_counts.get("thinks", 0) + 1
        elif "[tool-call]" in output:
            self.event_counts["tool_calls"] = self.event_counts.get("tool_calls", 0) + 1
            # Extract tool name
            if "tool-call]" in output:
                parts = output.split()
                for i, p in enumerate(parts):
                    if p.endswith("[tool-call]"):
                        if i + 2 < len(parts):
                            self.tool_calls.append(parts[i + 2])
        elif "[completed]" in output:
            self.event_counts["completions"] = (
                self.event_counts.get("completions", 0) + 1
            )

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def report(self) -> str:
        lines = ["Metrics Report:", "-" * 30]
        for key, count in self.event_counts.items():
            lines.append(f"  {key}: {count}")
        if self.tool_calls:
            lines.append(f"  tools used: {', '.join(self.tool_calls)}")
        return "\n".join(lines)


# === Usage Demo ===


async def main() -> None:
    print("=" * 60)
    print("1. RING BUFFER SINK - Capture events for analysis")
    print("=" * 60 + "\n")

    buffer = RingBufferSink(max_size=10)

    observer = Observer(level="progress")
    observer.add_sink(buffer)

    agent = Agent(
        name="Researcher",
        model="gpt-4o-mini",
        tools=[search, summarize],
        observer=observer,
    )

    await agent.run("Search for 'AI agents' and summarize the results")

    print(f"\n📋 Buffer captured {len(buffer.get_events())} events:")
    for event in buffer.get_events():
        print(f"   {event}")

    print("\n" + "=" * 60)
    print("2. FILTER SINK - Capture only tool events")
    print("=" * 60 + "\n")

    tool_buffer = RingBufferSink()
    tool_sink = FilterSink(tool_buffer, lambda s: "tool" in s.lower())

    observer2 = Observer(level="verbose")
    observer2.add_sink(tool_sink)

    agent2 = Agent(
        name="Analyst",
        model="gpt-4o-mini",
        tools=[search],
        observer=observer2,
    )

    await agent2.run("Search for 'machine learning'")

    print(f"\n🔧 Tool events captured: {len(tool_buffer.get_events())}")
    for event in tool_buffer.get_events():
        print(f"   {event}")

    print("\n" + "=" * 60)
    print("3. METRICS SINK - Extract metrics from agent run")
    print("=" * 60 + "\n")

    metrics = MetricsSink()

    observer3 = Observer(level="progress")
    observer3.add_sink(metrics)

    agent3 = Agent(
        name="Worker",
        model="gpt-4o-mini",
        tools=[search, summarize],
        observer=observer3,
    )

    await agent3.run("Search for 'Python' then summarize what you found")

    print(f"\n📊 {metrics.report()}")

    print("\n" + "=" * 60)
    print("4. WEBHOOK SINK - Batch and send events")
    print("=" * 60 + "\n")

    webhook = WebhookSink(
        url="https://example.com/events",
        batch_size=3,
    )

    observer4 = Observer(level="progress")
    observer4.add_sink(webhook)

    agent4 = Agent(
        name="Notifier",
        model="gpt-4o-mini",
        tools=[search],
        observer=observer4,
    )

    await agent4.run("Search for 'observability'")

    webhook.close()  # Flush remaining events


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
