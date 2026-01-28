"""
Event - The fundamental unit of observability.

Events are immutable records of things that happened in the system.
Unlike v1's rigid TraceType enum, event types are simple strings
that support pattern matching (e.g., "agent.*", "*.error").

Example:
    ```python
    from cogent.observability import Event, create_event

    # Create event with factory
    event = create_event(
        "agent.thinking",
        agent_name="Researcher",
        iteration=1,
    )

    # Access properties
    print(event.type)       # "agent.thinking"
    print(event.category)   # "agent"
    print(event.data)       # {"agent_name": "Researcher", "iteration": 1}
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cogent.core.utils import generate_id, now_utc


@dataclass(frozen=True, slots=True)
class Event:
    """
    Immutable event record.

    Attributes:
        type: Event type string (e.g., "agent.thinking", "tool.called")
        data: Event payload - arbitrary key-value data
        timestamp: When the event occurred (UTC)
        source: Origin of the event (agent name, "system", etc.)
        correlation_id: ID for correlating related events
        event_id: Unique identifier for this event
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=now_utc)
    source: str = "system"
    correlation_id: str | None = None
    event_id: str = field(default_factory=lambda: generate_id(8))

    @property
    def category(self) -> str:
        """
        Extract category from type.

        Examples:
            "agent.thinking" → "agent"
            "tool.called" → "tool"
            "my_app.order.placed" → "my_app"
            "custom" → "custom"
        """
        return self.type.split(".")[0] if "." in self.type else self.type

    @property
    def action(self) -> str:
        """
        Extract action from type (everything after the first dot).

        Examples:
            "agent.thinking" → "thinking"
            "tool.called" → "called"
            "my_app.order.placed" → "order.placed"
            "custom" → "custom"
        """
        parts = self.type.split(".", 1)
        return parts[1] if len(parts) > 1 else self.type

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from event data with default."""
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to data."""
        return self.data[key]

    def matches(self, pattern: str) -> bool:
        """
        Check if event type matches a glob pattern.

        Patterns:
            "agent.*" - matches agent.thinking, agent.responded, etc.
            "*.error" - matches agent.error, tool.error, etc.
            "*" - matches everything
            "agent.thinking" - exact match

        Args:
            pattern: Glob pattern to match against

        Returns:
            True if event type matches pattern
        """
        import fnmatch

        return fnmatch.fnmatch(self.type, pattern)


def create_event(
    event_type: str,
    *,
    source: str = "system",
    correlation_id: str | None = None,
    timestamp: datetime | None = None,
    **data: Any,
) -> Event:
    """
    Create an event with the given type and data.

    Factory function that provides a cleaner API than the dataclass constructor.

    Args:
        event_type: Event type string (e.g., "agent.thinking")
        source: Origin of the event
        correlation_id: ID for correlating related events
        timestamp: Event timestamp (defaults to now)
        **data: Event payload as keyword arguments

    Returns:
        New Event instance

    Example:
        ```python
        event = create_event(
            "tool.called",
            source="Researcher",
            tool_name="web_search",
            args={"query": "python tutorials"},
        )
        ```
    """
    return Event(
        type=event_type,
        data=data,
        timestamp=timestamp or now_utc(),
        source=source,
        correlation_id=correlation_id,
    )


# Common event type constants (for convenience, not required)
class EventTypes:
    """
    Common event type strings.

    These are provided for convenience and IDE autocomplete.
    You can use any string as an event type.
    """

    # Agent events
    AGENT_INVOKED = "agent.invoked"
    AGENT_THINKING = "agent.thinking"
    AGENT_REASONING = "agent.reasoning"
    AGENT_ACTING = "agent.acting"
    AGENT_RESPONDED = "agent.responded"
    AGENT_ERROR = "agent.error"

    # Tool events
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"

    # Task events
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    # Stream events
    STREAM_START = "stream.start"
    STREAM_TOKEN = "stream.token"
    STREAM_END = "stream.end"
    STREAM_ERROR = "stream.error"

    # LLM events
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"

    # Memory events
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    MEMORY_SEARCH = "memory.search"
