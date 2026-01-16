"""Core event types for orchestration.

These events drive application logic (reactive flows, external triggers, etc.).
They are *not* observability events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agenticflow.core.utils import generate_id, now_utc


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """A core event used for orchestration.

    Events are the unit of communication in the event-driven architecture.
    All reactors receive and emit events.

    Attributes:
        name: Event type (e.g., 'agent.done', 'task.created')
        source: Who/what emitted this event
        data: Event payload
        id: Unique event identifier
        correlation_id: Request chain tracking
        timestamp: When the event was created
        metadata: Additional context (not part of core data)

    Example:
        ```python
        # Create directly
        event = Event(name="task.created", source="user", data={"task": "..."})

        # Use factory methods
        event = Event.done("researcher", output="Research results...")
        event = Event.error("writer", error="Failed to generate content")
        ```
    """

    name: str
    """Event name/type (e.g., 'agent.done', 'task.created')."""

    source: str = "system"
    """Who/what emitted this event."""

    data: dict[str, Any] = field(default_factory=dict)
    """Event payload."""

    id: str = field(default_factory=generate_id)
    """Unique event id."""

    correlation_id: str | None = None
    """Correlation id for tracing across components."""

    timestamp: datetime = field(default_factory=now_utc)
    """Event timestamp."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional context (e.g., flow_id, thread_id)."""

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        source: str = "system",
        data: dict[str, Any] | None = None,
        *,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Create an event with optional parameters."""
        return cls(
            name=name,
            source=source,
            data=data or {},
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

    @classmethod
    def done(
        cls,
        source: str,
        output: str,
        *,
        correlation_id: str | None = None,
        **extra_data: Any,
    ) -> Event:
        """Create an 'agent.done' event for successful completion."""
        return cls(
            name="agent.done",
            source=source,
            data={"output": output, **extra_data},
            correlation_id=correlation_id,
        )

    @classmethod
    def error(
        cls,
        source: str,
        error: str,
        *,
        correlation_id: str | None = None,
        exception: Exception | None = None,
        **extra_data: Any,
    ) -> Event:
        """Create an 'agent.error' event for failures."""
        data: dict[str, Any] = {"error": error, **extra_data}
        if exception:
            data["exception_type"] = type(exception).__name__
        return cls(
            name="agent.error",
            source=source,
            data=data,
            correlation_id=correlation_id,
        )

    @classmethod
    def task_created(
        cls,
        task: str,
        source: str = "user",
        *,
        correlation_id: str | None = None,
        **extra_data: Any,
    ) -> Event:
        """Create a 'task.created' event to start a flow."""
        return cls(
            name="task.created",
            source=source,
            data={"task": task, **extra_data},
            correlation_id=correlation_id,
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def with_correlation(self, correlation_id: str) -> Event:
        """Return a copy with a new correlation_id."""
        return Event(
            name=self.name,
            source=self.source,
            data=self.data,
            id=self.id,
            correlation_id=correlation_id,
            timestamp=self.timestamp,
            metadata=self.metadata,
        )

    def with_metadata(self, **kwargs: Any) -> Event:
        """Return a copy with additional metadata."""
        return Event(
            name=self.name,
            source=self.source,
            data=self.data,
            id=self.id,
            correlation_id=self.correlation_id,
            timestamp=self.timestamp,
            metadata={**self.metadata, **kwargs},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "name": self.name,
            "source": self.source,
            "data": self.data,
            "id": self.id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        """Deserialize event from dictionary."""
        timestamp = d.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            name=d["name"],
            source=d.get("source", "system"),
            data=d.get("data", {}),
            id=d.get("id", generate_id()),
            correlation_id=d.get("correlation_id"),
            timestamp=timestamp or now_utc(),
            metadata=d.get("metadata", {}),
        )
