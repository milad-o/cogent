"""Base class for event sinks.

Event sinks send events from Flow to external systems
(webhooks, message queues, databases, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agenticflow.events.event import Event


class EventSink(Protocol):
    """Protocol for outbound event sinks.

    Event sinks receive events from Flow and deliver them
    to external systems. Uses Protocol for structural typing,
    so any class implementing these methods can be used as a sink.

    Example:
        ```python
        class MyDatabaseSink:
            async def send(self, event: Event) -> None:
                await db.insert("events", event.to_dict())

            async def close(self) -> None:
                await db.close()
        ```
    """

    async def send(self, event: Event) -> None:
        """Send an event to the external system.

        Args:
            event: The event to send.

        Raises:
            Exception: If delivery fails (will be logged but not crash flow).
        """
        ...

    async def close(self) -> None:
        """Clean up resources (optional).

        Called when the flow shuts down. Override to close connections.
        Default implementation does nothing.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name for this sink (used in logging)."""
        ...
