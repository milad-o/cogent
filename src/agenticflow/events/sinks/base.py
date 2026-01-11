"""Base class for event sinks.

Event sinks send events from EventFlow to external systems
(webhooks, message queues, databases, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenticflow.events.event import Event


@dataclass
class EventSink(ABC):
    """Abstract base class for outbound event sinks.

    Event sinks receive events from EventFlow and deliver them
    to external systems.

    Example:
        ```python
        class MyDatabaseSink(EventSink):
            async def send(self, event: Event) -> None:
                await db.insert("events", event.to_dict())

            async def close(self) -> None:
                await db.close()
        ```
    """

    @abstractmethod
    async def send(self, event: "Event") -> None:
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
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable name for this sink (used in logging)."""
        return self.__class__.__name__
