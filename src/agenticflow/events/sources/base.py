"""Base class for external event sources.

Event sources are adapters that ingest events from external systems
(webhooks, message queues, file systems, etc.) and emit them into
the EventFlow for reactive processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from agenticflow.events.event import Event


# Type alias for the emit callback
EmitCallback = Callable[["Event"], Awaitable[None]]


@dataclass
class EventSource(ABC):
    """Abstract base class for external event sources.

    Event sources connect external systems to EventFlow. They listen for
    external triggers (HTTP requests, file changes, queue messages) and
    emit corresponding events into the flow.

    Lifecycle:
        1. `start(emit)` - Begin listening, call `emit(event)` when events arrive
        2. `stop()` - Stop listening and clean up resources

    Example:
        ```python
        class MySource(EventSource):
            async def start(self, emit: EmitCallback) -> None:
                # Connect to external system
                while self._running:
                    data = await self.poll()
                    await emit(Event(name="my.event", data=data))

            async def stop(self) -> None:
                self._running = False
        ```
    """

    @abstractmethod
    async def start(self, emit: EmitCallback) -> None:
        """Start listening for external events.

        This method should run indefinitely (or until stop() is called),
        calling `emit(event)` whenever an external event is received.

        Args:
            emit: Async callback to emit events into the flow.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop listening and clean up resources.

        Called when the flow is shutting down. Should gracefully stop
        any running listeners and release resources.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name for this source (used in logging)."""
        return self.__class__.__name__
