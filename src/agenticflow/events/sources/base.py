"""Base class for external event sources.

Event sources are adapters that ingest events from external systems
(webhooks, message queues, file systems, etc.) and emit them into
the Flow for event-driven processing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agenticflow.events.event import Event


# Type alias for the emit callback
EmitCallback = Callable[["Event"], Awaitable[None]]


class EventSource(Protocol):
    """Protocol for external event sources.

    Event sources connect external systems to Flow. They listen for
    external triggers (HTTP requests, file changes, queue messages) and
    emit corresponding events into the flow. Uses Protocol for structural
    typing, so any class implementing these methods can be used as a source.

    Lifecycle:
        1. `start(emit)` - Begin listening, call `emit(event)` when events arrive
        2. `stop()` - Stop listening and clean up resources

    Example:
        ```python
        class MySource:
            async def start(self, emit: EmitCallback) -> None:
                # Connect to external system
                while self._running:
                    data = await self.poll()
                    await emit(Event(name="my.event", data=data))

            async def stop(self) -> None:
                self._running = False
        ```
    """

    async def start(self, emit: EmitCallback) -> None:
        """Start listening for external events.

        This method should run indefinitely (or until stop() is called),
        calling `emit(event)` whenever an external event is received.

        Args:
            emit: Async callback to emit events into the flow.
        """
        ...

    async def stop(self) -> None:
        """Stop listening and clean up resources.

        Called when the flow is shutting down. Should gracefully stop
        any running listeners and release resources.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name for this source (used in logging)."""
        ...
