"""Core EventBus for orchestration.

This is intentionally minimal and does not include observability concerns
(e.g., websocket broadcast, TraceType enums, tracing output formatting).

Supports optional transport backends for distributed event delivery.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from agenticflow.events.event import Event

if TYPE_CHECKING:
    from agenticflow.events.transport import Transport


EventHandler = Callable[[Event], None] | Callable[[Event], Awaitable[None]]
EventPattern = str | re.Pattern[str]


class EventBus:
    """A lightweight async pub/sub bus for core orchestration events.

    Supports optional distributed transport for cross-process communication.

    Args:
        max_history: Maximum events to keep in history
        transport: Optional transport backend (LocalTransport, RedisTransport, etc.)

    Example:
        ```python
        # Local (single process)
        bus = EventBus()

        # Distributed (Redis)
        from agenticflow.events.transport import RedisTransport
        transport = RedisTransport(url="redis://localhost:6379")
        await transport.connect()
        bus = EventBus(transport=transport)
        ```
    """

    def __init__(
        self,
        max_history: int = 10_000,
        transport: Transport | None = None,
    ) -> None:
        self._handlers: dict[EventPattern, list[EventHandler]] = defaultdict(list)
        self._global_handlers: list[EventHandler] = []
        self._event_history: list[Event] = []
        self._lock = asyncio.Lock()
        self._max_history = max_history
        self._transport = transport
        self._transport_subscriptions: dict[EventPattern, str] = {}

    @property
    def history_size(self) -> int:
        return len(self._event_history)

    def get_history(self, *, limit: int = 100) -> list[Event]:
        if limit <= 0:
            return []
        return self._event_history[-limit:]

    def subscribe(self, event: EventPattern, handler: EventHandler) -> None:
        if handler not in self._handlers[event]:
            self._handlers[event].append(handler)

            # Subscribe via transport if available
            if self._transport and event not in self._transport_subscriptions:
                pattern_str = event if isinstance(event, str) else str(event.pattern)

                # Create wrapper to handle transport events
                async def transport_handler(transport_event: Event) -> None:
                    await _call_handler(handler, transport_event)

                # Subscribe and store subscription ID
                import asyncio
                try:
                    # Fire-and-forget subscription for transport layer
                    asyncio.create_task(
                        self._transport.subscribe(pattern_str, transport_handler)
                    )
                except Exception:
                    pass

    def subscribe_all(self, handler: EventHandler) -> None:
        if handler not in self._global_handlers:
            self._global_handlers.append(handler)

    def unsubscribe(self, event: EventPattern, handler: EventHandler) -> None:
        if handler in self._handlers.get(event, []):
            self._handlers[event].remove(handler)

    def unsubscribe_all(self, handler: EventHandler) -> None:
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)

    def clear_subscriptions(self) -> None:
        self._handlers.clear()
        self._global_handlers.clear()

    async def publish(self, event: Event | str, data: dict[str, Any] | None = None) -> Event:
        """Publish an event.

        Supports:
        - publish(Event(...))
        - publish("event.name", {..})

        If transport is configured, event is also published to distributed backend.
        """
        if isinstance(event, str):
            event = Event(name=event, data=data or {})

        async with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history :]

        # Publish to transport if available
        if self._transport:
            try:
                await self._transport.publish(event)
            except Exception:
                # Don't fail local publish if transport fails
                pass

        # Pattern handlers
        for pattern, handlers in list(self._handlers.items()):
            if _matches(pattern, event.name):
                for handler in list(handlers):
                    await _call_handler(handler, event)

        # Global handlers
        for handler in list(self._global_handlers):
            await _call_handler(handler, event)

        return event


def _matches(pattern: EventPattern, event_name: str) -> bool:
    if isinstance(pattern, re.Pattern):
        return pattern.match(event_name) is not None
    # glob support
    if "*" in pattern:
        regex = pattern.replace(".", r"\.").replace("*", ".*")
        return re.match(regex, event_name) is not None
    return pattern == event_name


async def _call_handler(handler: EventHandler, event: Event) -> None:
    try:
        if inspect.iscoroutinefunction(handler):
            await handler(event)
        else:
            handler(event)
    except Exception:
        # Core bus should not crash orchestrator due to handler errors.
        # Observability should capture these separately if desired.
        return
