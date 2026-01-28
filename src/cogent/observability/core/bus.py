"""
EventBus - Simple pub/sub for event distribution.

A lightweight event bus that supports:
- Pattern-based subscriptions (e.g., "agent.*")
- Global subscriptions
- Sync handlers
"""

from __future__ import annotations

import fnmatch
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cogent.observability.core.event import Event

# Handler type
EventHandler = Callable[["Event"], None]


class EventBus:
    """
    Simple pub/sub event bus.

    Supports pattern-based subscriptions using glob patterns.

    Example:
        ```python
        bus = EventBus()

        # Subscribe to specific pattern
        bus.subscribe("agent.*", handle_agent_events)
        bus.subscribe("*.error", handle_errors)

        # Subscribe to all events
        bus.subscribe_all(log_event)

        # Publish event
        bus.publish(event)
        ```
    """

    def __init__(self) -> None:
        """Initialize empty event bus."""
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._global_handlers: list[EventHandler] = []
        # Stats tracking
        self._event_count = 0
        self._event_counts_by_type: dict[str, int] = defaultdict(int)

    def subscribe(self, pattern: str, handler: EventHandler) -> Callable[[], None]:
        """
        Subscribe to events matching a pattern.

        Patterns use glob syntax:
            "agent.*" - all agent events
            "*.error" - all error events
            "tool.called" - exact match

        Args:
            pattern: Glob pattern to match event types
            handler: Function to call when matching event is published

        Returns:
            Unsubscribe function
        """
        if handler not in self._handlers[pattern]:
            self._handlers[pattern].append(handler)

        def unsubscribe() -> None:
            self.unsubscribe(pattern, handler)

        return unsubscribe

    def subscribe_all(self, handler: EventHandler) -> Callable[[], None]:
        """
        Subscribe to all events.

        Args:
            handler: Function to call for every event

        Returns:
            Unsubscribe function
        """
        if handler not in self._global_handlers:
            self._global_handlers.append(handler)

        def unsubscribe() -> None:
            self.unsubscribe_all(handler)

        return unsubscribe

    def unsubscribe(self, pattern: str, handler: EventHandler) -> None:
        """
        Unsubscribe a handler from a pattern.

        Args:
            pattern: Pattern the handler was subscribed to
            handler: Handler to remove
        """
        if handler in self._handlers[pattern]:
            self._handlers[pattern].remove(handler)

    def unsubscribe_all(self, handler: EventHandler) -> None:
        """
        Remove a global handler.

        Args:
            handler: Handler to remove
        """
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)

    def publish(self, event: Event) -> None:
        """
        Publish an event to all matching subscribers.

        Args:
            event: Event to publish
        """
        import contextlib

        # Track stats
        self._event_count += 1
        self._event_counts_by_type[event.type] += 1

        # Global handlers first
        for handler in self._global_handlers:
            with contextlib.suppress(Exception):
                handler(event)

        # Pattern-matched handlers
        for pattern, handlers in self._handlers.items():
            if fnmatch.fnmatch(event.type, pattern):
                for handler in handlers:
                    with contextlib.suppress(Exception):
                        handler(event)

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._handlers.clear()
        self._global_handlers.clear()

    @property
    def subscriber_count(self) -> int:
        """Total number of subscriptions (including global)."""
        pattern_count = sum(len(handlers) for handlers in self._handlers.values())
        return pattern_count + len(self._global_handlers)

    def stats(self) -> dict[str, object]:
        """
        Get event statistics.

        Returns:
            Dictionary with event counts
        """
        by_category: dict[str, int] = defaultdict(int)
        for event_type, count in self._event_counts_by_type.items():
            category = event_type.split(".")[0] if "." in event_type else event_type
            by_category[category] += count

        return {
            "total_events": self._event_count,
            "by_type": dict(self._event_counts_by_type),
            "by_category": dict(by_category),
        }

    def reset_stats(self) -> None:
        """Reset event statistics."""
        self._event_count = 0
        self._event_counts_by_type.clear()
