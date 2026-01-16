"""Middleware base classes and protocols.

Middleware provides cross-cutting concerns for Flow execution,
such as logging, retry logic, timeouts, and tracing.

Middleware wraps reactor execution and can:
- Modify events before processing
- Transform results after processing
- Add retry logic, timeouts, logging
- Collect metrics and traces
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agenticflow.events import Event
    from agenticflow.reactors.base import Reactor


class Middleware(Protocol):
    """Protocol for middleware that wraps reactor execution.

    Middleware can intercept events before/after reactor processing.
    Both methods are optional - implement what you need.

    Example:
        ```python
        class TimingMiddleware:
            async def before(self, event: Event, reactor: Reactor) -> Event | None:
                event.data["_start_time"] = time.time()
                return event

            async def after(
                self,
                result: Event | list[Event] | None,
                event: Event,
                reactor: Reactor,
            ) -> Event | list[Event] | None:
                elapsed = time.time() - event.data.get("_start_time", 0)
                print(f"Reactor took {elapsed:.2f}s")
                return result
        ```
    """

    async def before(
        self,
        event: Event,
        reactor: Reactor,
    ) -> Event | None:
        """Called before reactor processes event.

        Args:
            event: The event being processed
            reactor: The reactor that will process the event

        Returns:
            Modified event, or None to use original event
        """
        ...

    async def after(
        self,
        result: Event | list[Event] | None,
        event: Event,
        reactor: Reactor,
    ) -> Event | list[Event] | None:
        """Called after reactor processes event.

        Args:
            result: The reactor's result (events or None)
            event: The original event
            reactor: The reactor that processed the event

        Returns:
            Modified result, or None to use original result
        """
        ...


class BaseMiddleware(ABC):
    """Base class for middleware implementations.

    Provides default pass-through implementations for before/after hooks.
    Override only the methods you need.

    Example:
        ```python
        class LoggingMiddleware(BaseMiddleware):
            async def before(self, event: Event, reactor: Reactor) -> Event | None:
                print(f"Processing {event.type}")
                return None  # Use original event

            async def after(self, result, event, reactor):
                print(f"Completed {event.type}")
                return None  # Use original result
        ```
    """

    async def before(
        self,
        event: Event,
        reactor: Reactor,
    ) -> Event | None:
        """Called before reactor. Override to modify event."""
        return None

    async def after(
        self,
        result: Event | list[Event] | None,
        event: Event,
        reactor: Reactor,
    ) -> Event | list[Event] | None:
        """Called after reactor. Override to modify result."""
        return None


class MiddlewareChain:
    """Chain multiple middleware together.

    Middleware is applied in order for `before`, and reverse order for `after`.

    Example:
        ```python
        chain = MiddlewareChain([
            LoggingMiddleware(),
            RetryMiddleware(max_retries=3),
            TimeoutMiddleware(timeout=30),
        ])

        # Before: Logging → Retry → Timeout
        # After: Timeout → Retry → Logging
        ```
    """

    def __init__(self, middleware: list[Middleware] | None = None) -> None:
        self._middleware: list[Middleware] = middleware or []

    def add(self, middleware: Middleware) -> MiddlewareChain:
        """Add middleware to the chain."""
        self._middleware.append(middleware)
        return self

    async def before(
        self,
        event: Event,
        reactor: Reactor,
    ) -> Event:
        """Run all before hooks in order."""
        current = event
        for mw in self._middleware:
            if hasattr(mw, "before"):
                result = await mw.before(current, reactor)
                if result is not None:
                    current = result
        return current

    async def after(
        self,
        result: Event | list[Event] | None,
        event: Event,
        reactor: Reactor,
    ) -> Event | list[Event] | None:
        """Run all after hooks in reverse order."""
        current = result
        for mw in reversed(self._middleware):
            if hasattr(mw, "after"):
                modified = await mw.after(current, event, reactor)
                if modified is not None:
                    current = modified
        return current

    def __len__(self) -> int:
        return len(self._middleware)

    def __iter__(self):
        return iter(self._middleware)
