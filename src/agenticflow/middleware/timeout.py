"""Timeout middleware for Flow execution.

Provides timeout enforcement for reactor executions.
"""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agenticflow.middleware.base import BaseMiddleware

if TYPE_CHECKING:
    from agenticflow.events import Event
    from agenticflow.reactors.base import Reactor


class TimeoutError(Exception):
    """Raised when reactor execution exceeds timeout."""

    def __init__(self, reactor_name: str, timeout: float) -> None:
        self.reactor_name = reactor_name
        self.timeout = timeout
        super().__init__(
            f"Reactor '{reactor_name}' timed out after {timeout}s"
        )


@dataclass
class TimeoutMiddleware(BaseMiddleware):
    """Enforce timeout on reactor execution.

    Cancels reactor execution if it exceeds the configured timeout.

    Attributes:
        timeout: Default timeout in seconds
        per_reactor: Optional dict of reactor-specific timeouts
        on_timeout: Optional callback when timeout occurs

    Example:
        ```python
        from agenticflow.middleware import TimeoutMiddleware

        flow = Flow()
        flow.use(TimeoutMiddleware(timeout=30))
        ```

    Example with per-reactor timeouts:
        ```python
        flow.use(TimeoutMiddleware(
            timeout=30,
            per_reactor={
                "slow_agent": 120,
                "fast_processor": 5,
            }
        ))
        ```
    """

    timeout: float = 30.0
    per_reactor: dict[str, float] | None = None
    on_timeout: Callable[[str, float], None] | None = None

    def _get_timeout(self, reactor: Reactor) -> float:
        """Get timeout for specific reactor."""
        if self.per_reactor:
            reactor_name = getattr(reactor, "name", None)
            if reactor_name and reactor_name in self.per_reactor:
                return self.per_reactor[reactor_name]
        return self.timeout

    async def wrap_execution(
        self,
        reactor: Reactor,
        event: Event,
        execute: Callable[[], Event | list[Event] | None],
    ) -> Event | list[Event] | None:
        """Execute with timeout enforcement.

        Args:
            reactor: The reactor being executed
            event: The event being processed
            execute: The execution function to wrap

        Returns:
            Reactor result or raises TimeoutError
        """
        timeout = self._get_timeout(reactor)
        reactor_name = getattr(reactor, "name", type(reactor).__name__)

        try:
            return await asyncio.wait_for(execute(), timeout=timeout)
        except builtins.TimeoutError:
            if self.on_timeout:
                self.on_timeout(reactor_name, timeout)
            raise TimeoutError(reactor_name, timeout)


@dataclass
class AggressiveTimeoutMiddleware(TimeoutMiddleware):
    """Aggressive timeout with shorter default.

    Example:
        ```python
        flow.use(AggressiveTimeoutMiddleware())  # 10s default
        ```
    """

    timeout: float = 10.0
