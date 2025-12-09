"""Base classes and protocols for flow orchestration.

This module defines the common interface and shared functionality
for all flow types (imperative and reactive).
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agenticflow.core.utils import generate_id, now_utc
from agenticflow.observability.bus import EventBus
from agenticflow.observability.event import Event, EventType

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent
    from agenticflow.observability.observer import Observer


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowResult:
    """
    Base result from any flow execution.

    All flow types return this or a subclass.
    """

    output: str
    """Final output from the flow."""

    execution_time_ms: float = 0.0
    """Total execution time in milliseconds."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional execution metadata."""


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class FlowProtocol(Protocol):
    """
    Protocol defining the interface for all flow types.

    This enables structural typing - any class implementing these
    methods is considered a valid flow.
    """

    @property
    def agents(self) -> list[str]:
        """Names of registered agents."""
        ...

    @property
    def observer(self) -> Observer | None:
        """Attached observer for monitoring."""
        ...

    @property
    def bus(self) -> EventBus:
        """Event bus for internal communication."""
        ...

    async def run(self, task: str, **kwargs: Any) -> FlowResult:
        """
        Execute the flow with a task.

        Args:
            task: The task/prompt to execute.
            **kwargs: Flow-specific options.

        Returns:
            FlowResult with output and metadata.
        """
        ...

    def stop(self) -> None:
        """Stop flow execution gracefully."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class BaseFlow:
    """
    Abstract base class for all flow implementations.

    Provides common functionality:
    - Event bus for internal communication
    - Observer integration for monitoring
    - Observability event emission
    - Graceful stop mechanism

    Subclasses must implement:
    - `run()`: Execute the flow
    - `agents` property: Return list of agent names
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        observer: Observer | None = None,
    ) -> None:
        """
        Initialize the base flow.

        Args:
            event_bus: Shared event bus (creates new one if not provided)
            observer: Optional observer for monitoring
        """
        self._bus = event_bus or EventBus()
        self._observer = observer
        self._running = False

        # Attach observer to the event bus
        if self._observer:
            self._observer.attach(self._bus)

    @property
    def bus(self) -> EventBus:
        """Event bus for internal communication."""
        return self._bus

    @property
    def observer(self) -> Observer | None:
        """Attached observer for monitoring."""
        return self._observer

    @property
    def is_running(self) -> bool:
        """Check if flow is currently executing."""
        return self._running

    @property
    @abstractmethod
    def agents(self) -> list[str]:
        """Names of registered agents."""
        ...

    @abstractmethod
    async def run(self, task: str, **kwargs: Any) -> FlowResult:
        """Execute the flow with a task."""
        ...

    def stop(self) -> None:
        """Stop flow execution gracefully."""
        self._running = False

    def attach_observer(self, observer: Observer) -> None:
        """
        Attach an observer for monitoring.

        Args:
            observer: The observer to attach
        """
        self._observer = observer
        observer.attach(self._bus)

    def _observe(
        self,
        event_type: EventType,
        data: dict[str, Any],
    ) -> None:
        """
        Emit an observability event.

        Args:
            event_type: Type of event to emit
            data: Event data
        """
        if self._observer:
            event = Event(
                id=generate_id(),
                type=event_type,
                timestamp=now_utc(),
                data=data,
            )
            # Use internal handler for synchronous handling
            self._observer._handle_event(event)

    async def _publish(
        self,
        event_type: EventType,
        data: dict[str, Any],
    ) -> None:
        """
        Publish an event to the bus.

        Args:
            event_type: Type of event to publish
            data: Event data
        """
        await self._bus.publish(event_type.value, data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agents={self.agents})"
