"""Reactor base protocol and types.

Reactors are the fundamental building blocks of event-driven flows.
They handle events and optionally emit new events.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agenticflow.events import Event
    from agenticflow.flow.context import Context


class FanInMode(StrEnum):
    """How to handle multiple incoming events."""

    WAIT_ALL = "wait_all"
    """Wait for all expected events before proceeding."""

    FIRST_WINS = "first_wins"
    """Proceed with the first event, ignore others."""

    STREAMING = "streaming"
    """Process each event as it arrives."""

    QUORUM = "quorum"
    """Wait for N of M events (requires collect parameter)."""


class HandoverStrategy(StrEnum):
    """How to pass data between reactors."""

    FULL_OUTPUT = "full_output"
    """Pass the complete output string."""

    SUMMARY = "summary"
    """Summarize before passing (requires summarizer)."""

    STRUCTURED = "structured"
    """Extract structured data from output."""

    ACCUMULATED = "accumulated"
    """Accumulate into growing context window."""


class ErrorPolicy(StrEnum):
    """How to handle reactor errors."""

    FAIL_FAST = "fail_fast"
    """Stop the flow on first error."""

    CONTINUE = "continue"
    """Skip the failed reactor, continue flow."""

    RETRY = "retry"
    """Retry the reactor (uses middleware)."""

    FALLBACK = "fallback"
    """Use a fallback reactor."""


@runtime_checkable
class Reactor(Protocol):
    """Protocol for all event reactors.

    Reactors handle events and optionally emit new events.
    They are the building blocks of event-driven flows.

    Built-in reactors:
    - AgentReactor: Wraps an Agent
    - FunctionReactor: Wraps a function
    - Aggregator: Collects multiple events (fan-in)
    - Router: Routes events based on conditions
    - Transform: Transforms event data
    - Gateway: Bridges external systems

    Example:
        ```python
        class MyReactor:
            name = "my_reactor"

            async def handle(self, event: Event, ctx: Context) -> Event | None:
                # Process event
                result = do_something(event.data)
                # Emit result event
                return Event.done(self.name, output=result)
        ```
    """

    @property
    def name(self) -> str:
        """Unique name for this reactor."""
        ...

    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> Event | list[Event] | None:
        """Handle an event, optionally emit result event(s).

        Args:
            event: The incoming event
            ctx: Flow execution context

        Returns:
            - Event: Single result event
            - list[Event]: Multiple result events
            - None: No event emitted
        """
        ...


@dataclass(kw_only=True)
class ReactorConfig:
    """Configuration for reactor registration.

    Defines how a reactor responds to events.
    """

    on: str | list[str]
    """Event pattern(s) to react to."""

    when: Callable[[Event], bool] | None = None
    """Optional condition filter."""

    after: str | None = None
    """React after events from this source (shorthand for when=from_source(...))."""

    priority: int = 0
    """Higher priority = executed first when multiple reactors match."""

    emits: str | None = None
    """Override the event name emitted by this reactor."""

    # Fan-in configuration
    collect: int | None = None
    """Number of events to collect before handling (for Aggregator)."""

    fan_in: FanInMode = FanInMode.WAIT_ALL
    """How to handle multiple incoming events."""

    # Handover
    handover: HandoverStrategy = HandoverStrategy.FULL_OUTPUT
    """How to pass data to next reactor."""

    # Delegation
    can_delegate: list[str] | None = None
    """List of reactor names this reactor can delegate to."""

    can_reply: bool = False
    """Whether this reactor can reply to delegated requests."""

    # Error handling
    on_error: ErrorPolicy = ErrorPolicy.FAIL_FAST
    """How to handle errors in this reactor."""

    fallback: Reactor | None = None
    """Fallback reactor if this one fails (when on_error=FALLBACK)."""

    # Metadata
    role: str | None = None
    """Role description for this reactor in the flow."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""


class BaseReactor:
    """Base class for reactor implementations.

    Provides common functionality for reactors.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> Event | list[Event] | None:
        """Handle an event."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
