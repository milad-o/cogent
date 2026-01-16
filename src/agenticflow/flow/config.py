"""Flow configuration dataclasses.

This module defines configuration options for Flow execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agenticflow.events import Event


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowConfig:
    """Configuration for Flow execution.

    Attributes:
        max_rounds: Maximum event processing rounds (prevents infinite loops)
        max_concurrent: Maximum concurrent reactor executions
        event_timeout: Timeout for waiting on events (seconds)
        enable_history: Whether to record all events for debugging
        stop_on_idle: Stop when no more events to process
        stop_events: Event types that signal flow completion
        flow_id: Optional fixed flow ID (auto-generated if None)
        checkpoint_every: Checkpoint after every N rounds (0 = disabled)
        error_policy: How to handle errors (fail_fast, continue, retry)

    Example:
        ```python
        config = FlowConfig(
            max_rounds=50,
            max_concurrent=5,
            stop_events=frozenset({"done", "error"}),
        )

        flow = Flow(config=config)
        ```
    """

    max_rounds: int = 100
    max_concurrent: int = 10
    event_timeout: float = 30.0
    enable_history: bool = True
    stop_on_idle: bool = True
    stop_events: frozenset[str] = frozenset({"flow.done", "flow.error"})
    flow_id: str | None = None
    checkpoint_every: int = 0
    error_policy: Literal["fail_fast", "continue", "retry"] = "fail_fast"


@dataclass(frozen=True, slots=True, kw_only=True)
class ReactorBinding:
    """Internal binding of a reactor to event pattern(s).

    Attributes:
        reactor_id: Unique identifier for the reactor
        patterns: Event type patterns to match (supports wildcards)
        priority: Execution priority (higher = first)
        condition: Optional filter condition
        emits: Event type to emit after execution (optional)
    """

    reactor_id: str
    patterns: frozenset[str]
    priority: int = 0
    condition: Callable[[Event], bool] | None = None
    emits: str | None = None


@dataclass(slots=True, kw_only=True)
class FlowResult:
    """Result of Flow execution.

    Attributes:
        success: Whether the flow completed successfully
        output: Final output value
        error: Error message if failed
        events_processed: Total events processed
        event_history: Full event log if enabled
        final_event: The event that terminated the flow
        flow_id: Flow execution ID
        metadata: Additional result data
    """

    success: bool = True
    output: Any = None
    error: str | None = None
    events_processed: int = 0
    event_history: list[Any] = field(default_factory=list)  # list[Event]
    final_event: Any = None  # Event | None
    flow_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Result is truthy if successful."""
        return self.success

    def raise_for_error(self) -> None:
        """Raise an exception if the flow failed."""
        if not self.success:
            raise RuntimeError(self.error or "Flow execution failed")
