"""Aggregator reactor - collects multiple events (fan-in pattern)."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agenticflow.events import Event
from agenticflow.reactors.base import BaseReactor, FanInMode

if TYPE_CHECKING:
    from agenticflow.flow.context import Context


@dataclass
class AggregationState:
    """State for an ongoing aggregation."""

    events: list[Event] = field(default_factory=list)
    """Collected events."""

    expected: int = 0
    """Number of events expected."""

    complete: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when aggregation is complete."""


class Aggregator(BaseReactor):
    """Collects multiple events before emitting a combined result.

    Implements fan-in patterns where you need to wait for multiple
    events before proceeding.

    Example:
        ```python
        # Wait for 3 worker completions
        flow.register(
            Aggregator(
                collect=3,
                emit="workers.done",
            ),
            on="agent.done",
            when=lambda e: e.source in ["worker1", "worker2", "worker3"],
        )

        # Wait for all events matching a pattern
        flow.register(
            Aggregator(
                collect=3,
                emit="reviews.done",
                mode=FanInMode.WAIT_ALL,
            ),
            on="review.*.done",
        )
        ```
    """

    def __init__(
        self,
        collect: int,
        emit: str,
        *,
        name: str | None = None,
        mode: FanInMode = FanInMode.WAIT_ALL,
        timeout: float | None = None,
        combine_fn: callable | None = None,
    ) -> None:
        """Initialize aggregator.

        Args:
            collect: Number of events to collect
            emit: Event name to emit when complete
            name: Reactor name
            mode: Fan-in mode (how to handle events)
            timeout: Timeout in seconds (None = no timeout)
            combine_fn: Custom function to combine events (default: list outputs)
        """
        super().__init__(name or f"aggregator_{emit}")
        self._collect = collect
        self._emit = emit
        self._mode = mode
        self._timeout = timeout
        self._combine_fn = combine_fn or self._default_combine

        # State per correlation_id (allows multiple concurrent aggregations)
        self._states: dict[str, AggregationState] = defaultdict(
            lambda: AggregationState(expected=collect)
        )
        self._lock = asyncio.Lock()

    def _default_combine(self, events: list[Event]) -> dict[str, Any]:
        """Default combiner: collect outputs by source."""
        outputs = {}
        for e in events:
            source = e.source
            output = e.data.get("output", e.data)
            outputs[source] = output
        return {
            "outputs": outputs,
            "count": len(events),
            "sources": list(outputs.keys()),
        }

    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> Event | None:
        """Collect event, emit when all collected."""
        # Use correlation_id or flow_id as aggregation key
        key = event.correlation_id or ctx.flow_id

        async with self._lock:
            state = self._states[key]
            state.events.append(event)

            # Check if we have enough events
            if self._mode == FanInMode.FIRST_WINS:
                # Emit immediately on first event
                result = self._create_result(state.events, event)
                del self._states[key]
                return result

            if self._mode == FanInMode.STREAMING:
                # Emit for each event (don't aggregate)
                return Event(
                    name=self._emit,
                    source=self.name,
                    data={
                        "event": event.to_dict(),
                        "index": len(state.events),
                        "expected": state.expected,
                    },
                    correlation_id=event.correlation_id,
                )

            # WAIT_ALL or QUORUM: check if complete
            if len(state.events) >= self._collect:
                result = self._create_result(state.events, event)
                del self._states[key]
                return result

            # Not complete yet
            return None

    def _create_result(self, events: list[Event], trigger: Event) -> Event:
        """Create the combined result event."""
        combined = self._combine_fn(events)

        return Event(
            name=self._emit,
            source=self.name,
            data=combined,
            correlation_id=trigger.correlation_id,
        )

    def reset(self, key: str | None = None) -> None:
        """Reset aggregation state.

        Args:
            key: Specific aggregation key, or None for all
        """
        if key:
            self._states.pop(key, None)
        else:
            self._states.clear()

    @property
    def pending_count(self) -> int:
        """Number of pending aggregations."""
        return len(self._states)


class FirstWins(Aggregator):
    """Shorthand for Aggregator with FIRST_WINS mode.

    Emits immediately when the first event arrives.

    Example:
        ```python
        flow.register(
            FirstWins(emit="first.response"),
            on="agent.done",
        )
        ```
    """

    def __init__(self, emit: str, **kwargs: Any) -> None:
        super().__init__(
            collect=1,
            emit=emit,
            mode=FanInMode.FIRST_WINS,
            **kwargs,
        )


class WaitAll(Aggregator):
    """Shorthand for Aggregator with WAIT_ALL mode.

    Example:
        ```python
        flow.register(
            WaitAll(collect=3, emit="all.done"),
            on="worker.done",
        )
        ```
    """

    def __init__(self, collect: int, emit: str, **kwargs: Any) -> None:
        super().__init__(
            collect=collect,
            emit=emit,
            mode=FanInMode.WAIT_ALL,
            **kwargs,
        )
