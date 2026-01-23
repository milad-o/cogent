"""State management for coordination patterns.

This module provides stateful coordination primitives for the Flow system,
enabling reactors to wait for events from ALL specified sources before executing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class CoordinationState:
    """Tracks completion state for a single coordination point.

    A coordination point waits for events from all required sources before
    completing. Once all sources have emitted, the coordination is marked
    as completed and can optionally auto-reset for the next cycle.

    Attributes:
        required_sources: Immutable set of source names that must all emit.
        seen_sources: Mutable set of sources that have emitted so far.
        completed: Whether all required sources have emitted.
        timeout_at: Optional timeout deadline (for future timeout support).
    """

    required_sources: frozenset[str]
    seen_sources: set[str] = field(default_factory=set)
    completed: bool = False
    timeout_at: float | None = None

    def add_source(self, source: str) -> bool:
        """Add a source to the seen set.

        Args:
            source: Name of the source that emitted an event.

        Returns:
            True if this source completes the coordination (all sources now seen),
            False otherwise.

        Note:
            If the source is not in required_sources, it is ignored.
            If the coordination is already completed, returns False.
        """
        if source in self.required_sources:
            self.seen_sources.add(source)
            self.completed = self.seen_sources == self.required_sources
            return self.completed
        return False

    def reset(self) -> None:
        """Clear state for reuse in the next coordination cycle.

        This resets both the seen_sources set and the completed flag,
        allowing the coordination to trigger again when all sources
        emit in the next cycle.
        """
        self.seen_sources.clear()
        self.completed = False


class CoordinationManager:
    """Manages all coordination states for a Flow instance.

    This class provides thread-safe management of multiple independent
    coordination points. Each coordination is identified by a unique
    binding_id (generated per reactor registration).

    Thread Safety:
        All public methods are thread-safe via a single lock.
        The lock is held for minimal duration (just state updates).
    """

    def __init__(self) -> None:
        """Initialize an empty coordination manager."""
        self._coordinations: dict[str, CoordinationState] = {}
        self._lock = Lock()

    def register_coordination(
        self, binding_id: str, sources: frozenset[str], timeout: float | None = None
    ) -> None:
        """Register a new coordination point for a reactor binding.

        Args:
            binding_id: Unique identifier for the reactor binding.
            sources: Set of source names that must all emit.
            timeout: Optional timeout in seconds (for future timeout support).

        Raises:
            ValueError: If sources is empty.
        """
        if not sources:
            raise ValueError("Coordination requires at least one source")

        with self._lock:
            self._coordinations[binding_id] = CoordinationState(
                required_sources=sources, timeout_at=timeout
            )

    def check_coordination(self, binding_id: str, source: str) -> bool:
        """Check if an event from a source completes the coordination.

        This is the core method called on each event to determine if
        the coordination point should trigger.

        Args:
            binding_id: Unique identifier for the reactor binding.
            source: Name of the source that emitted the event.

        Returns:
            True if this event completes the coordination (all required
            sources have now emitted), False otherwise.

        Note:
            - Returns False if binding_id is unknown (coordination not registered)
            - Returns False if coordination is already completed this cycle
            - Returns True only once per cycle (when last required source emits)

        Thread Safety:
            This method is thread-safe. Multiple threads can call it concurrently
            without causing race conditions.
        """
        with self._lock:
            if binding_id not in self._coordinations:
                return False
            coordination = self._coordinations[binding_id]
            if coordination.completed:
                return False  # Already completed this cycle
            return coordination.add_source(source)

    def reset_coordination(self, binding_id: str) -> None:
        """Manually reset a coordination point.

        This clears the seen sources and allows the coordination to
        trigger again in the next cycle. Typically used when auto-reset
        is disabled (reset_after=False).

        Args:
            binding_id: Unique identifier for the reactor binding.

        Note:
            Silently ignores unknown binding_id (no error raised).
        """
        with self._lock:
            if binding_id in self._coordinations:
                self._coordinations[binding_id].reset()

    def remove_coordination(self, binding_id: str) -> None:
        """Remove a coordination point (cleanup).

        This should be called when a reactor is unregistered to prevent
        memory leaks from accumulating coordination state.

        Args:
            binding_id: Unique identifier for the reactor binding.

        Note:
            Silently ignores unknown binding_id (no error raised).
        """
        with self._lock:
            self._coordinations.pop(binding_id, None)

    def get_coordination_state(self, binding_id: str) -> CoordinationState | None:
        """Get the current state of a coordination point (for debugging/observability).

        Args:
            binding_id: Unique identifier for the reactor binding.

        Returns:
            The CoordinationState for this binding, or None if not found.

        Warning:
            The returned state is a reference to the internal state.
            Do not modify it directly. This method is primarily for
            debugging and observability purposes.
        """
        with self._lock:
            return self._coordinations.get(binding_id)
