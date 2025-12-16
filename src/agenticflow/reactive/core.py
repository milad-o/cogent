"""Core event-driven orchestration primitives.

This module provides the foundational building blocks for event-driven
agent orchestration: triggers, conditions, and reactions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from agenticflow.events.event import Event

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent


class ReactionType(Enum):
    """How an agent reacts when triggered."""

    RUN = "run"  # Run agent with event data as context
    EMIT = "emit"  # Agent emits new events
    DELEGATE = "delegate"  # Delegate to sub-agents
    TRANSFORM = "transform"  # Transform event and re-emit


# Type aliases
TriggerCondition = Callable[[Event], bool]
EventPattern = str | re.Pattern[str] | Any


def _event_name(event: Any) -> str:
    """Get the effective event name for both core and legacy observability events."""
    # Core events
    if hasattr(event, "name"):
        return str(getattr(event, "name"))

    # Legacy observability events
    data = getattr(event, "data", None) or {}
    if isinstance(data, dict) and data.get("event_name"):
        return str(data.get("event_name"))

    evt_type = getattr(event, "type", None)
    if evt_type is not None and hasattr(evt_type, "value"):
        return str(getattr(evt_type, "value"))

    return ""


@dataclass(frozen=True, slots=True, kw_only=True)
class Trigger:
    """
    Defines when an agent should be activated.

    A trigger consists of:
    - An event pattern to match (EventType, string, or regex)
    - Optional condition function for filtering
    - Reaction type (what to do when triggered)
    - Optional event to emit after completion

    Example:
        ```python
        # Simple trigger on event type
        Trigger(on=EventType.TASK_CREATED)

        # Trigger with condition
        Trigger(
            on="research.requested",
            condition=lambda e: e.data.get("priority") == "high",
        )

        # Trigger that emits completion event
        Trigger(
            on="task.assigned",
            emits="task.completed",
        )
        ```
    """

    on: EventPattern
    """Event pattern to match. Can be EventType, string, or regex pattern."""

    condition: TriggerCondition | None = None
    """Optional filter function. Returns True if agent should activate."""

    reaction: ReactionType = ReactionType.RUN
    """How the agent should react when triggered."""

    emits: str | None = None
    """Event type to emit after agent completes (for chaining)."""

    priority: int = 0
    """Higher priority triggers are evaluated first."""

    def matches(self, event: Event) -> bool:
        """
        Check if this trigger matches an event.

        Args:
            event: The event to check

        Returns:
            True if the trigger matches and condition passes
        """
        # Check pattern match first
        if not self._matches_pattern(event):
            return False

        # Check condition if present
        if self.condition is not None:
            try:
                return bool(self.condition(event))
            except Exception:
                return False

        return True

    def _matches_pattern(self, event: Event) -> bool:
        """Check if event matches the pattern."""
        event_name = _event_name(event)

        # Regex pattern
        if isinstance(self.on, re.Pattern):
            return self.on.match(event_name) is not None

        # String pattern (exact or glob)
        if isinstance(self.on, str):
            if "*" in self.on:
                regex = self.on.replace(".", r"\.").replace("*", ".*")
                return re.match(regex, event_name) is not None
            return event_name == self.on

        # Enum-like patterns (legacy): match by `.value` if present
        if hasattr(self.on, "value"):
            return event_name == str(getattr(self.on, "value"))

        return False


@dataclass(frozen=True, slots=True, kw_only=True)
class Reaction:
    """
    Describes how an agent reacted to a trigger.

    Used for audit logging and debugging the event flow.
    """

    agent_name: str
    """Name of the agent that reacted."""

    trigger: Trigger
    """The trigger that activated the agent."""

    event: Event
    """The event that triggered the reaction."""

    output: str | None = None
    """Agent output if reaction completed."""

    emitted_events: list[str] = field(default_factory=list)
    """Events emitted as a result of this reaction."""

    error: str | None = None
    """Error message if reaction failed."""


class TriggerBuilder:
    """
    Fluent builder for creating triggers.

    Example:
        ```python
        trigger = (
            react_to("task.created")
            .when(lambda e: e.data.get("type") == "research")
            .emits("research.started")
            .with_priority(10)
            .build()
        )
        ```
    """

    def __init__(self, pattern: EventPattern) -> None:
        self._pattern = pattern
        self._condition: TriggerCondition | None = None
        self._reaction = ReactionType.RUN
        self._emits: str | None = None
        self._priority = 0

    def when(self, condition: TriggerCondition) -> TriggerBuilder:
        """Add a condition to the trigger."""
        self._condition = condition
        return self

    def emits(self, event_type: str) -> TriggerBuilder:
        """Specify event to emit after agent completes."""
        self._emits = event_type
        return self

    def with_priority(self, priority: int) -> TriggerBuilder:
        """Set trigger priority (higher = evaluated first)."""
        self._priority = priority
        return self

    def with_reaction(self, reaction: ReactionType) -> TriggerBuilder:
        """Set the reaction type."""
        self._reaction = reaction
        return self

    def build(self) -> Trigger:
        """Build the trigger."""
        return Trigger(
            on=self._pattern,
            condition=self._condition,
            reaction=self._reaction,
            emits=self._emits,
            priority=self._priority,
        )

    # Allow direct use without .build()
    def __iter__(self):
        """Allow unpacking: triggers=[react_to("event")]."""
        yield self.build()


def react_to(pattern: EventPattern) -> TriggerBuilder:
    """
    Create a trigger builder for the given event pattern.

    This is the primary API for defining triggers fluently.
    Agents "react to" events - hence the name.

    Args:
        pattern: Event type, string pattern, or regex to match

    Returns:
        TriggerBuilder for fluent configuration

    Example:
        ```python
        # Basic trigger
        react_to(EventType.TASK_CREATED)

        # With condition
        react_to("task.*").when(lambda e: e.data.get("priority") == "high")

        # Full chain
        react_to("research.requested").emits("research.completed").with_priority(5)
        ```
    """
    return TriggerBuilder(pattern)


# Backward compatibility alias
on = react_to


def when(condition: TriggerCondition) -> TriggerCondition:
    """
    Helper for creating named conditions.

    Args:
        condition: The condition function

    Returns:
        The same condition (for documentation/naming purposes)

    Example:
        ```python
        is_high_priority = when(lambda e: e.data.get("priority") == "high")

        trigger = react_to("task.created").when(is_high_priority)
        ```
    """
    return condition


@dataclass(kw_only=True)
class AgentTriggerConfig:
    """
    Configuration for an agent's event triggers.

    This is attached to agents to define their reactive behavior.
    """

    triggers: list[Trigger] = field(default_factory=list)
    """Triggers that activate this agent."""

    auto_emit_completion: bool = True
    """Automatically emit {agent_name}.completed event after run."""

    emit_on_error: bool = True
    """Emit {agent_name}.error event on failure."""

    def add_trigger(self, trigger: Trigger | TriggerBuilder) -> None:
        """Add a trigger to this agent."""
        if isinstance(trigger, TriggerBuilder):
            trigger = trigger.build()
        self.triggers.append(trigger)

    def get_matching_triggers(self, event: Event) -> list[Trigger]:
        """Get all triggers that match an event."""
        matching = [t for t in self.triggers if t.matches(event)]
        return sorted(matching, key=lambda t: -t.priority)
