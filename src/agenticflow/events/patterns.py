"""Event pattern matching utilities.

Provides pattern matching for event subscriptions:
- Exact match: "task.created"
- Wildcard: "task.*", "*.done"
- Regex: re.compile(r"agent\\..*\\.done")
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from agenticflow.events.event import Event

# Type aliases
EventPattern = str | re.Pattern[str]
EventCondition = Callable[[Event], bool]


def matches(pattern: EventPattern, event_name: str) -> bool:
    """Check if an event name matches a pattern.

    Args:
        pattern: Pattern to match (string with optional wildcards, or regex)
        event_name: Event name to check

    Returns:
        True if the event name matches the pattern

    Examples:
        ```python
        matches("task.created", "task.created")  # True
        matches("task.*", "task.created")        # True
        matches("*.done", "agent.done")          # True
        matches("agent.**", "agent.sub.done")    # True (** = multiple segments)
        ```
    """
    if isinstance(pattern, re.Pattern):
        return pattern.match(event_name) is not None

    # Exact match
    if "*" not in pattern:
        return pattern == event_name

    # Glob-style wildcards
    # ** matches multiple segments, * matches single segment
    regex = pattern
    regex = regex.replace(".", r"\.")  # Escape dots
    regex = regex.replace("**", "§§")  # Temp placeholder for **
    regex = regex.replace("*", r"[^.]*")  # * = single segment
    regex = regex.replace("§§", r".*")  # ** = any segments
    regex = f"^{regex}$"

    return re.match(regex, event_name) is not None


def matches_event(pattern: EventPattern, event: Event) -> bool:
    """Check if an event matches a pattern.

    Args:
        pattern: Pattern to match
        event: Event to check

    Returns:
        True if the event matches
    """
    return matches(pattern, event.name)


@dataclass(frozen=True, slots=True)
class EventMatcher:
    """Combines pattern matching with optional conditions.

    Example:
        ```python
        matcher = EventMatcher(
            pattern="agent.done",
            condition=lambda e: e.source == "researcher",
        )

        matcher.matches(event)  # Check if event matches
        ```
    """

    pattern: EventPattern
    """Event name pattern to match."""

    condition: EventCondition | None = None
    """Optional condition function for additional filtering."""

    def matches(self, event: Event) -> bool:
        """Check if an event matches this matcher."""
        # Check pattern first
        if not matches_event(self.pattern, event):
            return False

        # Check condition if present
        if self.condition is not None:
            try:
                return bool(self.condition(event))
            except Exception:
                return False

        return True


# -------------------------------------------------------------------------
# Common Conditions
# -------------------------------------------------------------------------


def from_source(source: str) -> EventCondition:
    """Create a condition that matches events from a specific source.

    Example:
        ```python
        flow.register(writer, on="agent.done", when=from_source("researcher"))
        ```
    """
    return lambda e: e.source == source


def after(source: str) -> EventCondition:
    """Alias for from_source - more readable for chaining.

    Example:
        ```python
        flow.register(writer, on="agent.done", when=after("researcher"))
        ```
    """
    return from_source(source)


def has_data(key: str, value: object = ...) -> EventCondition:
    """Create a condition that checks event data.

    Args:
        key: Data key to check
        value: Expected value (default: just check key exists)

    Example:
        ```python
        # Check key exists
        flow.register(agent, on="task.*", when=has_data("priority"))

        # Check specific value
        flow.register(agent, on="task.*", when=has_data("priority", "high"))
        ```
    """
    if value is ...:
        return lambda e: key in e.data
    return lambda e: e.data.get(key) == value


def all_of(*conditions: EventCondition) -> EventCondition:
    """Combine conditions with AND logic.

    Example:
        ```python
        flow.register(
            agent,
            on="agent.done",
            when=all_of(
                from_source("researcher"),
                has_data("success", True),
            ),
        )
        ```
    """
    def combined(event: Event) -> bool:
        return all(cond(event) for cond in conditions)
    return combined


def any_of(*conditions: EventCondition) -> EventCondition:
    """Combine conditions with OR logic.

    Example:
        ```python
        flow.register(
            agent,
            on="agent.done",
            when=any_of(
                from_source("researcher"),
                from_source("analyst"),
            ),
        )
        ```
    """
    def combined(event: Event) -> bool:
        return any(cond(event) for cond in conditions)
    return combined


def not_(condition: EventCondition) -> EventCondition:
    """Negate a condition.

    Example:
        ```python
        flow.register(agent, on="*", when=not_(from_source("system")))
        ```
    """
    return lambda e: not condition(e)
