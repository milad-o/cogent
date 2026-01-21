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


# -------------------------------------------------------------------------
# Source-Based Filtering
# -------------------------------------------------------------------------


class SourceFilter:
    """Composable event source filter with boolean operators.
    
    Supports composition via `&` (AND), `|` (OR), and `~` (NOT) operators.
    
    Example:
        ```python
        # Combine filters
        filter1 = from_source("agent1")
        filter2 = from_source("agent2")
        combined = filter1 | filter2  # Match either
        
        # Negate
        not_system = ~from_source("system")
        
        # Complex conditions
        high_priority_from_api = (
            from_source("api") & 
            (lambda e: e.data.get("priority") == "high")
        )
        ```
    """
    
    def __init__(self, predicate: Callable[[Event], bool]) -> None:
        """Initialize source filter.
        
        Args:
            predicate: Function that takes Event and returns bool.
        """
        self._predicate = predicate
    
    def __call__(self, event: Event) -> bool:
        """Evaluate filter on event.
        
        Args:
            event: Event to test.
            
        Returns:
            True if event passes filter, False otherwise.
        """
        return self._predicate(event)
    
    def __and__(self, other: Callable[[Event], bool]) -> SourceFilter:
        """Combine filters with AND logic.
        
        Args:
            other: Another filter or callable.
            
        Returns:
            New filter that passes only if both filters pass.
        """
        return SourceFilter(lambda e: self(e) and other(e))
    
    def __or__(self, other: Callable[[Event], bool]) -> SourceFilter:
        """Combine filters with OR logic.
        
        Args:
            other: Another filter or callable.
            
        Returns:
            New filter that passes if either filter passes.
        """
        return SourceFilter(lambda e: self(e) or other(e))
    
    def __invert__(self) -> SourceFilter:
        """Negate the filter (NOT logic).
        
        Returns:
            New filter that passes when this filter fails.
        """
        return SourceFilter(lambda e: not self(e))
    
    def __repr__(self) -> str:
        return f"SourceFilter({self._predicate!r})"


def from_source(source: str | list[str]) -> SourceFilter:
    """Filter events from specific source(s).
    
    Supports exact matches, lists (OR logic), and wildcard patterns.
    
    Args:
        source: Source name(s) to match. Can be:
            - Single string: Exact match or wildcard pattern
            - List of strings: Match any (OR logic)
    
    Returns:
        SourceFilter that matches the specified source(s).
    
    Examples:
        ```python
        # Exact match
        from_source("researcher")
        
        # Multiple sources (OR)
        from_source(["agent1", "agent2", "agent3"])
        
        # Wildcard patterns
        from_source("agent*")  # agent1, agent2, agentX, etc.
        from_source("api_*")   # api_webhook, api_rest, etc.
        from_source("*_dev")   # test_dev, staging_dev, etc.
        ```
    """
    from fnmatch import fnmatch
    
    if isinstance(source, str):
        # Wildcard pattern
        if "*" in source or "?" in source:
            return SourceFilter(lambda e: fnmatch(e.source, source))
        # Exact match
        return SourceFilter(lambda e: e.source == source)
    else:
        # List - OR logic
        sources_set = set(source)
        return SourceFilter(lambda e: e.source in sources_set)


def not_from_source(source: str | list[str]) -> SourceFilter:
    """Exclude events from specific source(s).
    
    Args:
        source: Source name(s) to exclude.
    
    Returns:
        SourceFilter that rejects the specified source(s).
    
    Examples:
        ```python
        # Exclude system events
        not_from_source("system")
        
        # Exclude multiple sources
        not_from_source(["test", "debug", "dev"])
        
        # Exclude with wildcards
        not_from_source("*_internal")
        ```
    """
    return ~from_source(source)


def any_source(sources: list[str]) -> SourceFilter:
    """Match any of the given sources (OR logic).
    
    Convenience alias for `from_source(list)`.
    
    Args:
        sources: List of source names to match.
    
    Returns:
        SourceFilter that matches any of the sources.
    
    Example:
        ```python
        any_source(["agent1", "agent2", "agent3"])
        ```
    """
    return from_source(sources)


def matching_sources(pattern: str) -> SourceFilter:
    """Match sources using wildcard pattern.
    
    Convenience alias for `from_source(pattern)` with explicit pattern intent.
    
    Args:
        pattern: Wildcard pattern (*, ?).
    
    Returns:
        SourceFilter that matches the pattern.
    
    Example:
        ```python
        matching_sources("agent*")
        matching_sources("api_*_prod")
        ```
    """
    return from_source(pattern)
