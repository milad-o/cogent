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
from typing import Any, TYPE_CHECKING

from agenticflow.events.event import Event
from agenticflow.flow.state import CoordinationState

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


def from_source(source: str | list[str], flow: Any = None) -> SourceFilter:
    """Filter events from specific source(s).
    
    Supports exact matches, lists (OR logic), wildcard patterns, and group references.
    
    Args:
        source: Source name(s) to match. Can be:
            - Single string: Exact match, wildcard pattern, or :group reference
            - List of strings: Match any (OR logic)
        flow: Flow instance (required for :group references)
    
    Returns:
        SourceFilter that matches the specified source(s).
    
    Raises:
        ValueError: If :group syntax used without flow parameter
    
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
        
        # Group reference (requires flow)
        from_source(":analysts", flow=flow)  # Uses flow.get_source_group("analysts")
        from_source(":agents", flow=flow)    # Built-in :agents group
        ```
    """
    from fnmatch import fnmatch
    
    # Handle group reference
    if isinstance(source, str) and source.startswith(":"):
        if flow is None:
            raise ValueError(
                f"Flow instance required for group reference '{source}'. "
                "Group references can only be used with 'after' parameter or pattern syntax."
            )
        group_name = source[1:]  # Remove : prefix
        sources = flow.get_source_group(group_name)
        if not sources:
            # Empty group or nonexistent group - match nothing
            return SourceFilter(lambda e: False)
        # Convert to OR filter for all sources in group
        return SourceFilter(lambda e: e.source in sources)
    
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


# -------------------------------------------------------------------------
# Stateful Coordination Patterns
# -------------------------------------------------------------------------


class StatefulSourceFilter(SourceFilter):
    """Source filter that tracks state across multiple events (coordination pattern).
    
    Unlike regular SourceFilter which is stateless, this filter maintains internal state
    to implement coordination patterns like "wait for ALL sources to emit".
    
    Each instance has its own independent CoordinationState that tracks which sources
    have emitted. No external manager or initialization required.
    
    Example:
        ```python
        # Simple and clean - just create and use
        coordination = all_sources(["worker_0", "worker_1", "worker_2"])
        
        # Use directly as event filter
        if coordination(event):
            print("All workers complete!")
        
        # Or with Flow.register()
        flow.register(reducer, on="task.done", when=coordination)
        ```
    """
    
    def __init__(
        self,
        sources: frozenset[str],
        timeout: float | None = None
    ) -> None:
        """Initialize stateful source filter with internal state.
        
        Args:
            sources: Set of source names to coordinate.
            timeout: Optional timeout in seconds (future feature).
        """
        self._sources = sources
        self._timeout = timeout
        
        # Create internal coordination state
        self._state = CoordinationState(
            required_sources=sources,
            timeout_at=None  # Timeout support is future work
        )
        
        # Predicate that checks coordination completion
        super().__init__(self._check_coordination)
    
    def _check_coordination(self, event: Event) -> bool:
        """Check if this event completes the coordination.
        
        Args:
            event: Event to check.
        
        Returns:
            True if event completes coordination (all sources seen), False otherwise.
        
        Note:
            Automatically resets after completion to allow repeated coordination cycles.
        """
        # Add source and check if coordination complete
        completed = self._state.add_source(event.source)
        
        # Always auto-reset after completion
        if completed:
            self._state.reset()
        
        return completed
    
    def reset(self) -> None:
        """Manually reset coordination state.
        
        Use this when reset_after=False to manually reset the coordination
        and allow it to trigger again.
        
        Example:
            ```python
            coordination = all_sources(["a", "b"], reset_after=False)
            # ... coordination triggers once ...
            coordination.reset()  # Allow it to trigger again
            ```
        """
        self._state.reset()
    
    @property
    def completed(self) -> bool:
        """Check if coordination is currently completed."""
        return self._state.completed
    
    @property
    def seen_sources(self) -> set[str]:
        """Get set of sources that have emitted so far."""
        return self._state.seen_sources.copy()
    
    @property
    def remaining_sources(self) -> set[str]:
        """Get set of sources that haven't emitted yet."""
        return set(self._state.required_sources - self._state.seen_sources)
    
    def once(self) -> SourceFilter:
        """Return a one-time version of this filter.
        
        The returned filter will only trigger once, then always return False.
        Perfect for deployment gates or one-time coordination events.
        
        Returns:
            OneTimeFilter that wraps this coordination filter.
        
        Example:
            ```python
            # Repeated coordination (default)
            batch_coord = all_sources(["w1", "w2"])
            # Triggers every time all workers complete
            
            # One-time coordination
            deploy_gate = all_sources(["build", "test", "security"]).once()
            # Triggers once when all checks pass, then never again
            ```
        """
        return OneTimeFilter(self)
    
    def __repr__(self) -> str:
        sources_str = ", ".join(sorted(self._sources))
        return f"StatefulSourceFilter(sources=[{sources_str}])"


class OneTimeFilter(SourceFilter):
    """Wrapper that makes any filter trigger only once.
    
    After the wrapped filter returns True once, this filter will always
    return False, regardless of subsequent events.
    
    Example:
        ```python
        # One-time coordination
        gate = all_sources(["a", "b", "c"]).once()
        
        # First time all sources emit: True
        # All subsequent times: False (even if coordination completes again)
        ```
    """
    
    def __init__(self, wrapped: SourceFilter) -> None:
        """Initialize one-time filter.
        
        Args:
            wrapped: The filter to wrap (typically a StatefulSourceFilter).
        """
        self._wrapped = wrapped
        self._triggered = False
        super().__init__(self._check_once)
    
    def _check_once(self, event: Event) -> bool:
        """Check wrapped filter, but only trigger once.
        
        Args:
            event: Event to check.
        
        Returns:
            True only the first time wrapped filter returns True.
        """
        if self._triggered:
            return False
        
        if self._wrapped(event):
            self._triggered = True
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset to allow triggering again.
        
        This also resets the wrapped filter if it has a reset() method.
        """
        self._triggered = False
        if hasattr(self._wrapped, 'reset'):
            self._wrapped.reset()
    
    @property
    def triggered(self) -> bool:
        """Check if this filter has already triggered."""
        return self._triggered
    
    def __repr__(self) -> str:
        return f"OneTimeFilter({self._wrapped!r})"


def all_sources(
    sources: list[str],
    *,
    timeout: float | None = None
) -> StatefulSourceFilter:
    """Create a coordination point that waits for ALL specified sources.
    
    This is a stateful filter that tracks which sources have emitted events
    and only triggers when ALL required sources have emitted at least once.
    
    **Automatically resets** after each completion, allowing repeated coordination
    cycles. For one-time gates, use simple application logic (boolean flag).
    
    Each filter instance maintains its own independent state - no external
    manager or initialization required. Just create and use!
    
    Common use cases:
    - Map-reduce: Wait for all workers to complete before aggregating
    - Fan-in: Wait for all parallel tasks to finish before proceeding
    - Multi-reviewer approval: Wait for all reviewers to approve
    - Checkpoint: Wait for all services to reach a checkpoint
    
    Args:
        sources: List of source names to wait for (e.g., ["worker_0", "worker_1"]).
        timeout: Optional timeout in seconds (future feature, currently ignored).
    
    Returns:
        StatefulSourceFilter that auto-resets after each completion.
    
    Examples:
        ```python
        # Simple - just create and use
        coordination = all_sources(["worker_0", "worker_1", "worker_2"])
        
        # Use directly as event filter
        event = Event(name="task.done", source="worker_0")
        if coordination(event):
            print("All workers complete!")
        
        # Batch processing (auto-resets every cycle)
        batch_coord = all_sources(["w1", "w2"])
        for batch in batches:
            if batch_coord(event):
                process_batch()  # Happens every cycle
        
        # One-time gate (deployment, approval, etc.)
        deploy_gate = all_sources(["build", "test", "security"]).once()
        if deploy_gate(event):
            deploy()  # Only happens once, automatically
        
        # With Flow.register()
        flow.register(
            reducer,
            on="task.done",
            when=all_sources(["worker_0", "worker_1", "worker_2"])
        )
        
        # Combine with other filters
        coordination = all_sources(["w1", "w2"]) & (lambda e: e.data.get("success"))
        
        # Check state (introspection)
        print(f"Completed: {coordination.completed}")
        print(f"Seen: {coordination.seen_sources}")
        print(f"Remaining: {coordination.remaining_sources}")
        ```
    
    Notes:
        - Each instance has its own independent state.
        - Automatically resets after completion for repeated cycles.
        - Duplicate events from same source are ignored (counted only once).
        - Events from sources not in the list are ignored.
    
    See Also:
        - `from_source()`: Stateless filtering by source
        - `StatefulSourceFilter.once()`: One-time coordination gate
        - `StatefulSourceFilter.reset()`: Manual reset (rarely needed)
    """
    if not sources:
        raise ValueError("all_sources() requires at least one source")
    
    return StatefulSourceFilter(
        sources=frozenset(sources),
        timeout=timeout
    )

