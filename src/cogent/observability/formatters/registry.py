"""
Formatter Registry - Pattern-based formatter lookup.

Manages a collection of formatters and routes events to the
appropriate formatter based on pattern matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cogent.observability.core.config import FormatConfig
    from cogent.observability.core.event import Event
    from cogent.observability.formatters.base import Formatter


class FormatterRegistry:
    """
    Registry for event formatters.

    Routes events to formatters based on pattern matching.
    Formatters are checked in order; first match wins.

    Example:
        ```python
        registry = FormatterRegistry()
        registry.register(AgentFormatter())
        registry.register(ToolFormatter())
        registry.register(DefaultFormatter())  # Fallback

        output = registry.format(event, config)
        ```
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._formatters: list[Formatter] = []

    def register(self, formatter: Formatter, priority: int = 0) -> None:
        """
        Register a formatter.

        Higher priority formatters are checked first.

        Args:
            formatter: Formatter to register
            priority: Priority (higher = checked first)
        """
        # Insert at correct position based on priority
        # For now, just append (FIFO order)
        self._formatters.append(formatter)

    def unregister(self, formatter: Formatter) -> None:
        """
        Remove a formatter.

        Args:
            formatter: Formatter to remove
        """
        if formatter in self._formatters:
            self._formatters.remove(formatter)

    def format(self, event: Event, config: FormatConfig) -> str | None:
        """
        Format an event using the first matching formatter.

        Args:
            event: Event to format
            config: Formatting configuration

        Returns:
            Formatted string, or None if no formatter matched
        """
        for formatter in self._formatters:
            if formatter.can_format(event):
                return formatter.format(event, config)
        return None

    def clear(self) -> None:
        """Remove all formatters."""
        self._formatters.clear()

    @property
    def formatter_count(self) -> int:
        """Number of registered formatters."""
        return len(self._formatters)

    @classmethod
    def with_defaults(cls) -> FormatterRegistry:
        """
        Create a registry with default console formatters.

        Returns:
            Registry pre-populated with standard formatters
        """
        from cogent.observability.formatters.console import (
            AgentFormatter,
            DefaultFormatter,
            LLMFormatter,
            StreamFormatter,
            TaskFormatter,
            ToolFormatter,
        )

        registry = cls()
        registry.register(AgentFormatter())
        registry.register(ToolFormatter())
        registry.register(TaskFormatter())
        registry.register(LLMFormatter())
        registry.register(StreamFormatter())
        registry.register(DefaultFormatter())  # Fallback last
        return registry
