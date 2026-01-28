"""
Formatter Protocol - Interface for event formatters.

Formatters transform Event objects into formatted strings for output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cogent.observability.core.config import FormatConfig
    from cogent.observability.core.event import Event


class Formatter(Protocol):
    """
    Protocol for event formatters.

    Implement this protocol to create custom formatters.

    Example:
        ```python
        class MyFormatter:
            def can_format(self, event: Event) -> bool:
                return event.type.startswith("my_app.")

            def format(self, event: Event, config: FormatConfig) -> str | None:
                return f"[MY APP] {event.type}: {event.data}"
        ```
    """

    def can_format(self, event: Event) -> bool:
        """
        Check if this formatter can handle the given event.

        Args:
            event: Event to check

        Returns:
            True if this formatter should handle the event
        """
        ...

    def format(self, event: Event, config: FormatConfig) -> str | None:
        """
        Format an event to a string.

        Args:
            event: Event to format
            config: Formatting configuration

        Returns:
            Formatted string, or None to skip output
        """
        ...


class BaseFormatter:
    """
    Base class for formatters with common utilities.

    Provides helper methods for formatting. Subclasses should
    implement `patterns` and `format`.
    """

    patterns: list[str] = []
    """Glob patterns this formatter handles (e.g., ["agent.*"])."""

    def can_format(self, event: Event) -> bool:
        """Check if event matches any of our patterns."""
        return any(event.matches(p) for p in self.patterns)

    def format(self, event: Event, config: FormatConfig) -> str | None:
        """Format the event. Subclasses must implement this."""
        raise NotImplementedError

    # === Utility Methods ===

    @staticmethod
    def truncate(text: str, max_len: int) -> str:
        """Truncate text to max length, adding ellipsis if needed."""
        if max_len <= 0 or len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    @staticmethod
    def format_duration(ms: float) -> str:
        """Format duration in human-readable form."""
        if ms < 1000:
            return f"{ms:.0f}ms"
        elif ms < 60000:
            return f"{ms / 1000:.1f}s"
        else:
            mins = int(ms / 60000)
            secs = (ms % 60000) / 1000
            return f"{mins}m {secs:.0f}s"

    @staticmethod
    def format_name(name: str) -> str:
        """Format agent/tool name with brackets."""
        return f"[{name}]"
