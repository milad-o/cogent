"""
Observer - Main entry point for observability.

This is a minimal, focused class that coordinates:
- Event publishing via EventBus
- Event formatting via FormatterRegistry
- Output via Sinks
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, TextIO

from cogent.observability.core.bus import EventBus
from cogent.observability.core.config import (
    FormatConfig,
    Level,
    ObserverConfig,
    get_preset,
)
from cogent.observability.core.event import Event, create_event
from cogent.observability.formatters.registry import FormatterRegistry
from cogent.observability.sinks.console import ConsoleSink

if TYPE_CHECKING:
    from cogent.observability.formatters.base import Formatter
    from cogent.observability.sinks.base import Sink


# Event type -> Level mapping
EVENT_LEVELS: dict[str, Level] = {
    # Result level (Level.RESULT = 1)
    "task.completed": Level.RESULT,
    "task.failed": Level.RESULT,
    "agent.responded": Level.RESULT,
    "agent.error": Level.RESULT,
    # Progress level (Level.PROGRESS = 2)
    "task.started": Level.PROGRESS,
    "agent.invoked": Level.PROGRESS,
    "agent.thinking": Level.PROGRESS,
    "tool.called": Level.PROGRESS,
    "tool.result": Level.PROGRESS,
    "tool.error": Level.PROGRESS,
    "stream.start": Level.PROGRESS,
    "stream.end": Level.PROGRESS,
    "stream.error": Level.PROGRESS,
    # Debug level (Level.DEBUG = 4)
    "llm.request": Level.DEBUG,
    "llm.response": Level.DEBUG,
    "agent.reasoning": Level.DEBUG,
    "agent.acting": Level.DEBUG,
    # Trace level (Level.TRACE = 5)
    "stream.token": Level.TRACE,
}


def _get_event_level(event_type: str) -> Level:
    """Get the log level for an event type."""
    if event_type in EVENT_LEVELS:
        return EVENT_LEVELS[event_type]
    # Default based on category
    category = event_type.split(".")[0] if "." in event_type else event_type
    if category in ("stream",):
        return Level.TRACE
    return Level.PROGRESS


class Observer:
    """
    Main observability interface.

    Coordinates event publishing, formatting, and output.

    Example:
        ```python
        # Simple usage with level preset
        observer = Observer(level="progress")
        observer.emit("agent.invoked", agent_name="MyAgent")

        # Custom configuration
        config = ObserverConfig(level=Level.DEBUG, format=FormatConfig(use_colors=False))
        observer = Observer(config=config)

        # Subscribe to events
        observer.on("tool.*", lambda e: print(f"Tool event: {e.type}"))
        ```
    """

    def __init__(
        self,
        config: ObserverConfig | None = None,
        *,
        level: str | Level | None = None,
        stream: TextIO | None = None,
    ) -> None:
        """
        Initialize observer.

        Args:
            config: Full configuration object
            level: Level preset name or Level enum (shortcut for config)
            stream: Output stream (shortcut for config)
        """
        # Build config from shortcuts or use provided
        if config is None:
            if isinstance(level, str):
                config = get_preset(level)
            elif isinstance(level, Level):
                config = ObserverConfig(level=level)
            else:
                config = get_preset("progress")  # Default

        # Override stream if provided
        if stream is not None:
            config = ObserverConfig(
                level=config.level,
                format=config.format,
                stream=stream,
                include=config.include,
                exclude=config.exclude,
            )

        self._config = config
        self._bus = EventBus()
        self._formatter_registry = FormatterRegistry.with_defaults()

        # Setup default sink
        self._sinks: list[Sink] = []
        self._default_sink = ConsoleSink(stream=config.stream or sys.stderr)
        self._sinks.append(self._default_sink)

        # State
        self._enabled = config.level != Level.OFF
        self._streaming_mode = False

    # === Core API ===

    def emit(self, event_type: str, **data: object) -> Event | None:
        """
        Emit an event.

        Args:
            event_type: Event type (e.g., "agent.invoked")
            **data: Event data as keyword arguments

        Returns:
            The created Event, or None if filtered/disabled
        """
        if not self._enabled:
            return None

        # Check level
        event_level = _get_event_level(event_type)
        if event_level > self._config.level:
            return None

        # Check include/exclude filters
        if not self._should_include(event_type):
            return None

        # Create event
        event = create_event(event_type, **data)

        # Publish to bus (for subscribers)
        self._bus.publish(event)

        # Format and output
        self._output(event)

        return event

    def emit_event(self, event: Event) -> None:
        """
        Emit a pre-created event.

        Args:
            event: Event to emit
        """
        if not self._enabled:
            return

        event_level = _get_event_level(event.type)
        if event_level > self._config.level:
            return

        if not self._should_include(event.type):
            return

        self._bus.publish(event)
        self._output(event)

    def on(self, pattern: str, handler: Callable[[Event], None]) -> Callable[[], None]:
        """
        Subscribe to events matching a pattern.

        Args:
            pattern: Glob pattern (e.g., "agent.*", "tool.called")
            handler: Function called with matching events

        Returns:
            Unsubscribe function
        """
        return self._bus.subscribe(pattern, handler)

    def on_all(self, handler: Callable[[Event], None]) -> Callable[[], None]:
        """
        Subscribe to all events.

        Args:
            handler: Function called with every event

        Returns:
            Unsubscribe function
        """
        return self._bus.subscribe_all(handler)

    # === Configuration ===

    @property
    def level(self) -> Level:
        """Current log level."""
        return self._config.level

    @level.setter
    def level(self, value: Level | str) -> None:
        """Set log level."""
        if isinstance(value, str):
            value = Level.from_string(value)
        self._config = ObserverConfig(
            level=value,
            format=self._config.format,
            stream=self._config.stream,
            include=self._config.include,
            exclude=self._config.exclude,
        )
        self._enabled = value != Level.OFF

    @property
    def enabled(self) -> bool:
        """Whether observer is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable observer."""
        self._enabled = value

    @property
    def config(self) -> ObserverConfig:
        """Current configuration."""
        return self._config

    # === Sinks ===

    def add_sink(self, sink: Sink) -> None:
        """
        Add an output sink.

        Args:
            sink: Sink to add
        """
        self._sinks.append(sink)

    def remove_sink(self, sink: Sink) -> None:
        """
        Remove an output sink.

        Args:
            sink: Sink to remove
        """
        if sink in self._sinks:
            self._sinks.remove(sink)

    # === Formatters ===

    def add_formatter(self, formatter: Formatter) -> None:
        """
        Add a custom formatter.

        Args:
            formatter: Formatter to add (checked before defaults)
        """
        # Insert at beginning so custom formatters take priority
        self._formatter_registry._formatters.insert(0, formatter)

    # === Streaming ===

    def stream_token(self, token: str) -> None:
        """
        Output a streaming token directly.

        Bypasses normal formatting for real-time token output.

        Args:
            token: Token string to output
        """
        if not self._enabled:
            return
        if self._config.level < Level.TRACE:
            return

        # Write directly to console sink
        if hasattr(self._default_sink, "write_raw"):
            self._default_sink.write_raw(token)

    def start_streaming(self) -> None:
        """Enter streaming mode."""
        self._streaming_mode = True

    def end_streaming(self) -> None:
        """Exit streaming mode, flush output."""
        self._streaming_mode = False
        self._default_sink.write_raw("\n")
        self.flush()

    # === Lifecycle ===

    def flush(self) -> None:
        """Flush all sinks."""
        for sink in self._sinks:
            sink.flush()

    def close(self) -> None:
        """Close all sinks."""
        for sink in self._sinks:
            sink.close()
        self._sinks.clear()

    def __enter__(self) -> Observer:
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Context manager exit."""
        self.close()

    # === Statistics ===

    def summary(self) -> str:
        """
        Get a summary of observed events.

        Returns:
            Human-readable summary string
        """
        stats = self._bus.stats()
        lines = [
            f"Events: {stats['total_events']}",
        ]
        if stats.get("by_category"):
            for cat, count in sorted(stats["by_category"].items()):
                lines.append(f"  {cat}: {count}")
        return "\n".join(lines)

    def stats(self) -> dict[str, object]:
        """
        Get event statistics.

        Returns:
            Dictionary with event counts by type/category
        """
        return self._bus.stats()

    # === Internal ===

    def _should_include(self, event_type: str) -> bool:
        """Check if event passes include/exclude filters."""
        # If include list specified, event must match at least one
        if self._config.include:
            event = create_event(event_type, {})
            if not any(event.matches(p) for p in self._config.include):
                return False

        # If exclude list specified, event must not match any
        if self._config.exclude:
            event = create_event(event_type, {})
            if any(event.matches(p) for p in self._config.exclude):
                return False

        return True

    def _output(self, event: Event) -> None:
        """Format and output an event."""
        output = self._formatter_registry.format(event, self._config.format)
        if output:
            for sink in self._sinks:
                sink.write(output)


# === Factory Functions ===


def create_observer(
    level: str | Level = "progress",
    *,
    stream: TextIO | None = None,
    use_colors: bool = True,
    show_timestamps: bool = False,
) -> Observer:
    """
    Create an observer with common settings.

    Args:
        level: Log level preset or Level enum
        stream: Output stream
        use_colors: Enable colored output
        show_timestamps: Show timestamps in output

    Returns:
        Configured Observer instance
    """
    if isinstance(level, str):
        config = get_preset(level)
    else:
        config = ObserverConfig(level=level)

    # Override format settings
    config = ObserverConfig(
        level=config.level,
        format=FormatConfig(
            use_colors=use_colors,
            show_timestamps=show_timestamps,
            show_duration=config.format.show_duration,
            show_trace_ids=config.format.show_trace_ids,
            truncate=config.format.truncate,
            indent=config.format.indent,
        ),
        stream=stream,
        include=config.include,
        exclude=config.exclude,
    )

    return Observer(config=config)
