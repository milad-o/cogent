"""
Observability v2 - Extensible, composable observability for Cogent.

A clean-slate redesign focused on:
- Extensibility: Add custom events, formatters, sinks without modifying core
- Composability: Mix and match components
- Simplicity: <2000 total lines vs 6500+ in v1
- Single Agent First: Optimized for the common case

Quick Start:
    ```python
    from cogent.observability import Observer

    # Use string-based levels (same API as v1)
    observer = Observer(level="verbose")

    # Attach to agent
    agent = Agent(name="Assistant", model=model, observer=observer)
    await agent.run("Hello!")

    # Get summary
    print(observer.summary())
    ```

Custom Events:
    ```python
    from cogent.observability import Observer, Event, create_event

    observer = Observer(level="debug")

    # Emit custom events
    observer.observe(create_event(
        "my_app.order.placed",
        order_id="12345",
        customer="john@example.com",
    ))
    ```

Custom Formatter:
    ```python
    from cogent.observability import Observer, Formatter, Event

    class MyFormatter:
        def can_format(self, event: Event) -> bool:
            return event.type.startswith("my_app.")

        def format(self, event: Event, config) -> str | None:
            return f"[MY APP] {event.type}: {event.data}"

    observer = Observer(level="debug")
    observer.add_formatter(MyFormatter())
    ```
"""

from cogent.observability.core.bus import EventBus
from cogent.observability.core.config import (
    FormatConfig,
    Level,
    ObserverConfig,
    get_preset,
)
from cogent.observability.core.event import Event, EventTypes, create_event
from cogent.observability.formatters.base import BaseFormatter, Formatter
from cogent.observability.formatters.console import (
    AgentFormatter,
    DefaultFormatter,
    StreamFormatter,
    Styler,
    TaskFormatter,
    ToolFormatter,
)
from cogent.observability.formatters.json import JSONFormatter
from cogent.observability.formatters.registry import FormatterRegistry
from cogent.observability.handlers import (
    ConsoleEventHandler,
    FileEventHandler,
    FilteringEventHandler,
    MetricsEventHandler,
)
from cogent.observability.logger import (
    LogEntry,
    LogLevel,
    ObservabilityLogger,
)
from cogent.observability.metrics.collector import (
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
)
from cogent.observability.observer import Observer, create_observer

# ============================================================================
# Backward Compatibility Imports (v1 exports)
# These are kept for backward compatibility during migration.
# ============================================================================
from cogent.observability.progress import (
    Colors,
    OutputConfig,
    OutputFormat,
    ProgressEvent,
    ProgressStyle,
    ProgressTracker,
    Symbols,
    Verbosity,
    configure_output,
    create_executor_callback,
    create_on_step_callback,
    render_dag_ascii,
)
from cogent.observability.sinks.base import BaseSink, Sink
from cogent.observability.sinks.callback import CallbackSink
from cogent.observability.sinks.console import ConsoleSink
from cogent.observability.sinks.file import FileSink
from cogent.observability.tracing.span import Span, SpanContext, SpanStatus
from cogent.observability.tracing.tracer import Tracer, current_span, get_tracer

# Compatibility aliases for v1 types
Channel = Level  # v1 Channel is now Level
ObservabilityLevel = Level  # v1 ObservabilityLevel is now Level


# v1 items that don't exist in v2 - create placeholders
class Timer:
    """Placeholder for v1 Timer - use MetricsCollector.histogram() instead."""

    pass


class Dashboard:
    """Placeholder for v1 Dashboard - not yet migrated."""

    pass


class DashboardConfig:
    """Placeholder for v1 DashboardConfig - not yet migrated."""

    pass


class AgentInspector:
    """Placeholder for v1 AgentInspector - use Observer.summary() instead."""

    pass


class EventInspector:
    """Placeholder for v1 EventInspector - use Observer.get_events() instead."""

    pass


class SystemInspector:
    """Placeholder for v1 SystemInspector - not yet migrated."""

    pass


class TaskInspector:
    """Placeholder for v1 TaskInspector - not yet migrated."""

    pass


# SpanKind enum for v1 compat - define inline with values
class SpanKind:
    """Span kind for tracing."""

    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


__all__ = [
    # Core
    "Event",
    "EventTypes",
    "create_event",
    "EventBus",
    "Level",
    "ObserverConfig",
    "FormatConfig",
    "get_preset",
    # Formatter Protocol & Base
    "Formatter",
    "BaseFormatter",
    "FormatterRegistry",
    # Built-in Formatters
    "AgentFormatter",
    "ToolFormatter",
    "TaskFormatter",
    "StreamFormatter",
    "DefaultFormatter",
    "JSONFormatter",
    "Styler",
    # Sink Protocol & Base
    "Sink",
    "BaseSink",
    # Built-in Sinks
    "ConsoleSink",
    "FileSink",
    "CallbackSink",
    # Main Observer
    "Observer",
    "create_observer",
    # Tracing
    "Span",
    "SpanContext",
    "SpanStatus",
    "SpanKind",
    "Tracer",
    "get_tracer",
    "current_span",
    # Metrics
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    # ========================================
    # Backward Compatibility (v1 exports)
    # ========================================
    "Colors",
    "OutputConfig",
    "OutputFormat",
    "ProgressEvent",
    "ProgressStyle",
    "ProgressTracker",
    "Symbols",
    "Verbosity",
    "configure_output",
    "create_executor_callback",
    "create_on_step_callback",
    "render_dag_ascii",
    "ConsoleEventHandler",
    "FileEventHandler",
    "FilteringEventHandler",
    "MetricsEventHandler",
    # Compatibility aliases
    "Channel",
    "ObservabilityLevel",
    "Timer",
    "LogEntry",
    "LogLevel",
    "ObservabilityLogger",
    "Dashboard",
    "DashboardConfig",
    "AgentInspector",
    "EventInspector",
    "SystemInspector",
    "TaskInspector",
]
