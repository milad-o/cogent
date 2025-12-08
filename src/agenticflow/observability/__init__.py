"""Observability module for AgenticFlow.

Provides comprehensive monitoring, tracing, metrics, events, and progress output
for understanding system behavior at runtime.

Includes:
- EventBus: Central pub/sub for events
- Event: Immutable event records
- Observer: Unified observability for agents, flows, teams
- Tracer: Distributed tracing
- Metrics: Counters, gauges, histograms
- Dashboard: Real-time monitoring UI
"""

from agenticflow.observability.event import Event
from agenticflow.observability.bus import EventBus, get_event_bus, set_event_bus
from agenticflow.observability.handlers import (
    ConsoleEventHandler,
    FileEventHandler,
    FilteringEventHandler,
    MetricsEventHandler,
)
from agenticflow.observability.websocket import (
    WebSocketServer,
    start_websocket_server,
    websocket_handler,
)
from agenticflow.observability.tracer import (
    Tracer,
    Span,
    SpanContext,
    SpanKind,
)
from agenticflow.observability.metrics import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Timer,
)
from agenticflow.observability.logger import (
    ObservabilityLogger,
    LogLevel,
    LogEntry,
)
from agenticflow.observability.dashboard import (
    Dashboard,
    DashboardConfig,
)
from agenticflow.observability.inspector import (
    SystemInspector,
    AgentInspector,
    TaskInspector,
    EventInspector,
)
from agenticflow.observability.observer import (
    Observer,
    ObservabilityLevel,
    Channel,
)
from agenticflow.observability.trace import (
    ExecutionTracer,
    TracingObserver,
    TraceLevel,
    NodeType,
    NodeStatus,
    NodeTrace,
    ToolTrace,
    ExecutionSpan,
)
from agenticflow.observability.progress import (
    # Configuration
    OutputConfig,
    Verbosity,
    OutputFormat,
    ProgressStyle,
    Theme,
    # Core classes
    ProgressTracker,
    ProgressEvent,
    # Renderers
    BaseRenderer,
    TextRenderer,
    RichRenderer,
    JSONRenderer,
    MinimalRenderer,
    # Styling
    Styler,
    Colors,
    Symbols,
    # Callbacks
    create_on_step_callback,
    create_executor_callback,
    # Convenience
    get_tracker,
    set_tracker,
    configure_output,
    # Visualization
    render_dag_ascii,
)

__all__ = [
    # Event System
    "Event",
    "EventBus",
    "get_event_bus",
    "set_event_bus",
    # Event Handlers
    "ConsoleEventHandler",
    "FileEventHandler",
    "FilteringEventHandler",
    "MetricsEventHandler",
    # WebSocket
    "WebSocketServer",
    "start_websocket_server",
    "websocket_handler",
    # Tracing
    "Tracer",
    "Span",
    "SpanContext",
    "SpanKind",
    # Deep Execution Tracing
    "ExecutionTracer",
    "TracingObserver",
    "TraceLevel",
    "NodeType",
    "NodeStatus",
    "NodeTrace",
    "ToolTrace",
    "ExecutionSpan",
    # Metrics
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    # Logging
    "ObservabilityLogger",
    "LogLevel",
    "LogEntry",
    # Dashboard
    "Dashboard",
    "DashboardConfig",
    # Inspection
    "SystemInspector",
    "AgentInspector",
    "TaskInspector",
    "EventInspector",
    # Observer (unified observability for agents, flows, teams)
    "Observer",
    "ObservabilityLevel",
    "Channel",
    # Progress & Output
    "OutputConfig",
    "Verbosity",
    "OutputFormat",
    "ProgressStyle",
    "Theme",
    "ProgressTracker",
    "ProgressEvent",
    "BaseRenderer",
    "TextRenderer",
    "RichRenderer",
    "JSONRenderer",
    "MinimalRenderer",
    "Styler",
    "Colors",
    "Symbols",
    "create_on_step_callback",
    "create_executor_callback",
    "get_tracker",
    "set_tracker",
    "configure_output",
    "render_dag_ascii",
]
