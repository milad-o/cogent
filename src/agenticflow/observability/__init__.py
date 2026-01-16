"""Observability module for AgenticFlow.

Provides comprehensive monitoring, tracing, metrics, events, and progress output
for understanding system behavior at runtime.

Includes:
- TraceBus: Central pub/sub for trace records
- Trace: Immutable trace records for observability
- TraceType: All event/trace types in the system
- Observer: Unified observability for agents, flows, teams
- Tracer: Distributed tracing
- Metrics: Counters, gauges, histograms
- Dashboard: Real-time monitoring UI

Note:
    For core orchestration events (agent-to-agent routing), use
    `agenticflow.events.Event` instead of `Trace`.
"""

from agenticflow.observability.bus import (
    TraceBus,
    get_trace_bus,
    set_trace_bus,
)
from agenticflow.observability.dashboard import (
    Dashboard,
    DashboardConfig,
)
from agenticflow.observability.handlers import (
    ConsoleEventHandler,
    FileEventHandler,
    FilteringEventHandler,
    MetricsEventHandler,
)
from agenticflow.observability.inspector import (
    AgentInspector,
    EventInspector,
    SystemInspector,
    TaskInspector,
)
from agenticflow.observability.logger import (
    LogEntry,
    LogLevel,
    ObservabilityLogger,
)
from agenticflow.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    Timer,
)
from agenticflow.observability.observer import (
    Channel,
    ObservabilityLevel,
    Observer,
)
from agenticflow.observability.progress import (
    # Renderers
    BaseRenderer,
    Colors,
    JSONRenderer,
    MinimalRenderer,
    # Configuration
    OutputConfig,
    OutputFormat,
    ProgressEvent,
    ProgressStyle,
    # Core classes
    ProgressTracker,
    RichRenderer,
    # Styling
    Styler,
    Symbols,
    TextRenderer,
    Theme,
    Verbosity,
    configure_output,
    create_executor_callback,
    # Callbacks
    create_on_step_callback,
    # Convenience
    get_tracker,
    # Visualization
    render_dag_ascii,
    set_tracker,
)
from agenticflow.observability.trace import (
    ExecutionSpan,
    ExecutionTracer,
    NodeStatus,
    NodeTrace,
    NodeType,
    ToolTrace,
    TraceLevel,
    TracingObserver,
)
from agenticflow.observability.trace_record import Trace, TraceType
from agenticflow.observability.tracer import (
    Span,
    SpanContext,
    SpanKind,
    Tracer,
)
from agenticflow.observability.websocket import (
    WebSocketServer,
    start_websocket_server,
    websocket_handler,
)

__all__ = [
    # Trace System (observability)
    "Trace",
    "TraceType",
    "TraceBus",
    "get_trace_bus",
    "set_trace_bus",
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
