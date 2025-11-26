"""
FlowObserver - pluggable observability for Flow execution.

Provides a unified interface for monitoring all aspects of flow execution
with configurable channels, verbosity levels, and output formats.

Example:
    ```python
    from agenticflow import Flow, Agent
    from agenticflow.observability import FlowObserver, ObservabilityLevel
    
    # Quick setup with preset levels
    observer = FlowObserver.verbose()
    
    # Or full control
    observer = FlowObserver(
        level=ObservabilityLevel.DEBUG,
        channels=["agents", "tools", "messages"],
        on_event=my_callback,
    )
    
    # Attach to flow
    flow = Flow(
        name="my-flow",
        agents=[agent1, agent2],
        topology="pipeline",
        observer=observer,  # <-- plugged in
    )
    
    # Or attach later
    flow.observer = observer
    
    # Run with full observability
    result = await flow.run("Do something")
    
    # Get timeline, metrics, etc.
    print(observer.timeline())
    print(observer.metrics())
    ```
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum, auto
from typing import Any, Callable, TextIO, TYPE_CHECKING

from agenticflow.core.enums import EventType
from agenticflow.core.utils import generate_id, now_utc
from agenticflow.models.event import Event
from agenticflow.observability.progress import (
    OutputConfig,
    ProgressTracker,
    Verbosity,
    OutputFormat,
    ProgressStyle,
    Styler,
    Colors,
)

if TYPE_CHECKING:
    from agenticflow.events.bus import EventBus


class ObservabilityLevel(IntEnum):
    """
    Preset observability levels.
    
    Each level determines what information is shown:
    
    - OFF: No output at all
    - RESULT: Only final results
    - PROGRESS: Key milestones (agent transitions)
    - DETAILED: Tool calls, retries, timing
    - DEBUG: Everything including internal events
    - TRACE: Maximum detail for debugging
    """
    
    OFF = 0
    RESULT = 1
    PROGRESS = 2
    DETAILED = 3
    DEBUG = 4
    TRACE = 5


class Channel(str, Enum):
    """
    Event channels for filtering.
    
    Subscribe to specific channels to see only relevant events:
    
    - AGENTS: Agent thinking, acting, status changes
    - TOOLS: Tool calls, results, errors, retries
    - MESSAGES: Inter-agent communication
    - TASKS: Task lifecycle events
    - SYSTEM: System-level events
    - RESILIENCE: Retries, circuit breakers, fallbacks
    - ALL: Everything
    """
    
    AGENTS = "agents"
    TOOLS = "tools"
    MESSAGES = "messages"
    TASKS = "tasks"
    SYSTEM = "system"
    RESILIENCE = "resilience"
    ALL = "all"


# Map channels to event types
CHANNEL_EVENTS: dict[Channel, set[EventType]] = {
    Channel.AGENTS: {
        EventType.AGENT_REGISTERED,
        EventType.AGENT_UNREGISTERED,
        EventType.AGENT_INVOKED,
        EventType.AGENT_THINKING,
        EventType.AGENT_ACTING,
        EventType.AGENT_RESPONDED,
        EventType.AGENT_ERROR,
        EventType.AGENT_STATUS_CHANGED,
    },
    Channel.TOOLS: {
        EventType.TOOL_REGISTERED,
        EventType.TOOL_CALLED,
        EventType.TOOL_RESULT,
        EventType.TOOL_ERROR,
    },
    Channel.MESSAGES: {
        EventType.MESSAGE_SENT,
        EventType.MESSAGE_RECEIVED,
        EventType.MESSAGE_BROADCAST,
    },
    Channel.TASKS: {
        EventType.TASK_CREATED,
        EventType.TASK_SCHEDULED,
        EventType.TASK_STARTED,
        EventType.TASK_BLOCKED,
        EventType.TASK_UNBLOCKED,
        EventType.TASK_COMPLETED,
        EventType.TASK_FAILED,
        EventType.TASK_CANCELLED,
        EventType.TASK_RETRYING,
        EventType.SUBTASK_SPAWNED,
        EventType.SUBTASK_COMPLETED,
        EventType.SUBTASKS_AGGREGATED,
    },
    Channel.SYSTEM: {
        EventType.SYSTEM_STARTED,
        EventType.SYSTEM_STOPPED,
        EventType.SYSTEM_ERROR,
        EventType.CLIENT_CONNECTED,
        EventType.CLIENT_DISCONNECTED,
        EventType.CLIENT_MESSAGE,
    },
    Channel.RESILIENCE: {
        EventType.TASK_RETRYING,
        EventType.TOOL_ERROR,
        EventType.AGENT_ERROR,
    },
}


@dataclass
class ObservedEvent:
    """An event captured by the observer."""
    
    event: Event
    channel: Channel
    timestamp: datetime = field(default_factory=now_utc)
    formatted: str = ""


@dataclass
class ObserverConfig:
    """
    Configuration for FlowObserver.
    
    Controls what events are captured and how they're displayed.
    """
    
    level: ObservabilityLevel = ObservabilityLevel.PROGRESS
    channels: set[Channel] = field(default_factory=lambda: {Channel.ALL})
    
    # Output settings
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    format: OutputFormat = OutputFormat.RICH
    show_timestamps: bool = False
    show_duration: bool = True
    show_trace_ids: bool = False
    truncate: int = 200
    use_colors: bool = True
    
    # Callbacks
    on_event: Callable[[Event], None] | None = None
    on_agent: Callable[[str, str, dict], None] | None = None  # name, action, data
    on_tool: Callable[[str, str, dict], None] | None = None   # name, action, data
    on_message: Callable[[str, str, str], None] | None = None  # from, to, content
    on_error: Callable[[str, Exception | str], None] | None = None
    
    # Filtering
    include_agents: set[str] | None = None  # None = all
    exclude_agents: set[str] | None = None
    include_tools: set[str] | None = None
    exclude_tools: set[str] | None = None


class FlowObserver:
    """
    Pluggable observability for Flow execution.
    
    FlowObserver provides a unified interface for monitoring all aspects
    of flow execution. It can be configured with:
    
    - Preset levels (OFF, RESULT, PROGRESS, DETAILED, DEBUG, TRACE)
    - Specific channels (AGENTS, TOOLS, MESSAGES, etc.)
    - Custom callbacks for events
    - Output format and styling
    
    Example - Quick setup:
        ```python
        # Use presets
        observer = FlowObserver.off()       # No output
        observer = FlowObserver.minimal()   # Results only
        observer = FlowObserver.progress()  # Key milestones
        observer = FlowObserver.detailed()  # Tool calls, timing
        observer = FlowObserver.debug()     # Everything
        
        flow = Flow(..., observer=observer)
        ```
    
    Example - Channel filtering:
        ```python
        # Only see agent events
        observer = FlowObserver(channels=[Channel.AGENTS])
        
        # See tools and messages
        observer = FlowObserver(channels=[Channel.TOOLS, Channel.MESSAGES])
        
        # Everything except system events
        observer = FlowObserver(channels=[
            Channel.AGENTS, Channel.TOOLS, Channel.MESSAGES, Channel.TASKS
        ])
        ```
    
    Example - Custom callbacks:
        ```python
        def on_tool_call(tool_name, action, data):
            if action == "error":
                send_alert(f"Tool {tool_name} failed: {data}")
        
        observer = FlowObserver(
            on_tool=on_tool_call,
            on_error=lambda source, err: log.error(f"{source}: {err}"),
        )
        ```
    
    Example - Detailed configuration:
        ```python
        observer = FlowObserver(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.AGENTS, Channel.TOOLS],
            show_timestamps=True,
            show_duration=True,
            truncate=500,
            include_agents={"Researcher", "Writer"},  # Filter to specific agents
            exclude_tools={"internal_tool"},
        )
        ```
    """
    
    def __init__(
        self,
        level: ObservabilityLevel = ObservabilityLevel.PROGRESS,
        channels: list[Channel] | set[Channel] | None = None,
        *,
        # Output
        stream: TextIO | None = None,
        format: OutputFormat = OutputFormat.RICH,
        show_timestamps: bool = False,
        show_duration: bool = True,
        show_trace_ids: bool = False,
        truncate: int = 200,
        use_colors: bool = True,
        # Callbacks
        on_event: Callable[[Event], None] | None = None,
        on_agent: Callable[[str, str, dict], None] | None = None,
        on_tool: Callable[[str, str, dict], None] | None = None,
        on_message: Callable[[str, str, str], None] | None = None,
        on_error: Callable[[str, Exception | str], None] | None = None,
        # Filtering
        include_agents: set[str] | list[str] | None = None,
        exclude_agents: set[str] | list[str] | None = None,
        include_tools: set[str] | list[str] | None = None,
        exclude_tools: set[str] | list[str] | None = None,
    ) -> None:
        """
        Create a FlowObserver.
        
        Args:
            level: Verbosity level (OFF through TRACE)
            channels: Which event channels to observe (default: ALL)
            stream: Output stream (default: stdout)
            format: Output format (TEXT, RICH, JSON)
            show_timestamps: Show timestamps in output
            show_duration: Show operation duration
            show_trace_ids: Show correlation IDs
            truncate: Max chars for results (0 = no limit)
            use_colors: Use ANSI colors
            on_event: Callback for every event
            on_agent: Callback for agent events (name, action, data)
            on_tool: Callback for tool events (name, action, data)
            on_message: Callback for messages (from, to, content)
            on_error: Callback for errors (source, error)
            include_agents: Only these agents (None = all)
            exclude_agents: Exclude these agents
            include_tools: Only these tools (None = all)
            exclude_tools: Exclude these tools
        """
        self.config = ObserverConfig(
            level=level,
            channels=set(channels) if channels else {Channel.ALL},
            stream=stream or sys.stdout,
            format=format,
            show_timestamps=show_timestamps,
            show_duration=show_duration,
            show_trace_ids=show_trace_ids,
            truncate=truncate,
            use_colors=use_colors,
            on_event=on_event,
            on_agent=on_agent,
            on_tool=on_tool,
            on_message=on_message,
            on_error=on_error,
            include_agents=set(include_agents) if include_agents else None,
            exclude_agents=set(exclude_agents) if exclude_agents else None,
            include_tools=set(include_tools) if include_tools else None,
            exclude_tools=set(exclude_tools) if exclude_tools else None,
        )
        
        self._events: list[ObservedEvent] = []
        self._start_time: datetime | None = None
        self._metrics: dict[str, Any] = defaultdict(int)
        self._styler = Styler(OutputConfig(
            use_colors=use_colors,
            format=format,
        ))
        self._attached_bus: EventBus | None = None
    
    # ==================== Factory Methods (Presets) ====================
    
    @classmethod
    def off(cls) -> FlowObserver:
        """Create observer with no output."""
        return cls(level=ObservabilityLevel.OFF)
    
    @classmethod
    def minimal(cls) -> FlowObserver:
        """Create observer showing only results."""
        return cls(
            level=ObservabilityLevel.RESULT,
            channels=[Channel.TASKS],
        )
    
    @classmethod
    def progress(cls) -> FlowObserver:
        """Create observer showing key milestones (default)."""
        return cls(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.AGENTS, Channel.TASKS],
        )
    
    @classmethod
    def normal(cls) -> FlowObserver:
        """Alias for progress() - key milestones."""
        return cls.progress()
    
    @classmethod
    def detailed(cls) -> FlowObserver:
        """Create observer showing tool calls and timing."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.AGENTS, Channel.TOOLS, Channel.TASKS],
            show_timestamps=True,
            show_duration=True,
        )
    
    @classmethod
    def debug(cls) -> FlowObserver:
        """Create observer showing everything."""
        return cls(
            level=ObservabilityLevel.DEBUG,
            channels=[Channel.ALL],
            show_timestamps=True,
            show_duration=True,
            show_trace_ids=True,
            truncate=0,
        )
    
    @classmethod
    def trace(cls) -> FlowObserver:
        """Create observer with maximum detail for debugging."""
        return cls(
            level=ObservabilityLevel.TRACE,
            channels=[Channel.ALL],
            show_timestamps=True,
            show_duration=True,
            show_trace_ids=True,
            truncate=0,
        )
    
    @classmethod
    def json(cls, stream: TextIO | None = None) -> FlowObserver:
        """Create observer with JSON output (for log aggregation)."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.ALL],
            stream=stream,
            format=OutputFormat.JSON,
            show_timestamps=True,
            use_colors=False,
        )
    
    @classmethod
    def agents_only(cls) -> FlowObserver:
        """Create observer showing only agent events."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.AGENTS],
            show_timestamps=True,
        )
    
    @classmethod
    def tools_only(cls) -> FlowObserver:
        """Create observer showing only tool events."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.TOOLS],
            show_timestamps=True,
        )
    
    @classmethod
    def messages_only(cls) -> FlowObserver:
        """Create observer showing only inter-agent messages."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.MESSAGES],
        )
    
    @classmethod
    def resilience_only(cls) -> FlowObserver:
        """Create observer showing retries, errors, and recovery."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.RESILIENCE, Channel.TOOLS],
            show_timestamps=True,
        )
    
    # ==================== Attach/Detach ====================
    
    def attach(self, event_bus: EventBus) -> None:
        """
        Attach observer to an event bus.
        
        Called automatically when observer is passed to Flow.
        
        Args:
            event_bus: The event bus to observe.
        """
        if self._attached_bus is not None:
            self.detach()
        
        self._attached_bus = event_bus
        self._start_time = now_utc()
        event_bus.subscribe_all(self._handle_event)
    
    def detach(self) -> None:
        """Detach observer from the current event bus."""
        if self._attached_bus is not None:
            self._attached_bus.unsubscribe_all(self._handle_event)
            self._attached_bus = None
    
    # ==================== Event Handling ====================
    
    def _should_observe(self, event: Event) -> tuple[bool, Channel | None]:
        """Check if event should be observed based on config."""
        if self.config.level == ObservabilityLevel.OFF:
            return False, None
        
        # Find channel for this event
        event_channel: Channel | None = None
        for channel, event_types in CHANNEL_EVENTS.items():
            if event.type in event_types:
                event_channel = channel
                break
        
        if event_channel is None:
            event_channel = Channel.SYSTEM
        
        # Check if channel is subscribed
        if Channel.ALL not in self.config.channels:
            if event_channel not in self.config.channels:
                return False, None
        
        # Check agent filters
        agent_name = event.data.get("agent_name") or event.data.get("agent")
        if agent_name:
            if self.config.include_agents and agent_name not in self.config.include_agents:
                return False, None
            if self.config.exclude_agents and agent_name in self.config.exclude_agents:
                return False, None
        
        # Check tool filters
        tool_name = event.data.get("tool")
        if tool_name:
            if self.config.include_tools and tool_name not in self.config.include_tools:
                return False, None
            if self.config.exclude_tools and tool_name in self.config.exclude_tools:
                return False, None
        
        # Check level requirements
        level_required = self._get_level_for_event(event)
        if self.config.level < level_required:
            return False, None
        
        return True, event_channel
    
    def _get_level_for_event(self, event: Event) -> ObservabilityLevel:
        """Get minimum level required to see this event."""
        # Result-level events
        if event.type in {EventType.TASK_COMPLETED, EventType.TASK_FAILED}:
            return ObservabilityLevel.RESULT
        
        # Progress-level events - these are the main milestones
        if event.type in {
            EventType.AGENT_INVOKED,
            EventType.AGENT_RESPONDED,
            EventType.AGENT_THINKING,  # Show thinking at progress level
            EventType.TASK_STARTED,
        }:
            return ObservabilityLevel.PROGRESS
        
        # Detailed-level events
        if event.type in {
            EventType.TOOL_CALLED,
            EventType.TOOL_RESULT,
            EventType.TOOL_ERROR,
            EventType.AGENT_ACTING,
            EventType.TASK_RETRYING,
        }:
            return ObservabilityLevel.DETAILED
        
        # Debug-level events
        if event.type in {
            EventType.AGENT_STATUS_CHANGED,
            EventType.MESSAGE_SENT,
            EventType.MESSAGE_RECEIVED,
        }:
            return ObservabilityLevel.DEBUG
        
        # Trace-level: everything else
        return ObservabilityLevel.TRACE
    
    def _handle_event(self, event: Event) -> None:
        """Handle an incoming event."""
        # Always track metrics and call callbacks regardless of level
        self._metrics["total_events"] += 1
        self._metrics[f"events.{event.type.value}"] += 1
        
        # Determine channel for this event
        event_channel = self._get_channel_for_event(event)
        
        # Store event (for metrics/history)
        observed = ObservedEvent(
            event=event,
            channel=event_channel,
            formatted="",
        )
        self._events.append(observed)
        
        # Always call callbacks (they're opt-in)
        self._dispatch_callbacks(event, event_channel)
        
        # Check if we should display this event
        should_display, _ = self._should_observe(event)
        if not should_display:
            return
        
        # Format and output
        formatted = self._format_event(event)
        observed.formatted = formatted
        
        if formatted and self.config.level > ObservabilityLevel.OFF:
            self.config.stream.write(formatted + "\n")
            self.config.stream.flush()
    
    def _get_channel_for_event(self, event: Event) -> Channel:
        """Get the channel for an event."""
        for channel, event_types in CHANNEL_EVENTS.items():
            if event.type in event_types:
                return channel
        return Channel.SYSTEM
    
    def _dispatch_callbacks(self, event: Event, channel: Channel | None) -> None:
        """Dispatch event to registered callbacks."""
        # General callback
        if self.config.on_event:
            self.config.on_event(event)
        
        # Agent callback
        if self.config.on_agent and channel == Channel.AGENTS:
            agent_name = event.data.get("agent_name") or event.data.get("agent", "unknown")
            action = event.type.value.split(".")[-1]  # e.g., "thinking" from "agent.thinking"
            self.config.on_agent(agent_name, action, event.data)
        
        # Tool callback
        if self.config.on_tool and channel == Channel.TOOLS:
            tool_name = event.data.get("tool", "unknown")
            action = event.type.value.split(".")[-1]
            self.config.on_tool(tool_name, action, event.data)
        
        # Message callback
        if self.config.on_message and channel == Channel.MESSAGES:
            sender = event.data.get("sender_id", "unknown")
            receiver = event.data.get("receiver_id", "broadcast")
            content = event.data.get("content", "")
            self.config.on_message(sender, receiver, content)
        
        # Error callback
        if self.config.on_error:
            if event.type in {EventType.AGENT_ERROR, EventType.TOOL_ERROR, EventType.TASK_FAILED}:
                source = event.data.get("agent_name") or event.data.get("tool") or "unknown"
                error = event.data.get("error") or event.data.get("message", "Unknown error")
                self.config.on_error(source, error)
    
    def _format_event(self, event: Event) -> str:
        """Format an event for output."""
        if self.config.format == OutputFormat.JSON:
            return event.to_json()
        
        s = self._styler
        parts = []
        
        # Timestamp
        if self.config.show_timestamps:
            ts = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
            parts.append(s.timestamp(f"[{ts}]"))
        
        # Trace ID
        if self.config.show_trace_ids and event.correlation_id:
            parts.append(s.dim(f"[{event.correlation_id[:8]}]"))
        
        # Format based on event type
        event_type = event.type
        data = event.data
        
        # Agent events
        if event_type == EventType.AGENT_INVOKED:
            parts.append(s.agent(f"[{data.get('agent_name', '?')}]"))
            parts.append("started")
        
        elif event_type == EventType.AGENT_THINKING:
            parts.append(s.agent(f"[{data.get('agent_name', '?')}]"))
            parts.append(s.dim("thinking..."))
        
        elif event_type == EventType.AGENT_ACTING:
            parts.append(s.agent(f"[{data.get('agent_name', '?')}]"))
            parts.append(s.dim("acting..."))
        
        elif event_type == EventType.AGENT_RESPONDED:
            parts.append(s.agent(f"[{data.get('agent_name', '?')}]"))
            parts.append(s.success("✓"))
            result = data.get("result_preview", "")
            if self.config.truncate and len(result) > self.config.truncate:
                result = result[:self.config.truncate] + "..."
            if result:
                parts.append(s.dim(result))
        
        elif event_type == EventType.AGENT_ERROR:
            parts.append(s.agent(f"[{data.get('agent_name', '?')}]"))
            parts.append(s.error(f"✗ {data.get('error', 'error')}"))
        
        # Tool events
        elif event_type == EventType.TOOL_CALLED:
            parts.append(s.info("→"))
            parts.append(s.tool(data.get("tool", "?")))
            args = data.get("args", {})
            if args and self.config.level >= ObservabilityLevel.DETAILED:
                args_str = str(args)
                if self.config.truncate and len(args_str) > self.config.truncate:
                    args_str = args_str[:self.config.truncate] + "..."
                parts.append(s.dim(f"({args_str})"))
        
        elif event_type == EventType.TOOL_RESULT:
            parts.append(s.success("✓"))
            parts.append(s.tool(data.get("tool", "")))
            result = data.get("result_preview", str(data.get("result", "")))
            if self.config.truncate and len(result) > self.config.truncate:
                result = result[:self.config.truncate] + "..."
            parts.append(s.dim(result))
            if self.config.show_duration and "duration_ms" in data:
                parts.append(s.dim(f"({data['duration_ms']:.0f}ms)"))
        
        elif event_type == EventType.TOOL_ERROR:
            parts.append(s.error("✗"))
            parts.append(s.tool(data.get("tool", "?")))
            parts.append(s.error(data.get("error", "error")))
        
        # Task events
        elif event_type == EventType.TASK_STARTED:
            parts.append(s.info("🚀"))
            parts.append(s.bold(data.get("task_name", data.get("task", "task"))))
        
        elif event_type == EventType.TASK_COMPLETED:
            parts.append(s.success("✅"))
            parts.append("Completed")
            if self.config.show_duration and "duration_ms" in data:
                parts.append(s.dim(f"({data['duration_ms']:.0f}ms)"))
        
        elif event_type == EventType.TASK_FAILED:
            parts.append(s.error("❌"))
            parts.append(s.error(f"Failed: {data.get('error', 'unknown')}"))
        
        elif event_type == EventType.TASK_RETRYING:
            attempt = data.get("attempt", "?")
            max_retries = data.get("max_retries", "?")
            parts.append(s.warning("🔄"))
            parts.append(f"Retry {attempt}/{max_retries}")
            if "delay" in data:
                parts.append(s.dim(f"in {data['delay']:.1f}s"))
        
        # Message events
        elif event_type == EventType.MESSAGE_SENT:
            sender = data.get("sender_id", "?")
            receiver = data.get("receiver_id", "?")
            parts.append(s.dim("📤"))
            parts.append(f"{sender} → {receiver}")
            content = data.get("content", "")[:50]
            if content:
                parts.append(s.dim(f'"{content}"'))
        
        elif event_type == EventType.MESSAGE_BROADCAST:
            sender = data.get("sender_id", "?")
            parts.append(s.dim("📢"))
            parts.append(f"{sender} broadcast")
        
        # Default
        else:
            parts.append(s.dim(event_type.value))
            if data:
                parts.append(s.dim(str(data)[:100]))
        
        return " ".join(parts)
    
    # ==================== Query API ====================
    
    def events(
        self,
        channel: Channel | None = None,
        event_type: EventType | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """
        Get observed events.
        
        Args:
            channel: Filter by channel
            event_type: Filter by event type
            limit: Max events to return
        
        Returns:
            List of events (most recent last)
        """
        result = self._events
        
        if channel:
            result = [e for e in result if e.channel == channel]
        
        if event_type:
            result = [e for e in result if e.event.type == event_type]
        
        if limit:
            result = result[-limit:]
        
        return [e.event for e in result]
    
    def timeline(self) -> str:
        """
        Get a timeline visualization of observed events.
        
        Returns:
            ASCII timeline string
        """
        if not self._events:
            return "No events observed"
        
        lines = ["Timeline:", "─" * 50]
        start = self._events[0].timestamp
        
        for observed in self._events:
            delta = (observed.timestamp - start).total_seconds()
            time_str = f"+{delta:>6.2f}s"
            lines.append(f"  {time_str}  {observed.formatted}")
        
        return "\n".join(lines)
    
    def metrics(self) -> dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Dict with event counts, timing, etc.
        """
        result = dict(self._metrics)
        
        # Add computed metrics
        if self._start_time:
            result["duration_seconds"] = (now_utc() - self._start_time).total_seconds()
        
        # Count by channel
        channel_counts: dict[str, int] = defaultdict(int)
        for observed in self._events:
            channel_counts[observed.channel.value] += 1
        result["by_channel"] = dict(channel_counts)
        
        return result
    
    def summary(self) -> str:
        """
        Get a human-readable summary.
        
        Returns:
            Summary string
        """
        m = self.metrics()
        lines = [
            "Observer Summary",
            "─" * 30,
            f"Total events: {m.get('total_events', 0)}",
        ]
        
        if "duration_seconds" in m:
            lines.append(f"Duration: {m['duration_seconds']:.2f}s")
        
        if "by_channel" in m:
            lines.append("By channel:")
            for channel, count in m["by_channel"].items():
                lines.append(f"  {channel}: {count}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear observed events and metrics."""
        self._events.clear()
        self._metrics.clear()
        self._start_time = now_utc()
    
    def to_tracker(self) -> ProgressTracker:
        """
        Convert observer config to a ProgressTracker.
        
        Useful when you need the ProgressTracker API.
        
        Returns:
            ProgressTracker with equivalent config
        """
        verbosity_map = {
            ObservabilityLevel.OFF: Verbosity.SILENT,
            ObservabilityLevel.RESULT: Verbosity.MINIMAL,
            ObservabilityLevel.PROGRESS: Verbosity.NORMAL,
            ObservabilityLevel.DETAILED: Verbosity.VERBOSE,
            ObservabilityLevel.DEBUG: Verbosity.DEBUG,
            ObservabilityLevel.TRACE: Verbosity.TRACE,
        }
        
        config = OutputConfig(
            verbosity=verbosity_map.get(self.config.level, Verbosity.NORMAL),
            format=self.config.format,
            show_timestamps=self.config.show_timestamps,
            show_duration=self.config.show_duration,
            show_trace_ids=self.config.show_trace_ids,
            truncate_results=self.config.truncate,
            use_colors=self.config.use_colors,
            stream=self.config.stream,
        )
        
        return ProgressTracker(config)
