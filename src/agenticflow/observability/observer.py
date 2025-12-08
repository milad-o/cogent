"""
Observer - Unified observability system for agents, flows, and teams.

This is your single entry point for ALL observability needs:
- Live console output (see what's happening)
- Deep execution tracing (graph, timeline, spans)
- Metrics and statistics
- Export to JSON, Mermaid, OpenTelemetry

Works with:
- Single agents: agent.add_observer(observer) or Agent(..., observer=observer)
- Flows: Flow(..., observer=observer)
- Teams: Team(..., observer=observer)

Example - Agent with Observer:
    ```python
    from agenticflow import Agent, Observer
    
    observer = Observer.verbose()
    agent = Agent(
        name="Assistant",
        model=model,
        observer=observer,  # Attach directly
    )
    
    await agent.run("Do something")
    print(observer.summary())
    ```

Example - Flow with Observer:
    ```python
    from agenticflow import Flow, Agent, Observer
    
    observer = Observer.trace()  # Maximum detail + graph
    
    flow = Flow(
        name="my-flow",
        agents=[agent1, agent2],
        topology="pipeline",
        observer=observer,
    )
    
    await flow.run("Do something")
    print(observer.graph())  # Execution graph (Mermaid)
    ```

Example - Full Control:
    ```python
    observer = Observer(
        level=ObservabilityLevel.DEBUG,
        channels=[Channel.AGENTS, Channel.TOOLS],
        show_timestamps=True,
        on_event=my_callback,
        on_agent=lambda name, action, data: print(f"{name}: {action}"),
        on_tool=lambda name, action, data: log_tool(name, data),
    )
    ```
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Callable, TextIO, TYPE_CHECKING

from agenticflow.core.enums import EventType
from agenticflow.core.utils import generate_id, now_utc, to_local
from agenticflow.observability.event import Event
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
    from agenticflow.observability.bus import EventBus


class ObservabilityLevel(IntEnum):
    """
    Preset observability levels.
    
    Each level determines what information is shown:
    
    - OFF: No output at all
    - RESULT: Only final results
    - PROGRESS: Key milestones (agent transitions)
    - DETAILED: Tool calls, retries, timing
    - DEBUG: Everything including internal events
    - TRACE: Maximum detail + execution graph
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
    - STREAMING: Token-by-token LLM output streaming
    - MEMORY: Memory read/write/search operations
    - RETRIEVAL: RAG retrieval pipeline events
    - DOCUMENTS: Document loading and splitting
    - SYSTEM: System-level events
    - RESILIENCE: Retries, circuit breakers, fallbacks
    - ALL: Everything
    """
    
    AGENTS = "agents"
    TOOLS = "tools"
    MESSAGES = "messages"
    TASKS = "tasks"
    STREAMING = "streaming"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"
    DOCUMENTS = "documents"
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
        EventType.AGENT_REASONING,
        EventType.AGENT_ACTING,
        EventType.AGENT_RESPONDED,
        EventType.AGENT_ERROR,
        EventType.AGENT_STATUS_CHANGED,
        EventType.LLM_REQUEST,
        EventType.LLM_RESPONSE,
        EventType.LLM_TOOL_DECISION,
        # Spawning events
        EventType.AGENT_SPAWNED,
        EventType.AGENT_SPAWN_COMPLETED,
        EventType.AGENT_SPAWN_FAILED,
        EventType.AGENT_DESPAWNED,
        # User/Output events (part of agent interaction)
        EventType.USER_INPUT,
        EventType.USER_FEEDBACK,
        EventType.OUTPUT_GENERATED,
        EventType.OUTPUT_STREAMED,
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
    Channel.STREAMING: {
        EventType.STREAM_START,
        EventType.TOKEN_STREAMED,
        EventType.STREAM_TOOL_CALL,
        EventType.STREAM_END,
        EventType.STREAM_ERROR,
    },
    Channel.MEMORY: {
        EventType.MEMORY_READ,
        EventType.MEMORY_WRITE,
        EventType.MEMORY_SEARCH,
        EventType.MEMORY_DELETE,
        EventType.MEMORY_CLEAR,
        EventType.THREAD_CREATED,
        EventType.THREAD_MESSAGE_ADDED,
    },
    Channel.RETRIEVAL: {
        EventType.RETRIEVAL_START,
        EventType.RETRIEVAL_COMPLETE,
        EventType.RETRIEVAL_ERROR,
        EventType.RERANK_START,
        EventType.RERANK_COMPLETE,
        EventType.FUSION_APPLIED,
        EventType.VECTORSTORE_ADD,
        EventType.VECTORSTORE_SEARCH,
        EventType.VECTORSTORE_DELETE,
    },
    Channel.DOCUMENTS: {
        EventType.DOCUMENT_LOADED,
        EventType.DOCUMENT_SPLIT,
        EventType.DOCUMENT_ENRICHED,
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
    Configuration for Observer.
    
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
    on_stream: Callable[[str, str, dict], None] | None = None  # agent, token/action, data
    on_error: Callable[[str, Exception | str], None] | None = None
    
    # Filtering
    include_agents: set[str] | None = None  # None = all
    exclude_agents: set[str] | None = None
    include_tools: set[str] | None = None
    exclude_tools: set[str] | None = None


class Observer:
    """
    Pluggable observability for Flow execution.
    
    Observer provides a unified interface for monitoring all aspects
    of flow execution. It can be configured with:
    
    - Preset levels (OFF, RESULT, PROGRESS, DETAILED, DEBUG, TRACE)
    - Specific channels (AGENTS, TOOLS, MESSAGES, etc.)
    - Custom callbacks for events
    - Output format and styling
    
    Example - Quick setup:
        ```python
        # Use presets
        observer = Observer.off()       # No output
        observer = Observer.minimal()   # Results only
        observer = Observer.progress()  # Key milestones
        observer = Observer.detailed()  # Tool calls, timing
        observer = Observer.debug()     # Everything
        
        flow = Flow(..., observer=observer)
        ```
    
    Example - Channel filtering:
        ```python
        # Only see agent events
        observer = Observer(channels=[Channel.AGENTS])
        
        # See tools and messages
        observer = Observer(channels=[Channel.TOOLS, Channel.MESSAGES])
        
        # Everything except system events
        observer = Observer(channels=[
            Channel.AGENTS, Channel.TOOLS, Channel.MESSAGES, Channel.TASKS
        ])
        ```
    
    Example - Custom callbacks:
        ```python
        def on_tool_call(tool_name, action, data):
            if action == "error":
                send_alert(f"Tool {tool_name} failed: {data}")
        
        observer = Observer(
            on_tool=on_tool_call,
            on_error=lambda source, err: log.error(f"{source}: {err}"),
        )
        ```
    
    Example - Detailed configuration:
        ```python
        observer = Observer(
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
        on_stream: Callable[[str, str, dict], None] | None = None,
        on_error: Callable[[str, Exception | str], None] | None = None,
        # Filtering
        include_agents: set[str] | list[str] | None = None,
        exclude_agents: set[str] | list[str] | None = None,
        include_tools: set[str] | list[str] | None = None,
        exclude_tools: set[str] | list[str] | None = None,
    ) -> None:
        """
        Create a Observer.
        
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
            on_stream: Callback for streaming events (agent, token/action, data)
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
            on_stream=on_stream,
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
        
        # === Deep Tracing (integrated - no separate tracer needed!) ===
        self._trace_id = generate_id()
        
        # Execution graph tracking
        self._nodes: dict[str, dict] = {}  # node_id -> {name, type, status, ...}
        self._edges: list[tuple[str, str, str]] = []  # (from, to, label)
        self._current_agents: dict[str, str] = {}  # agent_name -> node_id
        
        # Tool call tracking
        self._tool_calls: list[dict] = []
        self._current_tools: dict[str, dict] = {}  # tool_name -> trace info
        
        # Span tracking (for nested operations)
        self._spans: list[dict] = []
        self._span_stack: list[dict] = []
        
        # Streaming state tracking
        self._streaming_agents: dict[str, dict] = {}  # agent_name -> {tokens, start_time, ...}
        self._stream_buffer: dict[str, str] = {}  # agent_name -> accumulated content
        
        # Thinking event deduplication - track iteration count per agent
        self._agent_thinking_count: dict[str, int] = {}  # agent_name -> iteration count
    
    # ==================== Factory Methods (Presets) ====================
    
    @classmethod
    def off(cls) -> Observer:
        """Create observer with no output."""
        return cls(level=ObservabilityLevel.OFF)
    
    @classmethod
    def minimal(cls) -> Observer:
        """Create observer showing only results."""
        return cls(
            level=ObservabilityLevel.RESULT,
            channels=[Channel.TASKS],
        )
    
    @classmethod
    def progress(cls) -> Observer:
        """Create observer showing key milestones (default)."""
        return cls(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.AGENTS, Channel.TASKS],
        )
    
    @classmethod
    def normal(cls) -> Observer:
        """Alias for progress() - key milestones."""
        return cls.progress()
    
    @classmethod
    def verbose(cls) -> Observer:
        """Create observer showing agent thoughts with full content."""
        return cls(
            level=ObservabilityLevel.PROGRESS,
            channels=[Channel.AGENTS, Channel.TASKS],
            show_duration=True,
            truncate=500,  # Show substantial content
        )
    
    @classmethod
    def detailed(cls) -> Observer:
        """Create observer showing tool calls and timing."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.AGENTS, Channel.TOOLS, Channel.TASKS],
            show_timestamps=True,
            show_duration=True,
        )
    
    @classmethod
    def debug(cls) -> Observer:
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
    def trace(cls) -> Observer:
        """
        Create observer with maximum observability.
        
        Shows everything + builds execution graph.
        After running, call:
        - observer.graph() - Mermaid diagram of execution
        - observer.timeline(detailed=True) - chronological view
        - observer.summary() - stats and metrics
        - observer.execution_trace() - structured data for export
        """
        return cls(
            level=ObservabilityLevel.TRACE,
            channels=[Channel.ALL],
            show_timestamps=True,
            show_duration=True,
            show_trace_ids=True,
            truncate=0,
        )
    
    @classmethod
    def json(cls, stream: TextIO | None = None, colored: bool = True) -> Observer:
        """
        Create observer with structured JSON-like output.
        
        Outputs a readable, structured format ideal for log analysis.
        Set colored=False for plain text (e.g., when piping to files).
        """
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.ALL],
            stream=stream,
            format=OutputFormat.JSON,
            show_timestamps=True,
            use_colors=colored,
        )
    
    @classmethod
    def agents_only(cls) -> Observer:
        """Create observer showing only agent events."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.AGENTS],
            show_timestamps=True,
        )
    
    @classmethod
    def tools_only(cls) -> Observer:
        """Create observer showing only tool events."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.TOOLS],
            show_timestamps=True,
        )
    
    @classmethod
    def messages_only(cls) -> Observer:
        """Create observer showing only inter-agent messages."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.MESSAGES],
        )
    
    @classmethod
    def resilience_only(cls) -> Observer:
        """Create observer showing retries, errors, and recovery."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.RESILIENCE, Channel.TOOLS],
            show_timestamps=True,
        )
    
    @classmethod
    def streaming(cls, show_tokens: bool = True) -> Observer:
        """
        Create observer optimized for streaming output.
        
        Shows real-time token streaming from LLM responses.
        
        Args:
            show_tokens: If True, show individual tokens (DEBUG level).
                        If False, show only start/end (DETAILED level).
        
        Example:
            ```python
            observer = Observer.streaming()
            flow = Flow(..., observer=observer)
            
            # Will show tokens as they arrive:
            # ▸ [Agent] streaming... (gpt-4o)
            # Hello world...  <- tokens appear in real-time
            # ✓ [Agent] stream complete (1.2s, 45 tok/s)
            ```
        """
        return cls(
            level=ObservabilityLevel.DEBUG if show_tokens else ObservabilityLevel.DETAILED,
            channels=[Channel.AGENTS, Channel.STREAMING, Channel.TASKS],
            show_timestamps=True,
            show_duration=True,
        )
    
    @classmethod
    def streaming_only(cls) -> Observer:
        """Create observer showing only streaming events."""
        return cls(
            level=ObservabilityLevel.DEBUG,
            channels=[Channel.STREAMING],
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
            EventType.AGENT_REASONING,  # Show reasoning at progress level
            EventType.TASK_STARTED,
            # Spawning events at progress - important milestones
            EventType.AGENT_SPAWNED,
            EventType.AGENT_SPAWN_COMPLETED,
            EventType.AGENT_SPAWN_FAILED,
        }:
            return ObservabilityLevel.PROGRESS
        
        # Detailed-level events
        if event.type in {
            EventType.TOOL_CALLED,
            EventType.TOOL_RESULT,
            EventType.TOOL_ERROR,
            EventType.AGENT_ACTING,
            EventType.TASK_RETRYING,
            EventType.LLM_TOOL_DECISION,  # Show tool decisions at detailed level
            EventType.AGENT_DESPAWNED,  # Cleanup at detailed level
        }:
            return ObservabilityLevel.DETAILED
        
        # Debug-level events - LLM requests/responses (more verbose)
        if event.type in {
            EventType.LLM_REQUEST,
            EventType.LLM_RESPONSE,
        }:
            return ObservabilityLevel.DEBUG
        
        # Debug-level events
        if event.type in {
            EventType.AGENT_STATUS_CHANGED,
            EventType.MESSAGE_SENT,
            EventType.MESSAGE_RECEIVED,
        }:
            return ObservabilityLevel.DEBUG
        
        # Streaming events - DETAILED level (same as tool calls)
        # STREAM_START/END at DETAILED, TOKEN_STREAMED at DEBUG for less noise
        if event.type in {
            EventType.STREAM_START,
            EventType.STREAM_END,
            EventType.STREAM_TOOL_CALL,
            EventType.STREAM_ERROR,
        }:
            return ObservabilityLevel.DETAILED
        
        if event.type == EventType.TOKEN_STREAMED:
            return ObservabilityLevel.DEBUG  # Individual tokens at debug level
        
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
        
        # === Build execution trace graph ===
        self._trace_event(event)
        
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
            if event.type in {EventType.AGENT_ERROR, EventType.TOOL_ERROR, EventType.TASK_FAILED, EventType.STREAM_ERROR}:
                source = event.data.get("agent_name") or event.data.get("tool") or "unknown"
                error = event.data.get("error") or event.data.get("message", "Unknown error")
                self.config.on_error(source, error)
        
        # Streaming callback
        if self.config.on_stream and channel == Channel.STREAMING:
            agent_name = event.data.get("agent_name") or event.data.get("agent", "unknown")
            if event.type == EventType.TOKEN_STREAMED:
                token = event.data.get("token", event.data.get("content", ""))
                self.config.on_stream(agent_name, token, event.data)
            else:
                action = event.type.value.split(".")[-1]  # "start", "end", "tool_call", "error"
                self.config.on_stream(agent_name, action, event.data)
    
    def _format_event(self, event: Event) -> str | None:
        """Format an event for output with rich colors and formatting.
        
        Returns None if the event should be suppressed (e.g., duplicate thinking events).
        """
        if self.config.format == OutputFormat.JSON:
            return self._format_event_json(event)
        
        s = self._styler
        c = Colors  # Direct access for custom colors
        lines: list[str] = []
        
        # Timestamp prefix - use cyan for visibility
        ts_prefix = ""
        if self.config.show_timestamps:
            local_ts = to_local(event.timestamp)
            ts = local_ts.strftime("%H:%M:%S.%f")[:-3]
            ts_prefix = s.info(f"[{ts}]") + " "
        
        # Trace ID
        trace_prefix = ""
        if self.config.show_trace_ids and event.correlation_id:
            trace_prefix = s.dim(f"({event.correlation_id[:8]}) ")
        
        prefix = ts_prefix + trace_prefix
        
        # Format based on event type
        event_type = event.type
        data = event.data
        
        # ═══════════════════════════════════════════════════════════
        # USER & OUTPUT EVENTS - Clear visibility into user interactions
        # ═══════════════════════════════════════════════════════════
        
        if event_type == EventType.USER_INPUT:
            content = data.get("content", data.get("input", ""))
            source = data.get("source", "user")
            # Truncate long inputs
            max_len = min(self.config.truncate or 100, 100)
            if len(content) > max_len:
                content = content[:max_len].replace("\n", " ").strip() + "..."
            else:
                content = content.replace("\n", " ").strip()
            return f"{prefix}{s.info('👤')} {s.bold('User:')} {content}"
        
        elif event_type == EventType.USER_FEEDBACK:
            feedback = data.get("feedback", data.get("content", ""))
            decision = data.get("decision", "")
            if decision:
                return f"{prefix}{s.info('👤')} {s.bold('User feedback:')} {decision}"
            max_len = min(self.config.truncate or 80, 80)
            if len(feedback) > max_len:
                feedback = feedback[:max_len].replace("\n", " ").strip() + "..."
            return f"{prefix}{s.info('👤')} {s.bold('User feedback:')} {feedback}"
        
        elif event_type == EventType.OUTPUT_GENERATED:
            content = data.get("content", data.get("output", ""))
            agent_name = data.get("agent_name", "")
            # Duration if available
            duration_str = ""
            if self.config.show_duration and "duration_ms" in data:
                ms = data['duration_ms']
                if ms > 1000:
                    duration_str = s.success(f" ({ms/1000:.1f}s)")
                else:
                    duration_str = s.dim(f" ({ms:.0f}ms)")
            # Truncate output preview
            max_len = min(self.config.truncate or 120, 120)
            if len(content) > max_len:
                content = content[:max_len].replace("\n", " ").strip() + "..."
            else:
                content = content.replace("\n", " ").strip()
            agent_ctx = f" [{agent_name}]" if agent_name else ""
            return f"{prefix}{s.success('📤')} {s.bold('Output')}{agent_ctx}{duration_str}: {content}"
        
        elif event_type == EventType.OUTPUT_STREAMED:
            # Similar to TOKEN_STREAMED but for final user-facing output
            token = data.get("token", data.get("content", ""))
            if self.config.level >= ObservabilityLevel.DEBUG:
                self.config.stream.write(token)
                self.config.stream.flush()
                return ""  # No newline for streaming
            return ""  # Don't show at lower levels
        
        # ═══════════════════════════════════════════════════════════
        # AGENT EVENTS - Purple/Magenta theme with emojis
        # ═══════════════════════════════════════════════════════════
        
        elif event_type == EventType.AGENT_INVOKED:
            agent_name = data.get('agent_name', '?')
            return f"{prefix}{s.agent(f'▶ [{agent_name}]')} {s.dim('starting...')}"
        
        elif event_type == EventType.AGENT_THINKING:
            agent_name = data.get('agent_name', '?')
            iteration = data.get('iteration', 1)
            
            # Track thinking iterations per agent to reduce noise
            if agent_name not in self._agent_thinking_count:
                self._agent_thinking_count[agent_name] = 0
            self._agent_thinking_count[agent_name] += 1
            count = self._agent_thinking_count[agent_name]
            
            # Only show first thinking event per agent - subsequent work shows via tool calls
            if count == 1:
                return f"{prefix}{s.agent(f'🧠 [{agent_name}]')} {s.dim('thinking...')}"
            else:
                # Suppress subsequent thinking events - tool calls will show the actual work
                return None
        
        elif event_type == EventType.AGENT_REASONING:
            agent_name = data.get('agent_name', '?')
            round_num = data.get('round', 1)
            reasoning_type = data.get('reasoning_type', 'thinking')
            thought_preview = data.get('thought_preview', '')
            
            # Type-specific icons
            type_icons = {
                'analysis': '🔍',
                'plan': '📋',
                'reflection': '🪞',
                'correction': '🔄',
            }
            icon = type_icons.get(reasoning_type, '💭')
            
            header = f"{prefix}{s.agent(f'{icon} [{agent_name}]')} {s.dim(f'reasoning (round {round_num})...')}"
            
            if thought_preview and self.config.level >= ObservabilityLevel.DETAILED:
                # Show thought preview for detailed level
                lines.append(header)
                for line in thought_preview.split('\n')[:5]:
                    lines.append(f"      {s.dim(line)}")
                return "\n".join(lines)
            else:
                return header
        
        elif event_type == EventType.AGENT_ACTING:
            agent_name = data.get('agent_name', '?')
            return f"{prefix}{s.agent(f'⚡ [{agent_name}]')} {s.dim('acting...')}"
        
        elif event_type == EventType.AGENT_RESPONDED:
            agent_name = data.get('agent_name', '?')
            result = data.get("response_preview") or data.get("response") or data.get("result_preview", "")
            
            # Reset thinking count for this agent
            if agent_name in self._agent_thinking_count:
                del self._agent_thinking_count[agent_name]
            
            # Duration formatting
            duration_str = ""
            if self.config.show_duration and "duration_ms" in data:
                ms = data['duration_ms']
                if ms > 1000:
                    duration_str = s.success(f" ({ms/1000:.1f}s)")
                else:
                    duration_str = s.dim(f" ({ms:.0f}ms)")
            
            # Header with checkmark
            header = f"{prefix}{s.success('✓')} {s.agent(f'[{agent_name}]')}{duration_str}"
            
            if result:
                # Truncate more aggressively for cleaner output
                truncate_len = min(self.config.truncate or 150, 150)
                result_clean = result.replace("\n", " ").strip()
                if len(result_clean) > truncate_len:
                    result_clean = result_clean[:truncate_len] + "..."
                return f"{header}\n      {result_clean}"
            else:
                return header
        
        elif event_type == EventType.AGENT_ERROR:
            agent_name = data.get('agent_name', '?')
            error = data.get('error', 'Unknown error')
            return f"{prefix}{s.error(f'✗ [{agent_name}] ERROR:')} {s.error(error)}"
        
        # ═══════════════════════════════════════════════════════════
        # TOOL EVENTS - Blue/Cyan theme with agent context
        # ═══════════════════════════════════════════════════════════
        
        elif event_type == EventType.TOOL_CALLED:
            agent_name = data.get("agent_name", "")
            tool_name = data.get("tool_name", data.get("tool", "?"))
            args = data.get("args", {})
            
            # Agent context for spawned agents
            agent_ctx = f"{s.dim(f'[{agent_name}]')} " if agent_name else ""
            
            args_str = ""
            if args and self.config.level >= ObservabilityLevel.DETAILED:
                args_preview = str(args)
                if self.config.truncate and len(args_preview) > 80:
                    args_preview = args_preview[:80] + "..."
                args_str = f"\n{prefix}      {s.dim(f'args: {args_preview}')}"
            
            return f"{prefix}   {agent_ctx}{s.info('↳')} {s.tool(f'🔧 {tool_name}')}{args_str}"
        
        elif event_type == EventType.TOOL_RESULT:
            agent_name = data.get("agent_name", "")
            tool_name = data.get("tool_name", data.get("tool", "?"))
            result = data.get("result_preview", str(data.get("result", "")))
            
            # Agent context
            agent_ctx = f"{s.dim(f'[{agent_name}]')} " if agent_name else ""
            
            # Duration
            duration_str = ""
            if self.config.show_duration and "duration_ms" in data:
                ms = data['duration_ms']
                if ms > 1000:
                    duration_str = s.success(f" ({ms/1000:.1f}s)")
                else:
                    duration_str = s.dim(f" ({ms:.0f}ms)")
            
            # Truncate result - show more context
            truncate_len = min(self.config.truncate or 200, 200)
            if len(result) > truncate_len:
                result = result[:truncate_len] + "..."
            
            # Clean up result for single line display
            result_clean = result.replace("\n", " ").strip()
            result_preview = f"\n{prefix}      {s.dim(f'→ {result_clean}')}" if result_clean else ""
            
            return f"{prefix}   {agent_ctx}{s.success('✓')} {s.tool(f'🔧 {tool_name}')}{duration_str}{result_preview}"
        
        elif event_type == EventType.TOOL_ERROR:
            agent_name = data.get("agent_name", "")
            tool_name = data.get("tool_name", data.get("tool", "?"))
            error = data.get("error", "Unknown error")
            agent_ctx = f"{s.dim(f'[{agent_name}]')} " if agent_name else ""
            return f"{prefix}   {agent_ctx}{s.error('✗')} {s.tool(f'🔧 {tool_name}')} {s.error(f'FAILED: {error}')}"
        
        # ═══════════════════════════════════════════════════════════
        # TASK EVENTS - Green theme with clear status
        # ═══════════════════════════════════════════════════════════
        
        elif event_type == EventType.TASK_STARTED:
            task_name = data.get("task_name", data.get("task", "task"))
            return f"{prefix}{s.success('🚀')} {s.bold('Task started:')} {task_name}"
        
        elif event_type == EventType.TASK_COMPLETED:
            duration_str = ""
            if self.config.show_duration and "duration_ms" in data:
                ms = data['duration_ms']
                if ms > 1000:
                    duration_str = f" in {s.success(f'{ms/1000:.1f}s')}"
                else:
                    duration_str = f" in {s.dim(f'{ms:.0f}ms')}"
            return f"{prefix}{s.success('✅ Task completed')}{duration_str}"
        
        elif event_type == EventType.TASK_FAILED:
            error = data.get('error', 'unknown')
            return f"{prefix}{s.error(f'❌ Task FAILED: {error}')}"
        
        elif event_type == EventType.TASK_RETRYING:
            attempt = data.get("attempt", "?")
            max_retries = data.get("max_retries", "?")
            delay_str = ""
            if "delay" in data:
                delay_str = s.dim(f" in {data['delay']:.1f}s")
            return f"{prefix}{s.warning(f'🔄 Retrying ({attempt}/{max_retries})')}{delay_str}"
        
        # ═══════════════════════════════════════════════════════════
        # MESSAGE EVENTS - Communication between agents (concise)
        # ═══════════════════════════════════════════════════════════
        
        elif event_type == EventType.MESSAGE_SENT:
            sender = data.get("sender_id", "?")
            receiver = data.get("receiver_id", "?")
            content = data.get("content", "")
            # Always truncate message content aggressively for readability
            max_len = min(self.config.truncate or 80, 80)
            if len(content) > max_len:
                # Show just the start for context
                content = content[:max_len].replace("\n", " ").strip() + "..."
            else:
                content = content.replace("\n", " ").strip()
            content_str = f' "{content}"' if content else ""
            return f"{prefix}{s.dim('📤')} {sender} {s.dim('→')} {receiver}{s.dim(content_str)}"
        
        elif event_type == EventType.MESSAGE_RECEIVED:
            receiver = data.get("agent_name", data.get("agent", "?"))
            sender = data.get("from", "?")
            content = data.get("content", "")
            # Always truncate message content aggressively for readability
            max_len = min(self.config.truncate or 80, 80)
            if len(content) > max_len:
                content = content[:max_len].replace("\n", " ").strip() + "..."
            else:
                content = content.replace("\n", " ").strip()
            content_str = f' "{content}"' if content else ""
            return f"{prefix}{s.dim('📥')} {sender} {s.dim('→')} {receiver}{s.dim(content_str)}"
        
        elif event_type == EventType.MESSAGE_BROADCAST:
            sender = data.get("sender_id", "?")
            return f"{prefix}{s.dim('📢')} {sender} {s.dim('broadcast')}"
        
        # ═══════════════════════════════════════════════════════════
        # STREAMING EVENTS - Real-time LLM output
        # ═══════════════════════════════════════════════════════════
        
        elif event_type == EventType.STREAM_START:
            agent_name = data.get("agent_name", data.get("agent", "?"))
            model = data.get("model", "")
            model_str = f" ({model})" if model else ""
            # Track streaming state
            self._streaming_agents[agent_name] = {
                "start_time": event.timestamp,
                "token_count": 0,
            }
            self._stream_buffer[agent_name] = ""
            return f"{prefix}{s.info('▸')} {s.agent(f'[{agent_name}]')} {s.dim('streaming...')}{s.dim(model_str)}"
        
        elif event_type == EventType.TOKEN_STREAMED:
            agent_name = data.get("agent_name", data.get("agent", "?"))
            token = data.get("token", data.get("content", ""))
            # Track token count
            if agent_name in self._streaming_agents:
                self._streaming_agents[agent_name]["token_count"] += 1
            # Accumulate content
            if agent_name in self._stream_buffer:
                self._stream_buffer[agent_name] += token
            # At DEBUG level, show each token (can be noisy)
            # Use end="" style output for inline streaming effect
            if self.config.level >= ObservabilityLevel.DEBUG:
                # Return empty - we'll print directly for streaming effect
                self.config.stream.write(s.dim(token))
                self.config.stream.flush()
                return ""  # Don't add newline
            return ""  # Don't display individual tokens at lower levels
        
        elif event_type == EventType.STREAM_TOOL_CALL:
            agent_name = data.get("agent_name", data.get("agent", "?"))
            tool_name = data.get("tool_name", data.get("tool", "?"))
            tool_args = data.get("args", {})
            args_preview = str(tool_args)[:50] if tool_args else ""
            return f"{prefix}   {s.info('↳')} {s.tool(f'🔧 {tool_name}')} {s.dim('(during stream)')}{s.dim(f' {args_preview}') if args_preview else ''}"
        
        elif event_type == EventType.STREAM_END:
            agent_name = data.get("agent_name", data.get("agent", "?"))
            # Calculate stats
            token_count = 0
            duration_str = ""
            if agent_name in self._streaming_agents:
                stream_info = self._streaming_agents[agent_name]
                token_count = stream_info.get("token_count", 0)
                if "start_time" in stream_info and self.config.show_duration:
                    start = stream_info["start_time"]
                    duration_ms = (event.timestamp - start).total_seconds() * 1000
                    tokens_per_sec = token_count / (duration_ms / 1000) if duration_ms > 0 else 0
                    if duration_ms > 1000:
                        duration_str = s.success(f" ({duration_ms/1000:.1f}s, {tokens_per_sec:.0f} tok/s)")
                    else:
                        duration_str = s.dim(f" ({duration_ms:.0f}ms, {token_count} tokens)")
                # Clean up
                del self._streaming_agents[agent_name]
            
            # Get accumulated content preview
            content_preview = ""
            if agent_name in self._stream_buffer:
                content = self._stream_buffer[agent_name]
                if content:
                    truncate_len = self.config.truncate or 200
                    content_preview = content[:truncate_len]
                    if len(content) > truncate_len:
                        content_preview += "..."
                del self._stream_buffer[agent_name]
            
            # If we were at DEBUG level, add newline after streaming tokens
            if self.config.level >= ObservabilityLevel.DEBUG:
                self.config.stream.write("\n")
            
            header = f"{prefix}{s.success('✓')} {s.agent(f'[{agent_name}]')} {s.dim('stream complete')}{duration_str}"
            if content_preview and self.config.level >= ObservabilityLevel.DETAILED:
                return f"{header}\n      {s.dim(content_preview)}"
            return header
        
        elif event_type == EventType.STREAM_ERROR:
            agent_name = data.get("agent_name", data.get("agent", "?"))
            error = data.get("error", "Unknown streaming error")
            # Clean up streaming state
            self._streaming_agents.pop(agent_name, None)
            self._stream_buffer.pop(agent_name, None)
            return f"{prefix}{s.error('✗')} {s.agent(f'[{agent_name}]')} {s.error(f'Stream error: {error}')}"
        
        # ═══════════════════════════════════════════════════════════
        # LLM OBSERVABILITY EVENTS - Concise LLM interaction logging
        # ═══════════════════════════════════════════════════════════
        
        elif event_type == EventType.LLM_REQUEST:
            agent_name = data.get("agent_name", "?")
            message_count = data.get("message_count", 0)
            tools_available = data.get("tools_available", [])
            prompt = data.get("prompt", "")
            system_prompt = data.get("system_prompt", "")
            
            # Build concise header
            tools_count = f", {len(tools_available)} tools" if tools_available else ""
            header = f"{prefix}{s.info('📤')} {s.agent(f'[{agent_name}]')} LLM request {s.dim(f'({message_count} msgs{tools_count})')}"
            
            # Only show prompt preview at DETAILED+ level, single line
            if self.config.level >= ObservabilityLevel.DETAILED and (prompt or system_prompt):
                lines.append(header)
                if system_prompt:
                    sys_preview = system_prompt[:60].replace("\n", " ").strip()
                    if len(system_prompt) > 60:
                        sys_preview += "..."
                    lines.append(f"      {s.dim(f'System: {sys_preview}')}")
                if prompt:
                    prompt_preview = prompt[:80].replace("\n", " ").strip()
                    if len(prompt) > 80:
                        prompt_preview += "..."
                    lines.append(f"      {s.dim(f'Prompt: {prompt_preview}')}")
                return "\n".join(lines)
            return header
        
        elif event_type == EventType.LLM_RESPONSE:
            agent_name = data.get("agent_name", "?")
            content = data.get("content", "")
            tool_calls = data.get("tool_calls", [])
            finish_reason = data.get("finish_reason", "")
            duration_ms = data.get("duration_ms", 0)
            
            # Duration formatting
            duration_str = ""
            if duration_ms:
                if duration_ms > 1000:
                    duration_str = s.success(f" ({duration_ms/1000:.1f}s)")
                else:
                    duration_str = s.dim(f" ({duration_ms:.0f}ms)")
            
            tool_str = f", {len(tool_calls)} tools" if tool_calls else ""
            reason_str = f" {s.dim(f'({finish_reason})')}" if finish_reason else ""
            header = f"{prefix}{s.success('📥')} {s.agent(f'[{agent_name}]')} LLM response{duration_str}{reason_str}{s.dim(tool_str)}"
            
            # Show brief content preview at DETAILED+ level
            if self.config.level >= ObservabilityLevel.DETAILED and content:
                lines.append(header)
                # Single-line preview only
                content_preview = content[:100].replace("\n", " ").strip()
                if len(content) > 100:
                    content_preview += "..."
                lines.append(f"      {s.dim(content_preview)}")
                return "\n".join(lines)
            
            return header
        
        elif event_type == EventType.LLM_TOOL_DECISION:
            agent_name = data.get("agent_name", "?")
            tools_selected = data.get("tools_selected", [])
            reasoning = data.get("reasoning", "")
            
            if tools_selected:
                tools_str = ", ".join(tools_selected[:5])
                if len(tools_selected) > 5:
                    tools_str += f" ... (+{len(tools_selected) - 5})"
                header = f"{prefix}{s.info('🎯')} {s.agent(f'[{agent_name}]')} {s.bold('Tool Decision:')} {s.tool(tools_str)}"
            else:
                header = f"{prefix}{s.info('🎯')} {s.agent(f'[{agent_name}]')} {s.bold('Tool Decision:')} {s.dim('(no tools selected)')}"
            
            if reasoning:
                lines.append(header)
                reasoning_preview = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                lines.append(f"      {s.dim('Reasoning:')} {reasoning_preview}")
                return "\n".join(lines)
            
            return header
        
        # ═══════════════════════════════════════════════════════════
        # SPAWNING EVENTS - Dynamic agent creation
        # ═══════════════════════════════════════════════════════════
        
        elif event_type == EventType.AGENT_SPAWNED:
            parent = data.get("parent_agent", "?")
            role = data.get("role", "?")
            task = data.get("task", "")[:80]
            depth = data.get("depth", 1)
            active = data.get("active_spawns", 0)
            total = data.get("total_spawns", 0)
            
            # Tree-style indentation for nested spawns
            indent = "   " * depth
            tree_char = "├─" if depth > 1 else ""
            
            header = f"{prefix}{indent}{tree_char}{s.info('🚀')} {s.agent(f'[{parent}]')} spawned {s.success(s.bold(role))}"
            lines.append(header)
            if task:
                lines.append(f"{prefix}{indent}   {s.dim('Task:')} {s.dim(task)}")
            lines.append(f"{prefix}{indent}   {s.dim(f'Active: {active}, Total: {total}, Depth: {depth}')}")
            return "\n".join(lines)
        
        elif event_type == EventType.AGENT_SPAWN_COMPLETED:
            role = data.get("role", "?")
            result = data.get("result_preview", "")
            depth = data.get("depth", 1)
            
            # Tree-style indentation
            indent = "   " * depth
            tree_char = "└─" if depth > 1 else ""
            
            header = f"{prefix}{indent}{tree_char}{s.success('✓')} {s.success(role)} completed"
            if result:
                # Truncate and clean up result preview
                result_clean = result[:120].replace("\n", " ").strip()
                if len(result) > 120:
                    result_clean += "..."
                lines.append(header)
                lines.append(f"{prefix}{indent}   {s.dim(result_clean)}")
                return "\n".join(lines)
            return header
        
        elif event_type == EventType.AGENT_SPAWN_FAILED:
            role = data.get("role", "?")
            error = data.get("error", "Unknown error")
            depth = data.get("depth", 1)
            
            indent = "   " * depth
            tree_char = "└─" if depth > 1 else ""
            return f"{prefix}{indent}{tree_char}{s.error('✗')} {s.error(role)} {s.error(f'FAILED: {error[:80]}')}"
        
        elif event_type == EventType.AGENT_DESPAWNED:
            role = data.get("role", "?")
            depth = data.get("depth", 1)
            indent = "   " * depth
            return f"{prefix}{indent}{s.dim(f'🗑️ {role} cleaned up')}"
        
        # ═══════════════════════════════════════════════════════════
        # DEFAULT - System/custom events (dimmed)
        # ═══════════════════════════════════════════════════════════
        
        else:
            data_preview = str(data)[:100] if data else ""
            return f"{prefix}{s.dim(event_type.value)} {s.dim(data_preview)}"
    
    def _format_event_json(self, event: Event) -> str:
        """Format event as readable, colorized structured output."""
        s = self._styler
        data = event.data
        event_type = event.type
        lines: list[str] = []
        
        # Timestamp in cyan (converted to local)
        local_ts = to_local(event.timestamp)
        ts = local_ts.strftime("%H:%M:%S.%f")[:-3]
        
        # Color and emoji based on event type
        if event_type.category == "agent":
            if "responded" in event_type.value:
                icon = s.success("✓")
            elif "thinking" in event_type.value:
                icon = "🧠"
            elif "error" in event_type.value:
                icon = s.error("✗")
            else:
                icon = "▶"
            type_str = s.agent(event_type.value)
        elif event_type.category == "tool":
            if "result" in event_type.value:
                icon = s.success("✓")
            elif "error" in event_type.value:
                icon = s.error("✗")
            else:
                icon = "🔧"
            type_str = s.tool(event_type.value)
        elif event_type.category == "task":
            if "completed" in event_type.value:
                icon = s.success("✅")
            elif "failed" in event_type.value:
                icon = s.error("❌")
            else:
                icon = "🚀"
            type_str = s.success(event_type.value)
        else:
            icon = "•"
            type_str = s.dim(event_type.value)
        
        # Header with cyan timestamp
        lines.append(f"{s.info(ts)} {icon} {type_str}")
        
        # Details based on event type - simple indentation, no tree lines
        if event_type.category == "agent":
            agent = data.get("agent_name", "?")
            lines.append(f"    agent: {s.agent(agent)}")
            
            if "response_preview" in data or "response" in data:
                response = data.get("response_preview") or data.get("response", "")
                if len(response) > 150:
                    response = response[:150] + "..."
                first_line = response.split('\n')[0]
                lines.append(f"    response: {first_line}")
            
            if "duration_ms" in data:
                ms = data["duration_ms"]
                dur_str = f"{ms/1000:.1f}s" if ms > 1000 else f"{ms:.0f}ms"
                lines.append(f"    duration: {s.success(dur_str)}")
            
            if "error" in data:
                lines.append(f"    error: {s.error(data['error'])}")
        
        elif event_type.category == "tool":
            tool = data.get("tool", "?")
            lines.append(f"    tool: {s.tool(tool)}")
            
            if "args" in data and data["args"]:
                args_str = str(data["args"])[:80]
                lines.append(f"    args: {s.dim(args_str)}")
            
            if "result" in data or "result_preview" in data:
                result = data.get("result_preview", str(data.get("result", "")))[:80]
                lines.append(f"    result: {result}")
            
            if "duration_ms" in data:
                ms = data["duration_ms"]
                dur_str = f"{ms/1000:.1f}s" if ms > 1000 else f"{ms:.0f}ms"
                lines.append(f"    duration: {s.success(dur_str)}")
            
            if "error" in data:
                lines.append(f"    error: {s.error(data['error'])}")
        
        elif event_type.category == "task":
            task = data.get("task_name", data.get("task", ""))
            if task:
                lines.append(f"    task: {s.bold(task)}")
            
            if "duration_ms" in data:
                ms = data["duration_ms"]
                dur_str = f"{ms/1000:.1f}s" if ms > 1000 else f"{ms:.0f}ms"
                lines.append(f"    duration: {s.success(dur_str)}")
            
            if "error" in data:
                lines.append(f"    error: {s.error(data['error'])}")
        
        else:
            # Generic display for other events - simple indentation
            items = list(data.items())[:4]
            for key, value in items:
                val_str = str(value)[:60]
                lines.append(f"    {key}: {val_str}")
        
        return "\n".join(lines)
    
    # ==================== Deep Tracing ====================
    
    def _trace_event(self, event: Event) -> None:
        """Build execution trace graph from events."""
        event_type = event.type
        data = event.data
        ts = event.timestamp
        
        # Agent lifecycle tracking - start on INVOKED or THINKING
        if event_type in (EventType.AGENT_INVOKED, EventType.AGENT_THINKING):
            agent_name = data.get("agent_name", "unknown")
            
            # Only create node if agent not already tracked
            if agent_name not in self._current_agents:
                node_id = f"agent_{agent_name}_{generate_id()[:6]}"
                self._nodes[node_id] = {
                    "id": node_id,
                    "name": agent_name,
                    "type": "agent",
                    "status": "running",
                    "start_time": ts,
                    "end_time": None,
                    "input": data.get("input", ""),
                    "output": None,
                    "children": [],
                    "tools_called": [],
                }
                self._current_agents[agent_name] = node_id
                
                # Add edge from previous completed agent
                prev_completed = [
                    n for n in self._nodes.values() 
                    if n["type"] == "agent" and n["status"] == "completed"
                ]
                if prev_completed:
                    last = prev_completed[-1]
                    self._edges.append((last["id"], node_id, "→"))
        
        elif event_type == EventType.AGENT_RESPONDED:
            agent_name = data.get("agent_name", "unknown")
            if agent_name in self._current_agents:
                node_id = self._current_agents[agent_name]
                if node_id in self._nodes:
                    node = self._nodes[node_id]
                    node["status"] = "completed"
                    node["end_time"] = ts
                    node["output"] = data.get("response_preview") or data.get("response", "")[:200]
                    if node["start_time"]:
                        delta = (ts - node["start_time"]).total_seconds() * 1000
                        node["duration_ms"] = delta
                # Remove from current so next invocation creates new node
                del self._current_agents[agent_name]
        
        elif event_type == EventType.AGENT_ERROR:
            agent_name = data.get("agent_name", "unknown")
            if agent_name in self._current_agents:
                node_id = self._current_agents[agent_name]
                if node_id in self._nodes:
                    node = self._nodes[node_id]
                    node["status"] = "failed"
                    node["end_time"] = ts
                    node["error"] = data.get("error", "Unknown error")
                # Remove from current
                del self._current_agents[agent_name]
        
        # Tool call tracking
        elif event_type == EventType.TOOL_CALLED:
            tool_name = data.get("tool", "unknown")
            tool_id = f"tool_{tool_name}_{generate_id()[:6]}"
            tool_record = {
                "id": tool_id,
                "name": tool_name,
                "type": "tool",
                "status": "running",
                "start_time": ts,
                "args": data.get("args", {}),
                "result": None,
            }
            self._tool_calls.append(tool_record)
            self._current_tools[tool_name] = tool_record
            
            # Link to current agent
            for agent_name, agent_node_id in self._current_agents.items():
                if agent_node_id in self._nodes:
                    agent_node = self._nodes[agent_node_id]
                    if agent_node["status"] == "running":
                        agent_node["tools_called"].append(tool_id)
                        self._edges.append((agent_node_id, tool_id, "calls"))
        
        elif event_type == EventType.TOOL_RESULT:
            tool_name = data.get("tool", "unknown")
            if tool_name in self._current_tools:
                tool_record = self._current_tools[tool_name]
                tool_record["status"] = "completed"
                tool_record["end_time"] = ts
                tool_record["result"] = data.get("result_preview") or str(data.get("result", ""))[:200]
                if tool_record["start_time"]:
                    delta = (ts - tool_record["start_time"]).total_seconds() * 1000
                    tool_record["duration_ms"] = delta
        
        elif event_type == EventType.TOOL_ERROR:
            tool_name = data.get("tool", "unknown")
            if tool_name in self._current_tools:
                tool_record = self._current_tools[tool_name]
                tool_record["status"] = "failed"
                tool_record["end_time"] = ts
                tool_record["error"] = data.get("error", "Unknown error")
    
    def graph(self, style: str = "mermaid") -> str:
        """
        Get execution graph visualization.
        
        Shows the flow of execution through agents and tools.
        
        Args:
            style: 'mermaid' (default) or 'ascii'
        
        Returns:
            Graph visualization string
        
        Example:
            >>> print(observer.graph())
            ```mermaid
            graph TD
                agent_Researcher_a1b2c3[Researcher ✓ 1234ms]
                tool_search_d4e5f6[🔧 search ✓]
                agent_Researcher_a1b2c3 -->|calls| tool_search_d4e5f6
            ```
        """
        if not self._nodes and not self._tool_calls:
            return "No execution trace captured"
        
        if style == "mermaid":
            return self._graph_mermaid()
        else:
            return self._graph_ascii()
    
    def _graph_mermaid(self) -> str:
        """Generate Mermaid flowchart of execution."""
        lines = ["```mermaid", "graph TD"]
        
        # Style definitions
        lines.append("    classDef agent fill:#e1f5fe,stroke:#01579b")
        lines.append("    classDef tool fill:#fff3e0,stroke:#e65100")
        lines.append("    classDef error fill:#ffebee,stroke:#c62828")
        
        # Agent nodes
        for node in self._nodes.values():
            status_icon = "✓" if node["status"] == "completed" else "✗" if node["status"] == "failed" else "⋯"
            duration = f" {node.get('duration_ms', 0):.0f}ms" if "duration_ms" in node else ""
            label = f"{node['name']} {status_icon}{duration}"
            css_class = "error" if node["status"] == "failed" else "agent"
            lines.append(f"    {node['id']}[{label}]:::{css_class}")
        
        # Tool nodes
        for tool in self._tool_calls:
            status_icon = "✓" if tool["status"] == "completed" else "✗" if tool["status"] == "failed" else "⋯"
            duration = f" {tool.get('duration_ms', 0):.0f}ms" if "duration_ms" in tool else ""
            label = f"🔧 {tool['name']} {status_icon}{duration}"
            css_class = "error" if tool["status"] == "failed" else "tool"
            lines.append(f"    {tool['id']}[{label}]:::{css_class}")
        
        # Edges
        for from_id, to_id, label in self._edges:
            if label:
                lines.append(f"    {from_id} -->|{label}| {to_id}")
            else:
                lines.append(f"    {from_id} --> {to_id}")
        
        lines.append("```")
        return "\n".join(lines)
    
    def _graph_ascii(self) -> str:
        """Generate ASCII art graph of execution."""
        lines = ["Execution Graph:", "─" * 50]
        
        for node in self._nodes.values():
            status = "✓" if node["status"] == "completed" else "✗" if node["status"] == "failed" else "…"
            duration = f" ({node.get('duration_ms', 0):.0f}ms)" if "duration_ms" in node else ""
            lines.append(f"  [{status}] {node['name']}{duration}")
            
            # Show tools called by this agent
            for tool_id in node.get("tools_called", []):
                tool = next((t for t in self._tool_calls if t["id"] == tool_id), None)
                if tool:
                    t_status = "✓" if tool["status"] == "completed" else "✗"
                    t_duration = f" ({tool.get('duration_ms', 0):.0f}ms)" if "duration_ms" in tool else ""
                    lines.append(f"      └─ 🔧 {tool['name']} [{t_status}]{t_duration}")
        
        return "\n".join(lines)
    
    def execution_trace(self) -> dict[str, Any]:
        """
        Get full execution trace as structured data.
        
        Returns:
            Dict with nodes, edges, tools, and metadata
        
        Useful for:
        - Exporting to JSON
        - Building custom visualizations
        - Integration with tracing backends (Jaeger, etc.)
        """
        return {
            "trace_id": self._trace_id,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "nodes": list(self._nodes.values()),
            "edges": [{"from": f, "to": t, "label": l} for f, t, l in self._edges],
            "tools": self._tool_calls,
            "spans": self._spans,
            "metrics": dict(self._metrics),
        }
    
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
    
    def timeline(self, detailed: bool = False) -> str:
        """
        Get a timeline visualization of execution.
        
        Args:
            detailed: If True, show tool calls and nested details
        
        Returns:
            ASCII timeline string
        
        Example:
            >>> print(observer.timeline())
            Timeline:
            ──────────────────────────────────────────────────
            +  0.00s  [Researcher] started
            +  0.05s  └─ 🔧 web_search (345ms)
            +  0.40s  [Researcher] completed (401ms)
            +  0.41s  [Writer] started
            +  1.23s  [Writer] completed (820ms)
        """
        if not self._events and not self._nodes:
            return "No events observed"
        
        lines = ["Timeline:", "─" * 60]
        
        # Use trace nodes if available (richer data)
        if self._nodes:
            all_items: list[tuple[datetime, str, str]] = []
            
            for node in self._nodes.values():
                # Start event
                if node.get("start_time"):
                    all_items.append((
                        node["start_time"],
                        "start",
                        f"[{node['name']}] started"
                    ))
                    
                    # Show tools called (if detailed)
                    if detailed:
                        for tool_id in node.get("tools_called", []):
                            tool = next((t for t in self._tool_calls if t["id"] == tool_id), None)
                            if tool and tool.get("start_time"):
                                t_dur = f" ({tool.get('duration_ms', 0):.0f}ms)" if "duration_ms" in tool else ""
                                status = "✓" if tool["status"] == "completed" else "✗"
                                all_items.append((
                                    tool["start_time"],
                                    "tool",
                                    f"└─ 🔧 {tool['name']} {status}{t_dur}"
                                ))
                
                # End event
                if node.get("end_time"):
                    status = "✓" if node["status"] == "completed" else "✗"
                    duration = f" ({node.get('duration_ms', 0):.0f}ms)" if "duration_ms" in node else ""
                    all_items.append((
                        node["end_time"],
                        "end",
                        f"[{node['name']}] {node['status']}{duration}"
                    ))
            
            # Sort by timestamp
            all_items.sort(key=lambda x: x[0])
            
            if all_items:
                start_time = all_items[0][0]
                for ts, item_type, text in all_items:
                    delta = (ts - start_time).total_seconds()
                    indent = "    " if item_type == "tool" else "  "
                    lines.append(f"{indent}+{delta:>6.2f}s  {text}")
        
        # Fallback to event-based timeline
        else:
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
        Get a human-readable summary of execution.
        
        Includes:
        - Event counts
        - Duration
        - Agent execution stats
        - Tool call stats
        
        Returns:
            Summary string
        """
        m = self.metrics()
        lines = [
            "═" * 50,
            "Execution Summary",
            "═" * 50,
        ]
        
        # Timing
        if "duration_seconds" in m:
            lines.append(f"⏱  Duration: {m['duration_seconds']:.2f}s")
        
        lines.append(f"📊 Total events: {m.get('total_events', 0)}")
        
        # Agent stats
        if self._nodes:
            completed = sum(1 for n in self._nodes.values() if n["status"] == "completed")
            failed = sum(1 for n in self._nodes.values() if n["status"] == "failed")
            lines.append("")
            lines.append("Agents:")
            lines.append(f"  ✓ Completed: {completed}")
            if failed:
                lines.append(f"  ✗ Failed: {failed}")
            
            # Show each agent's timing
            for node in self._nodes.values():
                status = "✓" if node["status"] == "completed" else "✗"
                duration = f" ({node.get('duration_ms', 0):.0f}ms)" if "duration_ms" in node else ""
                lines.append(f"    [{status}] {node['name']}{duration}")
        
        # Tool stats
        if self._tool_calls:
            completed = sum(1 for t in self._tool_calls if t["status"] == "completed")
            failed = sum(1 for t in self._tool_calls if t["status"] == "failed")
            lines.append("")
            lines.append("Tools:")
            lines.append(f"  🔧 Calls: {len(self._tool_calls)}")
            lines.append(f"  ✓ Success: {completed}")
            if failed:
                lines.append(f"  ✗ Failed: {failed}")
        
        # Channel breakdown
        if "by_channel" in m and m["by_channel"]:
            lines.append("")
            lines.append("Events by channel:")
            for channel, count in m["by_channel"].items():
                lines.append(f"  {channel}: {count}")
        
        lines.append("═" * 50)
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all observed events, metrics, and trace data."""
        self._events.clear()
        self._metrics.clear()
        self._start_time = now_utc()
        
        # Clear trace data
        self._trace_id = generate_id()
        self._nodes.clear()
        self._edges.clear()
        self._current_agents.clear()
        self._tool_calls.clear()
        self._current_tools.clear()
        self._spans.clear()
        self._span_stack.clear()
    
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
