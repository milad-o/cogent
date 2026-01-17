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

Example - Subscribe to LLM Events (Opt-in):
    ```python
    # By default, LLM events show subtle presence only
    # To see detailed LLM request/response content, subscribe to LLM channel
    observer = Observer(
        level=ObservabilityLevel.DEBUG,
        channels=[Channel.AGENTS, Channel.TOOLS, Channel.LLM],  # Add LLM channel
    )

    # Or use TRACE level to see all LLM details
    observer = Observer.trace()  # Includes all channels
    ```
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from agenticflow.core.utils import generate_id, now_utc, to_local
from agenticflow.observability.progress import (
    OutputConfig,
    OutputFormat,
    ProgressTracker,
    Styler,
    Verbosity,
)
from agenticflow.observability.trace_record import Trace, TraceType

if TYPE_CHECKING:
    from agenticflow.observability.bus import TraceBus


# =============================================================================
# Formatting Utilities - Consistent output formatting
# =============================================================================

def _format_agent_name(name: str) -> str:
    """Format agent name with brackets (no truncation)."""
    return f"[{name}]"


def _format_duration(duration_ms: float) -> str:
    """Consistent duration formatting."""
    if duration_ms < 1000:
        return f"({duration_ms:.0f}ms)"
    elif duration_ms < 60000:
        return f"({duration_ms/1000:.1f}s)"
    else:
        mins = int(duration_ms / 60000)
        secs = (duration_ms % 60000) / 1000
        return f"({mins}m {secs:.0f}s)"


def _truncate_smart(text: str, max_chars: int = 500) -> str:
    """Smart truncation that respects word boundaries and adds clear marker."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    # Find last space before max_chars
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')

    # Use the space if it's reasonably close to the limit
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]

    return truncated + "... (truncated)"


def _wrap_content(text: str, indent: str, max_width: int = 80) -> str:
    """Wrap and indent content consistently."""
    if not text:
        return ""

    words = text.split()
    lines = []
    current_line = []
    current_length = len(indent)

    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_length + word_len > max_width and current_line:
            lines.append(indent + " ".join(current_line))
            current_line = [word]
            current_length = len(indent) + len(word)
        else:
            current_line.append(word)
            current_length += word_len

    if current_line:
        lines.append(indent + " ".join(current_line))

    return "\n".join(lines)


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
    - LLM: Raw LLM request/response events (opt-in for deep debugging)
    - STREAMING: Token-by-token LLM output streaming
    - MEMORY: Memory read/write/search operations
    - RETRIEVAL: RAG retrieval pipeline events
    - DOCUMENTS: Document loading and splitting
    - MCP: Model Context Protocol server events
    - REACTIVE: Reactive flow orchestration events
    - SYSTEM: System-level events
    - RESILIENCE: Retries, circuit breakers, fallbacks
    - ALL: Everything
    """

    AGENTS = "agents"
    TOOLS = "tools"
    MESSAGES = "messages"
    TASKS = "tasks"
    LLM = "llm"
    STREAMING = "streaming"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"
    DOCUMENTS = "documents"
    MCP = "mcp"
    REACTIVE = "reactive"
    SYSTEM = "system"
    RESILIENCE = "resilience"
    ALL = "all"


# Map channels to event types
CHANNEL_EVENTS: dict[Channel, set[TraceType]] = {
    Channel.AGENTS: {
        TraceType.AGENT_REGISTERED,
        TraceType.AGENT_UNREGISTERED,
        TraceType.AGENT_INVOKED,
        TraceType.AGENT_THINKING,
        TraceType.AGENT_REASONING,
        TraceType.AGENT_ACTING,
        TraceType.AGENT_RESPONDED,
        TraceType.AGENT_ERROR,
        TraceType.AGENT_STATUS_CHANGED,
        # Spawning events
        TraceType.AGENT_SPAWNED,
        TraceType.AGENT_SPAWN_COMPLETED,
        TraceType.AGENT_SPAWN_FAILED,
        TraceType.AGENT_DESPAWNED,
        # User/Output events (part of agent interaction)
        TraceType.USER_INPUT,
        TraceType.USER_FEEDBACK,
        TraceType.OUTPUT_GENERATED,
        TraceType.OUTPUT_STREAMED,
    },
    Channel.LLM: {
        TraceType.LLM_REQUEST,
        TraceType.LLM_RESPONSE,
        TraceType.LLM_TOOL_DECISION,
    },
    Channel.TOOLS: {
        TraceType.TOOL_REGISTERED,
        TraceType.TOOL_CALLED,
        TraceType.TOOL_RESULT,
        TraceType.TOOL_ERROR,
    },
    Channel.MESSAGES: {
        TraceType.MESSAGE_SENT,
        TraceType.MESSAGE_RECEIVED,
        TraceType.MESSAGE_BROADCAST,
    },
    Channel.TASKS: {
        TraceType.TASK_CREATED,
        TraceType.TASK_SCHEDULED,
        TraceType.TASK_STARTED,
        TraceType.TASK_BLOCKED,
        TraceType.TASK_UNBLOCKED,
        TraceType.TASK_COMPLETED,
        TraceType.TASK_FAILED,
        TraceType.TASK_CANCELLED,
        TraceType.TASK_RETRYING,
        TraceType.SUBTASK_SPAWNED,
        TraceType.SUBTASK_COMPLETED,
        TraceType.SUBTASKS_AGGREGATED,
    },
    Channel.SYSTEM: {
        TraceType.SYSTEM_STARTED,
        TraceType.SYSTEM_STOPPED,
        TraceType.SYSTEM_ERROR,
        TraceType.CLIENT_CONNECTED,
        TraceType.CLIENT_DISCONNECTED,
        TraceType.CLIENT_MESSAGE,
    },
    Channel.RESILIENCE: {
        TraceType.TASK_RETRYING,
        TraceType.TOOL_ERROR,
        TraceType.AGENT_ERROR,
    },
    Channel.STREAMING: {
        TraceType.STREAM_START,
        TraceType.TOKEN_STREAMED,
        TraceType.STREAM_TOOL_CALL,
        TraceType.STREAM_END,
        TraceType.STREAM_ERROR,
    },
    Channel.MEMORY: {
        TraceType.MEMORY_READ,
        TraceType.MEMORY_WRITE,
        TraceType.MEMORY_SEARCH,
        TraceType.MEMORY_DELETE,
        TraceType.MEMORY_CLEAR,
        TraceType.THREAD_CREATED,
        TraceType.THREAD_MESSAGE_ADDED,
    },
    Channel.RETRIEVAL: {
        TraceType.RETRIEVAL_START,
        TraceType.RETRIEVAL_COMPLETE,
        TraceType.RETRIEVAL_ERROR,
        TraceType.RERANK_START,
        TraceType.RERANK_COMPLETE,
        TraceType.FUSION_APPLIED,
        TraceType.VECTORSTORE_ADD,
        TraceType.VECTORSTORE_SEARCH,
        TraceType.VECTORSTORE_DELETE,
    },
    Channel.DOCUMENTS: {
        TraceType.DOCUMENT_LOADED,
        TraceType.DOCUMENT_SPLIT,
        TraceType.DOCUMENT_ENRICHED,
    },
    Channel.MCP: {
        TraceType.MCP_SERVER_CONNECTING,
        TraceType.MCP_SERVER_CONNECTED,
        TraceType.MCP_SERVER_DISCONNECTED,
        TraceType.MCP_SERVER_ERROR,
        TraceType.MCP_TOOLS_DISCOVERED,
        TraceType.MCP_TOOL_CALLED,
        TraceType.MCP_TOOL_RESULT,
        TraceType.MCP_TOOL_ERROR,
    },
    Channel.REACTIVE: {
        TraceType.REACTIVE_FLOW_STARTED,
        TraceType.REACTIVE_FLOW_COMPLETED,
        TraceType.REACTIVE_FLOW_FAILED,
        TraceType.REACTIVE_EVENT_EMITTED,
        TraceType.REACTIVE_EVENT_PROCESSED,
        TraceType.REACTIVE_AGENT_TRIGGERED,
        TraceType.REACTIVE_AGENT_COMPLETED,
        TraceType.REACTIVE_AGENT_FAILED,
        TraceType.REACTIVE_NO_MATCH,
        TraceType.REACTIVE_ROUND_STARTED,
        TraceType.REACTIVE_ROUND_COMPLETED,
        # Skill events
        TraceType.SKILL_ACTIVATED,
        TraceType.SKILL_DEACTIVATED,
    },
}


@dataclass
class ObservedEvent:
    """An event captured by the observer."""

    event: Trace
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
    truncate: int = 0  # General content truncation (0 = no limit)
    truncate_tool_args: int = 0  # Tool arguments (0 = no limit)
    truncate_tool_results: int = 0  # Tool results (0 = no limit)
    truncate_messages: int = 0  # Message content (0 = no limit)
    use_colors: bool = True

    # Token tracking
    track_tokens: bool = True  # Track token usage from LLM responses
    show_token_usage: bool = True  # Display token counts in output
    show_cost_estimates: bool = False  # Show estimated costs (requires model pricing)

    # Progress tracking
    show_progress_steps: bool = False  # Show step N/M progress indicators

    # Callbacks
    on_event: Callable[[Trace], None] | None = None
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
        # Mid-level API: single output limit (applies to all content types)
        max_output: int | None = None,
        # Low-level API: fine-grained truncation (advanced users)
        truncate: int | None = None,
        truncate_tool_args: int | None = None,
        truncate_tool_results: int | None = None,
        truncate_messages: int | None = None,
        use_colors: bool = True,
        # Token tracking
        track_tokens: bool = True,
        show_token_usage: bool = True,
        show_cost_estimates: bool = False,
        # Progress tracking
        show_progress_steps: bool = False,
        # Callbacks
        on_event: Callable[[Trace], None] | None = None,
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

        API Levels:
            - **High level**: Use presets like `Observer.minimal()`, `Observer.verbose()`, etc.
            - **Mid level**: Use `max_output` for simple output limiting
            - **Low level**: Use individual `truncate_*` params for fine control

        Args:
            level: Verbosity level (OFF through TRACE)
            channels: Which event channels to observe (default: ALL)
            stream: Output stream (default: stdout)
            format: Output format (TEXT, RICH, JSON)
            show_timestamps: Show timestamps in output
            show_duration: Show operation duration
            show_trace_ids: Show correlation IDs
            max_output: Max chars for ALL content types (None = no limit).
                        Simple API - use this instead of individual truncate params.
            truncate: Max chars for general content (advanced, overrides max_output)
            truncate_tool_args: Max chars for tool arguments (advanced)
            truncate_tool_results: Max chars for tool results (advanced)
            truncate_messages: Max chars for messages (advanced)
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
        # Resolve truncation: low-level overrides mid-level, default is 0 (no limit)
        base_limit = max_output if max_output is not None else 0
        final_truncate = truncate if truncate is not None else base_limit
        final_truncate_tool_args = truncate_tool_args if truncate_tool_args is not None else base_limit
        final_truncate_tool_results = truncate_tool_results if truncate_tool_results is not None else base_limit
        final_truncate_messages = truncate_messages if truncate_messages is not None else base_limit

        self.config = ObserverConfig(
            level=level,
            channels=set(channels) if channels else {Channel.ALL},
            stream=stream or sys.stdout,
            format=format,
            show_timestamps=show_timestamps,
            show_duration=show_duration,
            show_trace_ids=show_trace_ids,
            truncate=final_truncate,
            truncate_tool_args=final_truncate_tool_args,
            truncate_tool_results=final_truncate_tool_results,
            truncate_messages=final_truncate_messages,
            use_colors=use_colors,
            track_tokens=track_tokens,
            show_token_usage=show_token_usage,
            show_cost_estimates=show_cost_estimates,
            show_progress_steps=show_progress_steps,
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
        self._attached_bus: TraceBus | None = None

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

        # Token usage tracking
        self._token_usage: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})  # agent_name -> token counts
        self._total_tokens: dict[str, int] = {"input": 0, "output": 0, "total": 0}

        # Progress step tracking
        self._progress_steps: dict[str, dict[str, Any]] = {}  # agent_name -> {current: int, total: int, description: str}

        # State change tracking (for diff visualization)
        self._state_snapshots: dict[str, dict[str, Any]] = {}  # entity_id -> latest state

        # Error context enhancement
        self._error_suggestions: dict[str, list[str]] = {  # error_pattern -> suggestions
            "permission denied": [
                "Check file/directory permissions",
                "Verify the process has necessary access rights",
                "Consider running with elevated permissions if appropriate"
            ],
            "connection refused": [
                "Verify the target service is running",
                "Check network connectivity",
                "Confirm the port number is correct"
            ],
            "timeout": [
                "Check network connectivity",
                "Verify the service is responding",
                "Consider increasing timeout duration"
            ],
            "not found": [
                "Verify the resource exists",
                "Check the path or identifier is correct",
                "Ensure required dependencies are installed"
            ],
            "invalid credentials": [
                "Verify API keys and secrets are correct",
                "Check credentials haven't expired",
                "Ensure environment variables are set"
            ],
        }

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
            channels=[Channel.AGENTS, Channel.TASKS, Channel.REACTIVE],
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
            channels=[Channel.AGENTS, Channel.TASKS, Channel.REACTIVE],
            show_duration=True,
            truncate=500,  # Show substantial content
        )

    @classmethod
    def detailed(cls) -> Observer:
        """Create observer showing tool calls and timing."""
        return cls(
            level=ObservabilityLevel.DETAILED,
            channels=[Channel.AGENTS, Channel.TOOLS, Channel.TASKS, Channel.REACTIVE],
            show_timestamps=True,
            show_duration=True,
        )

    @classmethod
    def debug(cls) -> Observer:
        """
        Create observer showing everything except raw LLM content.

        To include LLM request/response details, add Channel.LLM explicitly:
            observer = Observer(
                level=ObservabilityLevel.DEBUG,
                channels=[Channel.AGENTS, Channel.TOOLS, Channel.LLM, ...],
            )
        """
        return cls(
            level=ObservabilityLevel.DEBUG,
            channels=[
                Channel.AGENTS, Channel.TOOLS, Channel.MESSAGES,
                Channel.TASKS, Channel.STREAMING, Channel.MEMORY,
                Channel.RETRIEVAL, Channel.DOCUMENTS, Channel.MCP,
                Channel.REACTIVE, Channel.SYSTEM, Channel.RESILIENCE,
            ],
            show_timestamps=True,
            show_duration=True,
            show_trace_ids=True,
            truncate=0,
        )

    @classmethod
    def trace(cls) -> Observer:
        """
        Create observer with maximum observability (excluding raw LLM content).

        Shows everything + builds execution graph.
        After running, call:
        - observer.graph() - Mermaid diagram of execution
        - observer.timeline(detailed=True) - chronological view
        - observer.summary() - stats and metrics
        - observer.execution_trace() - structured data for export

        To include LLM request/response details, add Channel.LLM explicitly:
            observer = Observer(
                level=ObservabilityLevel.TRACE,
                channels=[Channel.AGENTS, Channel.TOOLS, Channel.LLM, ...],
            )
        """
        return cls(
            level=ObservabilityLevel.TRACE,
            channels=[
                Channel.AGENTS, Channel.TOOLS, Channel.MESSAGES,
                Channel.TASKS, Channel.STREAMING, Channel.MEMORY,
                Channel.RETRIEVAL, Channel.DOCUMENTS, Channel.MCP,
                Channel.REACTIVE, Channel.SYSTEM, Channel.RESILIENCE,
            ],
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

    def attach(self, event_bus: TraceBus) -> None:
        """
        Attach observer to an event bus.

        Called automatically when observer is passed to Flow.

        Args:
            event_bus: The event bus to observe.
        """
        # Prevent duplicate attachment to the same bus
        if self._attached_bus is event_bus:
            return

        # Detach from previous bus if any
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

    def _should_observe(self, event: Trace) -> tuple[bool, Channel | None]:
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

    def _get_level_for_event(self, event: Trace) -> ObservabilityLevel:
        """Get minimum level required to see this event."""
        # Result-level events
        if event.type in {TraceType.TASK_COMPLETED, TraceType.TASK_FAILED}:
            return ObservabilityLevel.RESULT

        # Progress-level events - these are the main milestones
        if event.type in {
            TraceType.AGENT_INVOKED,
            TraceType.AGENT_RESPONDED,
            TraceType.AGENT_THINKING,  # Show thinking at progress level
            TraceType.AGENT_REASONING,  # Show reasoning at progress level
            TraceType.TASK_STARTED,
            # User interaction events at progress level
            TraceType.USER_INPUT,
            TraceType.OUTPUT_GENERATED,
            # Spawning events at progress - important milestones
            TraceType.AGENT_SPAWNED,
            TraceType.AGENT_SPAWN_COMPLETED,
            TraceType.AGENT_SPAWN_FAILED,
            # MCP server connection at progress level (important milestones)
            TraceType.MCP_SERVER_CONNECTED,
            TraceType.MCP_TOOLS_DISCOVERED,
        }:
            return ObservabilityLevel.PROGRESS

        # Detailed-level events
        if event.type in {
            TraceType.TOOL_CALLED,
            TraceType.TOOL_RESULT,
            TraceType.TOOL_ERROR,
            TraceType.AGENT_ACTING,
            TraceType.TASK_RETRYING,
            TraceType.AGENT_DESPAWNED,  # Cleanup at detailed level
            # MCP tool calls at detailed level (same as regular tools)
            TraceType.MCP_TOOL_CALLED,
            TraceType.MCP_TOOL_RESULT,
            TraceType.MCP_TOOL_ERROR,
            # VectorStore operations at detailed level
            TraceType.VECTORSTORE_ADD,
            TraceType.VECTORSTORE_SEARCH,
            TraceType.VECTORSTORE_DELETE,
        }:
            return ObservabilityLevel.DETAILED

        # LLM events - opt-in only (requires explicit LLM channel subscription)
        # Show subtle presence at DEBUG, full details at TRACE
        if event.type in {
            TraceType.LLM_REQUEST,
            TraceType.LLM_RESPONSE,
            TraceType.LLM_TOOL_DECISION,
        }:
            return ObservabilityLevel.DEBUG

        # Debug-level events
        if event.type in {
            TraceType.AGENT_STATUS_CHANGED,
            TraceType.MESSAGE_SENT,
            TraceType.MESSAGE_RECEIVED,
            # MCP connecting/disconnecting at debug level
            TraceType.MCP_SERVER_CONNECTING,
            TraceType.MCP_SERVER_DISCONNECTED,
            TraceType.MCP_SERVER_ERROR,
        }:
            return ObservabilityLevel.DEBUG

        # Streaming events - DETAILED level (same as tool calls)
        # STREAM_START/END at DETAILED, TOKEN_STREAMED at DEBUG for less noise
        if event.type in {
            TraceType.STREAM_START,
            TraceType.STREAM_END,
            TraceType.STREAM_TOOL_CALL,
            TraceType.STREAM_ERROR,
        }:
            return ObservabilityLevel.DETAILED

        if event.type == TraceType.TOKEN_STREAMED:
            return ObservabilityLevel.DEBUG  # Individual tokens at debug level

        # Reactive flow events - tiered visibility
        # Core milestones at PROGRESS level
        if event.type in {
            TraceType.REACTIVE_FLOW_STARTED,
            TraceType.REACTIVE_FLOW_COMPLETED,
            TraceType.REACTIVE_FLOW_FAILED,
            TraceType.REACTIVE_AGENT_TRIGGERED,
            TraceType.REACTIVE_AGENT_COMPLETED,
            TraceType.REACTIVE_AGENT_FAILED,
            TraceType.SKILL_ACTIVATED,  # Skill activation is a key milestone
        }:
            return ObservabilityLevel.PROGRESS

        # Detailed reactive events
        if event.type in {
            TraceType.REACTIVE_EVENT_EMITTED,
            TraceType.REACTIVE_EVENT_PROCESSED,
            TraceType.REACTIVE_NO_MATCH,
        }:
            return ObservabilityLevel.DETAILED

        # Round info at DEBUG (verbose)
        if event.type in {
            TraceType.REACTIVE_ROUND_STARTED,
            TraceType.REACTIVE_ROUND_COMPLETED,
        }:
            return ObservabilityLevel.DEBUG

        # Trace-level: everything else
        return ObservabilityLevel.TRACE

    def _handle_event(self, event: Trace) -> None:
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

    def _get_channel_for_event(self, event: Trace) -> Channel:
        """Get the channel for an event."""
        for channel, event_types in CHANNEL_EVENTS.items():
            if event.type in event_types:
                return channel
        return Channel.SYSTEM

    def _dispatch_callbacks(self, event: Trace, channel: Channel | None) -> None:
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
            if event.type in {TraceType.AGENT_ERROR, TraceType.TOOL_ERROR, TraceType.TASK_FAILED, TraceType.STREAM_ERROR}:
                source = event.data.get("agent_name") or event.data.get("tool") or "unknown"
                error = event.data.get("error") or event.data.get("message", "Unknown error")
                self.config.on_error(source, error)

        # Streaming callback
        if self.config.on_stream and channel == Channel.STREAMING:
            agent_name = event.data.get("agent_name") or event.data.get("agent", "unknown")
            if event.type == TraceType.TOKEN_STREAMED:
                token = event.data.get("token", event.data.get("content", ""))
                self.config.on_stream(agent_name, token, event.data)
            else:
                action = event.type.value.split(".")[-1]  # "start", "end", "tool_call", "error"
                self.config.on_stream(agent_name, action, event.data)

    def _format_event(self, event: Trace) -> str | None:
        """Format an event for output with rich colors and formatting.

        Returns None if the event should be suppressed (e.g., duplicate thinking events).
        """
        if self.config.format == OutputFormat.JSON:
            return self._format_event_json(event)

        def _normalize_content(content: Any) -> str:
            """Normalize arbitrary content to a plain string for safe rendering."""
            if content is None:
                return ""
            if isinstance(content, (list, tuple)):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(item.get("text", "") or item.get("content", "") or str(item))
                    else:
                        parts.append(str(item))
                return " ".join(p for p in parts if p)
            try:
                return str(content)
            except Exception:
                return ""

        def _truncate_text(text: str, *, max_len: int | None = None) -> str:
            """Truncate text according to observer config.

            If `self.config.truncate` (or `max_len`) is 0, truncation is disabled.
            """
            if not text:
                return ""
            limit = self.config.truncate if max_len is None else max_len
            if not limit:
                return text
            if len(text) <= limit:
                return text
            return text[:limit] + "..."

        s = self._styler
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
        # USER & OUTPUT EVENTS - Consistent [Name] format
        # Format: [User] 👤 input: content
        #         [Agent] 📤 output (duration)
        # ═══════════════════════════════════════════════════════════

        if event_type == TraceType.USER_INPUT:
            content = _normalize_content(data.get("content", data.get("input", "")))
            data.get("source", "user")
            content = _truncate_smart(content.replace("\n", " ").strip(), self.config.truncate)
            formatted_name = _format_agent_name("User")
            return f"{prefix}{s.info(formatted_name)} {s.dim('[input]')}: {content}"

        elif event_type == TraceType.USER_FEEDBACK:
            feedback = data.get("feedback", data.get("content", ""))
            decision = data.get("decision", "")
            formatted_name = _format_agent_name("User")
            if decision:
                return f"{prefix}{s.info(formatted_name)} {s.dim('[feedback]')}: {decision}"
            feedback = _truncate_smart(str(feedback).replace("\n", " ").strip(), self.config.truncate)
            return f"{prefix}{s.info(formatted_name)} {s.dim('[feedback]')}: {feedback}"

        elif event_type == TraceType.OUTPUT_GENERATED:
            content = _normalize_content(data.get("content", data.get("output", "")))
            agent_name = data.get("agent_name", "")
            flow_name = data.get("flow_name", "")

            # Duration if available
            duration_str = ""
            if self.config.show_duration and "duration_ms" in data:
                duration_str = " " + _format_duration(data['duration_ms'])

            # Format: [Agent] [output] OR [flow-name] [flow-output] (no duration - shown in completed)
            #         content wrapped and indented
            if agent_name:
                formatted_name = _format_agent_name(agent_name)
                header = f"{prefix}{s.agent(formatted_name)} {s.dim('[output]')}"
            elif flow_name:
                formatted_flow = _format_agent_name(flow_name)
                # Flow output keeps duration since there's no separate "completed" event
                header = f"{prefix}{s.info(formatted_flow)} {s.dim('[flow-output]')}{s.success(duration_str)}"
            else:
                header = f"{prefix}{s.dim('[output]')}"

            if content:
                content_clean = _truncate_smart(content.replace("\n", " ").strip(), self.config.truncate)
                indent = " " * 2
                wrapped = _wrap_content(content_clean, indent)
                return f"{header}\n{wrapped}"
            else:
                return header

        elif event_type == TraceType.OUTPUT_STREAMED:
            # Similar to TOKEN_STREAMED but for final user-facing output
            token = data.get("token", data.get("content", ""))
            if self.config.level >= ObservabilityLevel.DEBUG:
                self.config.stream.write(token)
                self.config.stream.flush()
                return ""  # No newline for streaming
            return ""  # Don't show at lower levels

        # ═══════════════════════════════════════════════════════════
        # AGENT EVENTS - Purple/Magenta theme with consistent formatting
        # Format: [AgentName] ICON status (duration) label
        #         content indented consistently
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.AGENT_INVOKED:
            agent_name = data.get('agent_name', '?')
            formatted_name = _format_agent_name(agent_name)
            return f"{prefix}{s.agent(formatted_name)} {s.dim('[starting]')}"

        elif event_type == TraceType.AGENT_THINKING:
            agent_name = data.get('agent_name', '?')
            prompt_preview = data.get('prompt_preview', '')
            data.get('iteration', 1)
            data.get('max_iterations', 0)
            current_step = data.get('current_step', 0)
            total_steps = data.get('total_steps', 0)
            step_description = data.get('step_description', '')

            # Track thinking iterations per agent to reduce noise
            if agent_name not in self._agent_thinking_count:
                self._agent_thinking_count[agent_name] = 0
            self._agent_thinking_count[agent_name] += 1
            count = self._agent_thinking_count[agent_name]

            # Update progress tracking if step info provided
            if current_step and total_steps:
                self._progress_steps[agent_name] = {
                    "current": current_step,
                    "total": total_steps,
                    "description": step_description
                }

            # Show thinking event with prompt preview
            formatted_name = _format_agent_name(agent_name)

            # Add progress indicator if enabled and available
            progress_suffix = ""
            if self.config.show_progress_steps and agent_name in self._progress_steps:
                progress = self._progress_steps[agent_name]
                step_text = f" Step {progress['current']}/{progress['total']}"
                if progress['description']:
                    step_text += f": {progress['description']}"
                progress_suffix = s.dim(step_text)

            base_msg = f"{prefix}{s.agent(formatted_name)} {s.dim('[thinking]')}"
            
            # By default, don't show the prompt we send to LLM
            # The actual AI thoughts are shown in AGENT_REASONING events
            if prompt_preview and self.config.level >= ObservabilityLevel.DETAILED:
                preview = prompt_preview.strip()
                base_msg += f" {s.dim(preview)}"
            
            return base_msg + progress_suffix if progress_suffix else base_msg

        elif event_type == TraceType.AGENT_REASONING:
            agent_name = data.get('agent_name', '?')
            round_num = data.get('round', 1)
            reasoning_type = data.get('reasoning_type', 'thinking')
            thought_preview = data.get('thought_preview', '')
            phase = data.get('phase', 'thinking')

            # For start phase, just show a simple indicator
            if phase == 'start':
                formatted_name = _format_agent_name(agent_name)
                return f"{prefix}{s.agent(formatted_name)} {s.dim('[reasoning started]')}"
            
            # Skip complete phase
            if phase == 'complete':
                return None

            formatted_name = _format_agent_name(agent_name)
            header = (
                f"{prefix}{s.agent(formatted_name)} "
                f"{s.dim(f'[reasoning round {round_num}]')}"
            )

            if thought_preview and self.config.level >= ObservabilityLevel.PROGRESS:
                # Show actual AI thoughts at PROGRESS level - NO truncation
                lines.append(header)
                preview = str(thought_preview)
                indent = " " * (len(formatted_name) + 1)
                for line in preview.split('\n'):
                    lines.append(f"{indent}{s.dim(line)}")
                return "\n".join(lines)
            else:
                return header

        elif event_type == TraceType.AGENT_ACTING:
            agent_name = data.get('agent_name', '?')
            formatted_name = _format_agent_name(agent_name)
            return f"{prefix}{s.agent(formatted_name)} {s.dim('[acting]')}"

        elif event_type == TraceType.AGENT_RESPONDED:
            agent_name = data.get('agent_name', '?')
            result = data.get("response_preview") or data.get("response") or data.get("result_preview", "")
            result = _normalize_content(result)

            # Reset thinking count for this agent
            if agent_name in self._agent_thinking_count:
                del self._agent_thinking_count[agent_name]

            # Duration formatting
            duration_str = ""
            if self.config.show_duration and "duration_ms" in data:
                duration_str = " " + _format_duration(data['duration_ms'])

            # Format: [Agent] [completed] (Xs) - just status, no content duplication
            formatted_name = _format_agent_name(agent_name)
            return f"{prefix}{s.agent(formatted_name)} {s.success('[completed]')}{s.success(duration_str)}"

        elif event_type == TraceType.AGENT_ERROR:
            agent_name = data.get('agent_name', '?')
            error = data.get('error', 'Unknown error')
            error_str = str(error)
            formatted_name = _format_agent_name(agent_name)

            # Build error message with context
            error_lines = [f"{prefix}{s.agent(formatted_name)} {s.error('[error]')} {s.error(error_str)}"]

            # Add error suggestions if pattern matches
            error_lower = error_str.lower()
            suggestions = []
            for pattern, pattern_suggestions in self._error_suggestions.items():
                if pattern in error_lower:
                    suggestions.extend(pattern_suggestions)
                    break

            # Add context from data if available
            context_info = []
            if "file" in data or "path" in data:
                file_path = data.get("file") or data.get("path")
                context_info.append(f"File: {file_path}")
            if "line" in data:
                context_info.append(f"Line: {data['line']}")
            if "tool" in data:
                context_info.append(f"Tool: {data['tool']}")

            # Add suggestions and context at DEBUG level
            if self.config.level >= ObservabilityLevel.DEBUG:
                indent = " " * (len(formatted_name) + 1)

                if context_info:
                    for info in context_info:
                        error_lines.append(f"{indent}{s.dim(info)}")

                if suggestions:
                    error_lines.append(f"{indent}{s.warning('Suggestions:')}")
                    for suggestion in suggestions[:3]:  # Limit to top 3
                        error_lines.append(f"{indent}  • {s.dim(suggestion)}")

            return "\n".join(error_lines)

        elif event_type == TraceType.AGENT_STATUS_CHANGED:
            # State change diff visualization
            agent_name = data.get('agent_name', '?')
            entity_id = data.get('entity_id', agent_name)
            old_state = data.get('old_state', {})
            new_state = data.get('new_state', {})
            formatted_name = _format_agent_name(agent_name)

            # Track state snapshots
            if entity_id and new_state:
                self._state_snapshots[entity_id] = new_state.copy()

            # Build diff visualization
            diff_lines = [f"{prefix}{s.agent(formatted_name)} {s.dim('[state-change]')}"]

            # Show at DETAILED level or higher
            if self.config.level >= ObservabilityLevel.DETAILED:
                indent = " " * (len(formatted_name) + 1)

                # Find changed fields
                all_keys = set(old_state.keys()) | set(new_state.keys())
                changes = []

                for key in sorted(all_keys):
                    old_val = old_state.get(key)
                    new_val = new_state.get(key)

                    if old_val != new_val:
                        # Format change
                        old_str = str(old_val) if old_val is not None else "null"
                        new_str = str(new_val) if new_val is not None else "null"

                        # Truncate long values
                        if len(old_str) > 40:
                            old_str = old_str[:37] + "..."
                        if len(new_str) > 40:
                            new_str = new_str[:37] + "..."

                        # Color: old in dim, new in bright
                        change_line = f"{key}: {s.dim(old_str)} {s.info('→')} {s.success(new_str)}"
                        changes.append(change_line)

                # Display changes
                if changes:
                    for change in changes[:5]:  # Limit to 5 changes
                        diff_lines.append(f"{indent}{change}")

                    if len(changes) > 5:
                        diff_lines.append(f"{indent}{s.dim(f'... and {len(changes) - 5} more changes')}")
                else:
                    diff_lines.append(f"{indent}{s.dim('(no changes detected)')}")

            return "\n".join(diff_lines)

        # ═══════════════════════════════════════════════════════════
        # TOOL EVENTS - Shown as children of agents with consistent indentation
        # Format: [Agent]   ↳ 🔧 tool_name
        #         [Agent]     → result preview
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.TOOL_CALLED:
            agent_name = data.get("agent_name", "")
            tool_name = data.get("tool_name", data.get("tool", "?"))
            args = data.get("args", {})

            formatted_name = _format_agent_name(agent_name) if agent_name else " " * 12

            args_str = ""
            if args and self.config.level >= ObservabilityLevel.DETAILED:
                args_preview = str(args)
                args_preview = _truncate_smart(args_preview, self.config.truncate_tool_args)
                indent = " " * (len(formatted_name) + 3)  # Align under tool name
                args_str = f"\n{indent}{s.dim(args_preview)}"

            return f"{prefix}{s.agent(formatted_name)} {s.dim('[tool-call]')} {s.tool(tool_name)}{args_str}"

        elif event_type == TraceType.TOOL_RESULT:
            agent_name = data.get("agent_name", "")
            tool_name = data.get("tool_name", data.get("tool", "?"))
            result = data.get("result_preview", str(data.get("result", "")))

            formatted_name = _format_agent_name(agent_name) if agent_name else " " * 12

            # Duration
            duration_str = ""
            if self.config.show_duration and "duration_ms" in data:
                duration_str = " " + _format_duration(data['duration_ms'])

            # Truncate result - show inline preview if short enough
            result_str = _truncate_smart(str(result), max_chars=self.config.truncate_tool_results)
            result_clean = result_str.replace("\n", " ").strip()

            # Format: [Agent] [tool-result] tool_name (duration) ⮕ result
            result_part = ""
            if result_clean and len(result_clean) < 120:
                result_part = f" {s.dim('⮕')} {result_clean}"
            elif result_clean:
                # Longer results on next line
                indent = " " * (len(formatted_name) + 3)
                result_part = f"\n{indent}{s.dim('⮕')} {result_clean}"

            return f"{prefix}{s.agent(formatted_name)} {s.success('[tool-result]')} {s.tool(tool_name)}{s.success(duration_str)}{result_part}"

        elif event_type == TraceType.TOOL_ERROR:
            agent_name = data.get("agent_name", "")
            tool_name = data.get("tool_name", data.get("tool", "?"))
            error = data.get("error", "Unknown error")
            error_str = str(error)
            formatted_name = _format_agent_name(agent_name) if agent_name else " " * 12

            # Build error message
            error_lines = [f"{prefix}{s.agent(formatted_name)} {s.error('[tool-error]')} {s.tool(tool_name)}: {s.error(error_str)}"]

            # Add error suggestions if pattern matches
            error_lower = error_str.lower()
            suggestions = []
            for pattern, pattern_suggestions in self._error_suggestions.items():
                if pattern in error_lower:
                    suggestions.extend(pattern_suggestions)
                    break

            # Add suggestions at DEBUG level
            if self.config.level >= ObservabilityLevel.DEBUG and suggestions:
                indent = " " * (len(formatted_name) + 1)
                error_lines.append(f"{indent}{s.warning('Suggestions:')}")
                for suggestion in suggestions[:3]:
                    error_lines.append(f"{indent}  • {s.dim(suggestion)}")

            return "\n".join(error_lines)

        # ═══════════════════════════════════════════════════════════
        # TASK EVENTS - Consistent task lifecycle tracking
        # Format: [Task] icon status (duration) description
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.TASK_CREATED:
            task = data.get("task", data.get("task_name", ""))
            task_truncated = _truncate_smart(task, max_chars=100)
            event_name = data.get("event_name", "")
            if event_name and event_name != "task.created":
                return f"{prefix}{s.success('📋 created')} {task_truncated} {s.dim(f'({event_name})')}"
            return f"{prefix}{s.success('📋 created')} {task_truncated}"

        elif event_type == TraceType.TASK_STARTED:
            task_name = data.get("task_name", data.get("task", "task"))
            task_truncated = _truncate_smart(task_name, max_chars=100)
            return f"{prefix}{s.success('▶ started')} {task_truncated}"

        elif event_type == TraceType.TASK_COMPLETED:
            duration_str = ""
            if self.config.show_duration and "duration_ms" in data:
                duration_str = f" {_format_duration(data['duration_ms'])}"
            task_result = data.get("result", "")
            if task_result:
                result_preview = _truncate_smart(str(task_result), max_chars=80)
                return f"{prefix}{s.success(f'✓ completed{duration_str}')}\n  {s.dim(result_preview)}"
            return f"{prefix}{s.success(f'✓ completed{duration_str}')}"

        elif event_type == TraceType.TASK_FAILED:
            error = data.get('error', 'unknown')
            error_truncated = _truncate_smart(str(error), max_chars=100)
            return f"{prefix}{s.error(f'✗ failed: {error_truncated}')}"

        elif event_type == TraceType.TASK_RETRYING:
            attempt = data.get("attempt", "?")
            max_retries = data.get("max_retries", "?")
            delay_str = ""
            if "delay" in data:
                delay_str = s.dim(f" in {data['delay']:.1f}s")
            return f"{prefix}{s.warning(f'🔄 retrying ({attempt}/{max_retries})')}{delay_str}"

        # ═══════════════════════════════════════════════════════════
        # MESSAGE EVENTS - Agent-to-agent communication (concise)
        # Format: 📤 Sender → Receiver: "preview"
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.MESSAGE_SENT:
            sender = data.get("sender_id", "?")
            receiver = data.get("receiver_id", "?")
            content = _normalize_content(data.get("content", ""))
            content_preview = _truncate_smart(content.replace("\n", " ").strip(), max_chars=self.config.truncate_messages)
            content_str = f': "{content_preview}"' if content_preview else ""
            return f"{prefix}{s.dim('📤')} {s.agent(sender)} {s.dim('→')} {s.agent(receiver)}{s.dim(content_str)}"

        elif event_type == TraceType.MESSAGE_RECEIVED:
            receiver = data.get("agent_name", data.get("agent", "?"))
            sender = data.get("from", "?")
            content = _normalize_content(data.get("content", ""))
            content_preview = _truncate_smart(content.replace("\n", " ").strip(), max_chars=self.config.truncate_messages)
            content_str = f': "{content_preview}"' if content_preview else ""
            return f"{prefix}{s.dim('📥')} {s.agent(sender)} {s.dim('→')} {s.agent(receiver)}{s.dim(content_str)}"

        elif event_type == TraceType.MESSAGE_BROADCAST:
            sender = data.get("sender_id", "?")
            topic = data.get("topic", "")
            topic_str = f" ({topic})" if topic else ""
            return f"{prefix}{s.dim('📢')} {s.agent(sender)} {s.dim(f'broadcast{topic_str}')}"

        # ═══════════════════════════════════════════════════════════
        # STREAMING EVENTS - Real-time LLM output with consistent formatting
        # Format: [Agent] ▸ streaming... → [Agent] ✓ stream complete (duration, tok/s)
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.STREAM_START:
            agent_name = data.get("agent_name", data.get("agent", "?"))
            formatted_name = _format_agent_name(agent_name)
            model = data.get("model", "")
            model_str = f" {s.dim(f'({model})')}" if model else ""
            # Track streaming state
            self._streaming_agents[agent_name] = {
                "start_time": event.timestamp,
                "token_count": 0,
            }
            self._stream_buffer[agent_name] = ""
            return f"{prefix}{s.agent(formatted_name)}  {s.info('▸ streaming...')}{model_str}"

        elif event_type == TraceType.TOKEN_STREAMED:
            agent_name = data.get("agent_name", data.get("agent", "?"))
            token = _normalize_content(data.get("token", data.get("content", "")))
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

        elif event_type == TraceType.STREAM_TOOL_CALL:
            agent_name = data.get("agent_name", data.get("agent", "?"))
            formatted_name = _format_agent_name(agent_name)
            tool_name = data.get("tool_name", data.get("tool", "?"))
            tool_args = data.get("args", {})
            args_preview = _truncate_smart(str(tool_args), max_chars=100) if tool_args else ""
            args_part = f"\n               {s.dim(args_preview)}" if args_preview else ""
            return f"{prefix}{s.agent(formatted_name)}    {s.info('↳')} {s.tool(f'🔧 {tool_name}')} {s.dim('(during stream)')}{args_part}"

        elif event_type == TraceType.STREAM_END:
            agent_name = data.get("agent_name", data.get("agent", "?"))
            formatted_name = _format_agent_name(agent_name)
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
                    duration_str = f" {_format_duration(duration_ms)}"
                    if tokens_per_sec > 0:
                        duration_str = f" {s.success(f'({duration_ms/1000:.1f}s, {tokens_per_sec:.0f} tok/s)')}"
                # Clean up
                del self._streaming_agents[agent_name]

            # Get accumulated content preview
            content_preview = ""
            if agent_name in self._stream_buffer:
                content = self._stream_buffer[agent_name]
                if content:
                    content_preview = _truncate_smart(content, max_chars=200)
                del self._stream_buffer[agent_name]

            # If we were at DEBUG level, add newline after streaming tokens
            if self.config.level >= ObservabilityLevel.DEBUG:
                self.config.stream.write("\n")

            header = f"{prefix}{s.agent(formatted_name)}  {s.success(f'✓ stream complete{duration_str}')}"
            if content_preview and self.config.level >= ObservabilityLevel.DETAILED:
                wrapped = _wrap_content(content_preview, indent="  ")
                return f"{header}\n{wrapped}"
            return header

        elif event_type == TraceType.STREAM_ERROR:
            agent_name = data.get("agent_name", data.get("agent", "?"))
            formatted_name = _format_agent_name(agent_name)
            error = data.get("error", "Unknown streaming error")
            error_truncated = _truncate_smart(str(error), max_chars=100)
            # Clean up streaming state
            self._streaming_agents.pop(agent_name, None)
            self._stream_buffer.pop(agent_name, None)
            return f"{prefix}{s.agent(formatted_name)}  {s.error(f'✗ stream error: {error_truncated}')}"

        # ═══════════════════════════════════════════════════════════
        # LLM OBSERVABILITY EVENTS - Subtle presence, details opt-in
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.LLM_REQUEST:
            agent_name = data.get("agent_name", "?")
            message_count = data.get("message_count", 0)
            tools_available = data.get("tools_available", [])
            formatted_name = _format_agent_name(agent_name)

            # Professional format with brackets
            tools_str = f", {len(tools_available)} tools" if tools_available else ""
            header = f"{prefix}{s.agent(formatted_name)} {s.dim(f'[llm-request] ({message_count} messages{tools_str})')}"

            # Details only at TRACE level or if explicitly subscribed to LLM channel
            if self.config.level >= ObservabilityLevel.TRACE:
                prompt = _normalize_content(data.get("prompt", ""))
                system_prompt = _normalize_content(data.get("system_prompt", ""))
                if system_prompt or prompt:
                    lines.append(header)
                    if system_prompt:
                        sys_preview = _truncate_text(system_prompt.replace("\n", " ").strip())
                        lines.append(f"      {s.dim(f'System: {sys_preview}')}")
                    if prompt:
                        prompt_preview = _truncate_text(prompt.replace("\n", " ").strip())
                        lines.append(f"      {s.dim(f'Prompt: {prompt_preview}')}")
                    return "\n".join(lines)
            return header

        elif event_type == TraceType.LLM_RESPONSE:
            agent_name = data.get("agent_name", "?")
            tool_calls = data.get("tool_calls", [])
            duration_ms = data.get("duration_ms", 0)
            formatted_name = _format_agent_name(agent_name)

            # Token usage tracking
            input_tokens = data.get("input_tokens", 0) or data.get("prompt_tokens", 0)
            output_tokens = data.get("output_tokens", 0) or data.get("completion_tokens", 0)
            total_tokens = data.get("total_tokens", 0) or (input_tokens + output_tokens)

            # Track tokens if enabled
            if self.config.track_tokens and total_tokens > 0:
                self._token_usage[agent_name]["input"] += input_tokens
                self._token_usage[agent_name]["output"] += output_tokens
                self._token_usage[agent_name]["total"] += total_tokens
                self._total_tokens["input"] += input_tokens
                self._total_tokens["output"] += output_tokens
                self._total_tokens["total"] += total_tokens

            # Duration formatting
            duration_str = ""
            if duration_ms:
                duration_str = " " + _format_duration(duration_ms)

            # Token usage display
            token_str = ""
            if self.config.show_token_usage and total_tokens > 0:
                # Format: ~850 tokens (650 in, 200 out)
                if input_tokens and output_tokens:
                    token_str = f" ~{total_tokens} tokens ({input_tokens} in, {output_tokens} out)"
                else:
                    token_str = f" ~{total_tokens} tokens"

            # Professional format with brackets
            tool_str = f", {len(tool_calls)} tools" if tool_calls else ""
            header = f"{prefix}{s.agent(formatted_name)} {s.dim(f'[llm-response]{duration_str}{token_str}{tool_str}')}"

            # Details only at TRACE level
            if self.config.level >= ObservabilityLevel.TRACE:
                content = _normalize_content(data.get("content", ""))
                if content:
                    lines.append(header)
                    content_preview = _truncate_text(content.replace("\n", " ").strip())
                    lines.append(f"      {s.dim(content_preview)}")
                    return "\n".join(lines)

            return header

        elif event_type == TraceType.LLM_TOOL_DECISION:
            agent_name = data.get("agent_name", "?")
            tools_selected = data.get("tools_selected", [])
            formatted_name = _format_agent_name(agent_name)

            # Professional format
            if tools_selected:
                tools_str = ", ".join(tools_selected[:3])
                if len(tools_selected) > 3:
                    tools_str += f" (+{len(tools_selected) - 3})"
                header = f"{prefix}{s.agent(formatted_name)} {s.dim('[tool-decision]')} {s.tool(tools_str)}"
            else:
                header = f"{prefix}{s.agent(formatted_name)} {s.dim('[tool-decision] none')}"

            # Show reasoning only at TRACE level
            if self.config.level >= ObservabilityLevel.TRACE:
                reasoning = data.get("reasoning", "")
                if reasoning:
                    lines.append(header)
                    reasoning_preview = _truncate_text(str(reasoning).replace("\n", " ").strip())
                    lines.append(f"      {s.dim('Reasoning:')} {reasoning_preview}")
                    return "\n".join(lines)

            return header

        # ═══════════════════════════════════════════════════════════
        # SPAWNING EVENTS - Dynamic agent creation
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.AGENT_SPAWNED:
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

        elif event_type == TraceType.AGENT_SPAWN_COMPLETED:
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

        elif event_type == TraceType.AGENT_SPAWN_FAILED:
            role = data.get("role", "?")
            error = data.get("error", "Unknown error")
            depth = data.get("depth", 1)

            indent = "   " * depth
            tree_char = "└─" if depth > 1 else ""
            return f"{prefix}{indent}{tree_char}{s.error('✗')} {s.error(role)} {s.error(f'FAILED: {error[:80]}')}"

        elif event_type == TraceType.AGENT_DESPAWNED:
            role = data.get("role", "?")
            depth = data.get("depth", 1)
            indent = "   " * depth
            return f"{prefix}{indent}{s.dim(f'🗑️ {role} cleaned up')}"

        # ═══════════════════════════════════════════════════════════
        # VECTORSTORE EVENTS - Document indexing and search
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.VECTORSTORE_ADD:
            count = data.get("count", 0)
            embedded = data.get("embedded", count)
            return f"{prefix}{s.success('📚')} {s.bold('VectorStore')} Added {s.agent(str(count))} documents ({embedded} embedded)"

        elif event_type == TraceType.VECTORSTORE_SEARCH:
            query = data.get("query", "?")[:50]
            k = data.get("k", "?")
            results = data.get("results", 0)
            top_score = data.get("top_score")
            score_str = f" (top: {top_score:.2f})" if top_score else ""
            return f"{prefix}{s.info('🔍')} {s.bold('VectorStore')} Search k={k} → {s.agent(str(results))} results{score_str}: {s.dim(query)}"

        elif event_type == TraceType.VECTORSTORE_DELETE:
            count = data.get("count", 0)
            clear = data.get("clear", False)
            if clear:
                return f"{prefix}{s.warning('🗑️')} {s.bold('VectorStore')} Cleared all {s.agent(str(count))} documents"
            return f"{prefix}{s.warning('🗑️')} {s.bold('VectorStore')} Deleted {s.agent(str(count))} documents"

        # ═══════════════════════════════════════════════════════════
        # MCP EVENTS - Model Context Protocol server interactions
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.MCP_SERVER_CONNECTING:
            server = data.get("server_name", "?")
            transport = data.get("transport", "?")
            return f"{prefix}{s.info('🔌')} {s.bold('MCP')} Connecting to {s.agent(server)} ({transport})..."

        elif event_type == TraceType.MCP_SERVER_CONNECTED:
            server = data.get("server_name", "?")
            transport = data.get("transport", "?")
            return f"{prefix}{s.success('✓')} {s.bold('MCP')} Connected to {s.agent(server)} ({transport})"

        elif event_type == TraceType.MCP_SERVER_DISCONNECTED:
            count = data.get("server_count", 0)
            tools = data.get("tool_count", 0)
            return f"{prefix}{s.dim('🔌')} {s.bold('MCP')} Disconnected ({count} servers, {tools} tools)"

        elif event_type == TraceType.MCP_SERVER_ERROR:
            server = data.get("server_name", "?")
            error = data.get("error", "Unknown error")[:80]
            return f"{prefix}{s.error('✗')} {s.bold('MCP')} {s.agent(server)} {s.error(f'Error: {error}')}"

        elif event_type == TraceType.MCP_TOOLS_DISCOVERED:
            server = data.get("server_name", "?")
            tool_names = data.get("tools", [])
            count = len(tool_names)
            if count > 0:
                tools_preview = ", ".join(tool_names[:5])
                if count > 5:
                    tools_preview += f" (+{count - 5} more)"
                return f"{prefix}{s.info('🔧')} {s.bold('MCP')} {s.agent(server)} discovered {s.tool(f'{count} tools')}: {tools_preview}"
            return f"{prefix}{s.dim('🔧')} {s.bold('MCP')} {s.agent(server)} no tools discovered"

        elif event_type == TraceType.MCP_TOOL_CALLED:
            server = data.get("server_name", "?")
            tool = data.get("tool_name", "?")
            args = data.get("arguments", {})
            args_preview = str(args)[:60] if args else ""
            return f"{prefix}{s.info('▶')} {s.bold('MCP')} {s.tool(tool)} ({server}) {s.dim(args_preview)}"

        elif event_type == TraceType.MCP_TOOL_RESULT:
            server = data.get("server_name", "?")
            tool = data.get("tool_name", "?")
            result = str(data.get("result_preview", ""))[:80]
            return f"{prefix}{s.success('✓')} {s.bold('MCP')} {s.tool(tool)} {s.dim(result)}"

        elif event_type == TraceType.MCP_TOOL_ERROR:
            server = data.get("server_name", "?")
            tool = data.get("tool_name", "?")
            error = data.get("error", "Unknown error")[:80]
            return f"{prefix}{s.error('✗')} {s.bold('MCP')} {s.tool(tool)} {s.error(f'Error: {error}')}"

        # ═══════════════════════════════════════════════════════════
        # REACTIVE FLOW EVENTS - Event-driven orchestration
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.REACTIVE_FLOW_STARTED:
            task = data.get("task", "")[:80]
            agents = data.get("agents", [])
            agent_count = len(agents)
            agents_preview = ", ".join(agents[:4])
            if agent_count > 4:
                agents_preview += f" (+{agent_count - 4})"
            header = f"{prefix}{s.info('[Flow]')} {s.info('[started]')}"
            lines.append(header)
            if task:
                lines.append(f"      {s.dim('Task:')} {task}")
            if agents:
                lines.append(f"      {s.dim('Agents:')} {s.agent(agents_preview)}")
            return "\n".join(lines)

        elif event_type == TraceType.REACTIVE_FLOW_COMPLETED:
            events_processed = data.get("events_processed", 0)
            reactions = data.get("reactions", 0)
            rounds = data.get("rounds", 0)
            execution_time_ms = data.get("execution_time_ms", 0)
            output_length = data.get("output_length", 0)

            duration_str = ""
            if execution_time_ms > 1000:
                duration_str = s.success(f" in {execution_time_ms/1000:.1f}s")
            else:
                duration_str = s.dim(f" in {execution_time_ms:.0f}ms")

            return f"{prefix}{s.info('[Flow]')} {s.success('[completed]')}{duration_str} {s.dim(f'({reactions} reactions, {rounds} rounds, {events_processed} events)')}"

        elif event_type == TraceType.REACTIVE_FLOW_FAILED:
            error = data.get("error", "Unknown error")[:100]
            rounds = data.get("rounds", 0)
            return f"{prefix}{s.info('[Flow]')} {s.error('[FAILED]')} {s.error(error)} {s.dim(f'(after {rounds} rounds)')}"

        elif event_type == TraceType.REACTIVE_EVENT_EMITTED:
            event_name = data.get("event_name", "?")
            return f"{prefix}{s.dim('[Trace]')} {s.dim('[emitted]')} {s.info(event_name)}"

        elif event_type == TraceType.REACTIVE_EVENT_PROCESSED:
            # This can be very chatty. Keep it for DETAILED+ but suppress in
            # PROGRESS/minimal modes to improve signal-to-noise.
            if self.config.level <= ObservabilityLevel.PROGRESS:
                return None
            event_name = data.get("event_name", "?")
            round_num = data.get("round", 0)
            return f"{prefix}{s.dim('[Trace]')} {s.dim('[processing]')} {event_name} {s.dim(f'(round {round_num})')}"

        elif event_type == TraceType.REACTIVE_AGENT_TRIGGERED:
            agent = data.get("agent", "?")
            trigger_event = data.get("trigger_event", "?")
            formatted_name = _format_agent_name(agent)
            return f"{prefix}{s.agent(formatted_name)} {s.dim('[triggered]')} {s.dim('by')} {s.info(trigger_event)}"

        elif event_type == TraceType.REACTIVE_AGENT_COMPLETED:
            agent = data.get("agent", "?")
            output_length = data.get("output_length", 0)
            trigger_event = data.get("trigger_event", "")
            output_str = f" ({output_length} chars)" if output_length else ""
            by = f" {s.dim('←')} {s.dim(trigger_event)}" if trigger_event else ""
            formatted_name = _format_agent_name(agent)
            return f"{prefix}{s.agent(formatted_name)} {s.success('[completed]')}{s.dim(output_str)}{by}"

        elif event_type == TraceType.REACTIVE_AGENT_FAILED:
            agent = data.get("agent", "?")
            error = data.get("error", "Unknown error")[:80]
            formatted_name = _format_agent_name(agent)
            return f"{prefix}{s.agent(formatted_name)} {s.error('[FAILED]')} {s.error(error)}"

        elif event_type == TraceType.REACTIVE_NO_MATCH:
            # Useful for debugging, but noisy for normal runs.
            if self.config.level < ObservabilityLevel.DEBUG:
                return None
            event_name = data.get("event_name", "?")
            return f"{prefix}{s.dim('[Trace]')} {s.dim('[no-match]')} {s.dim(event_name)}"

        elif event_type == TraceType.REACTIVE_ROUND_STARTED:
            if self.config.level <= ObservabilityLevel.PROGRESS:
                return None
            round_num = data.get("round", 0)
            pending = data.get("pending_events", 0)
            return f"{prefix}[Round] {s.dim(f'{round_num}')} {s.dim(f'({pending} pending)')}"

        elif event_type == TraceType.REACTIVE_ROUND_COMPLETED:
            if self.config.level <= ObservabilityLevel.PROGRESS:
                return None
            round_num = data.get("round", 0)
            reactions = data.get("reactions", 0)
            total = data.get("total_reactions", 0)
            if reactions > 0:
                return f"{prefix}{s.dim('[Round]')} {s.dim(f'{round_num} done:')} {s.success(f'{reactions} reactions')} {s.dim(f'(total: {total})')}"
            return None  # Suppress empty rounds

        # ═══════════════════════════════════════════════════════════
        # SKILL EVENTS - Event-triggered behavioral specializations
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.SKILL_ACTIVATED:
            skill_name = data.get("skill", "?")
            agent = data.get("agent", "?")
            trigger_event = data.get("trigger_event", "")
            formatted_agent = _format_agent_name(agent)
            by = f" {s.dim('←')} {s.dim(trigger_event)}" if trigger_event else ""
            return f"{prefix}{formatted_agent} {s.info('[skill:')} {s.info(skill_name)}{s.info(']')}{by}"

        elif event_type == TraceType.SKILL_DEACTIVATED:
            # Only show at DEBUG level to reduce noise
            if self.config.level < ObservabilityLevel.DEBUG:
                return None
            skill_name = data.get("skill", "?")
            agent = data.get("agent", "?")
            formatted_agent = _format_agent_name(agent)
            return f"{prefix}{formatted_agent} {s.dim('[skill:')} {s.dim(skill_name)}{s.dim('] done')}"

        # ═══════════════════════════════════════════════════════════
        # CUSTOM EVENTS - User-defined events (often reactive)
        # ═══════════════════════════════════════════════════════════

        elif event_type == TraceType.CUSTOM:
            event_name = data.get("event_name", "")

            def _kv_preview(keys: list[str]) -> str:
                parts: list[str] = []
                for key in keys:
                    if key not in data:
                        continue
                    value = data.get(key)
                    if value is None:
                        continue
                    text = str(value)
                    if len(text) > 32:
                        text = text[:29] + "..."
                    parts.append(f"{s.dim(key + '=')}{s.info(text)}")
                return (" " + " ".join(parts)) if parts else ""

            # Reactive-style custom events (agent completions, etc.)
            if event_name:
                agent = data.get("agent", "")
                # Common domain IDs (e.g. case_id/scan_id) for reactive apps
                meta = _kv_preview([
                    "case_id",
                    "job_id",
                    "scan_id",
                    "customer_id",
                    "category",
                    "priority",
                    "decision",
                ])
                if agent:
                    return f"{prefix}{s.dim('📨')} {s.dim('Event:')} {s.info(event_name)}{meta} {s.dim('from')} {s.agent(f'[{agent}]')}"
                return f"{prefix}{s.dim('📨')} {s.dim('Event:')} {s.info(event_name)}{meta}"
            # Generic custom events
            data_preview = str(data)[:100] if data else ""
            return f"{prefix}{s.dim('📨')} {s.dim('Custom:')} {s.dim(data_preview)}"

        # ═══════════════════════════════════════════════════════════
        # DEFAULT - System/other events (dimmed)
        # ═══════════════════════════════════════════════════════════

        else:
            data_preview = str(data)[:100] if data else ""
            return f"{prefix}{s.dim(event_type.value)} {s.dim(data_preview)}"

    def _format_event_json(self, event: Trace) -> str:
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

    def _trace_event(self, event: Trace) -> None:
        """Build execution trace graph from events."""
        event_type = event.type
        data = event.data
        ts = event.timestamp

        # Agent lifecycle tracking - start on INVOKED or THINKING
        if event_type in (TraceType.AGENT_INVOKED, TraceType.AGENT_THINKING):
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

        elif event_type == TraceType.AGENT_RESPONDED:
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

        elif event_type == TraceType.AGENT_ERROR:
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
        elif event_type == TraceType.TOOL_CALLED:
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

        elif event_type == TraceType.TOOL_RESULT:
            tool_name = data.get("tool", "unknown")
            if tool_name in self._current_tools:
                tool_record = self._current_tools[tool_name]
                tool_record["status"] = "completed"
                tool_record["end_time"] = ts
                tool_record["result"] = data.get("result_preview") or str(data.get("result", ""))[:200]
                if tool_record["start_time"]:
                    delta = (ts - tool_record["start_time"]).total_seconds() * 1000
                    tool_record["duration_ms"] = delta

        elif event_type == TraceType.TOOL_ERROR:
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
        event_type: TraceType | None = None,
        limit: int | None = None,
    ) -> list[Trace]:
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

    def event_graph(self, include_timing: bool = False, format: str = "mermaid") -> str:
        """
        Generate a visualization of event-driven flow execution.

        Parses REACTIVE_* traces to build an event → reactor → event graph
        showing how events cascade through the reactive system.

        Args:
            include_timing: If True, annotate nodes with execution times
            format: Output format - currently only 'mermaid' supported

        Returns:
            Graph visualization in requested format (Mermaid flowchart by default)

        Example:
            ```python
            observer = Observer.trace()
            flow = Flow(observer=observer)
            
            flow.on("start", handler1, emits="processing")
            flow.on("processing", handler2, emits="complete")
            
            await flow.run("task")
            
            # Generate Mermaid diagram
            print(observer.event_graph())
            # Output:
            # graph LR
            #   start[\"start"/] --> handler1["handler1"]
            #   handler1 --> processing[\"processing"/]
            #   processing --> handler2["handler2"]
            #   handler2 --> complete[\"complete"/]
            
            # With timing annotations
            print(observer.event_graph(include_timing=True))
            # Output includes execution times on reactor nodes
            ```

        Use Cases:
            - **Debugging**: Visualize which reactors triggered and in what order
            - **Documentation**: Auto-generate flow diagrams from execution traces
            - **Performance**: Identify slow reactors by enabling timing annotations
            - **Validation**: Verify expected event chains occurred

        Note:
            Only processes REACTIVE_* trace events. Standard agent execution
            traces are not included in the event graph.
        """
        if format != "mermaid":
            raise ValueError(f"Unsupported format: {format}. Only 'mermaid' is currently supported.")

        # Extract reactive events
        reactive_events = [
            observed.event
            for observed in self._events
            if observed.event.type.value.startswith("reactive")
        ]
        
        # Also collect agent execution traces for timing data
        agent_traces = [
            observed.event
            for observed in self._events
            if observed.event.type.value.startswith("agent")
        ]

        if not reactive_events:
            return "graph LR\n  START[\"No reactive events captured\"]"

        # Build graph structure
        lines = ["graph LR"]
        seen_nodes: set[str] = set()
        edges: list[tuple[str, str, str | None]] = []  # (from, to, label)

        # Track event emissions and reactor triggers
        event_emissions: dict[str, list[str]] = {}  # event_type -> [source_reactor, ...]
        reactor_timings: dict[str, float] = {}  # reactor_name -> duration_ms
        
        # Collect timing from agent.responded traces (has duration_ms)
        for trace in agent_traces:
            if trace.type.value == "agent.responded":
                agent_name = trace.data.get("agent_name", trace.data.get("agent", "unknown"))
                duration = trace.data.get("duration_ms", 0)
                if duration > 0:
                    reactor_timings[agent_name] = duration

        # First pass: collect all events and reactor timings
        for trace in reactive_events:
            trace_type = trace.type
            data = trace.data

            if trace_type == TraceType.REACTIVE_EVENT_EMITTED:
                event_type = data.get("event_name", data.get("event_type", "unknown"))
                source = data.get("source", "system")
                
                if event_type not in event_emissions:
                    event_emissions[event_type] = []
                if source not in event_emissions[event_type]:
                    event_emissions[event_type].append(source)

            elif trace_type == TraceType.REACTIVE_AGENT_COMPLETED:
                agent_name = data.get("agent", "unknown")
                # Duration might be in reactive trace too (fallback)
                duration = data.get("duration_ms", 0)
                if duration > 0 and agent_name not in reactor_timings:
                    reactor_timings[agent_name] = duration

        # Second pass: build edges from REACTIVE_AGENT_TRIGGERED events
        for trace in reactive_events:
            if trace.type == TraceType.REACTIVE_AGENT_TRIGGERED:
                data = trace.data
                agent_name = data.get("agent", "unknown")
                trigger_event = data.get("trigger_event", data.get("event_type", "unknown"))

                # Find what events this agent emits
                emitted_events = [
                    evt for evt, sources in event_emissions.items()
                    if agent_name in sources and evt != trigger_event
                ]

                # Edge: trigger_event -> agent
                event_id = f"evt_{trigger_event.replace('.', '_')}"
                agent_id = f"agent_{agent_name.replace(' ', '_')}"

                if event_id not in seen_nodes:
                    lines.append(f'  {event_id}[\\"{trigger_event}"/]')
                    seen_nodes.add(event_id)

                # Create agent node with optional timing
                if agent_id not in seen_nodes:
                    if include_timing and agent_name in reactor_timings:
                        timing = reactor_timings[agent_name]
                        lines.append(f'  {agent_id}["{agent_name}<br/>({timing:.0f}ms)"]')
                    else:
                        lines.append(f'  {agent_id}["{agent_name}"]')
                    seen_nodes.add(agent_id)

                edges.append((event_id, agent_id, None))

                # Edges: agent -> emitted_events
                for emitted in emitted_events:
                    emitted_id = f"evt_{emitted.replace('.', '_')}"
                    if emitted_id not in seen_nodes:
                        lines.append(f'  {emitted_id}[\\"{emitted}"/]')
                        seen_nodes.add(emitted_id)
                    edges.append((agent_id, emitted_id, None))

        # Add edges
        for from_node, to_node, label in edges:
            if label:
                lines.append(f"  {from_node} -->|{label}| {to_node}")
            else:
                lines.append(f"  {from_node} --> {to_node}")

        # Add styling
        lines.extend([
            "",
            "  classDef eventNode fill:#e1f5ff,stroke:#01579b,stroke-width:2px",
            "  classDef agentNode fill:#fff3e0,stroke:#e65100,stroke-width:2px",
            "  class " + ",".join([n for n in seen_nodes if n.startswith("evt_")]) + " eventNode" if any(n.startswith("evt_") for n in seen_nodes) else "",
            "  class " + ",".join([n for n in seen_nodes if n.startswith("agent_")]) + " agentNode" if any(n.startswith("agent_") for n in seen_nodes) else "",
        ])

        return "\n".join([l for l in lines if l])  # Remove empty lines

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

        # Token usage stats (if tracked)
        if self.config.track_tokens and self._total_tokens["total"] > 0:
            lines.append("")
            lines.append("Token Usage:")
            lines.append(f"  📊 Total: {self._total_tokens['total']:,}")
            lines.append(f"  ⬇️  Input: {self._total_tokens['input']:,}")
            lines.append(f"  ⬆️  Output: {self._total_tokens['output']:,}")

            # Per-agent breakdown if multiple agents
            if len(self._token_usage) > 1:
                lines.append("")
                lines.append("  By agent:")
                for agent_name, tokens in sorted(self._token_usage.items(), key=lambda x: x[1]["total"], reverse=True):
                    if tokens["total"] > 0:
                        lines.append(f"    {agent_name}: {tokens['total']:,} ({tokens['input']:,} in, {tokens['output']:,} out)")

        # Channel breakdown
        if "by_channel" in m and m["by_channel"]:
            lines.append("")
            lines.append("Events by channel:")
            for channel, count in m["by_channel"].items():
                lines.append(f"  {channel}: {count}")

        lines.append("═" * 50)
        return "\n".join(lines)

    def export(self, filepath: str | Path, format: str = "jsonl") -> None:
        """
        Export captured events to a file for analysis.

        Supports structured export formats for post-processing, analytics,
        and integration with logging/monitoring systems.

        Args:
            filepath: Path to export file
            format: Export format - 'jsonl' (JSON Lines), 'json', or 'csv'

        Example:
            ```python
            observer = Observer.trace()
            await flow.run("task")

            # Export for analysis
            observer.export("trace.jsonl")  # One event per line
            observer.export("trace.json", format="json")  # Full array
            ```

        JSON Lines format (default):
            Each line is a complete JSON object:
            ```
            {"timestamp":"2026-01-03T10:30:45.123Z","event":"agent.invoked","agent":"Manager",...}
            {"timestamp":"2026-01-03T10:30:46.234Z","event":"tool.called","tool":"search",...}
            ```

        JSON format:
            Single JSON array with all events:
            ```json
            [
              {"timestamp": "...", "event": "agent.invoked", ...},
              {"timestamp": "...", "event": "tool.called", ...}
            ]
            ```
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            # JSON Lines: one event per line
            with filepath.open("w") as f:
                for observed in self._events:
                    event_data = {
                        "timestamp": observed.timestamp.isoformat(),
                        "event": observed.event.type.value,
                        "channel": observed.channel.value,
                        "data": observed.event.data,
                        "source": observed.event.source,
                        "correlation_id": observed.event.correlation_id,
                    }
                    f.write(json.dumps(event_data, default=str) + "\n")

        elif format == "json":
            # Single JSON array
            events = []
            for observed in self._events:
                event_data = {
                    "timestamp": observed.timestamp.isoformat(),
                    "event": observed.event.type.value,
                    "channel": observed.channel.value,
                    "data": observed.event.data,
                    "source": observed.event.source,
                    "correlation_id": observed.event.correlation_id,
                }
                events.append(event_data)

            with filepath.open("w") as f:
                json.dump(events, f, indent=2, default=str)

        elif format == "csv":
            # CSV with flattened data
            import csv

            with filepath.open("w", newline="") as f:
                # Determine all possible fields
                fields = ["timestamp", "event", "channel", "source", "correlation_id"]
                data_fields = set()
                for observed in self._events:
                    data_fields.update(observed.event.data.keys())
                fields.extend(sorted(data_fields))

                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()

                for observed in self._events:
                    row = {
                        "timestamp": observed.timestamp.isoformat(),
                        "event": observed.event.type.value,
                        "channel": observed.channel.value,
                        "source": observed.event.source,
                        "correlation_id": observed.event.correlation_id or "",
                    }
                    # Add flattened data fields
                    for key, value in observed.event.data.items():
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value, default=str)
                        else:
                            row[key] = str(value) if value is not None else ""
                    writer.writerow(row)

        else:
            raise ValueError(f"Unsupported export format: {format}. Use 'jsonl', 'json', or 'csv'.")

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

        # Clear new tracking structures
        self._token_usage.clear()
        self._total_tokens = {"input": 0, "output": 0, "total": 0}
        self._progress_steps.clear()
        self._state_snapshots.clear()
        self._agent_thinking_count.clear()
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
