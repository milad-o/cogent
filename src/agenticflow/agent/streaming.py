"""
Streaming support for Agent LLM responses.

This module provides token-by-token streaming from LLM providers,
enabling real-time output display and improved user experience.

Streaming Modes:
    - Token streaming: Individual tokens as they're generated
    - Event streaming: Structured events (start, token, tool_call, end)
    - Callback-based: Register handlers for stream events

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.models import ChatModel
    
    # Create agent with streaming-capable model
    agent = Agent(
        name="Assistant",
        model=ChatModel(model="gpt-4o"),
    )
    
    # Stream tokens as they arrive
    async for chunk in agent.think("Write a poem", stream=True):
        print(chunk.content, end="", flush=True)
    
    # Or use event-based streaming
    async for event in agent.stream_events("Write a poem"):
        if event.type == StreamTraceType.TOKEN:
            print(event.content, end="", flush=True)
        elif event.type == StreamTraceType.TOOL_CALL:
            print(f"\\n[Calling {event.tool_name}...]")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Protocol,
    TypeAlias,
)

from agenticflow.core.messages import AIMessage

if TYPE_CHECKING:
    pass


class StreamTraceType(Enum):
    """Types of streaming events."""
    
    STREAM_START = "stream_start"
    """Stream has started, contains metadata."""
    
    TOKEN = "token"
    """A token chunk from the LLM."""
    
    TOOL_CALL_START = "tool_call_start"
    """LLM is starting a tool call."""
    
    TOOL_CALL_ARGS = "tool_call_args"
    """Partial arguments for a tool call."""
    
    TOOL_CALL_END = "tool_call_end"
    """Tool call specification complete."""
    
    TOOL_RESULT = "tool_result"
    """Result from tool execution."""
    
    STREAM_END = "stream_end"
    """Stream has ended, contains final message."""
    
    ERROR = "error"
    """An error occurred during streaming."""


@dataclass
class StreamChunk:
    """
    A chunk of streamed content from the LLM.
    
    This is the simplest streaming interface - just content as it arrives.
    For more structured events, use StreamEvent.
    
    Attributes:
        content: The text content of this chunk.
        finish_reason: Why this chunk was emitted (None if not final).
        model: The model that generated this chunk.
        index: Position in the stream (0-indexed).
        token_count: Number of tokens in this chunk (if available).
    """
    
    content: str
    """The text content of this chunk."""
    
    finish_reason: str | None = None
    """Why streaming stopped: 'stop', 'length', 'tool_calls', etc."""
    
    model: str | None = None
    """The model that generated this chunk."""
    
    index: int = 0
    """Position in the stream (0-indexed)."""
    
    token_count: int | None = None
    """Number of tokens in this chunk."""
    
    raw: Any = None
    """The raw chunk from the provider (AIMessage)."""
    
    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        return f"StreamChunk({self.content!r})"
    
    def __add__(self, other: "StreamChunk") -> "StreamChunk":
        """Concatenate chunks."""
        return StreamChunk(
            content=self.content + other.content,
            finish_reason=other.finish_reason or self.finish_reason,
            model=self.model or other.model,
            index=other.index,
            token_count=(
                (self.token_count or 0) + (other.token_count or 0)
                if self.token_count is not None or other.token_count is not None
                else None
            ),
        )


@dataclass
class ToolCallChunk:
    """
    Partial tool call information streamed from the LLM.
    
    Tool calls are streamed incrementally:
    1. First chunk has id and name
    2. Subsequent chunks have args fragments
    3. Final chunk completes the call
    """
    
    id: str | None = None
    """Unique ID for this tool call."""
    
    name: str | None = None
    """Name of the tool being called."""
    
    args: str = ""
    """Partial JSON arguments (accumulated)."""
    
    index: int = 0
    """Index of this tool call in the response."""
    
    def is_complete(self) -> bool:
        """Check if we have enough info to parse the call."""
        return bool(self.id and self.name and self.args)


@dataclass
class StreamEvent:
    """
    A structured streaming event.
    
    Events provide more context than raw chunks, including:
    - Event type (token, tool call, error, etc.)
    - Timing information
    - Metadata from the model
    
    Attributes:
        type: The type of event.
        content: Text content (for TOKEN events).
        tool_call: Tool call info (for TOOL_CALL_* events).
        error: Error message (for ERROR events).
        metadata: Additional event metadata.
        timestamp: When this event occurred.
        accumulated: All content accumulated so far.
    """
    
    type: StreamTraceType
    """The type of streaming event."""
    
    content: str = ""
    """Text content for TOKEN events."""
    
    tool_call: ToolCallChunk | None = None
    """Tool call information for TOOL_CALL_* events."""
    
    tool_name: str | None = None
    """Tool name (convenience for TOOL_CALL events)."""
    
    tool_result: Any = None
    """Tool execution result for TOOL_RESULT events."""
    
    error: str | None = None
    """Error message for ERROR events."""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (model, tokens, etc.)."""
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When this event occurred (UTC)."""
    
    accumulated: str = ""
    """All content accumulated so far in the stream."""
    
    index: int = 0
    """Position in the event stream."""
    
    @property
    def is_final(self) -> bool:
        """Check if this is the final event."""
        return self.type in (StreamTraceType.STREAM_END, StreamTraceType.ERROR)
    
    @property
    def has_content(self) -> bool:
        """Check if this event has text content."""
        return bool(self.content)
    
    @property
    def is_tool_related(self) -> bool:
        """Check if this is a tool-related event."""
        return self.type in (
            StreamTraceType.TOOL_CALL_START,
            StreamTraceType.TOOL_CALL_ARGS,
            StreamTraceType.TOOL_CALL_END,
            StreamTraceType.TOOL_RESULT,
        )


@dataclass
class StreamConfig:
    """
    Configuration for streaming behavior.
    
    Attributes:
        emit_start_event: Emit STREAM_START at the beginning.
        emit_end_event: Emit STREAM_END at the end.
        include_tool_events: Include tool call events.
        include_accumulated: Track accumulated content.
        buffer_tool_args: Buffer tool args until complete.
        on_token: Callback for each token.
        on_tool_call: Callback when tool call is detected.
        on_error: Callback on streaming error.
    """
    
    emit_start_event: bool = True
    """Emit a STREAM_START event at the beginning."""
    
    emit_end_event: bool = True
    """Emit a STREAM_END event at the end."""
    
    include_tool_events: bool = True
    """Include TOOL_CALL_* events in the stream."""
    
    include_accumulated: bool = True
    """Track accumulated content in events."""
    
    buffer_tool_args: bool = True
    """Buffer tool call args until complete."""
    
    on_token: Callable[[str], None] | None = None
    """Synchronous callback for each token."""
    
    on_tool_call: Callable[[str, dict], None] | None = None
    """Callback when a tool call is detected (name, args)."""
    
    on_error: Callable[[Exception], None] | None = None
    """Callback on streaming errors."""
    
    @classmethod
    def minimal(cls) -> "StreamConfig":
        """Config that only yields tokens, no events."""
        return cls(
            emit_start_event=False,
            emit_end_event=False,
            include_tool_events=False,
            include_accumulated=False,
        )
    
    @classmethod
    def full(cls) -> "StreamConfig":
        """Config that includes all events and metadata."""
        return cls(
            emit_start_event=True,
            emit_end_event=True,
            include_tool_events=True,
            include_accumulated=True,
        )


class StreamCallback(Protocol):
    """Protocol for stream callbacks."""
    
    def on_token(self, token: str) -> None:
        """Called for each token."""
        ...
    
    def on_stream_start(self, metadata: dict[str, Any]) -> None:
        """Called when streaming starts."""
        ...
    
    def on_stream_end(self, full_response: str) -> None:
        """Called when streaming ends."""
        ...
    
    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Called when a tool call is detected."""
        ...
    
    def on_error(self, error: Exception) -> None:
        """Called on streaming errors."""
        ...


@dataclass
class PrintStreamCallback:
    """
    Simple callback that prints tokens to stdout.
    
    Example:
        ```python
        callback = PrintStreamCallback(end="", flush=True)
        async for chunk in agent.think("Hello", stream=True):
            callback.on_token(chunk.content)
        ```
    """
    
    end: str = ""
    """String to print after each token."""
    
    flush: bool = True
    """Whether to flush stdout after each token."""
    
    prefix: str = ""
    """Prefix before the stream starts."""
    
    suffix: str = "\n"
    """Suffix after the stream ends."""
    
    show_tool_calls: bool = True
    """Whether to print tool call notifications."""
    
    def on_token(self, token: str) -> None:
        print(token, end=self.end, flush=self.flush)
    
    def on_stream_start(self, metadata: dict[str, Any]) -> None:
        if self.prefix:
            print(self.prefix, end="", flush=self.flush)
    
    def on_stream_end(self, full_response: str) -> None:
        if self.suffix:
            print(self.suffix, end="", flush=True)
    
    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        if self.show_tool_calls:
            print(f"\n[Calling {name}...]", flush=self.flush)
    
    def on_error(self, error: Exception) -> None:
        print(f"\n[Error: {error}]", flush=True)


@dataclass
class CollectorStreamCallback:
    """
    Callback that collects all streamed content.
    
    Useful when you need both streaming display and the full response.
    
    Example:
        ```python
        collector = CollectorStreamCallback()
        async for chunk in agent.think("Hello", stream=True):
            collector.on_token(chunk.content)
            print(chunk.content, end="", flush=True)
        
        full_response = collector.get_full_response()
        ```
    """
    
    _tokens: list[str] = field(default_factory=list)
    _tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    _errors: list[Exception] = field(default_factory=list)
    _metadata: dict[str, Any] = field(default_factory=dict)
    
    def on_token(self, token: str) -> None:
        self._tokens.append(token)
    
    def on_stream_start(self, metadata: dict[str, Any]) -> None:
        self._metadata = metadata
    
    def on_stream_end(self, full_response: str) -> None:
        pass  # Already collected via tokens
    
    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self._tool_calls.append((name, args))
    
    def on_error(self, error: Exception) -> None:
        self._errors.append(error)
    
    def get_full_response(self) -> str:
        """Get the complete response."""
        return "".join(self._tokens)
    
    def get_tool_calls(self) -> list[tuple[str, dict]]:
        """Get all tool calls detected during streaming."""
        return list(self._tool_calls)
    
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return bool(self._errors)
    
    def clear(self) -> None:
        """Reset the collector."""
        self._tokens.clear()
        self._tool_calls.clear()
        self._errors.clear()
        self._metadata.clear()


# Type aliases for cleaner signatures
StreamHandler: TypeAlias = Callable[[StreamChunk], None]
EventHandler: TypeAlias = Callable[[StreamEvent], None]


def chunk_from_message(chunk: AIMessage, index: int = 0) -> StreamChunk:
    """
    Convert an AIMessage chunk to our StreamChunk.
    
    Args:
        chunk: The AIMessage chunk from streaming.
        index: Position in the stream.
        
    Returns:
        A StreamChunk with the content extracted.
    """
    content = chunk.content if isinstance(chunk.content, str) else ""
    
    return StreamChunk(
        content=content,
        finish_reason=None,
        index=index,
        token_count=None,
        raw=chunk,
    )


def extract_tool_calls(msg: AIMessage) -> list[ToolCallChunk]:
    """
    Extract tool call chunks from an AI message.
    
    Args:
        msg: The AIMessage to extract tool calls from.
        
    Returns:
        List of ToolCallChunk objects found in the message.
    """
    tool_calls = []
    
    # Check tool_calls list
    if msg.tool_calls:
        for i, tc in enumerate(msg.tool_calls):
            tool_calls.append(ToolCallChunk(
                id=tc.get("id"),
                name=tc.get("name"),
                args=str(tc.get("args", {})) if isinstance(tc.get("args"), dict) else tc.get("args", ""),
                index=i,
            ))
    
    return tool_calls


async def collect_stream(
    stream: AsyncIterator[StreamChunk],
) -> tuple[str, list[ToolCallChunk]]:
    """
    Collect all chunks from a stream into a complete response.
    
    Args:
        stream: An async iterator of StreamChunk objects.
        
    Returns:
        Tuple of (full_content, tool_calls).
        
    Example:
        ```python
        content, tool_calls = await collect_stream(agent.think("Hello", stream=True))
        print(f"Response: {content}")
        if tool_calls:
            print(f"Tool calls: {[tc.name for tc in tool_calls]}")
        ```
    """
    content_parts = []
    tool_calls = []
    
    async for chunk in stream:
        content_parts.append(chunk.content)
        if chunk.raw and hasattr(chunk.raw, "tool_call_chunks"):
            tool_calls.extend(extract_tool_calls(chunk.raw))
    
    return "".join(content_parts), tool_calls


async def print_stream(
    stream: AsyncIterator[StreamChunk],
    *,
    end: str = "",
    flush: bool = True,
    prefix: str = "",
    suffix: str = "\n",
) -> str:
    """
    Print a stream to stdout and return the full content.
    
    Convenience function for simple streaming display.
    
    Args:
        stream: An async iterator of StreamChunk objects.
        end: String to print after each chunk.
        flush: Whether to flush after each chunk.
        prefix: String to print before streaming starts.
        suffix: String to print after streaming ends.
        
    Returns:
        The complete streamed content.
        
    Example:
        ```python
        response = await print_stream(agent.think("Write a haiku", stream=True))
        # Prints tokens as they arrive, returns full response
        ```
    """
    if prefix:
        print(prefix, end="", flush=flush)
    
    content_parts = []
    async for chunk in stream:
        print(chunk.content, end=end, flush=flush)
        content_parts.append(chunk.content)
    
    if suffix:
        print(suffix, end="", flush=True)
    
    return "".join(content_parts)


@dataclass
class ObserverStreamCallback:
    """
    Streaming callback that integrates with Observer.
    
    Emits streaming events through the observer for unified observability.
    Also provides real-time console output when observer is configured for it.
    
    Example:
        ```python
        from agenticflow.observability import Observer
        from agenticflow.agent.streaming import ObserverStreamCallback
        
        observer = Observer.verbose()
        callback = ObserverStreamCallback(observer, agent_name="writer")
        
        async for chunk in agent.think("Write a story", stream=True):
            callback.on_token(chunk.content)
        ```
    
    Attributes:
        observer: The Observer instance to emit events to.
        agent_name: Name of the agent for event correlation.
        show_tokens: Print tokens in real-time (default: True if observer is verbose+).
        emit_events: Emit streaming events to observer (default: True).
    """
    
    observer: Any  # Observer, but we avoid circular import
    """The Observer to integrate with."""
    
    agent_name: str = "agent"
    """Name of the agent for event correlation."""
    
    show_tokens: bool | None = None
    """Print tokens in real-time. None = auto-detect from observer level."""
    
    emit_events: bool = True
    """Whether to emit events to the observer."""
    
    _accumulated: str = ""
    """Internal: accumulated content."""
    
    _token_count: int = 0
    """Internal: count of tokens received."""
    
    def __post_init__(self) -> None:
        """Initialize show_tokens based on observer level if not set."""
        if self.show_tokens is None:
            # Auto-detect from observer level
            # Import here to avoid circular imports
            try:
                from agenticflow.observability.observer import ObservabilityLevel
                self.show_tokens = self.observer.config.level >= ObservabilityLevel.DETAILED
            except (ImportError, AttributeError):
                self.show_tokens = True
    
    def on_token(self, token: str) -> None:
        """Handle each token."""
        self._accumulated += token
        self._token_count += 1
        
        # Print token if configured
        if self.show_tokens:
            print(token, end="", flush=True)
        
        # Emit token event to observer
        if self.emit_events and hasattr(self.observer, "_emit"):
            from agenticflow.observability.trace_record import TraceType
            self.observer._emit(
                TraceType.TOKEN_STREAMED,
                agent_name=self.agent_name,
                token=token,
                token_index=self._token_count,
                accumulated_length=len(self._accumulated),
            )
    
    def on_stream_start(self, metadata: dict[str, Any]) -> None:
        """Handle stream start."""
        if self.emit_events and hasattr(self.observer, "_emit"):
            from agenticflow.observability.trace_record import TraceType
            self.observer._emit(
                TraceType.STREAM_START,
                agent_name=self.agent_name,
                metadata=metadata,
            )
    
    def on_stream_end(self, full_response: str) -> None:
        """Handle stream end."""
        # Add newline after streaming output
        if self.show_tokens:
            print()
        
        if self.emit_events and hasattr(self.observer, "_emit"):
            from agenticflow.observability.trace_record import TraceType
            self.observer._emit(
                TraceType.STREAM_END,
                agent_name=self.agent_name,
                response_preview=full_response[:500] if len(full_response) > 500 else full_response,
                total_tokens=self._token_count,
            )
    
    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Handle tool call detection."""
        if self.show_tokens:
            print(f"\n[Calling {name}...]", flush=True)
        
        if self.emit_events and hasattr(self.observer, "_emit"):
            from agenticflow.observability.trace_record import TraceType
            self.observer._emit(
                TraceType.STREAM_TOOL_CALL,
                agent_name=self.agent_name,
                tool=name,
                args=args,
            )
    
    def on_error(self, error: Exception) -> None:
        """Handle streaming error."""
        if self.show_tokens:
            print(f"\n[Stream error: {error}]", flush=True)
        
        if self.emit_events and hasattr(self.observer, "_emit"):
            from agenticflow.observability.trace_record import TraceType
            self.observer._emit(
                TraceType.STREAM_ERROR,
                agent_name=self.agent_name,
                error=str(error),
            )
    
    def get_accumulated(self) -> str:
        """Get all accumulated content."""
        return self._accumulated
    
    def get_token_count(self) -> int:
        """Get the count of tokens received."""
        return self._token_count
    
    def reset(self) -> None:
        """Reset the callback for reuse."""
        self._accumulated = ""
        self._token_count = 0
