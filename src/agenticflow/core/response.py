"""Unified Response Protocol.

This module provides the core types for the unified response protocol used
throughout AgenticFlow. All agent operations (run, think, etc.) return a
Response[T] object that contains the result plus rich metadata.

The Response[T] generic container ensures:
- Type safety for response content
- Consistent metadata access (tokens, timing, model info)
- Tool call tracking
- Event integration
- Standardized error handling
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from agenticflow.events import Event

T = TypeVar("T")


@dataclass
class TokenUsage:
    """Token usage information from model execution.

    Attributes:
        prompt_tokens: Tokens in the prompt/input
        completion_tokens: Tokens in the completion/output
        total_tokens: Sum of prompt and completion tokens
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for serialization."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ToolCall:
    """Information about a tool invocation.

    Attributes:
        tool_name: Name of the tool that was called
        arguments: Arguments passed to the tool
        result: Result returned by the tool
        duration: Execution time in seconds
        success: Whether the call succeeded
        error: Error message if call failed
    """

    tool_name: str
    arguments: dict[str, Any]
    result: Any
    duration: float
    success: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "duration": self.duration,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ErrorInfo:
    """Error information from failed execution.

    Attributes:
        message: Human-readable error message
        type: Error type/category
        traceback: Stack trace (if available)
    """

    message: str
    type: str
    traceback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message": self.message,
            "type": self.type,
            "traceback": self.traceback,
        }


@dataclass
class ResponseMetadata:
    """Consistent metadata across all responses.

    Attributes:
        agent: Name of the agent that generated response
        model: Model name/identifier used (if applicable)
        tokens: Token usage information
        duration: Execution duration in seconds
        timestamp: Unix timestamp of response creation
        correlation_id: ID linking related operations
        trace_id: ID for distributed tracing
    """

    agent: str
    model: str | None = None
    tokens: TokenUsage | None = None
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)
    correlation_id: str | None = None
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent": self.agent,
            "model": self.model,
            "tokens": self.tokens.to_dict() if self.tokens else None,
            "duration": self.duration,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
        }


@dataclass
class Response(Generic[T]):
    """Unified response container for all agent/tool operations.

    This is the canonical response type used throughout AgenticFlow:
    - Agent.run() returns Response[T]
    - Agent.think() returns Response[str]
    - Tools return Response[ToolResult]
    - A2A delegation returns Response[Any]

    Attributes:
        content: The actual response data (generic over type)
        metadata: Consistent metadata (tokens, timing, model, etc)
        tool_calls: List of tool invocations during execution
        events: Events emitted during execution
        error: Error information if execution failed

    Examples:
        # Basic usage
        response = await agent.run("task")
        result = response.content
        tokens = response.metadata.tokens.total_tokens

        # Type-safe unwrap
        result = response.unwrap()  # Raises if error

        # Convert to event
        event = response.to_event("task.done", agent.name)

        # Check success
        if response.success:
            print(f"Used {response.metadata.tokens.total_tokens} tokens")
    """

    content: T
    metadata: ResponseMetadata
    tool_calls: list[ToolCall] = field(default_factory=list)
    events: list[Any] = field(default_factory=list)  # Will be list[Event] at runtime
    error: ErrorInfo | None = None

    @property
    def success(self) -> bool:
        """Whether execution succeeded (no error)."""
        return self.error is None

    def unwrap(self) -> T:
        """Get content or raise error if failed.

        Returns:
            The content value

        Raises:
            ResponseError: If response contains an error
        """
        if self.error:
            raise ResponseError(self.error.message, response=self)
        return self.content

    def to_event(self, name: str, source: str) -> Event:
        """Convert response to event with full metadata.

        Args:
            name: Event name (e.g., "task.done")
            source: Event source (e.g., agent name)

        Returns:
            Event with response data and metadata
        """
        from agenticflow.events import Event

        return Event(
            name=name,
            source=source,
            data={
                "content": self.content,
                "metadata": self.metadata.to_dict(),
                "tool_calls": [tc.to_dict() for tc in self.tool_calls],
                "success": self.success,
            },
            correlation_id=self.metadata.correlation_id,
            metadata=self.metadata.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "events": [e.to_dict() for e in self.events],
            "error": self.error.to_dict() if self.error else None,
            "success": self.success,
        }


class ResponseError(Exception):
    """Exception raised when unwrapping a failed Response.

    Attributes:
        response: The original Response object that failed
    """

    def __init__(self, message: str, response: Response) -> None:
        """Initialize ResponseError.

        Args:
            message: Error message
            response: The original Response object
        """
        super().__init__(message)
        self.response = response
