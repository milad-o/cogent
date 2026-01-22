"""
Native message types for AgenticFlow.

Lightweight message implementations compatible with chat API message format.
All major providers (OpenAI, Azure, Anthropic, etc.) use this same format.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


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
class MessageMetadata:
    """Metadata for message tracking and observability.

    Attributes:
        id: Unique message identifier
        timestamp: Unix timestamp of message creation
        model: Model name used (for AI messages)
        tokens: Token usage (for AI messages)
        finish_reason: Completion finish reason (for AI messages)
        duration: Generation duration in seconds (for AI messages)
        response_id: Provider response ID (for AI messages)
        correlation_id: ID linking related messages
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    model: str | None = None
    tokens: TokenUsage | None = None
    finish_reason: str | None = None
    duration: float | None = None
    response_id: str | None = None
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "model": self.model,
            "tokens": self.tokens.to_dict() if self.tokens else None,
            "finish_reason": self.finish_reason,
            "duration": self.duration,
            "response_id": self.response_id,
            "correlation_id": self.correlation_id,
        }


@dataclass
class BaseMessage:
    """Base class for all message types.

    Attributes:
        content: The message content
        role: Message role (system, user, assistant, tool)
        metadata: Optional metadata for tracking and observability
    """

    content: str
    role: str = "user"
    metadata: MessageMetadata = field(default_factory=MessageMetadata)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict format for chat APIs."""
        return {"role": self.role, "content": self.content}

    def to_dict_with_metadata(self) -> dict[str, Any]:
        """Convert to dict with full metadata."""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
        }

    # Alias for backward compatibility
    to_openai = to_dict


@dataclass
class SystemMessage(BaseMessage):
    """System message for setting agent behavior."""

    role: str = field(default="system", init=False)

    def __init__(self, content: str, metadata: MessageMetadata | None = None) -> None:
        self.content = content
        self.role = "system"
        self.metadata = metadata or MessageMetadata()


@dataclass
class HumanMessage(BaseMessage):
    """User/human message."""

    role: str = field(default="user", init=False)

    def __init__(self, content: str, metadata: MessageMetadata | None = None) -> None:
        self.content = content
        self.role = "user"
        self.metadata = metadata or MessageMetadata()


@dataclass
class AIMessage(BaseMessage):
    """Assistant/AI message with optional tool calls and rich metadata."""

    role: str = field(default="assistant", init=False)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def __init__(
        self,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        metadata: MessageMetadata | None = None,
    ) -> None:
        self.content = content
        self.role = "assistant"
        self.tool_calls = tool_calls or []
        self.metadata = metadata or MessageMetadata()

    def __str__(self) -> str:
        return self.content

    def __bool__(self) -> bool:
        return bool(self.content or self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict format for chat APIs."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("args", "{}") if isinstance(tc.get("args"), str) else __import__("json").dumps(tc.get("args", {})),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg

    def to_dict_with_metadata(self) -> dict[str, Any]:
        """Convert to dict with full metadata including tool calls."""
        result = super().to_dict_with_metadata()
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result

    # Alias for backward compatibility
    to_openai = to_dict


@dataclass
class ToolMessage(BaseMessage):
    """Tool/function result message."""

    role: str = field(default="tool", init=False)
    tool_call_id: str = ""

    def __init__(
        self,
        content: str,
        tool_call_id: str = "",
        metadata: MessageMetadata | None = None,
    ) -> None:
        self.content = content
        self.role = "tool"
        self.tool_call_id = tool_call_id
        self.metadata = metadata or MessageMetadata()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict format for chat APIs."""
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
        }

    # Alias for backward compatibility
    to_openai = to_dict


def messages_to_dict(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert a list of messages to dict format for chat APIs."""
    return [msg.to_dict() for msg in messages]


# Alias for backward compatibility
messages_to_openai = messages_to_dict


def parse_openai_response(response: Any) -> AIMessage:
    """Parse OpenAI API response into AIMessage with full metadata.

    Args:
        response: OpenAI ChatCompletion response object.

    Returns:
        AIMessage with content, tool_calls, and rich metadata.
    """
    choice = response.choices[0]
    message = choice.message

    content = message.content or ""
    tool_calls = []

    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": __import__("json").loads(tc.function.arguments) if tc.function.arguments else {},
            })

    # Extract metadata
    metadata = MessageMetadata(
        model=response.model,
        tokens=TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        ) if response.usage else None,
        finish_reason=choice.finish_reason,
        response_id=response.id,
    )

    return AIMessage(content=content, tool_calls=tool_calls, metadata=metadata)
