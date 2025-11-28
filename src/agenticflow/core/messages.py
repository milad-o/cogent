"""
Native message types for AgenticFlow.

Lightweight message implementations compatible with OpenAI API message format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseMessage:
    """Base class for all message types."""
    
    content: str
    role: str = "user"
    
    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        return {"role": self.role, "content": self.content}


@dataclass
class SystemMessage(BaseMessage):
    """System message for setting agent behavior."""
    
    role: str = field(default="system", init=False)
    
    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "system"


@dataclass
class HumanMessage(BaseMessage):
    """User/human message."""
    
    role: str = field(default="user", init=False)
    
    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "user"


@dataclass
class AIMessage(BaseMessage):
    """Assistant/AI message with optional tool calls."""
    
    role: str = field(default="assistant", init=False)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    
    def __init__(
        self,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.content = content
        self.role = "assistant"
        self.tool_calls = tool_calls or []
    
    def __str__(self) -> str:
        return self.content
    
    def __bool__(self) -> bool:
        return bool(self.content or self.tool_calls)
    
    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
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


@dataclass
class ToolMessage(BaseMessage):
    """Tool/function result message."""
    
    role: str = field(default="tool", init=False)
    tool_call_id: str = ""
    
    def __init__(self, content: str, tool_call_id: str = "") -> None:
        self.content = content
        self.role = "tool"
        self.tool_call_id = tool_call_id
    
    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI API format."""
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
        }


def messages_to_openai(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert a list of messages to OpenAI API format."""
    return [msg.to_openai() for msg in messages]


def parse_openai_response(response: Any) -> AIMessage:
    """Parse OpenAI API response into AIMessage.
    
    Args:
        response: OpenAI ChatCompletion response object.
        
    Returns:
        AIMessage with content and tool_calls.
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
    
    return AIMessage(content=content, tool_calls=tool_calls)
