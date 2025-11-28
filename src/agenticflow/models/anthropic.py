"""
Anthropic models for AgenticFlow.

Usage:
    from agenticflow.models.anthropic import Chat
    
    llm = Chat(model="claude-sonnet-4-20250514")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agenticflow.models.base import AIMessage, BaseChatModel


def _messages_to_anthropic(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert messages to Anthropic format, extracting system message."""
    system = None
    anthropic_messages = []
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            system = content
        elif role == "user":
            anthropic_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                # Anthropic uses content blocks for tool use
                content_blocks = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("name", tc.get("function", {}).get("name", "")),
                        "input": tc.get("args", tc.get("function", {}).get("arguments", {})),
                    })
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            else:
                anthropic_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content,
                }],
            })
    
    return system, anthropic_messages


def _tools_to_anthropic(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert tools to Anthropic format."""
    anthropic_tools = []
    for tool in tools:
        if hasattr(tool, "name") and hasattr(tool, "description"):
            schema = getattr(tool, "args_schema", {}) or {}
            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": schema,
            })
        elif isinstance(tool, dict):
            # Handle OpenAI format
            if "function" in tool:
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
            else:
                anthropic_tools.append(tool)
    return anthropic_tools


def _parse_response(response: Any) -> AIMessage:
    """Parse Anthropic response into AIMessage."""
    content = ""
    tool_calls = []
    
    for block in response.content:
        if block.type == "text":
            content += block.text
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "args": block.input,
            })
    
    return AIMessage(
        content=content,
        tool_calls=tool_calls,
    )


@dataclass
class AnthropicChat(BaseChatModel):
    """Anthropic chat model.
    
    High-performance chat model using Anthropic SDK directly.
    Supports Claude 3.5, Claude 3, and other Anthropic models.
    
    Example:
        from agenticflow.models.anthropic import Chat
        
        # Simple usage
        llm = Chat()  # Uses claude-sonnet-4-20250514 by default
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
        
        # With custom model
        llm = Chat(model="claude-3-opus-20240229", temperature=0.7)
        
        # With tools
        llm = Chat().bind_tools([search_tool, calc_tool])
        response = await llm.ainvoke(messages)
        
        # Streaming
        async for chunk in llm.astream(messages):
            print(chunk.content, end="")
    """
    
    model: str = "claude-sonnet-4-20250514"
    
    def _init_client(self) -> None:
        """Initialize Anthropic clients."""
        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: uv add anthropic"
            )
        
        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if api_key:
            kwargs["api_key"] = api_key
        
        self._client = Anthropic(**kwargs)
        self._async_client = AsyncAnthropic(**kwargs)
    
    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> "AnthropicChat":
        """Bind tools to the model."""
        self._ensure_initialized()
        
        new_model = AnthropicChat(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens or 4096,  # Anthropic requires max_tokens
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        new_model._tools = tools
        new_model._parallel_tool_calls = parallel_tool_calls
        new_model._client = self._client
        new_model._async_client = self._async_client
        new_model._initialized = True
        return new_model
    
    def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:
        """Invoke synchronously."""
        self._ensure_initialized()
        response = self._client.messages.create(**self._build_request(messages))
        return _parse_response(response)
    
    async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
        """Invoke asynchronously."""
        self._ensure_initialized()
        response = await self._async_client.messages.create(**self._build_request(messages))
        return _parse_response(response)
    
    async def astream(self, messages: list[dict[str, Any]]) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously."""
        self._ensure_initialized()
        kwargs = self._build_request(messages)
        kwargs["stream"] = True
        
        async with self._async_client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield AIMessage(content=text)
    
    def _build_request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Build API request."""
        system, anthropic_messages = _messages_to_anthropic(messages)
        
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens or 4096,  # Anthropic requires max_tokens
            "temperature": self.temperature,
        }
        if system:
            kwargs["system"] = system
        if self._tools:
            kwargs["tools"] = _tools_to_anthropic(self._tools)
        return kwargs
