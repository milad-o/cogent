"""
Anthropic models for AgenticFlow.

Usage:
    from cogent.models.anthropic import AnthropicChat

    llm = AnthropicChat(model="claude-sonnet-4-20250514")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from cogent.core.messages import MessageMetadata, TokenUsage
from cogent.models.base import (
    AIMessage,
    BaseChatModel,
    convert_messages,
    normalize_input,
)


def _messages_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
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
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get(
                                "name", tc.get("function", {}).get("name", "")
                            ),
                            "input": tc.get(
                                "args", tc.get("function", {}).get("arguments", {})
                            ),
                        }
                    )
                anthropic_messages.append(
                    {"role": "assistant", "content": content_blocks}
                )
            else:
                anthropic_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content,
                        }
                    ],
                }
            )

    return system, anthropic_messages


def _tools_to_anthropic(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert tools to Anthropic format."""
    anthropic_tools = []
    for tool in tools:
        if hasattr(tool, "name") and hasattr(tool, "description"):
            schema = getattr(tool, "args_schema", {}) or {}
            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": schema,
                }
            )
        elif isinstance(tool, dict):
            # Handle OpenAI format
            if "function" in tool:
                func = tool["function"]
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )
            else:
                anthropic_tools.append(tool)
    return anthropic_tools


def _parse_response(response: Any) -> AIMessage:
    """Parse Anthropic response into AIMessage with metadata."""
    from cogent.core.messages import MessageMetadata, TokenUsage

    content = ""
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            content += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "name": block.name,
                    "args": block.input,
                }
            )

    metadata = MessageMetadata(
        model=response.model,
        tokens=TokenUsage(
            prompt_tokens=response.usage.input_tokens
            if hasattr(response.usage, "input_tokens")
            else 0,
            completion_tokens=response.usage.output_tokens
            if hasattr(response.usage, "output_tokens")
            else 0,
            total_tokens=(response.usage.input_tokens + response.usage.output_tokens)
            if hasattr(response.usage, "input_tokens")
            else 0,
        )
        if hasattr(response, "usage") and response.usage
        else None,
        finish_reason=response.stop_reason
        if hasattr(response, "stop_reason")
        else None,
        response_id=response.id if hasattr(response, "id") else None,
    )

    return AIMessage(
        content=content,
        tool_calls=tool_calls,
        metadata=metadata,
    )


@dataclass
class AnthropicChat(BaseChatModel):
    """Anthropic chat model.

    High-performance chat model using Anthropic SDK directly.
    Supports Claude 3.5, Claude 3, and other Anthropic models.

    Example:
        from cogent.models.anthropic import AnthropicChat

        # Simple usage
        llm = AnthropicChat()  # Uses claude-sonnet-4-20250514 by default
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

        # With custom model
        llm = AnthropicChat(model="claude-3-opus-20240229", temperature=0.7)

        # With tools
        llm = AnthropicChat().bind_tools([search_tool, calc_tool])
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
            ) from None

        from cogent.config import get_api_key

        api_key = get_api_key("anthropic", self.api_key)

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
    ) -> AnthropicChat:
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

    def invoke(self, messages: str | list[dict[str, Any]] | list[Any]) -> AIMessage:
        """Invoke synchronously.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
        """
        self._ensure_initialized()
        response = self._client.messages.create(
            **self._build_request(normalize_input(messages))
        )
        return _parse_response(response)

    async def ainvoke(
        self, messages: str | list[dict[str, Any]] | list[Any]
    ) -> AIMessage:
        """Invoke asynchronously.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
        """
        self._ensure_initialized()
        response = await self._async_client.messages.create(
            **self._build_request(normalize_input(messages))
        )
        return _parse_response(response)

    async def astream(
        self, messages: str | list[dict[str, Any]] | list[Any]
    ) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously with metadata.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.

        Yields:
            AIMessage objects with incremental content and metadata.
        """
        self._ensure_initialized()
        kwargs = self._build_request(normalize_input(messages))

        start_time = time.time()
        chunk_metadata = {
            "id": None,
            "model": None,
            "finish_reason": None,
            "usage": None,
        }

        async with self._async_client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                # Accumulate metadata from the stream
                if hasattr(stream, "current_message_snapshot"):
                    msg = stream.current_message_snapshot
                    if hasattr(msg, "id") and msg.id:
                        chunk_metadata["id"] = msg.id
                    if hasattr(msg, "model") and msg.model:
                        chunk_metadata["model"] = msg.model
                    if hasattr(msg, "stop_reason") and msg.stop_reason:
                        chunk_metadata["finish_reason"] = msg.stop_reason
                    if hasattr(msg, "usage") and msg.usage:
                        chunk_metadata["usage"] = msg.usage

                metadata = MessageMetadata(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    model=chunk_metadata.get("model"),
                    tokens=None,
                    finish_reason=chunk_metadata.get("finish_reason"),
                    response_id=chunk_metadata.get("id"),
                    duration=time.time() - start_time,
                )
                yield AIMessage(content=text, metadata=metadata)

            # Yield final metadata chunk with complete usage
            if hasattr(stream, "current_message_snapshot"):
                msg = stream.current_message_snapshot
                if hasattr(msg, "usage") and msg.usage:
                    final_metadata = MessageMetadata(
                        id=str(uuid.uuid4()),
                        timestamp=time.time(),
                        model=chunk_metadata.get("model"),
                        tokens=TokenUsage(
                            prompt_tokens=msg.usage.input_tokens,
                            completion_tokens=msg.usage.output_tokens,
                            total_tokens=msg.usage.input_tokens
                            + msg.usage.output_tokens,
                        ),
                        finish_reason=chunk_metadata.get("finish_reason"),
                        response_id=chunk_metadata.get("id"),
                        duration=time.time() - start_time,
                    )
                    yield AIMessage(content="", metadata=final_metadata)

    def _build_request(
        self, messages: list[dict[str, Any]] | list[Any]
    ) -> dict[str, Any]:
        """Build API request."""
        # Convert to standard format first
        formatted = convert_messages(messages)
        system, anthropic_messages = _messages_to_anthropic(formatted)

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
            kwargs["tool_choice"] = {
                "type": "auto",
                "disable_parallel_tool_use": not self._parallel_tool_calls,
            }
        return kwargs
