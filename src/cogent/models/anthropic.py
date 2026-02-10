"""
Anthropic models for Cogent.

Supports Claude models with Extended Thinking for complex reasoning tasks.

Usage:
    from cogent.models.anthropic import AnthropicChat

    # Standard usage
    llm = AnthropicChat(model="claude-sonnet-4-20250514")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # With Extended Thinking (reasoning before responding)
    llm = AnthropicChat(
        model="claude-sonnet-4-20250514",
        thinking_budget=16000,  # Enable thinking with token budget
    )
    response = await llm.ainvoke([{"role": "user", "content": "Solve this complex problem..."}])
    print(response.thinking)  # Access thinking content
    print(response.content)   # Access final response

Extended Thinking models:
    - claude-opus-4-0 (claude-opus-4-20250514): Most capable, deepest reasoning
    - claude-sonnet-4-0 (claude-sonnet-4-20250514): Balanced performance
    - claude-3-7-sonnet (claude-3-7-sonnet-20250219): Previous flagship
    - claude-3-5-sonnet-20241022: Extended thinking support
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from cogent.core.messages import MessageMetadata, TokenUsage
from cogent.models.base import (
    AIMessage,
    BaseChatModel,
    convert_messages,
    normalize_input,
)

# Models that support Extended Thinking
THINKING_MODELS = {
    # Claude 4 series
    "claude-opus-4-20250514",
    "claude-opus-4-0",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-0",
    # Claude 3.7
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet",
    # Claude 3.5 (some versions)
    "claude-3-5-sonnet-20241022",
}


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


def _parse_response(response: Any, include_thinking: bool = False) -> AIMessage:
    """Parse Anthropic response into AIMessage with metadata.

    Args:
        response: Anthropic API response.
        include_thinking: Whether to include thinking blocks in response.

    Returns:
        AIMessage with content, tool_calls, thinking (if enabled), and metadata.
    """
    from cogent.core.messages import MessageMetadata, TokenUsage

    content = ""
    thinking = ""
    thinking_signature = None
    tool_calls = []

    for block in response.content:
        if block.type == "thinking":
            # Extended thinking block
            thinking += block.thinking
            if hasattr(block, "signature"):
                thinking_signature = block.signature
        elif block.type == "text":
            content += block.text
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "name": block.name,
                    "args": block.input,
                }
            )

    # Extract thinking tokens if available (cache_creation_input_tokens, cache_read_input_tokens)
    thinking_tokens = None
    if hasattr(response.usage, "cache_creation_input_tokens"):
        thinking_tokens = response.usage.cache_creation_input_tokens

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
            reasoning_tokens=thinking_tokens,
        )
        if hasattr(response, "usage") and response.usage
        else None,
        finish_reason=response.stop_reason
        if hasattr(response, "stop_reason")
        else None,
        response_id=response.id if hasattr(response, "id") else None,
    )

    # Create AIMessage with thinking if present
    msg = AIMessage(
        content=content,
        tool_calls=tool_calls,
        metadata=metadata,
    )

    # Add thinking as attribute if present
    if thinking and include_thinking:
        msg.thinking = thinking
        if thinking_signature:
            msg.thinking_signature = thinking_signature

    return msg


@dataclass
class AnthropicChat(BaseChatModel):
    """Anthropic chat model with Extended Thinking support.

    High-performance chat model using Anthropic SDK directly.
    Supports Claude 4, Claude 3.7, Claude 3.5, and other Anthropic models.

    Extended Thinking enables the model to "think" before responding,
    improving performance on complex reasoning tasks.

    Example:
        from cogent.models.anthropic import AnthropicChat

        # Simple usage
        llm = AnthropicChat()  # Uses claude-sonnet-4-20250514 by default
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

        # With Extended Thinking (for complex reasoning)
        llm = AnthropicChat(
            model="claude-sonnet-4-20250514",
            thinking_budget=16000,  # Token budget for thinking (1024-32768+)
        )
        response = await llm.ainvoke([{"role": "user", "content": "Solve..."}])

        # Access thinking process (if you want to see the reasoning)
        if hasattr(response, 'thinking'):
            print("Thinking:", response.thinking)
        print("Answer:", response.content)

        # With tools
        llm = AnthropicChat().bind_tools([search_tool, calc_tool])
        response = await llm.ainvoke(messages)

        # Streaming (includes thinking chunks when enabled)
        async for chunk in llm.astream(messages):
            print(chunk.content, end="")

    Attributes:
        model: Model name (default: claude-sonnet-4-20250514).
        thinking_budget: Token budget for thinking (enables Extended Thinking).
            - Minimum: 1024 tokens
            - Recommended: 10000-32000 for complex tasks
            - None: Disable thinking (default)
        temperature: Only used when thinking is disabled (0.0-2.0).
            Extended Thinking requires temperature=1.
    """

    model: str = "claude-sonnet-4-20250514"
    thinking_budget: int | None = field(default=None)

    # Internal state
    _include_thinking_in_response: bool = field(default=True, repr=False)

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
            thinking_budget=self.thinking_budget,  # Preserve thinking settings
        )
        new_model._tools = tools
        new_model._parallel_tool_calls = parallel_tool_calls
        new_model._include_thinking_in_response = self._include_thinking_in_response
        new_model._client = self._client
        new_model._async_client = self._async_client
        new_model._initialized = True
        return new_model

    def with_thinking(
        self,
        budget: int = 16000,
        *,
        include_in_response: bool = True,
    ) -> AnthropicChat:
        """Enable Extended Thinking with specified budget.

        Extended Thinking allows the model to reason through complex problems
        before providing a response, significantly improving accuracy.

        Args:
            budget: Token budget for thinking (minimum 1024, recommended 10000-32000).
            include_in_response: Whether to include thinking in response (default True).

        Returns:
            New AnthropicChat instance with thinking enabled.

        Example:
            llm = AnthropicChat().with_thinking(budget=16000)
            response = await llm.ainvoke("What's 127 * 893?")
            print(response.thinking)  # Shows reasoning steps
            print(response.content)   # Shows final answer

        Note:
            - Extended Thinking requires temperature=1 (set automatically)
            - Minimum budget is 1024 tokens
            - Supported on Claude 4, Claude 3.7 Sonnet, Claude 3.5 Sonnet (20241022+)
        """
        if budget < 1024:
            raise ValueError("thinking_budget must be at least 1024 tokens")

        self._ensure_initialized()

        new_model = AnthropicChat(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens or 16000,  # Higher default for thinking
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            thinking_budget=budget,
        )
        new_model._tools = getattr(self, "_tools", [])
        new_model._parallel_tool_calls = getattr(self, "_parallel_tool_calls", True)
        new_model._include_thinking_in_response = include_in_response
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
        return _parse_response(
            response,
            include_thinking=self._include_thinking_in_response
            and self.thinking_budget is not None,
        )

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
        return _parse_response(
            response,
            include_thinking=self._include_thinking_in_response
            and self.thinking_budget is not None,
        )

    async def astream(
        self, messages: str | list[dict[str, Any]] | list[Any]
    ) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously with metadata.

        When Extended Thinking is enabled, yields thinking chunks first,
        followed by text chunks. Each chunk has appropriate metadata.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.

        Yields:
            AIMessage objects with incremental content and metadata.
            - If thinking enabled: thinking chunks have `is_thinking=True` in metadata
            - Final chunk has complete token usage
        """
        self._ensure_initialized()
        kwargs = self._build_request(normalize_input(messages))

        start_time = time.time()
        chunk_metadata: dict[str, Any] = {
            "id": None,
            "model": None,
            "finish_reason": None,
            "usage": None,
        }
        accumulated_thinking = ""

        async with self._async_client.messages.stream(**kwargs) as stream:
            # Handle raw events for thinking support
            async for event in stream:
                # Update metadata from message snapshots
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

                # Handle different event types
                event_type = getattr(event, "type", None)

                # Thinking delta (Extended Thinking)
                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        delta_type = getattr(delta, "type", None)

                        # Thinking content
                        if delta_type == "thinking_delta":
                            thinking_text = getattr(delta, "thinking", "")
                            if thinking_text and self._include_thinking_in_response:
                                accumulated_thinking += thinking_text
                                metadata = MessageMetadata(
                                    id=str(uuid.uuid4()),
                                    timestamp=time.time(),
                                    model=chunk_metadata.get("model"),
                                    tokens=None,
                                    finish_reason=None,
                                    response_id=chunk_metadata.get("id"),
                                    duration=time.time() - start_time,
                                    is_thinking=True,  # Mark as thinking chunk
                                )
                                msg = AIMessage(content="", metadata=metadata)
                                msg.thinking = thinking_text
                                yield msg

                        # Regular text content
                        elif delta_type == "text_delta":
                            text = getattr(delta, "text", "")
                            if text:
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
                    final_msg = AIMessage(content="", metadata=final_metadata)
                    if accumulated_thinking and self._include_thinking_in_response:
                        final_msg.thinking = accumulated_thinking
                    yield final_msg

    def _build_request(
        self, messages: list[dict[str, Any]] | list[Any]
    ) -> dict[str, Any]:
        """Build API request with Extended Thinking support."""
        # Convert to standard format first
        formatted = convert_messages(messages)
        system, anthropic_messages = _messages_to_anthropic(formatted)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens or 4096,  # Anthropic requires max_tokens
        }

        # Extended Thinking configuration
        if self.thinking_budget is not None:
            # Validate model supports thinking
            if self.model not in THINKING_MODELS:
                # Check if it's a pattern match (e.g., claude-sonnet-4-* matches)
                is_thinking_model = any(
                    self.model.startswith(prefix)
                    for prefix in (
                        "claude-opus-4",
                        "claude-sonnet-4",
                        "claude-3-7-sonnet",
                        "claude-3-5-sonnet-2024",
                    )
                )
                if not is_thinking_model:
                    raise ValueError(
                        f"Model {self.model} does not support Extended Thinking. "
                        f"Use claude-opus-4-*, claude-sonnet-4-*, or claude-3-7-sonnet-*"
                    )

            # Enable thinking with budget
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
            # Extended Thinking requires temperature=1 (Anthropic enforces this)
            kwargs["temperature"] = 1
        else:
            # Normal mode - use specified temperature
            kwargs["temperature"] = self.temperature

        if system:
            kwargs["system"] = system
        if self._tools:
            kwargs["tools"] = _tools_to_anthropic(self._tools)
            kwargs["tool_choice"] = {
                "type": "auto",
                "disable_parallel_tool_use": not self._parallel_tool_calls,
            }
        return kwargs
