"""
DeepSeek models for Cogent.

Supports DeepSeek Chat and DeepSeek Reasoner models.

Usage:
    from cogent.models.deepseek import DeepSeekChat

    # Standard chat model
    llm = DeepSeekChat(model="deepseek-chat")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # Reasoning model (exposes Chain of Thought)
    llm = DeepSeekChat(model="deepseek-reasoner")
    response = await llm.ainvoke("9.11 and 9.8, which is greater?")

    # Access reasoning content (Chain of Thought)
    if hasattr(response, 'reasoning'):
        print("Reasoning:", response.reasoning)
    print("Answer:", response.content)

Available models:
    - deepseek-chat: General chat model
    - deepseek-reasoner: Reasoning model with Chain of Thought

Features:
    - Function/tool calling (deepseek-chat only)
    - JSON output mode
    - Chain of Thought reasoning (deepseek-reasoner)
    - Streaming support

Note:
    DeepSeek Reasoner does NOT support:
    - Function calling/tools
    - temperature, top_p, presence_penalty, frequency_penalty
    - logprobs, top_logprobs
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

# DeepSeek API base URL
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Models that expose reasoning content
REASONING_MODELS = {
    "deepseek-reasoner",
}


def _format_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert tools to DeepSeek/OpenAI API format."""
    formatted = []
    for tool in tools:
        if hasattr(tool, "to_dict"):
            formatted.append(tool.to_dict())
        elif hasattr(tool, "to_openai"):
            formatted.append(tool.to_openai())
        elif hasattr(tool, "name") and hasattr(tool, "description"):
            schema = getattr(tool, "args_schema", {}) or {}
            formatted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema,
                    },
                }
            )
        elif isinstance(tool, dict):
            formatted.append(tool)
    return formatted


def _parse_response(response: Any, is_reasoner: bool = False) -> AIMessage:
    """Parse DeepSeek response into AIMessage with metadata.

    Args:
        response: DeepSeek API response.
        is_reasoner: Whether this is from deepseek-reasoner model.

    Returns:
        AIMessage with content, tool_calls, reasoning (if reasoner), and metadata.
    """
    import json

    choice = response.choices[0]
    message = choice.message

    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
                }
            )

    # Extract token usage
    tokens = None
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        tokens = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
        # DeepSeek reasoner may include reasoning tokens
        if hasattr(usage, "reasoning_tokens") and usage.reasoning_tokens:
            tokens.reasoning_tokens = usage.reasoning_tokens

    metadata = MessageMetadata(
        model=response.model if hasattr(response, "model") else None,
        tokens=tokens,
        finish_reason=choice.finish_reason
        if hasattr(choice, "finish_reason")
        else None,
        response_id=response.id if hasattr(response, "id") else None,
    )

    msg = AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
        metadata=metadata,
    )

    # Extract reasoning content from deepseek-reasoner
    if (
        is_reasoner
        and hasattr(message, "reasoning_content")
        and message.reasoning_content
    ):
        msg.reasoning = message.reasoning_content

    return msg


@dataclass
class DeepSeekChat(BaseChatModel):
    """DeepSeek chat model.

    Uses the DeepSeek API (OpenAI-compatible) for DeepSeek models.

    Example:
        from cogent.models.deepseek import DeepSeekChat

        # Standard chat
        llm = DeepSeekChat(model="deepseek-chat")
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

        # Reasoning model with Chain of Thought
        llm = DeepSeekChat(model="deepseek-reasoner")
        response = await llm.ainvoke("9.11 and 9.8, which is greater?")

        # Access reasoning (Chain of Thought)
        if hasattr(response, 'reasoning'):
            print("Reasoning:", response.reasoning)
        print("Answer:", response.content)

    Available models:
        - deepseek-chat: General chat model with tool support
        - deepseek-reasoner: Reasoning model with CoT (no tools)

    Features:
        - Function calling (deepseek-chat only)
        - JSON output mode
        - Chain of Thought reasoning (deepseek-reasoner)
        - Streaming support

    Note:
        deepseek-reasoner does NOT support:
        - Function calling/tools
        - temperature, top_p, presence_penalty, frequency_penalty
    """

    model: str = "deepseek-chat"
    base_url: str = DEEPSEEK_BASE_URL

    _tool_choice: str | dict[str, Any] | None = field(default=None, repr=False)

    def _init_client(self) -> None:
        """Initialize DeepSeek client (OpenAI-compatible)."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: uv add openai"
            ) from None

        from cogent.config import get_api_key

        api_key = get_api_key("deepseek", self.api_key)
        if not api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> DeepSeekChat:
        """Bind tools to the model.

        Note: Only supported on deepseek-chat, NOT on deepseek-reasoner.

        Args:
            tools: List of tools to bind.
            tool_choice: Controls which tool is called.

        Returns:
            New DeepSeekChat instance with tools bound.

        Raises:
            ValueError: If trying to bind tools to deepseek-reasoner.
        """
        if self.model in REASONING_MODELS:
            raise ValueError(
                f"Model {self.model} does not support function calling. "
                f"Use deepseek-chat for tool support."
            )

        self._ensure_initialized()

        new_model = DeepSeekChat(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        new_model._tools = tools
        new_model._tool_choice = tool_choice
        new_model._client = self._client
        new_model._async_client = self._async_client
        new_model._initialized = True
        return new_model

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build API request parameters.

        Handles model-specific restrictions for deepseek-reasoner.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        # Check if model is reasoner (has restrictions)
        is_reasoner = self.model in REASONING_MODELS

        # Temperature (not supported by reasoner)
        if not is_reasoner and self.temperature is not None:
            params["temperature"] = self.temperature

        # Max tokens
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        # Tools (not supported by reasoner)
        if self._tools and not is_reasoner:
            params["tools"] = _format_tools(self._tools)
            if self._tool_choice:
                params["tool_choice"] = self._tool_choice

        # Response format (JSON mode)
        if hasattr(self, "_response_format") and self._response_format:
            params["response_format"] = self._response_format

        # Apply any overrides (but filter restricted params for reasoner)
        for key, value in kwargs.items():
            if is_reasoner and key in (
                "temperature",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
            ):
                continue  # Skip restricted params for reasoner
            params[key] = value

        return params

    async def ainvoke(
        self,
        messages: str | list[dict[str, Any]] | list[Any],
        **kwargs: Any,
    ) -> AIMessage:
        """Async invoke the model.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
            **kwargs: Additional arguments (max_tokens, etc.)

        Returns:
            AIMessage with response content, tool calls (if any), and reasoning (if reasoner).
        """
        self._ensure_initialized()
        converted_messages = convert_messages(normalize_input(messages))
        params = self._build_request(converted_messages, **kwargs)
        response = await self._async_client.chat.completions.create(**params)
        return _parse_response(response, is_reasoner=self.model in REASONING_MODELS)

    def invoke(
        self,
        messages: str | list[dict[str, Any]] | list[Any],
        **kwargs: Any,
    ) -> AIMessage:
        """Sync invoke the model.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
            **kwargs: Additional arguments.

        Returns:
            AIMessage with response content, tool calls (if any), and reasoning (if reasoner).
        """
        self._ensure_initialized()
        converted_messages = convert_messages(normalize_input(messages))
        params = self._build_request(converted_messages, **kwargs)
        response = self._client.chat.completions.create(**params)
        return _parse_response(response, is_reasoner=self.model in REASONING_MODELS)

    async def astream(
        self,
        messages: str | list[dict[str, Any]] | list[Any],
        **kwargs: Any,
    ) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously.

        Args:
            messages: Can be a string, list of dicts, or list of message objects.
            **kwargs: Additional arguments.

        Yields:
            AIMessage objects with incremental content. Final chunk has metadata.

        Note:
            For deepseek-reasoner, reasoning_content is streamed before content.
        """
        self._ensure_initialized()
        converted_messages = convert_messages(normalize_input(messages))
        params = self._build_request(converted_messages, **kwargs)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        is_reasoner = self.model in REASONING_MODELS
        start_time = time.time()
        chunk_metadata: dict[str, Any] = {
            "id": None,
            "model": None,
            "finish_reason": None,
            "usage": None,
        }
        accumulated_reasoning = ""

        stream = await self._async_client.chat.completions.create(**params)

        async for chunk in stream:
            # Accumulate metadata
            if chunk.id:
                chunk_metadata["id"] = chunk.id
            if chunk.model:
                chunk_metadata["model"] = chunk.model

            # Check for finish_reason
            if chunk.choices and chunk.choices[0].finish_reason:
                chunk_metadata["finish_reason"] = chunk.choices[0].finish_reason

            # Check for usage (final chunk)
            if hasattr(chunk, "usage") and chunk.usage:
                chunk_metadata["usage"] = chunk.usage
                # Build final metadata with token usage
                tokens = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )

                metadata = MessageMetadata(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    model=chunk_metadata["model"],
                    tokens=tokens,
                    finish_reason=chunk_metadata["finish_reason"],
                    response_id=chunk_metadata["id"],
                    duration=time.time() - start_time,
                )
                final_msg = AIMessage(content="", metadata=metadata)
                # Add accumulated reasoning to final message
                if is_reasoner and accumulated_reasoning:
                    final_msg.reasoning = accumulated_reasoning
                yield final_msg

            # Handle delta content
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta

                # Handle reasoning content (deepseek-reasoner)
                if (
                    is_reasoner
                    and hasattr(delta, "reasoning_content")
                    and delta.reasoning_content
                ):
                    accumulated_reasoning += delta.reasoning_content
                    # Yield reasoning chunk
                    metadata = MessageMetadata(
                        id=str(uuid.uuid4()),
                        timestamp=time.time(),
                        model=chunk_metadata["model"],
                    )
                    msg = AIMessage(content="", metadata=metadata)
                    msg.reasoning = delta.reasoning_content
                    msg.is_reasoning_chunk = True
                    yield msg

                # Handle regular content
                if delta.content:
                    metadata = MessageMetadata(
                        id=str(uuid.uuid4()),
                        timestamp=time.time(),
                        model=chunk_metadata["model"],
                        tokens=None,
                        finish_reason=chunk_metadata.get("finish_reason"),
                        response_id=chunk_metadata["id"],
                        duration=time.time() - start_time,
                    )
                    yield AIMessage(
                        content=delta.content,
                        metadata=metadata,
                    )

    def with_structured_output(
        self,
        schema: type | dict[str, Any],
        *,
        method: str = "json_mode",
    ) -> DeepSeekChat:
        """Configure model for structured output.

        Note: Only JSON mode is supported. JSON schema not available.

        Args:
            schema: Pydantic model or JSON schema dict (for reference only).
            method: Output method - only "json_mode" supported.

        Returns:
            New DeepSeekChat configured for JSON output.
        """
        self._ensure_initialized()

        new_model = DeepSeekChat(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        new_model._client = self._client
        new_model._async_client = self._async_client
        new_model._initialized = True
        new_model._tools = getattr(self, "_tools", [])
        new_model._tool_choice = self._tool_choice
        new_model._response_format = {"type": "json_object"}

        return new_model
