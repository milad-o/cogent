"""
xAI Grok models for Cogent.

Supports all Grok models via the xAI API (OpenAI-compatible).

Usage:
    from cogent.models.xai import XAIChat

    # Latest flagship model (Grok 4 - reasoning)
    llm = XAIChat(model="grok-4")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    # Fast agentic model (2M context, optimized for tool calling)
    llm = XAIChat(model="grok-4-1-fast")
    response = await llm.ainvoke(messages)

    # Non-reasoning variant (faster, cheaper)
    llm = XAIChat(model="grok-4-1-fast-non-reasoning")

    # With reasoning effort control (grok-3-mini only)
    llm = XAIChat(model="grok-3-mini", reasoning_effort="high")
    response = await llm.ainvoke("What is 101 * 3?")
    print(response.metadata.tokens.reasoning_tokens)

    # With tools
    bound = llm.bind_tools([search_tool])
    response = await bound.ainvoke(messages)

    # Vision (image understanding)
    llm = XAIChat(model="grok-2-vision-1212")
    response = await llm.ainvoke([
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "..."}}
        ]}
    ])

Available models:
    Flagship:
    - grok-4 (alias: grok-4-0709): Latest reasoning model, 256K context
    - grok-4-latest: Points to latest grok-4

    Fast/Agentic (2M context):
    - grok-4-1-fast: Frontier multimodal, optimized for agentic tool calling
    - grok-4-1-fast-reasoning: Same with explicit reasoning
    - grok-4-1-fast-non-reasoning: Without reasoning (faster)

    Legacy:
    - grok-3, grok-3-beta: Previous flagship
    - grok-3-mini, grok-3-mini-beta: Smaller, faster (supports reasoning_effort)

    Vision:
    - grok-2-vision-1212: Image understanding

    Code:
    - grok-code-fast-1: Optimized for coding tasks

Features:
    - Function/tool calling
    - Structured outputs (JSON mode)
    - Reasoning (model thinks before responding)
    - Reasoning effort control (grok-3-mini: low/high)
    - Image understanding (vision models)
    - 2M context window (grok-4-1-fast models)
    - Streaming support
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from cogent.core.messages import MessageMetadata, TokenUsage
from cogent.models.base import (
    AIMessage,
    BaseChatModel,
    convert_messages,
    normalize_input,
)

# xAI API base URL
XAI_BASE_URL = "https://api.x.ai/v1"

# Models that support reasoning (have reasoning_tokens in response)
REASONING_MODELS = {
    "grok-4",
    "grok-4-0709",
    "grok-4-latest",
    "grok-4-1-fast",
    "grok-4-1-fast-reasoning",
    "grok-4-1-fast-reasoning-latest",
    "grok-3-mini",
    "grok-3-mini-beta",
}

# Models that support reasoning_effort parameter (low/high)
REASONING_EFFORT_MODELS = {
    "grok-3-mini",
    "grok-3-mini-beta",
}

# Models that don't support temperature/presence_penalty/frequency_penalty
# (reasoning models have restrictions)
RESTRICTED_PARAMS_MODELS = {
    "grok-4",
    "grok-4-0709",
    "grok-4-latest",
}


def _format_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert tools to xAI/OpenAI API format."""
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


def _parse_response(response: Any) -> AIMessage:
    """Parse xAI response into AIMessage with metadata."""
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

    # Extract token usage including reasoning tokens
    tokens = None
    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        tokens = TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
        # Include reasoning tokens if available
        if hasattr(usage, "completion_tokens_details"):
            details = usage.completion_tokens_details
            if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                tokens.reasoning_tokens = details.reasoning_tokens

    metadata = MessageMetadata(
        model=response.model if hasattr(response, "model") else None,
        tokens=tokens,
        finish_reason=choice.finish_reason
        if hasattr(choice, "finish_reason")
        else None,
        response_id=response.id if hasattr(response, "id") else None,
    )

    return AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
        metadata=metadata,
    )


@dataclass
class XAIChat(BaseChatModel):
    """xAI Grok chat model.

    Uses the xAI API (OpenAI-compatible) for Grok models.

    Example:
        from cogent.models.xai import XAIChat

        # Flagship reasoning model
        llm = XAIChat(model="grok-4")

        # Fast agentic model (2M context, great for tools)
        llm = XAIChat(model="grok-4-1-fast")

        # Non-reasoning (faster, cheaper)
        llm = XAIChat(model="grok-4-1-fast-non-reasoning")

        # With reasoning effort (grok-3-mini only)
        llm = XAIChat(model="grok-3-mini", reasoning_effort="high")
        response = await llm.ainvoke("What is 101 * 3?")
        print(response.metadata.tokens.reasoning_tokens)

        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])

    Available models:
        - grok-4: Flagship reasoning model (256K context)
        - grok-4-1-fast: Agentic model optimized for tools (2M context)
        - grok-4-1-fast-non-reasoning: Fast without reasoning
        - grok-3, grok-3-mini: Legacy models (grok-3-mini supports reasoning_effort)
        - grok-2-vision-1212: Image understanding
        - grok-code-fast-1: Code-optimized

    Features:
        - Function calling (all models)
        - Structured outputs (JSON mode)
        - Reasoning (grok-4, grok-4-1-fast-reasoning, grok-3-mini)
        - Reasoning effort control (grok-3-mini: low/high)
        - Vision (grok-2-vision-1212)
        - 2M context (grok-4-1-fast models)

    Attributes:
        model: Model name (default: grok-4-1-fast).
        reasoning_effort: Reasoning effort for grok-3-mini models.
            - "low": Minimal thinking, fewer tokens, faster
            - "high": Maximum thinking, more tokens, better for complex problems
            - None: Use model default
    """

    model: str = "grok-4-1-fast"
    base_url: str = XAI_BASE_URL
    reasoning_effort: Literal["low", "high"] | None = field(default=None)

    _tool_choice: str | dict[str, Any] | None = field(default=None, repr=False)
    _parallel_tool_calls: bool = field(default=True, repr=False)

    def _init_client(self) -> None:
        """Initialize xAI client (OpenAI-compatible)."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: uv add openai"
            ) from None

        from cogent.config import get_api_key

        api_key = get_api_key("xai", self.api_key)
        if not api_key:
            raise ValueError(
                "xAI API key required. Set XAI_API_KEY environment variable "
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
        parallel_tool_calls: bool = True,
    ) -> XAIChat:
        """Bind tools to the model.

        Args:
            tools: List of tools to bind.
            tool_choice: Controls which tool is called:
                - "auto": Model decides (default)
                - "required": Must call a tool
                - "none": No tools
                - {"type": "function", "function": {"name": "..."}}: Specific tool
            parallel_tool_calls: Allow parallel tool calls (default True).

        Returns:
            New XAIChat instance with tools bound.
        """
        self._ensure_initialized()

        new_model = XAIChat(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            reasoning_effort=self.reasoning_effort,
        )
        new_model._tools = tools
        new_model._tool_choice = tool_choice
        new_model._parallel_tool_calls = parallel_tool_calls
        new_model._client = self._client
        new_model._async_client = self._async_client
        new_model._initialized = True
        return new_model

    def with_reasoning(
        self,
        effort: Literal["low", "high"] = "high",
    ) -> XAIChat:
        """Enable reasoning with specified effort level.

        Only supported on grok-3-mini models. Other models have
        reasoning enabled by default and cannot be controlled.

        Args:
            effort: Reasoning effort level.
                - "low": Minimal thinking, faster responses
                - "high": Maximum thinking, better for complex problems

        Returns:
            New XAIChat instance with reasoning enabled.

        Example:
            llm = XAIChat(model="grok-3-mini").with_reasoning("high")
            response = await llm.ainvoke("What is 127 * 893?")
            print(response.metadata.tokens.reasoning_tokens)

        Note:
            - Only supported on grok-3-mini models
            - Reasoning tokens are tracked in response metadata
        """
        self._ensure_initialized()

        # Validate model supports reasoning_effort
        if self.model not in REASONING_EFFORT_MODELS:
            raise ValueError(
                f"Model {self.model} does not support reasoning_effort. "
                f"Only grok-3-mini supports this parameter. "
                f"Other models (grok-4, grok-4-1-fast) have reasoning enabled by default."
            )

        new_model = XAIChat(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            reasoning_effort=effort,
        )
        new_model._tools = getattr(self, "_tools", [])
        new_model._tool_choice = self._tool_choice
        new_model._parallel_tool_calls = self._parallel_tool_calls
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

        Handles model-specific restrictions (e.g., reasoning models don't support
        temperature, presence_penalty, frequency_penalty, stop).
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        # Check if model has parameter restrictions
        is_restricted = self.model in RESTRICTED_PARAMS_MODELS

        # Temperature (not supported by Grok 4 reasoning models)
        if not is_restricted and self.temperature is not None:
            params["temperature"] = self.temperature

        # Max tokens
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        # Reasoning effort (grok-3-mini only)
        if self.reasoning_effort and self.model in REASONING_EFFORT_MODELS:
            params["reasoning_effort"] = self.reasoning_effort

        # Tools
        if self._tools:
            params["tools"] = _format_tools(self._tools)
            params["parallel_tool_calls"] = self._parallel_tool_calls
            if self._tool_choice:
                params["tool_choice"] = self._tool_choice

        # Structured output support
        if hasattr(self, "_response_format") and self._response_format:
            params["response_format"] = self._response_format

        # Apply any overrides (but filter restricted params for reasoning models)
        for key, value in kwargs.items():
            if is_restricted and key in (
                "temperature",
                "presence_penalty",
                "frequency_penalty",
                "stop",
            ):
                continue  # Skip restricted params for reasoning models
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
            AIMessage with response content and any tool calls.
        """
        self._ensure_initialized()
        converted_messages = convert_messages(normalize_input(messages))
        params = self._build_request(converted_messages, **kwargs)
        response = await self._async_client.chat.completions.create(**params)
        return _parse_response(response)

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
            AIMessage with response content and any tool calls.
        """
        self._ensure_initialized()
        converted_messages = convert_messages(normalize_input(messages))
        params = self._build_request(converted_messages, **kwargs)
        response = self._client.chat.completions.create(**params)
        return _parse_response(response)

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
        """
        self._ensure_initialized()
        converted_messages = convert_messages(normalize_input(messages))
        params = self._build_request(converted_messages, **kwargs)
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}

        start_time = time.time()
        chunk_metadata: dict[str, Any] = {
            "id": None,
            "model": None,
            "finish_reason": None,
            "usage": None,
        }

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
                # Include reasoning tokens if available
                if hasattr(chunk.usage, "completion_tokens_details"):
                    details = chunk.usage.completion_tokens_details
                    if (
                        hasattr(details, "reasoning_tokens")
                        and details.reasoning_tokens
                    ):
                        tokens.reasoning_tokens = details.reasoning_tokens

                metadata = MessageMetadata(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    model=chunk_metadata["model"],
                    tokens=tokens,
                    finish_reason=chunk_metadata["finish_reason"],
                    response_id=chunk_metadata["id"],
                    duration=time.time() - start_time,
                )
                yield AIMessage(content="", metadata=metadata)

            # Yield content chunks
            if chunk.choices and chunk.choices[0].delta.content:
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
                    content=chunk.choices[0].delta.content,
                    metadata=metadata,
                )

    def with_structured_output(
        self,
        schema: type | dict[str, Any],
        *,
        method: str = "json_schema",
    ) -> XAIChat:
        """Configure model for structured output.

        Args:
            schema: Pydantic model or JSON schema dict.
            method: Output method - "json_schema" (recommended) or "json_mode".

        Returns:
            New XAIChat configured for structured output.

        Example:
            from pydantic import BaseModel

            class Person(BaseModel):
                name: str
                age: int

            llm = XAIChat(model="grok-4-1-fast").with_structured_output(Person)
            response = await llm.ainvoke("Extract: John is 30 years old")
        """
        self._ensure_initialized()

        new_model = XAIChat(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            reasoning_effort=self.reasoning_effort,
        )
        new_model._client = self._client
        new_model._async_client = self._async_client
        new_model._initialized = True
        new_model._tools = self._tools
        new_model._tool_choice = self._tool_choice
        new_model._parallel_tool_calls = self._parallel_tool_calls

        # Build response format
        if method == "json_mode":
            new_model._response_format = {"type": "json_object"}
        else:
            # json_schema method
            if hasattr(schema, "model_json_schema"):
                # Pydantic model
                json_schema = schema.model_json_schema()
                schema_name = schema.__name__
            else:
                # Already a dict
                json_schema = schema
                schema_name = schema.get("title", "response")

            new_model._response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": True,
                },
            }

        return new_model
