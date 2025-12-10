"""
Mistral AI chat models for AgenticFlow.

Supports all Mistral models via the official API.

Usage:
    from agenticflow.models.mistral import MistralChat
    
    llm = MistralChat(model="mistral-large-latest")
    response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
    
    # With tools
    llm = MistralChat(model="mistral-large-latest")
    bound = llm.bind_tools([search_tool])
    response = await bound.ainvoke(messages)

Available models:
    - mistral-large-latest: State-of-the-art, best for complex tasks
    - mistral-small-latest: Budget-friendly, good for most tasks
    - codestral-latest: Optimized for coding tasks
    - ministral-8b-latest: Efficient smaller model
    - open-mistral-nemo: Open source 12B model

Note: Mistral does not currently offer embedding models via their API.
      Use OpenAI or another provider for embeddings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from agenticflow.models.base import AIMessage, BaseChatModel, convert_messages


def _format_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert tools to Mistral API format."""
    formatted = []
    for tool in tools:
        if hasattr(tool, "to_dict"):
            formatted.append(tool.to_dict())
        elif hasattr(tool, "to_openai"):
            formatted.append(tool.to_openai())
        elif hasattr(tool, "name") and hasattr(tool, "description"):
            schema = getattr(tool, "args_schema", {}) or {}
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            })
        elif isinstance(tool, dict):
            formatted.append(tool)
    return formatted


def _parse_response(response: Any) -> AIMessage:
    """Parse Mistral response into AIMessage."""
    choice = response.choices[0]
    message = choice.message
    
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = __import__("json").loads(args)
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": args,
            })
    
    return AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
    )


@dataclass
class MistralChat(BaseChatModel):
    """Mistral AI chat model.
    
    Uses the Mistral API (OpenAI-compatible) for high-performance inference.
    
    Example:
        from agenticflow.models.mistral import MistralChat
        
        # State-of-the-art model
        llm = MistralChat(model="mistral-large-latest")
        
        # Budget-friendly
        llm = MistralChat(model="mistral-small-latest")
        
        # For coding
        llm = MistralChat(model="codestral-latest")
        
        response = await llm.ainvoke([{"role": "user", "content": "Hello!"}])
        
    Available models:
        - mistral-large-latest: Best for complex reasoning
        - mistral-small-latest: Cost-effective, good quality
        - codestral-latest: Optimized for code
        - ministral-8b-latest: Efficient 8B model
        - open-mistral-nemo: Open 12B model
    """
    
    model: str = "mistral-small-latest"
    base_url: str = "https://api.mistral.ai/v1"
    
    _tool_choice: str | dict[str, Any] | None = field(default=None, repr=False)
    
    def _init_client(self) -> None:
        """Initialize Mistral client (OpenAI-compatible)."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")
        
        api_key = self.api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable "
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
    
    async def ainvoke(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AIMessage:
        """Async invoke the model.
        
        Args:
            messages: List of message dicts with role and content.
            **kwargs: Additional arguments (temperature, max_tokens, etc.)
            
        Returns:
            AIMessage with response content and any tool calls.
        """
        self._ensure_initialized()
        
        # Convert messages to dict format
        converted_messages = convert_messages(messages)
        
        params: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
        }
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        
        # Add tools if bound
        if self._tools:
            params["tools"] = _format_tools(self._tools)
            if self._tool_choice:
                params["tool_choice"] = self._tool_choice
        
        # Apply overrides
        params.update(kwargs)
        
        response = await self._async_client.chat.completions.create(**params)
        return _parse_response(response)
    
    def invoke(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AIMessage:
        """Sync invoke the model.
        
        Args:
            messages: List of message dicts with role and content.
            **kwargs: Additional arguments.
            
        Returns:
            AIMessage with response content and any tool calls.
        """
        self._ensure_initialized()
        
        # Convert messages to dict format
        converted_messages = convert_messages(messages)
        
        params: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
        }
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        
        if self._tools:
            params["tools"] = _format_tools(self._tools)
            if self._tool_choice:
                params["tool_choice"] = self._tool_choice
        
        params.update(kwargs)
        
        response = self._client.chat.completions.create(**params)
        return _parse_response(response)
    
    async def astream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[AIMessage]:
        """Stream responses from the model.
        
        Args:
            messages: List of message dicts.
            **kwargs: Additional arguments.
            
        Yields:
            AIMessage chunks as they arrive.
        """
        self._ensure_initialized()
        
        # Convert messages to dict format
        converted_messages = convert_messages(messages)
        
        params: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
            "stream": True,
        }
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        
        params.update(kwargs)
        
        stream = await self._async_client.chat.completions.create(**params)
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield AIMessage(content=chunk.choices[0].delta.content)
    
    def bind_tools(
        self,
        tools: list[Any],
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,  # Accept extra kwargs like parallel_tool_calls
    ) -> "MistralChat":
        """Bind tools to the model.
        
        Args:
            tools: List of tools (BaseTool instances or dicts).
            tool_choice: How to choose tools ("auto", "none", "any", or specific).
            **kwargs: Additional arguments (ignored for compatibility).
            
        Returns:
            New model instance with tools bound.
        """
        # Note: Mistral doesn't support parallel_tool_calls, so we ignore it
        new_model = MistralChat(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        new_model._tools = tools
        new_model._tool_choice = tool_choice
        new_model._client = self._client
        new_model._async_client = self._async_client
        new_model._initialized = True
        return new_model
