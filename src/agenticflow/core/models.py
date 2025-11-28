"""
Native model wrapper for AgenticFlow.

Provides a unified interface for LLM providers using their native SDKs.
Currently supports OpenAI, Anthropic, and compatible APIs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from agenticflow.core.messages import (
    AIMessage,
    BaseMessage,
    messages_to_openai,
    parse_openai_response,
)
from agenticflow.tools.base import BaseTool, tools_to_openai


@dataclass
class ChatModel:
    """Native chat model wrapper.
    
    Uses OpenAI SDK directly for maximum performance.
    Supports tool binding and async invocation.
    
    Example:
        model = ChatModel(model="gpt-4o-mini")
        response = await model.ainvoke([HumanMessage("Hello!")])
        
        # With tools:
        model_with_tools = model.bind_tools([search_tool])
        response = await model_with_tools.ainvoke([HumanMessage("Search for X")])
    """
    
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None
    
    # Bound tools (set via bind_tools)
    _tools: list[BaseTool] = field(default_factory=list, repr=False)
    _parallel_tool_calls: bool = True
    
    # Cached client
    _client: Any = field(default=None, repr=False)
    _async_client: Any = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize the OpenAI client."""
        self._init_clients()
    
    def _init_clients(self) -> None:
        """Initialize sync and async OpenAI clients."""
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: uv add openai")
        
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)
    
    def bind_tools(
        self,
        tools: list[BaseTool],
        parallel_tool_calls: bool = True,
    ) -> "ChatModel":
        """Create a new model with tools bound.
        
        Args:
            tools: List of tools to bind.
            parallel_tool_calls: Whether to allow parallel tool calls.
            
        Returns:
            New ChatModel with tools bound.
        """
        # Create a copy with tools
        new_model = ChatModel(
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        new_model._tools = tools
        new_model._parallel_tool_calls = parallel_tool_calls
        new_model._client = self._client
        new_model._async_client = self._async_client
        return new_model
    
    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Invoke the model synchronously.
        
        Args:
            messages: List of messages to send.
            
        Returns:
            AI response message.
        """
        kwargs = self._build_request(messages)
        response = self._client.chat.completions.create(**kwargs)
        return parse_openai_response(response)
    
    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Invoke the model asynchronously.
        
        Args:
            messages: List of messages to send.
            
        Returns:
            AI response message.
        """
        kwargs = self._build_request(messages)
        response = await self._async_client.chat.completions.create(**kwargs)
        return parse_openai_response(response)
    
    def _build_request(self, messages: list[BaseMessage]) -> dict[str, Any]:
        """Build the API request kwargs."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages_to_openai(messages),
            "temperature": self.temperature,
        }
        
        if self._tools:
            kwargs["tools"] = tools_to_openai(self._tools)
            kwargs["parallel_tool_calls"] = self._parallel_tool_calls
        
        return kwargs


# Convenience factory functions
def create_openai_model(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: str | None = None,
) -> ChatModel:
    """Create an OpenAI chat model.
    
    Args:
        model: Model name (e.g., "gpt-4o-mini", "gpt-4o").
        temperature: Sampling temperature.
        api_key: API key (uses OPENAI_API_KEY env var if not provided).
        
    Returns:
        Configured ChatModel.
    """
    return ChatModel(model=model, temperature=temperature, api_key=api_key)


def create_anthropic_model(
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.0,
    api_key: str | None = None,
) -> ChatModel:
    """Create an Anthropic chat model via OpenAI-compatible endpoint.
    
    Note: Requires anthropic package for native support.
    This uses OpenAI-compatible API if available.
    
    Args:
        model: Model name.
        temperature: Sampling temperature.
        api_key: API key (uses ANTHROPIC_API_KEY env var if not provided).
        
    Returns:
        Configured ChatModel.
    """
    # For now, return a placeholder that will fail gracefully
    # Full Anthropic support can be added later
    raise NotImplementedError(
        "Native Anthropic support coming soon. "
        "Use create_openai_model() with an OpenAI-compatible Anthropic endpoint."
    )


def create_model(
    provider: Literal["openai", "anthropic", "ollama"] = "openai",
    model: str | None = None,
    **kwargs: Any,
) -> ChatModel:
    """Create a chat model for the specified provider.
    
    Args:
        provider: Model provider name.
        model: Model name (uses provider default if not specified).
        **kwargs: Additional provider-specific arguments.
        
    Returns:
        Configured ChatModel.
    """
    defaults = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "ollama": "llama3.2",
    }
    
    model_name = model or defaults.get(provider, "gpt-4o-mini")
    
    if provider == "openai":
        return create_openai_model(model=model_name, **kwargs)
    elif provider == "anthropic":
        return create_anthropic_model(model=model_name, **kwargs)
    elif provider == "ollama":
        return ChatModel(
            model=model_name,
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Ollama doesn't need a real key
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
