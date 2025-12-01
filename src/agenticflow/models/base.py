"""
Base classes for AgenticFlow models.

All chat and embedding models inherit from these base classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

# Import AIMessage from core messages - single source of truth
from agenticflow.core.messages import AIMessage


class ModelProvider(str, Enum):
    """Supported model providers."""
    
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class BaseChatModel(ABC):
    """Abstract base class for all chat models.
    
    All chat model implementations must inherit from this class
    and implement the required methods.
    
    Features:
    - Lazy initialization (no API key needed at construction)
    - Sync and async invocation
    - Tool binding with parallel execution
    - Streaming support
    
    Example:
        class MyChatModel(BaseChatModel):
            model: str = "my-model"
            
            def _init_client(self) -> None:
                self._client = MyClient(api_key=self.api_key)
            
            def invoke(self, messages):
                self._ensure_initialized()
                return self._client.chat(messages)
    """
    
    model: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 2
    
    # Tool binding state
    _tools: list[Any] = field(default_factory=list, repr=False)
    _parallel_tool_calls: bool = field(default=True, repr=False)
    
    # Client state (lazy initialized)
    _client: Any = field(default=None, repr=False)
    _async_client: Any = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)
    
    def _ensure_initialized(self) -> None:
        """Lazily initialize clients on first use."""
        if not self._initialized:
            self._init_client()
            self._initialized = True
    
    @abstractmethod
    def _init_client(self) -> None:
        """Initialize the API client. Must be implemented by subclasses."""
        ...
    
    @abstractmethod
    def invoke(self, messages: list[dict[str, Any]]) -> AIMessage:
        """Invoke the model synchronously.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            
        Returns:
            AIMessage with response content and optional tool calls.
        """
        ...
    
    @abstractmethod
    async def ainvoke(self, messages: list[dict[str, Any]]) -> AIMessage:
        """Invoke the model asynchronously.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            
        Returns:
            AIMessage with response content and optional tool calls.
        """
        ...
    
    @abstractmethod
    def bind_tools(
        self,
        tools: list[Any],
        *,
        parallel_tool_calls: bool = True,
    ) -> "BaseChatModel":
        """Bind tools to the model.
        
        Args:
            tools: List of tools to bind.
            parallel_tool_calls: Allow parallel tool execution.
            
        Returns:
            New model instance with tools bound.
        """
        ...
    
    async def astream(
        self,
        messages: list[dict[str, Any]],
    ) -> AsyncIterator[AIMessage]:
        """Stream response asynchronously.
        
        Default implementation yields single response.
        Override for true streaming support.
        
        Args:
            messages: List of message dicts.
            
        Yields:
            AIMessage chunks with partial content.
        """
        response = await self.ainvoke(messages)
        yield response


@dataclass  
class BaseEmbedding(ABC):
    """Abstract base class for all embedding models.
    
    All embedding implementations must inherit from this class
    and implement the required methods.
    
    Features:
    - Lazy initialization
    - Single and batch embedding
    - Sync and async support
    - Configurable dimensions
    """
    
    model: str = ""
    dimensions: int | None = None
    api_key: str | None = None
    timeout: float = 60.0
    max_retries: int = 2
    batch_size: int = 100
    
    # Client state (lazy initialized)
    _client: Any = field(default=None, repr=False)
    _async_client: Any = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)
    
    def _ensure_initialized(self) -> None:
        """Lazily initialize clients on first use."""
        if not self._initialized:
            self._init_client()
            self._initialized = True
    
    @abstractmethod
    def _init_client(self) -> None:
        """Initialize the API client. Must be implemented by subclasses."""
        ...
    
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts synchronously.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        ...
    
    @abstractmethod
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts asynchronously.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        ...
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
        """
        return self.embed([text])[0]
    
    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query text asynchronously.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
        """
        result = await self.aembed([text])
        return result[0]
    
    # Aliases for common interface compatibility
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts (alias for embed)."""
        return self.embed(texts)
    
    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts async (alias for aembed)."""
        return await self.aembed(texts)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (alias for embed)."""
        return self.embed(texts)
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents async (alias for aembed)."""
        return await self.aembed(texts)
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension. Override in subclasses."""
        return self.dimensions or 1536  # Default to OpenAI dimension


# Message helper functions for easy message creation
def system(content: str) -> dict[str, str]:
    """Create a system message.
    
    Example:
        messages = [system("You are helpful"), user("Hello")]
    """
    return {"role": "system", "content": content}


def user(content: str) -> dict[str, str]:
    """Create a user message.
    
    Example:
        messages = [user("What is 2+2?")]
    """
    return {"role": "user", "content": content}


def assistant(content: str, tool_calls: list[dict] | None = None) -> dict[str, Any]:
    """Create an assistant message.
    
    Example:
        messages = [assistant("The answer is 4")]
    """
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def tool_result(content: str, tool_call_id: str) -> dict[str, str]:
    """Create a tool result message.
    
    Example:
        messages = [tool_result("42", "call_123")]
    """
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


# Default model names by provider
DEFAULT_CHAT_MODELS: dict[ModelProvider, str] = {
    ModelProvider.OPENAI: "gpt-4o-mini",
    ModelProvider.AZURE: "gpt-4o-mini",
    ModelProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    ModelProvider.GOOGLE: "gemini-2.0-flash-exp",
    ModelProvider.GROQ: "llama-3.3-70b-versatile",
    ModelProvider.OLLAMA: "llama3.2",
    ModelProvider.CUSTOM: "gpt-4o-mini",
}

DEFAULT_EMBEDDING_MODELS: dict[ModelProvider, str] = {
    ModelProvider.OPENAI: "text-embedding-3-small",
    ModelProvider.AZURE: "text-embedding-3-small",
    ModelProvider.ANTHROPIC: "voyage-3",
    ModelProvider.GOOGLE: "text-embedding-004",
    ModelProvider.GROQ: "nomic-embed-text",  # Via Groq
    ModelProvider.OLLAMA: "nomic-embed-text",
    ModelProvider.CUSTOM: "text-embedding-3-small",
}
