"""Mock models for testing.

Provides deterministic mock implementations of chat and embedding models
that don't require API calls. Perfect for unit tests and development.

Usage:
    from agenticflow.models.mock import MockEmbedding, MockChatModel
    
    # Create mock embeddings
    embeddings = MockEmbedding(dimensions=384)
    vectors = embeddings.embed(["Hello", "World"])
    
    # Create mock chat model
    model = MockChatModel(responses=["Hello!", "How can I help?"])
    response = await model.ainvoke([{"role": "user", "content": "Hi"}])
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from agenticflow.core.messages import AIMessage
from agenticflow.models.base import BaseChatModel, BaseEmbedding


@dataclass
class MockChatModel(BaseChatModel):
    """Mock chat model for testing without API calls.
    
    Returns predefined responses in sequence, cycling back to start
    when exhausted. Supports tool calls via mock_tool_calls parameter.
    
    Attributes:
        responses: List of responses to return in sequence.
        model_name: Model name (default: "mock-chat").
        mock_tool_calls: Optional list of tool calls to include in responses.
    
    Example:
        >>> from agenticflow.models.mock import MockChatModel
        >>> model = MockChatModel(responses=["Hello!", "Goodbye!"])
        >>> response = await model.ainvoke([{"role": "user", "content": "Hi"}])
        >>> response.content
        'Hello!'
        >>> response = await model.ainvoke([{"role": "user", "content": "Bye"}])
        >>> response.content
        'Goodbye!'
    """
    
    responses: list[str] = field(default_factory=lambda: ["Mock response"])
    model_name: str = "mock-chat"
    mock_tool_calls: list[dict[str, Any]] | None = None
    
    _response_index: int = field(default=0, init=False, repr=False)
    
    def _init_client(self) -> None:
        """No client needed for mock."""
        pass
    
    def _get_next_response(self) -> str:
        """Get the next response in sequence, cycling if needed."""
        if not self.responses:
            return "Mock response"
        response = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1
        return response
    
    def invoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> AIMessage:
        """Generate mock response synchronously.
        
        Args:
            messages: List of message dicts (ignored, returns next in sequence).
            **kwargs: Additional arguments (ignored).
            
        Returns:
            AIMessage with next response.
        """
        content = self._get_next_response()
        return AIMessage(
            content=content,
            tool_calls=self.mock_tool_calls or [],
        )
    
    async def ainvoke(self, messages: list[dict[str, Any]], **kwargs: Any) -> AIMessage:
        """Generate mock response asynchronously.
        
        Args:
            messages: List of message dicts (ignored, returns next in sequence).
            **kwargs: Additional arguments (ignored).
            
        Returns:
            AIMessage with next response.
        """
        return self.invoke(messages, **kwargs)
    
    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "MockChatModel":
        """Return self (tools binding is a no-op for mock).
        
        Args:
            tools: List of tools to bind (ignored).
            **kwargs: Additional arguments (ignored).
            
        Returns:
            Self (mock doesn't actually bind tools).
        """
        return self
    
    def reset(self) -> None:
        """Reset response index to start from beginning."""
        self._response_index = 0


@dataclass
class MockEmbedding(BaseEmbedding):
    """Mock embedding model for testing without API calls.
    
    Generates deterministic embeddings based on text hash.
    Same text always produces the same embedding vector.
    
    Attributes:
        dimensions: Dimension of generated embeddings (default: 384).
        model: Model name (default: "mock-embedding").
    
    Example:
        >>> from agenticflow.models.mock import MockEmbedding
        >>> embeddings = MockEmbedding(dimensions=128)
        >>> vectors = embeddings.embed(["Hello", "World"])
        >>> len(vectors[0])
        128
        >>> # Same text = same embedding
        >>> embeddings.embed(["Hello"])[0] == embeddings.embed(["Hello"])[0]
        True
    """
    
    model: str = "mock-embedding"
    dimensions: int = 384
    
    def _init_client(self) -> None:
        """No client needed for mock."""
        pass
    
    def _generate_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding from text hash.
        
        Uses SHA-256 hash of text to generate consistent embeddings.
        Result is normalized to unit length.
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        embedding = []
        for i in range(self.dimensions or 384):
            byte_idx = i % 32
            byte_val = int(text_hash[byte_idx * 2:(byte_idx + 1) * 2], 16)
            val = (byte_val / 127.5 - 1) * math.cos(i * 0.1)
            embedding.append(val)
        
        # Normalize to unit length
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings synchronously.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        return [self._generate_embedding(text) for text in texts]
    
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings asynchronously.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        return [self._generate_embedding(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.dimensions or 384


__all__ = ["MockEmbedding", "MockChatModel"]
