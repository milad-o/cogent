"""Embedding providers for vector store.

This module re-exports embeddings from agenticflow.models for backward compatibility.
Prefer importing directly from agenticflow.models for new code:

    from agenticflow.models import create_embedding
    from agenticflow.models.openai import OpenAIEmbedding
    from agenticflow.models.ollama import OllamaEmbedding

For testing without API calls, use MockEmbeddings defined here.

Example:
    >>> from agenticflow.vectorstore import MockEmbeddings
    >>> embeddings = MockEmbeddings()
    >>> vectors = embeddings.embed(["Hello", "World"])
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

# Re-export from models for backward compatibility
from agenticflow.models.openai import OpenAIEmbedding as OpenAIEmbeddings
from agenticflow.models.ollama import OllamaEmbedding as OllamaEmbeddings


# ============================================================
# Embedding Dimensions for Common Models (utility)
# ============================================================

EMBEDDING_DIMENSIONS: dict[str, int] = {
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Ollama
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}


def get_embedding_dimension(model: str) -> int:
    """Get the embedding dimension for a model.
    
    Args:
        model: Model name.
        
    Returns:
        Embedding dimension, or 1536 as default.
    """
    return EMBEDDING_DIMENSIONS.get(model, 1536)


# ============================================================
# Mock Embeddings (for testing)
# ============================================================

@dataclass
class MockEmbeddings:
    """Mock embedding provider for testing.
    
    Generates deterministic embeddings based on text hash.
    Useful for unit tests without API calls.
    
    Attributes:
        dimension: Dimension of generated embeddings (default: 384).
    
    Example:
        >>> embeddings = MockEmbeddings()
        >>> vectors = embeddings.embed(["Hello", "World"])
        >>> # Also works async
        >>> vectors = await embeddings.aembed(["Hello", "World"])
    """
    
    _dimension: int = 384
    
    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings."""
        return self._dimension
    
    def _generate_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding from text hash."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        embedding = []
        for i in range(self._dimension):
            byte_idx = i % 32
            byte_val = int(text_hash[byte_idx * 2:(byte_idx + 1) * 2], 16)
            val = (byte_val / 127.5 - 1) * math.cos(i * 0.1)
            embedding.append(val)
        
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    # ========== Sync API ==========
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings synchronously."""
        return [self._generate_embedding(text) for text in texts]
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents (alias for embed)."""
        return self.embed(texts)
    
    # ========== Async API ==========
    
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings asynchronously."""
        return [self._generate_embedding(text) for text in texts]
    
    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query asynchronously."""
        return self._generate_embedding(text)
    
    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts async (alias for aembed)."""
        return await self.aembed(texts)
    
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents async (alias for aembed)."""
        return await self.aembed(texts)
    
    # Legacy async methods (for backward compat with vectorstore tests)
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts (async for backward compat)."""
        return await self.aembed(texts)
    
    async def embed_query(self, text: str) -> list[float]:
        """Embed query (async for backward compat)."""
        return await self.aembed_query(text)


__all__ = [
    # Re-exports from models (backward compat)
    "OpenAIEmbeddings",
    "OllamaEmbeddings",
    # Testing
    "MockEmbeddings",
    # Utilities
    "EMBEDDING_DIMENSIONS",
    "get_embedding_dimension",
]
