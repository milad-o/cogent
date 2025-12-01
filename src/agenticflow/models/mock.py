"""Mock models for testing.

Provides deterministic mock implementations of chat and embedding models
that don't require API calls. Perfect for unit tests and development.

Usage:
    from agenticflow.models.mock import MockEmbedding
    
    # Create mock embeddings
    embeddings = MockEmbedding(dimensions=384)
    vectors = embeddings.embed(["Hello", "World"])
    
    # Async works too
    vectors = await embeddings.aembed(["Hello", "World"])
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from agenticflow.models.base import BaseEmbedding


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


__all__ = ["MockEmbedding"]
