"""Embedding providers for vector store.

Provides embedding generation using:
- OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- Ollama: nomic-embed-text, mxbai-embed-large, etc.

Example:
    >>> from agenticflow.vectorstore import OpenAIEmbeddings
    >>> embeddings = OpenAIEmbeddings()
    >>> vectors = await embeddings.embed_texts(["Hello", "World"])
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import httpx


# ============================================================
# Embedding Dimensions for Common Models
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
# OpenAI Embeddings
# ============================================================

@dataclass
class OpenAIEmbeddings:
    """OpenAI embedding provider.
    
    Uses the OpenAI API to generate embeddings. Supports text-embedding-3-small,
    text-embedding-3-large, and text-embedding-ada-002.
    
    Attributes:
        model: Model to use (default: text-embedding-3-small).
        api_key: OpenAI API key (default: from OPENAI_API_KEY env var).
        base_url: API base URL (for Azure or compatible APIs).
        batch_size: Maximum texts per API call.
        
    Example:
        >>> embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        >>> vectors = await embeddings.embed_texts(["Hello world"])
    """
    
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    batch_size: int = 100
    _cache: dict[str, list[float]] = field(default_factory=dict, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
    
    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings."""
        return get_embedding_dimension(self.model)
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.
        
        Uses batching for efficiency and caching for repeated texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            ValueError: If API key is not set.
            httpx.HTTPError: If API call fails.
        """
        if not self.api_key:
            msg = "OpenAI API key not set. Set OPENAI_API_KEY env var or pass api_key."
            raise ValueError(msg)
        
        # Check cache for each text
        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []
        
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                texts_to_embed.append((i, text))
        
        # Embed uncached texts in batches
        if texts_to_embed:
            for batch_start in range(0, len(texts_to_embed), self.batch_size):
                batch = texts_to_embed[batch_start:batch_start + self.batch_size]
                batch_texts = [text for _, text in batch]
                batch_embeddings = await self._embed_batch(batch_texts)
                
                for (idx, text), embedding in zip(batch, batch_embeddings):
                    results[idx] = embedding
                    self._cache[self._cache_key(text)] = embedding
        
        return [r for r in results if r is not None]  # type: ignore
    
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.
        
        For OpenAI, query and document embeddings are the same.
        
        Args:
            text: Query text to embed.
            
        Returns:
            Embedding vector.
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]
    
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via API.
        
        Args:
            texts: Batch of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": texts,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
        
        # Sort by index to maintain order
        embeddings_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings_data]
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()


# ============================================================
# Ollama Embeddings
# ============================================================

@dataclass
class OllamaEmbeddings:
    """Ollama embedding provider.
    
    Uses a local Ollama server to generate embeddings. Supports various models
    like nomic-embed-text, mxbai-embed-large, etc.
    
    Attributes:
        model: Model to use (default: nomic-embed-text).
        base_url: Ollama server URL (default: http://localhost:11434).
        
    Example:
        >>> embeddings = OllamaEmbeddings(model="nomic-embed-text")
        >>> vectors = await embeddings.embed_texts(["Hello world"])
    """
    
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    _cache: dict[str, list[float]] = field(default_factory=dict, repr=False)
    _dimension: int | None = field(default=None, repr=False)
    
    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings."""
        if self._dimension is not None:
            return self._dimension
        return get_embedding_dimension(self.model)
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.
        
        Ollama doesn't support batch embedding, so we process sequentially
        with caching for efficiency.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        results: list[list[float]] = []
        
        for text in texts:
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                embedding = await self._embed_single(text)
                self._cache[cache_key] = embedding
                results.append(embedding)
                
                # Update dimension on first embed
                if self._dimension is None:
                    self._dimension = len(embedding)
        
        return results
    
    async def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.
        
        Args:
            text: Query text to embed.
            
        Returns:
            Embedding vector.
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]
    
    async def _embed_single(self, text: str) -> list[float]:
        """Embed a single text via Ollama API.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
        
        return data["embedding"]
    
    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()


# ============================================================
# Mock Embeddings (for testing)
# ============================================================

@dataclass
class MockEmbeddings:
    """Mock embedding provider for testing.
    
    Generates deterministic embeddings based on text hash.
    Useful for unit tests without API calls.
    
    Attributes:
        dimension: Dimension of generated embeddings.
    """
    
    _dimension: int = 384
    
    @property
    def dimension(self) -> int:
        """Return the dimension of embeddings."""
        return self._dimension
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings.
        
        Uses text hash to generate deterministic vectors.
        """
        return [self._generate_embedding(text) for text in texts]
    
    async def embed_query(self, text: str) -> list[float]:
        """Generate mock embedding for query."""
        return self._generate_embedding(text)
    
    def _generate_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding from text hash."""
        import math
        
        # Use hash to seed the embedding
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Generate embedding from hash bytes
        embedding = []
        for i in range(self._dimension):
            # Use different parts of hash for each dimension
            byte_idx = i % 32
            byte_val = int(text_hash[byte_idx * 2:(byte_idx + 1) * 2], 16)
            # Normalize to [-1, 1] range with some variation
            val = (byte_val / 127.5 - 1) * math.cos(i * 0.1)
            embedding.append(val)
        
        # Normalize to unit vector
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
