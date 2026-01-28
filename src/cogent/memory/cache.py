"""Semantic caching for agent reasoning artifacts.

Based on arXiv:2601.16286 - "SemanticALLI: Caching Reasoning, Not Just Responses"

Key innovation: Cache structured intermediate representations (IRs) at each reasoning
stage, not just final outputs. Achieves 83% hit rate vs 38% for boundary caching.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from cogent.core.utils import now_utc

if TYPE_CHECKING:
    from cogent.models import EmbeddingModel


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for objects that can generate embeddings."""

    async def embed_query(self, query: str) -> Any: ...


@dataclass
class CachedArtifact:
    """A cacheable intermediate representation from agent reasoning.
    
    Artifacts represent structured outputs from specific reasoning stages:
    - Intent parsing: Extract goal from natural language
    - Plan generation: Decompose into executable steps  
    - Tool patterns: Common tool call sequences
    - Result synthesis: Output formatting patterns
    
    Attributes:
        stage: Reasoning stage identifier
        semantic_key: Embedding-based similarity key
        artifact: Structured data (JSON-serializable)
        created_at: Creation timestamp
        hit_count: Number of cache hits
        last_accessed: Last access timestamp
        context_hash: Hash of execution context (prompt, tools, model)
        ttl_seconds: Time-to-live for cache entry
    """

    stage: str
    semantic_key: str
    artifact: dict[str, Any]
    context_hash: str
    created_at: datetime = field(default_factory=now_utc)
    hit_count: int = 0
    last_accessed: datetime | None = None
    ttl_seconds: int = 86400  # 24 hours default

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        age = (now_utc() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hit_count += 1
        self.last_accessed = now_utc()


class SemanticCache:
    """Pipeline-aware semantic cache for agent reasoning.
    
    Caches structured intermediate representations at each reasoning stage:
    1. Intent parsing - What is the user asking?
    2. Plan generation - What steps are needed?
    3. Tool patterns - What tool sequences are common?
    4. Result synthesis - How to format output?
    
    Uses embedding similarity for cache key matching, achieving 83% hit rate
    compared to 38% for traditional boundary caching.
    
    Args:
        embedding: Embedding model instance with embed_query method
        similarity_threshold: Minimum cosine similarity for cache hit (0.85 = ~85% similar)
        max_entries: Maximum cache entries per stage (LRU eviction)
        default_ttl: Default time-to-live in seconds
    
    Example:
        ```python
        from cogent.models import create_embedding
        
        # Create cache with embedding model
        embed = create_embedding("openai", "text-embedding-3-small")
        cache = SemanticCache(embedding=embed)
        
        # Cache intent parsing
        intent = {"goal": "find_tutorials", "topic": "python"}
        await cache.put(
            stage="intent_parse",
            query="Show me Python tutorials",
            artifact=intent,
            context_hash="abc123",
        )
        
        # Later, similar query hits cache
        cached = await cache.get(
            stage="intent_parse",
            query="Find resources for learning Python",  # Different wording!
            context_hash="abc123",
        )
        # Returns intent artifact - semantic match!
        ```
    
    Performance (from research):
        - 83.10% cache hit rate (vs 38.7% baseline)
        - 2.66ms median latency per hit
        - 4,023 LLM calls saved per 1000 queries
    """

    def __init__(
        self,
        embedding: EmbeddingProvider | EmbeddingModel,
        similarity_threshold: float = 0.85,
        max_entries: int = 10000,
        default_ttl: int = 86400,
    ) -> None:
        self._embedding = embedding
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._default_ttl = default_ttl

        # Stage-specific caches (LRU ordering)
        self._caches: dict[str, OrderedDict[str, CachedArtifact]] = {}

        # Metrics
        self._hits = 0
        self._misses = 0
        self._invalidations = 0

    async def get(
        self,
        stage: str,
        query: str,
        context_hash: str,
    ) -> CachedArtifact | None:
        """Retrieve cached artifact if semantically similar.
        
        Args:
            stage: Reasoning stage identifier
            query: Query text to match
            context_hash: Execution context hash (for invalidation)
        
        Returns:
            Cached artifact if found and valid, None otherwise
        """
        # Get or create stage cache
        if stage not in self._caches:
            self._misses += 1
            return None

        stage_cache = self._caches[stage]

        # Generate embedding for query
        query_embedding = await self._embed(query)

        # Find best match
        best_match: CachedArtifact | None = None
        best_similarity = -1.0

        for cached in stage_cache.values():
            # Check context validity
            if cached.context_hash != context_hash:
                continue

            # Check expiration
            if cached.is_expired():
                continue

            # Compute similarity
            cached_embedding = await self._embed(cached.semantic_key)
            similarity = self._cosine_similarity(query_embedding, cached_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached

        # Check if best match meets threshold
        if best_match and best_similarity >= self._threshold:
            best_match.record_hit()
            # Move to end (most recently used)
            stage_cache.move_to_end(best_match.semantic_key)
            self._hits += 1
            return best_match

        self._misses += 1
        return None

    async def put(
        self,
        stage: str,
        query: str,
        artifact: dict[str, Any],
        context_hash: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store artifact for future retrieval.
        
        Args:
            stage: Reasoning stage identifier
            query: Query text (used as semantic key)
            artifact: Structured data to cache
            context_hash: Execution context hash
            ttl_seconds: Override default TTL
        """
        # Get or create stage cache
        if stage not in self._caches:
            self._caches[stage] = OrderedDict()

        stage_cache = self._caches[stage]

        # Create cached artifact
        cached = CachedArtifact(
            stage=stage,
            semantic_key=query,
            artifact=artifact,
            context_hash=context_hash,
            ttl_seconds=ttl_seconds or self._default_ttl,
        )

        # Store (will replace existing if same key)
        stage_cache[query] = cached

        # LRU eviction if over limit
        while len(stage_cache) > self._max_entries:
            stage_cache.popitem(last=False)  # Remove oldest

    def invalidate(self, stage: str | None = None, context_hash: str | None = None) -> int:
        """Invalidate cache entries.
        
        Args:
            stage: Specific stage to invalidate (None = all stages)
            context_hash: Specific context to invalidate (None = all contexts)
        
        Returns:
            Number of entries invalidated
        """
        count = 0

        # Invalidate all
        if stage is None and context_hash is None:
            for stage_cache in self._caches.values():
                count += len(stage_cache)
                stage_cache.clear()
            self._invalidations += count
            return count

        # Invalidate specific stage
        if stage and stage in self._caches:
            if context_hash is None:
                count = len(self._caches[stage])
                self._caches[stage].clear()
            else:
                # Invalidate specific context within stage
                stage_cache = self._caches[stage]
                to_remove = [
                    key
                    for key, cached in stage_cache.items()
                    if cached.context_hash == context_hash
                ]
                for key in to_remove:
                    del stage_cache[key]
                    count += 1

        self._invalidations += count
        return count

    def get_metrics(self) -> dict[str, Any]:
        """Get cache performance metrics.
        
        Returns:
            Dictionary with hit rate, size, and performance stats
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        total_entries = sum(len(cache) for cache in self._caches.values())

        # Estimate memory usage (rough approximation)
        avg_artifact_size = 1024  # 1KB per artifact (estimate)
        memory_mb = (total_entries * avg_artifact_size) / (1024 * 1024)

        return {
            "cache_hit_rate": hit_rate,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "total_requests": total_requests,
            "invalidations": self._invalidations,
            "total_entries": total_entries,
            "stages": len(self._caches),
            "memory_mb": round(memory_mb, 2),
        }

    def clear_expired(self) -> int:
        """Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        count = 0
        for stage_cache in self._caches.values():
            to_remove = [
                key for key, cached in stage_cache.items() if cached.is_expired()
            ]
            for key in to_remove:
                del stage_cache[key]
                count += 1

        return count

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding for text using the embedding model."""
        result = await self._embedding.embed_query(text)
        # EmbeddingResult.embeddings is list[list[float]], get first vector
        return result.embeddings[0]

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


def compute_context_hash(
    system_prompt: str,
    tool_names: list[str],
    model_identifier: str,
) -> str:
    """Compute hash of execution context for cache invalidation.
    
    Args:
        system_prompt: Agent system prompt
        tool_names: List of available tool names
        model_identifier: Model name/identifier
    
    Returns:
        16-character hex hash
    
    Example:
        ```python
        hash1 = compute_context_hash(
            system_prompt="You are a helpful assistant",
            tool_names=["search", "calculator"],
            model_identifier="gpt-4o",
        )
        
        # Different tools = different hash
        hash2 = compute_context_hash(
            system_prompt="You are a helpful assistant",
            tool_names=["search"],  # Changed!
            model_identifier="gpt-4o",
        )
        assert hash1 != hash2  # Cache invalidated
        ```
    """
    context_str = (
        system_prompt + "|" + "|".join(sorted(tool_names)) + "|" + model_identifier
    )
    return hashlib.sha256(context_str.encode()).hexdigest()[:16]
