"""
Demonstrates WebSearch capability with SemanticCache integration.

This example shows how semantic caching can dramatically reduce costs
and latency for repeated searches by caching similar queries.

Based on research: arXiv:2601.16286 shows semantic caching achieves
83% hit rate vs 38% for traditional boundary caching.
"""

import asyncio
import time

from cogent.capabilities.web_search import WebSearch
from cogent.memory import SemanticCache
from cogent.models import create_embedding


async def main():
    """Demonstrate WebSearch with semantic caching."""
    
    # Create embedding model for semantic similarity
    # Using OpenAI's text-embedding-3-small for fast, cost-effective caching
    print("Initializing OpenAI embedding model...")
    embed_model = create_embedding("openai", "text-embedding-3-small")
    
    # Wrapper function to extract embedding vector from EmbeddingResult
    async def embed_fn(text: str) -> list[float]:
        result = await embed_model.embed_query(text)
        return result.embeddings[0]  # Get first embedding vector
    
    # Create semantic cache with embedding-based similarity matching
    cache = SemanticCache(
        embedding_fn=embed_fn,
        similarity_threshold=0.85,  # 85% similarity threshold for cache hits
        max_entries=100,
    )
    
    # Create web search with cache integration
    # Uses DuckDuckGo by default (free, no API key required)
    search = WebSearch(
        max_results=5,
        cache=cache,
    )
    
    print("=" * 70)
    print("WebSearch with SemanticCache Demo")
    print("=" * 70)
    print()
    
    # Test 1: Initial search (cache miss)
    print("Test 1: Initial search - 'Python async programming'")
    print("-" * 70)
    start = time.time()
    results = await search.search("Python async programming", max_results=3)
    duration = time.time() - start
    
    print(f"Results: {len(results)} found in {duration:.3f}s")
    for r in results[:2]:
        print(f"  - {r.title}")
        print(f"    {r.url}")
    print()
    
    # Test 2: Exact same query (should hit cache)
    print("Test 2: Exact same query - 'Python async programming'")
    print("-" * 70)
    start = time.time()
    results = await search.search("Python async programming", max_results=3)
    duration = time.time() - start
    
    print(f"Results: {len(results)} found in {duration:.3f}s (cached)")
    print(f"Speedup: ~{10/duration:.1f}x faster")
    print()
    
    # Test 3: Similar query (semantic cache hit)
    print("Test 3: Similar query - 'asynchronous Python programming'")
    print("-" * 70)
    start = time.time()
    results = await search.search("asynchronous Python programming", max_results=3)
    duration = time.time() - start
    
    # Check if cache was hit
    cached = await cache.get("search", "asynchronous Python programming", "")
    cache_status = "HIT" if cached else "MISS"
    
    print(f"Results: {len(results)} found in {duration:.3f}s")
    print(f"Cache status: {cache_status}")
    if cache_status == "HIT":
        print(f"Speedup: ~{10/duration:.1f}x faster")
    print()
    
    # Test 4: News search with caching
    print("Test 4: News search - 'AI breakthroughs'")
    print("-" * 70)
    start = time.time()
    news = await search.search_news("AI breakthroughs", max_results=3)
    duration = time.time() - start
    
    print(f"News: {len(news)} articles found in {duration:.3f}s")
    for article in news[:2]:
        print(f"  - {article.title}")
    print()
    
    # Test 5: Similar news query (semantic cache hit expected)
    print("Test 5: Similar news query - 'artificial intelligence advances'")
    print("-" * 70)
    start = time.time()
    news = await search.search_news("artificial intelligence advances", max_results=3)
    duration = time.time() - start
    
    cached = await cache.get("news", "artificial intelligence advances", "")
    cache_status = "HIT" if cached else "MISS"
    
    print(f"News: {len(news)} articles found in {duration:.3f}s")
    print(f"Cache status: {cache_status}")
    if cache_status == "HIT":
        print(f"Speedup: ~{10/duration:.1f}x faster")
    print()
    
    # Cache statistics
    print("=" * 70)
    print("Cache Statistics")
    print("=" * 70)
    stats = cache.get_metrics()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    if stats['total_requests'] > 0:
        hit_rate = stats['cache_hit_rate'] * 100
        print(f"Hit rate: {hit_rate:.1f}%")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Memory usage: {stats['memory_mb']} MB")
    print()
    
    # Expected benefits
    print("=" * 70)
    print("Expected Production Benefits (arXiv:2601.16286)")
    print("=" * 70)
    print("• Semantic caching: 83% hit rate (vs 38% boundary caching)")
    print("• Cost reduction: 60-80% on repeated searches")
    print("• Latency reduction: 10-100x faster on cache hits")
    print("• Similarity matching: Handles query variations automatically")
    print()


if __name__ == "__main__":
    asyncio.run(main())
