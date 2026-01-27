# Semantic Caching

**High-performance similarity-based caching for LLM applications.**

## Overview

SemanticCache provides embedding-based caching with configurable similarity thresholds. When a query is "close enough" to a cached entry, return the cached result instead of making an expensive LLM or API call.

**Key Benefits:**
- **80%+ hit rates** — Cache similar queries, not just exact matches
- **7-10× speedup** — Cached responses return instantly
- **Cost reduction** — Fewer API calls = lower costs
- **Automatic eviction** — LRU policy and TTL expiration

## When to Use Semantic Caching

| Use Semantic Cache When | Don't Use When |
|-------------------------|----------------|
| User queries with variation | Need exact-match guarantees |
| Similar questions rephrased | Outputs must be deterministic |
| Intent-based matching | Query structure matters |
| High query volume | Low query volume |

## Basic Usage

```python
from cogent.memory import SemanticCache
from cogent.models import OpenAIEmbedding

# Create cache with embedding model
cache = SemanticCache(
    embedding_model=OpenAIEmbedding(model="text-embedding-3-small"),
    similarity_threshold=0.85,  # 85% similar = cache hit
    max_size=1000,              # Keep 1000 most recent entries
)

# Cache a result
cache.set("What is the capital of France?", "The capital of France is Paris.")

# Similar query hits cache
result = cache.get("What's the capital city of France?")
print(result)  # "The capital of France is Paris." (cache hit!)

# Dissimilar query misses cache
result = cache.get("What is the capital of Germany?")
print(result)  # None (cache miss)
```

## Configuration

```python
from cogent.memory import SemanticCache
from cogent.models import OpenAIEmbedding

cache = SemanticCache(
    embedding_model=OpenAIEmbedding(model="text-embedding-3-large"),
    similarity_threshold=0.90,  # Stricter matching (default: 0.85)
    max_size=5000,              # Larger cache (default: 1000)
    ttl=3600,                   # 1 hour TTL in seconds (default: None)
    eviction_policy="lru",      # LRU or FIFO (default: "lru")
)
```

### Similarity Threshold

Controls how similar a query must be to trigger a cache hit:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| **0.95-1.0** | Very strict, near-exact matches | Deterministic outputs required |
| **0.85-0.95** | Balanced, similar intent | General purpose (recommended) |
| **0.70-0.85** | Loose, broad matching | Exploratory queries |

**Recommendation:** Start with 0.85, adjust based on hit rate and quality.

## WebSearch Integration

Semantic caching dramatically improves WebSearch performance:

```python
from cogent.capabilities import WebSearch
from cogent.memory import SemanticCache
from cogent.models import OpenAIEmbedding

# WebSearch with semantic caching
search = WebSearch(
    cache=SemanticCache(
        embedding_model=OpenAIEmbedding(model="text-embedding-3-small"),
        similarity_threshold=0.85,
    )
)

# First search - hits DuckDuckGo API
results1 = await search.search("latest AI research")

# Similar query - cache hit (instant!)
results2 = await search.search("newest AI research papers")  # ~7× faster
```

### Performance Results

From [web_search_cached.py](../examples/capabilities/web_search_cached.py):

```
Search 1: "latest AI developments"     — 0.89s (cache miss)
Search 2: "recent AI advancements"     — 0.12s (cache hit, 7.4× faster)
Search 3: "newest AI breakthroughs"    — 0.13s (cache hit, 6.8× faster)
Search 4: "current AI innovations"     — 0.14s (cache hit, 6.4× faster)
Search 5: "quantum computing trends"   — 0.91s (cache miss, different topic)

Hit Rate: 60% (3/5)
Average Speedup: 7× on cache hits
```

## Agent Integration

Use semantic caching with agents to cache similar user queries:

```python
from cogent import Agent
from cogent.memory import SemanticCache
from cogent.models import OpenAIEmbedding

agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    tools=[search, fetch_data],
    semantic_cache=SemanticCache(
        embedding_model=OpenAIEmbedding(),
        similarity_threshold=0.85,
        max_size=500,
    ),
)

# First query
await agent.run("What are the best Python frameworks?")

# Similar query hits cache
await agent.run("What are the top Python frameworks?")  # Instant
```

## Pattern: Multi-Level Caching

Combine ACC (exact-match) and SemanticCache (similarity) for comprehensive caching:

```python
from cogent import Agent
from cogent.memory import ACC, SemanticCache
from cogent.models import OpenAIEmbedding

agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    tools=[expensive_tool],
    acc=ACC(capacity=100),                    # Level 1: Exact matches
    semantic_cache=SemanticCache(             # Level 2: Similar queries
        embedding_model=OpenAIEmbedding(),
        similarity_threshold=0.85,
    ),
)

# Query 1: "Get user 123" → Tool call, cache in both ACC and SemanticCache
# Query 2: "Get user 123" → ACC hit (exact match, fastest)
# Query 3: "Fetch user 123" → SemanticCache hit (similar, fast)
# Query 4: "Get user 456" → Cache miss (different user), new tool call
```

## Pattern: Intent Caching

Cache based on user intent rather than exact phrasing:

```python
from cogent.memory import SemanticCache
from cogent.models import OpenAIEmbedding

intent_cache = SemanticCache(
    embedding_model=OpenAIEmbedding(),
    similarity_threshold=0.80,  # Looser for intent matching
)

# Cache intent → response
intent_cache.set(
    "I want to know the weather",
    "To check the weather, use the get_weather tool with a city name."
)

# All similar intents hit cache
intents = [
    "How do I check the weather?",
    "Tell me the weather",
    "What's the weather like?",
    "Can you show me the weather?",
]

for intent in intents:
    response = intent_cache.get(intent)
    print(f"{intent} → {response}")  # All hit cache!
```

## Pattern: Time-Based Invalidation

Use TTL to automatically expire stale cache entries:

```python
from cogent.memory import SemanticCache
from cogent.models import OpenAIEmbedding

# Cache with 1 hour TTL
news_cache = SemanticCache(
    embedding_model=OpenAIEmbedding(),
    similarity_threshold=0.85,
    ttl=3600,  # 1 hour in seconds
)

# Cache news results
news_cache.set("latest tech news", fetch_news_results())

# 30 minutes later - still cached
result = news_cache.get("recent tech news")  # Cache hit

# 2 hours later - expired
result = news_cache.get("recent tech news")  # Cache miss (TTL expired)
```

## Embedding Model Selection

Choose embedding models based on use case:

| Model | Dimensions | Cost | Use Case |
|-------|------------|------|----------|
| `text-embedding-3-small` | 1536 | Lowest | High-volume, cost-sensitive |
| `text-embedding-3-large` | 3072 | Higher | Best quality, lower volume |
| `nomic-embed-text` (Ollama) | 768 | Free | Local, no API costs |

**Recommendation:** Use `text-embedding-3-small` for most cases — good quality at lowest cost.

## Monitoring Cache Performance

Track cache effectiveness:

```python
from cogent.memory import SemanticCache
from cogent.models import OpenAIEmbedding

cache = SemanticCache(
    embedding_model=OpenAIEmbedding(),
    similarity_threshold=0.85,
)

# Track hits and misses
hits = 0
misses = 0

for query in user_queries:
    result = cache.get(query)
    if result:
        hits += 1
    else:
        misses += 1
        result = expensive_operation(query)
        cache.set(query, result)

# Calculate metrics
total = hits + misses
hit_rate = hits / total * 100
print(f"Hit Rate: {hit_rate:.1f}% ({hits}/{total})")
print(f"Cache Size: {cache.size()}/{cache.max_size}")
```

## Eviction Policies

Control how entries are removed when cache is full:

### LRU (Least Recently Used) — Default

Evicts the entry that was least recently accessed:

```python
cache = SemanticCache(
    embedding_model=model,
    max_size=100,
    eviction_policy="lru",  # Default
)

# Entry accessed recently stays, old entries evicted
```

**Use when:** Access patterns vary over time (most scenarios)

### FIFO (First In, First Out)

Evicts the oldest added entry:

```python
cache = SemanticCache(
    embedding_model=model,
    max_size=100,
    eviction_policy="fifo",
)
```

**Use when:** Want time-based eviction regardless of access patterns

## Programmatic Access

Work with cache directly:

```python
from cogent.memory import SemanticCache
from cogent.models import OpenAIEmbedding

cache = SemanticCache(
    embedding_model=OpenAIEmbedding(),
    similarity_threshold=0.85,
)

# Set entry
cache.set("query", "response")

# Get entry with similarity score
result, score = cache.get("similar query", return_score=True)
print(f"Match: {result} (similarity: {score:.2f})")

# Check if entry exists (with threshold)
exists = cache.has("query")

# Clear all entries
cache.clear()

# Get current size
size = cache.size()

# Get all keys
keys = cache.keys()
```

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| `set(key, value)` | O(n) for embedding + O(1) for storage | Embedding is async cached |
| `get(key)` | O(n) for embedding + O(m) for comparison | m = cache size |
| `clear()` | O(1) | Reinitialize storage |

**Optimization:** Embeddings are cached per unique query, so repeated gets are O(m) only.

## Best Practices

1. **Start with 0.85 threshold** — Adjust based on hit rate and quality
2. **Use text-embedding-3-small** — Good quality, lowest cost
3. **Set appropriate max_size** — 500-5000 based on memory and query diversity
4. **Enable TTL for time-sensitive data** — News, prices, weather
5. **Monitor hit rates** — Track cache effectiveness
6. **Combine with ACC** — Use both for comprehensive caching
7. **Test threshold values** — Measure impact on quality and hit rate

## Common Pitfalls

| Issue | Problem | Solution |
|-------|---------|----------|
| **Low hit rate** | Threshold too high | Lower to 0.80-0.85 |
| **Poor quality hits** | Threshold too low | Raise to 0.90-0.95 |
| **High latency** | Embedding on every get | Embeddings are cached |
| **Memory growth** | No max_size set | Set max_size or enable TTL |

## Examples

See working examples:
- [examples/capabilities/web_search_cached.py](../examples/capabilities/web_search_cached.py) — WebSearch with caching
- [examples/advanced/semantic_caching.py](../examples/advanced/semantic_caching.py) — ACC + SemanticCache
- [tests/test_caching.py](../tests/test_caching.py) — Comprehensive tests

## API Reference

### SemanticCache Class

```python
class SemanticCache:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        similarity_threshold: float = 0.85,
        max_size: int = 1000,
        ttl: int | None = None,
        eviction_policy: Literal["lru", "fifo"] = "lru",
    ):
        """Initialize semantic cache with embedding model and config."""
    
    def set(self, key: str, value: str) -> None:
        """Cache a result for the given key."""
    
    def get(
        self,
        key: str,
        return_score: bool = False
    ) -> str | None | tuple[str, float]:
        """
        Retrieve cached result for similar key.
        
        Returns cached value if similarity >= threshold.
        If return_score=True, returns (value, similarity_score).
        """
    
    def has(self, key: str) -> bool:
        """Check if similar key exists in cache."""
    
    def clear(self) -> None:
        """Remove all entries from cache."""
    
    def size(self) -> int:
        """Get current number of cached entries."""
    
    def keys(self) -> list[str]:
        """Get all cached keys."""
```

## Further Reading

- [ACC Memory Control](acc.md) — Exact-match caching
- [Memory System](memory.md) — Overview of all memory components
- [WebSearch Capability](capabilities.md) — Using WebSearch with caching
