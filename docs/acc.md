# Agentive Context Control (ACC)

**Exact-match caching for reliable agent memory.**

## Overview

Agentive Context Control (ACC) implements bio-inspired bounded memory with exact-match caching. Unlike traditional conversation history that grows unbounded, ACC:

- **Prevents context drift** — Exact string matching ensures consistent retrieval¬
- **Bounds memory** — Automatic eviction when capacity is reached
- **Caches exactly** — No semantic similarity, no false positives
- **Fast lookups** — O(1) dictionary-based retrieval

## When to Use ACC

| Use ACC When | Don't Use ACC When |
|--------------|-------------------|
| Need deterministic memory retrieval | Need semantic similarity search |
| Caching exact tool outputs | Searching across documents |
| Preventing repeated API calls | Fuzzy matching required |
| Bounded context is critical | Need unbounded memory |

**Complementary to SemanticCache:** Use ACC for exact-match scenarios, SemanticCache for similarity-based caching.

## Basic Usage

```python
from cogent.memory import ACC

# Create ACC with capacity limit
acc = ACC(capacity=100)

# Store context
acc.add("What is the capital of France?", "Paris")
acc.add("2 + 2", "4")

# Exact match retrieval
result = acc.get("What is the capital of France?")
print(result)  # "Paris"

# No match (different string)
result = acc.get("what is the capital of france?")  # Case-sensitive
print(result)  # None
```

## Configuration

```python
from cogent.memory import ACC

acc = ACC(
    capacity=500,           # Maximum entries (default: 1000)
    eviction_policy="lru",  # "lru" or "fifo" (default: "lru")
)
```

### Eviction Policies

- **LRU (Least Recently Used)** — Evicts oldest accessed entries (default, recommended)
- **FIFO (First In, First Out)** — Evicts oldest added entries

## Agent Integration

ACC automatically integrates with agents for tool call caching:

```python
from cogent import Agent
from cogent.memory import ACC

# Enable ACC for agent
agent = Agent(
    name="Assistant",
    model="gpt-4o-mini",
    tools=[expensive_api_call],
    acc=ACC(capacity=100),  # Cache up to 100 tool calls
)

# First call - executes tool
result1 = await agent.run("Fetch data from API for user 123")

# Second call with exact same query - cached
result2 = await agent.run("Fetch data from API for user 123")  # Instant
```

## Caching Tool Outputs

ACC is particularly effective for caching expensive tool calls:

```python
from cogent import Agent, tool
from cogent.memory import ACC
import httpx

@tool
async def fetch_user(user_id: int) -> dict:
    """Fetch user from external API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/users/{user_id}")
        return response.json()

agent = Agent(
    name="UserAgent",
    model="gpt-4o-mini",
    tools=[fetch_user],
    acc=ACC(capacity=200),
)

# First call - hits API
await agent.run("Get user 42")

# Subsequent calls with same user_id - cached
await agent.run("Get user 42")  # Instant, no API call
```

## Pattern: Bounded Context Window

Use ACC to maintain a bounded conversation context:

```python
from cogent import Agent
from cogent.memory import ACC

# Keep only last 50 interactions
agent = Agent(
    name="ChatBot",
    model="gpt-4o-mini",
    acc=ACC(capacity=50, eviction_policy="lru"),
)

# As conversation grows, oldest entries are evicted
for i in range(100):
    await agent.run(f"Message {i}")

# Only last 50 messages retained
```

## Pattern: Deduplication

Prevent duplicate work with ACC:

```python
from cogent import Agent
from cogent.memory import ACC

agent = Agent(
    name="Processor",
    model="gpt-4o-mini",
    tools=[process_document],
    acc=ACC(capacity=1000),
)

documents = ["doc1.pdf", "doc2.pdf", "doc1.pdf"]  # doc1.pdf appears twice

for doc in documents:
    await agent.run(f"Process {doc}")
    # doc1.pdf processed once, second call cached
```

## Pattern: Session-Based Memory

Use scoped ACC for user sessions:

```python
from cogent import Agent
from cogent.memory import ACC

# Per-user ACC instances
user_accs = {}

def get_agent_for_user(user_id: str) -> Agent:
    if user_id not in user_accs:
        user_accs[user_id] = ACC(capacity=100)
    
    return Agent(
        name=f"Agent-{user_id}",
        model="gpt-4o-mini",
        acc=user_accs[user_id],
    )

# Each user gets independent cache
agent_alice = get_agent_for_user("alice")
agent_bob = get_agent_for_user("bob")
```

## Programmatic Access

Work with ACC directly for custom caching logic:

```python
from cogent.memory import ACC

acc = ACC(capacity=100)

# Check if key exists
if acc.has("query"):
    result = acc.get("query")
else:
    result = expensive_operation()
    acc.add("query", result)

# Get all keys
keys = acc.keys()

# Clear all entries
acc.clear()

# Get current size
size = acc.size()
```

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| `add(key, value)` | O(1) | Dictionary insert |
| `get(key)` | O(1) | Dictionary lookup |
| `has(key)` | O(1) | Dictionary check |
| `keys()` | O(n) | Returns all keys |
| `clear()` | O(1) | Reinitialize dict |

**Memory Usage:** ~56 bytes per entry (Python dict overhead) + key/value size

## Monitoring

Track ACC usage with built-in metrics:

```python
from cogent.memory import ACC

acc = ACC(capacity=100)

# Add some entries
for i in range(50):
    acc.add(f"key{i}", f"value{i}")

# Check metrics
print(f"Size: {acc.size()}")          # 50
print(f"Capacity: {acc.capacity}")     # 100
print(f"Utilization: {acc.size() / acc.capacity * 100:.1f}%")  # 50.0%
```

## Comparison: ACC vs SemanticCache

| Feature | ACC | SemanticCache |
|---------|-----|---------------|
| **Matching** | Exact string match | Semantic similarity |
| **Use Case** | Deterministic caching | Fuzzy matching |
| **Overhead** | O(1) dictionary | Embedding computation |
| **False Positives** | None | Possible |
| **Memory** | Low (dict only) | Higher (embeddings) |
| **Speed** | Fastest | Fast (cached embeddings) |

**Recommendation:** Use ACC for exact-match scenarios (API calls, tool outputs), SemanticCache for semantic similarity (user queries, intent matching).

## Best Practices

1. **Set appropriate capacity** — Based on use case (100-1000 typical)
2. **Use LRU eviction** — Better for most scenarios than FIFO
3. **Scope per user/session** — Prevent cache pollution across users
4. **Monitor utilization** — Track hit rates and capacity usage
5. **Clear periodically** — Reset cache for long-running agents
6. **Combine with SemanticCache** — Use both for comprehensive caching

## Examples

See working examples:
- [examples/advanced/semantic_caching.py](../examples/advanced/semantic_caching.py) — ACC + SemanticCache together
- [tests/test_acc.py](../tests/test_acc.py) — Comprehensive ACC tests

## API Reference

### ACC Class

```python
class ACC:
    def __init__(
        self,
        capacity: int = 1000,
        eviction_policy: Literal["lru", "fifo"] = "lru",
    ):
        """Initialize ACC with capacity and eviction policy."""
    
    def add(self, key: str, value: str) -> None:
        """Add entry to cache. Evicts oldest if at capacity."""
    
    def get(self, key: str) -> str | None:
        """Retrieve entry by exact key match. Returns None if not found."""
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
    
    def keys(self) -> list[str]:
        """Get all keys in cache."""
    
    def clear(self) -> None:
        """Remove all entries from cache."""
    
    def size(self) -> int:
        """Get current number of entries."""
```

## Further Reading

- [Memory System](memory.md) — Overview of all memory components
- [Semantic Cache](memory.md#semantic-cache) — Similarity-based caching
- [Agent Configuration](agent.md) — Configuring agents with ACC
