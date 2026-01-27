# Memory Module

The `cogent.memory` module provides a memory-first architecture where memory is a first-class citizen that can be wired to any entity (Agent, Team, Flow).

## Overview

Memory enables agents to:
- Persist knowledge across conversations
- Share state between agents
- Perform semantic search over memories
- Scope memories by user, team, or conversation

```python
from cogent import Agent
from cogent.memory import Memory

# Basic in-memory storage
memory = Memory()
await memory.remember("user_preference", "dark mode")
value = await memory.recall("user_preference")

# Wire to an agent
agent = Agent(name="assistant", model=model, memory=memory)
```

## Core Classes

### Memory

The main memory interface with simple remember/recall API:

```python
from cogent.memory import Memory

memory = Memory()

# Remember a value
await memory.remember("key", "value")
await memory.remember("user.name", "Alice")
await memory.remember("conversation.topic", "AI research")

# Recall a value
name = await memory.recall("user.name")  # "Alice"
missing = await memory.recall("unknown")  # None
missing = await memory.recall("unknown", default="N/A")  # "N/A"

# Check existence
exists = await memory.exists("user.name")  # True

# Delete a memory
await memory.forget("user.name")

# List all keys
keys = await memory.list_keys()  # ["conversation.topic"]

# Clear all memories
await memory.clear()
```

### Scoped Memory

Create isolated memory views for users, teams, or conversations:

```python
from cogent.memory import Memory

memory = Memory()

# Create scoped views
user_mem = memory.scoped("user:alice")
team_mem = memory.scoped("team:research")
conv_mem = memory.scoped("conv:thread-123")

# Each scope is isolated
await user_mem.remember("preference", "compact")
await team_mem.remember("preference", "detailed")

user_pref = await user_mem.recall("preference")  # "compact"
team_pref = await team_mem.recall("preference")  # "detailed"

# Scopes can be nested
project_mem = team_mem.scoped("project:alpha")
await project_mem.remember("status", "active")
```

### Shared Memory Between Agents

Wire the same memory to multiple agents for shared knowledge:

```python
from cogent import Agent
from cogent.memory import Memory

# Shared memory instance
shared = Memory()

# Both agents share the same memory
researcher = Agent(name="researcher", model=model, memory=shared)
writer = Agent(name="writer", model=model, memory=shared)

# Researcher stores findings
await shared.remember("findings", "Key insight: AI adoption is growing")

# Writer can access them
findings = await shared.recall("findings")
```

---

## Storage Backends

### InMemoryStore (Default)

Fast, no-persistence storage for development and testing:

```python
from cogent.memory import Memory, InMemoryStore

# Default - uses InMemoryStore
memory = Memory()

# Explicit
memory = Memory(store=InMemoryStore())
```

### SQLAlchemyStore

Persistent storage with SQLAlchemy 2.0 async support:

```python
from cogent.memory import Memory, SQLAlchemyStore

# SQLite (local file)
store = SQLAlchemyStore("sqlite+aiosqlite:///./memory.db")
memory = Memory(store=store)

# PostgreSQL
store = SQLAlchemyStore(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=10,
)
memory = Memory(store=store)

# Initialize tables (run once)
await store.initialize()

# Cleanup
await store.close()
```

**Context manager for cleanup:**

```python
async with SQLAlchemyStore("sqlite+aiosqlite:///./data.db") as store:
    memory = Memory(store=store)
    await memory.remember("key", "value")
```

### RedisStore

Distributed cache with native TTL support:

```python
from cogent.memory import Memory, RedisStore

store = RedisStore(
    url="redis://localhost:6379",
    prefix="myapp:",  # Key prefix
    default_ttl=3600,  # 1 hour default TTL
)
memory = Memory(store=store)

# With TTL per key
await memory.remember("session", {"user": "alice"}, ttl=1800)
```

---

## Memory Key Search

Memory provides intelligent key search with three methods that automatically cascade:

### 1. Fuzzy Matching (Default - Fast & Free)

The default search method uses fuzzy string matching for instant, offline key discovery:

```python
from cogent import Agent
from cogent.memory import Memory

# No special setup needed - fuzzy matching works out of the box
memory = Memory()

agent = Agent(name="assistant", model=model, memory=memory)

# Save memories
await agent.run("My name is Alice, I prefer dark mode, language is Python")

# Fuzzy matching finds similar keys instantly
await agent.run("What are my preferences?")
# → search_memories("preferences") finds "preferred_mode" and "preferred_language"
# Method: Fuzzy match (0.1ms, free, offline)
```

**Benefits:**
- ⚡ **2,800× faster** than semantic search (0.1ms vs 280ms)
- 💰 **Free** - no API calls
- 🔌 **Works offline** - no network required
- 📊 **62.5% accuracy** - good enough for most use cases
- 🧹 **Smart normalization** - handles underscores, hyphens, word order

**How it works:**
```python
# String normalization helps matching:
"preferred_mode" → "preferred mode"
"user_timezone" → "user timezone"
"notification-settings" → "notification settings"

# Fuzzy matching finds similarity:
Query: "preferences" → Matches: "preferred mode", "preferred language"
Query: "contact" → Matches: "email", "phone number"
Query: "settings" → Matches: "notification settings"
```

### 2. Semantic Search (Optional Fallback)

Enable semantic search by adding a vectorstore (used when fuzzy matching unavailable):

```python
from cogent import Agent
from cogent.memory import Memory
from cogent.vectorstore import VectorStore

# Add vectorstore for semantic fallback
memory = Memory(vectorstore=VectorStore())

agent = Agent(name="assistant", model=model, memory=memory)
```

**When semantic search is used:**
- Fuzzy matching library (rapidfuzz) not installed
- Fuzzy matching finds no matches (< 40% similarity)

**Trade-offs:**
- ✅ **75% accuracy** - better than fuzzy (but only 12.5% improvement)
- ❌ **280ms avg** - 2,800× slower than fuzzy
- ❌ **Costs money** - OpenAI API calls
- ❌ **Requires network** - API dependency

### 3. Keyword Search (Final Fallback)

Simple substring matching when all else fails:

```python
# Query: "mode" → Matches keys containing "mode": "preferred_mode", "dark_mode"
```

### Installation

**Recommended (fuzzy matching):**
```bash
uv add rapidfuzz  # For fast, free fuzzy matching
```

**Optional (semantic fallback):**
```python
from cogent.memory import Memory
from cogent.vectorstore import VectorStore

memory = Memory(vectorstore=VectorStore())  # Enables semantic fallback
```

### Performance Comparison

| Method | Speed | Accuracy | Cost | Offline |
|--------|-------|----------|------|---------|
| **Fuzzy** | 0.1ms | 62.5% | Free | ✅ Yes |
| **Semantic** | 280ms | 75.0% | $$ API | ❌ No |
| **Keyword** | 0.1ms | ~30% | Free | ✅ Yes |

**Recommendation:** Use fuzzy matching (default) for 99% of use cases.

### Example

See [examples/basics/memory_semantic_search.py](../examples/basics/memory_semantic_search.py) for a complete demo.

```python
from cogent import Agent
from cogent.memory import Memory

memory = Memory()  # Fuzzy matching by default

agent = Agent(name="assistant", model="gpt-4o", memory=memory)

# Save with specific key names
await memory.remember("preferred_mode", "dark")
await memory.remember("preferred_language", "Python")
await memory.remember("email", "alice@example.com")

# Agent finds them with fuzzy matching (instant!)
await agent.run("What are my preferences?")
# → search_memories("preferences") finds "preferred_mode" and "preferred_language"
# ⚡ 0.1ms, free, offline

await agent.run("How can I contact the user?")
# → search_memories("contact") finds "email"
```

---

## Memory Tools

Memory automatically exposes tools to agents for autonomous memory management:

```python
from cogent import Agent
from cogent.memory import Memory

# Memory is always agentic - tools auto-added
memory = Memory()

agent = Agent(
    name="assistant",
    model=model,
    memory=memory,
)

# Agent has 5 memory tools available:
# 1. remember(key, value) - Save facts to long-term memory
# 2. recall(key) - Retrieve specific facts
# 3. forget(key) - Remove facts
# 4. search_memories(query) - Search long-term facts (fuzzy matching by default)
# 5. search_conversation(query) - Search conversation history

# Agent can now use memory tools autonomously
result = await agent.run("Remember that my name is Alice")
result = await agent.run("What's my name?")
```

### Available Tools

**1. remember(key, value)** - Save important facts
```python
# Agent automatically calls when user shares information
await agent.run("My favorite language is Python")
# → Agent calls: remember("favorite_language", "Python")
```

**2. recall(key)** - Retrieve specific saved facts
```python
await agent.run("What's my favorite language?")
# → Agent calls: recall("favorite_language")
```

**3. forget(key)** - Remove facts (when user requests)
```python
await agent.run("Forget my favorite language")
# → Agent calls: forget("favorite_language")
```

**4. search_memories(query, k=5)** - Search long-term facts with intelligent matching
```python
# Default: Fast fuzzy matching (0.1ms, free, offline)
memory = Memory()
await agent.run("What are my preferences?")
# → Agent calls: search_memories("preferences")
# → Finds: "preferred_mode", "preferred_language" via fuzzy matching

# Optional: Add vectorstore for semantic fallback
memory = Memory(vectorstore=VectorStore())
# → Uses fuzzy matching first, falls back to semantic if needed
```

**5. search_conversation(query, max_results=5)** - Search conversation history
```python
# Critical for long conversations exceeding context window
await agent.run("What were the three projects I mentioned earlier?")
# → Agent calls: search_conversation("three projects")
```

### When Tools Are Used

The agent's system prompt instructs it to:

1. **At conversation start** → `search_memories("user")` to recall context
2. **When user shares info** → `remember(key, value)` immediately
3. **When asked about something** → Search before saying "I don't know"
   - For facts → `search_memories(query)` or `recall(key)`
   - For past conversation → `search_conversation(query)`
4. **In long conversations** → Use `search_conversation()` to find earlier context

**Shorthand** - `memory=True` creates a Memory instance:

```python
# Shorthand for Memory()
agent = Agent(name="assistant", model=model, memory=True)
```

---

## Usage Patterns

### Conversation History

```python
from cogent.memory import Memory

memory = Memory()

async def chat(user_id: str, message: str) -> str:
    user_mem = memory.scoped(f"user:{user_id}")
    
    # Load history
    history = await user_mem.recall("history", default=[])
    history.append({"role": "user", "content": message})
    
    # Get response (using agent)
    response = await agent.run(message, history=history)
    
    # Save updated history
    history.append({"role": "assistant", "content": response})
    await user_mem.remember("history", history)
    
    return response
```

### Team Knowledge Base

```python
from cogent.memory import Memory, SQLAlchemyStore
from cogent.vectorstore import VectorStore

# Persistent team memory with search
team_memory = Memory(
    store=SQLAlchemyStore("sqlite+aiosqlite:///./team.db"),
    vectorstore=VectorStore(),
)

# Store team knowledge
await team_memory.remember("policy:vacation", "Employees get 20 days PTO")
await team_memory.remember("policy:remote", "Remote work allowed 3 days/week")
await team_memory.remember("contact:hr", "hr@company.com")

# Search policies
results = await team_memory.search("time off work", k=3)
```

### Agent with Persistent Context

```python
from cogent import Agent
from cogent.memory import Memory, SQLAlchemyStore

store = SQLAlchemyStore("sqlite+aiosqlite:///./agent.db")
memory = Memory(store=store)

agent = Agent(
    name="assistant",
    model=model,
    memory=memory,
    instructions="""You have access to persistent memory.
    Use it to remember user preferences and context.""",
)

# First conversation
await agent.run("My favorite color is blue")

# Later conversation (same agent)
await agent.run("What's my favorite color?")  # Recalls "blue"
```

---

## Store Protocol

Implement custom storage backends:

```python
from typing import Protocol, Any

class Store(Protocol):
    """Protocol for memory storage backends."""
    
    async def get(self, key: str) -> Any | None:
        """Get a value by key."""
        ...
    
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value with optional TTL."""
        ...
    
    async def delete(self, key: str) -> bool:
        """Delete a key. Returns True if existed."""
        ...
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...
    
    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern."""
        ...
    
    async def clear(self) -> None:
        """Clear all keys."""
        ...
```

**Custom implementation example:**

```python
class DynamoDBStore:
    """Custom DynamoDB backend."""
    
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.client = boto3.resource("dynamodb")
        self.table = self.client.Table(table_name)
    
    async def get(self, key: str) -> Any | None:
        response = self.table.get_item(Key={"pk": key})
        item = response.get("Item")
        return item["value"] if item else None
    
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        item = {"pk": key, "value": value}
        if ttl:
            item["ttl"] = int(time.time()) + ttl
        self.table.put_item(Item=item)
    
    # ... implement other methods

# Use custom store
memory = Memory(store=DynamoDBStore("my-memories"))
```

---

## API Reference

### Memory

| Method | Description |
|--------|-------------|
| `remember(key, value, ttl?)` | Store a value |
| `recall(key, default?)` | Retrieve a value |
| `forget(key)` | Delete a value |
| `exists(key)` | Check if key exists |
| `list_keys(pattern?)` | List matching keys |
| `clear()` | Clear all memories |
| `scoped(prefix)` | Create scoped view |
| `search(query, k?)` | Semantic search (requires vectorstore) |

### Stores

| Store | Use Case |
|-------|----------|
| `InMemoryStore` | Development, testing, ephemeral |
| `SQLAlchemyStore` | Persistent, ACID, SQL databases |
| `RedisStore` | Distributed, TTL, high-throughput |

---

## Semantic Cache

SemanticCache provides embedding-based caching with configurable similarity thresholds. When a query is "close enough" to a cached entry, return the cached result instead of making an expensive LLM or API call.

**Key Benefits:**
- **80%+ hit rates** — Cache similar queries, not just exact matches
- **7-10× speedup** — Cached responses return instantly
- **Cost reduction** — Fewer API calls = lower costs
- **Automatic eviction** — LRU policy and TTL expiration

### Quick Start

Enable caching with `cache=True`:

```python
from cogent import Agent

agent = Agent(
    model="gpt-4o-mini",
    cache=True,  # Enable semantic cache with defaults
)

# First query
await agent.run("What are the best Python frameworks?")

# Similar query hits cache (instant!)
await agent.run("What are the top Python frameworks?")
```

### Custom Configuration

Pass a `SemanticCache` instance for custom settings:

```python
from cogent import Agent
from cogent.memory import SemanticCache

agent = Agent(
    model="gpt-4o-mini",
    cache=SemanticCache(
        similarity_threshold=0.90,  # Stricter matching (default: 0.85)
        max_size=5000,              # Larger cache (default: 1000)
        ttl=3600,                   # 1 hour TTL (default: None)
    ),
)
```

**Similarity Threshold:**

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| **0.95-1.0** | Very strict, near-exact | Deterministic outputs |
| **0.85-0.95** | Balanced, similar intent | General purpose (default) |
| **0.70-0.85** | Loose, broad matching | Exploratory queries |

### Tool-Level Caching

Use `@tool(cache=True)` to cache expensive tool calls:

```python
from cogent import Agent, tool

@tool(cache=True)
async def search_products(query: str) -> str:
    """Search products in the catalog."""
    return await product_api.search(query)

agent = Agent(
    model="gpt-4o-mini",
    tools=[search_products],
    cache=True,  # Required — tools use agent's cache
)

# First call executes the tool
await agent.run("Find running shoes")

# Similar query hits cache
await agent.run("Show me running sneakers")  # Cache hit!
```

See [tool-building.md](tool-building.md#semantic-caching) for more details.

### When to Use

| Use Semantic Cache When | Don't Use When |
|-------------------------|----------------|
| User queries with variation | Need exact-match guarantees |
| Similar questions rephrased | Outputs must be deterministic |
| Intent-based matching | Query structure matters |
| High query volume | Low query volume |
