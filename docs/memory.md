# Memory Module

The `agenticflow.memory` module provides a memory-first architecture where memory is a first-class citizen that can be wired to any entity (Agent, Team, Flow).

## Overview

Memory enables agents to:
- Persist knowledge across conversations
- Share state between agents
- Perform semantic search over memories
- Scope memories by user, team, or conversation

```python
from agenticflow import Agent
from agenticflow.memory import Memory

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
from agenticflow.memory import Memory

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
from agenticflow.memory import Memory

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
from agenticflow import Agent
from agenticflow.memory import Memory

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
from agenticflow.memory import Memory, InMemoryStore

# Default - uses InMemoryStore
memory = Memory()

# Explicit
memory = Memory(store=InMemoryStore())
```

### SQLAlchemyStore

Persistent storage with SQLAlchemy 2.0 async support:

```python
from agenticflow.memory import Memory, SQLAlchemyStore

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
from agenticflow.memory import Memory, RedisStore

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

## Semantic Search

Add a vector store to enable semantic search over memories:

```python
from agenticflow.memory import Memory, SQLAlchemyStore
from agenticflow.vectorstore import VectorStore

# Memory with semantic search
memory = Memory(
    store=SQLAlchemyStore("sqlite+aiosqlite:///./data.db"),
    vectorstore=VectorStore(),
)

# Store with text content (auto-embedded)
await memory.remember("doc1", "Python is a programming language")
await memory.remember("doc2", "Machine learning uses algorithms")
await memory.remember("doc3", "JavaScript runs in browsers")

# Semantic search
results = await memory.search("AI and coding", k=2)
for result in results:
    print(f"{result.key}: {result.value}")
```

---

## Memory Tools

Memory can automatically expose tools to agents when `agentic=True`:

```python
from agenticflow import Agent
from agenticflow.memory import Memory

# Agentic memory - tools auto-added to agent
memory = Memory(agentic=True)

agent = Agent(
    name="assistant",
    model=model,
    memory=memory,  # Tools: remember, recall, forget, search_memories
)

# Agent can now use memory tools
result = await agent.run("Remember that my name is Alice")
result = await agent.run("What's my name?")
```

**Non-agentic memory** (default) - for programmatic use only:

```python
# Non-agentic - no tools exposed, for code use
memory = Memory()  # agentic=False by default
agent = Agent(memory=memory)

# Memory available for your code, not the agent
await memory.remember("user_id", "12345")
```

**Shorthand** - `memory=True` creates agentic memory:

```python
# Shorthand for Memory(agentic=True)
agent = Agent(name="assistant", model=model, memory=True)
```

---

## Usage Patterns

### Conversation History

```python
from agenticflow.memory import Memory

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
from agenticflow.memory import Memory, SQLAlchemyStore
from agenticflow.vectorstore import VectorStore

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
from agenticflow import Agent
from agenticflow.memory import Memory, SQLAlchemyStore

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
