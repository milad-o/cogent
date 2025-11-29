# Memory-First Architecture

## Status: 🔴 Cleanup Required

The current `agenticflow.memory` module is a mess:
- Over-engineered: 12+ files, multiple abstract classes
- Duplicates: `indexes.py` rebuilds what `vectorstore` already does
- Fragmented: ConversationMemory, UserMemory, TeamMemory, WorkingMemory all similar

## Goal

**One `Memory` class. Reuse `VectorStore`. Namespace for isolation.**

```python
# Before (messy)
from agenticflow.memory import ConversationMemory, UserMemory, TeamMemory, WorkingMemory, MemoryManager, MemoryConfig, BaseMemory, MemoryBackend, ConversationBackend, MemoryScope...

# After (clean)
from agenticflow.memory import Memory
memory = Memory()
agent = Agent(memory=memory.scoped("agent:alice"))
```

---

## Current Files (to refactor)

```
memory/
├── __init__.py          # Exports 15+ things → simplify
├── base.py              # BaseMemory, MemoryBackend, MemoryScope → remove
├── manager.py           # MemoryManager (400+ lines) → remove
├── conversation.py      # ConversationMemory → namespace
├── user.py              # UserMemory → namespace
├── team.py              # TeamMemory → namespace
├── working.py           # WorkingMemory → namespace
├── tools.py             # Memory tools for agents → keep, simplify
├── core.py              # NEW Memory class → this is the core
├── stores.py            # Store impls → keep
├── indexes.py           # DELETED - use vectorstore
└── backends/
    ├── inmemory.py      # → merge into stores.py
    └── sqlite.py        # → merge into stores.py
```

---

## New Architecture

```
memory/
├── __init__.py          # Export: Memory, Store, InMemoryStore, SQLiteStore
├── memory.py            # Memory class (unified)
├── stores.py            # Store protocol + implementations
└── tools.py             # Agent memory tools
```

### Memory Class (unified)

```python
class Memory:
    def __init__(
        self,
        store: Store | None = None,
        vectorstore: VectorStore | None = None,  # Reuse existing!
        namespace: str = "",
    ): ...
    
    # Key-value
    async def remember(self, key: str, value: Any): ...
    async def recall(self, key: str) -> Any: ...
    async def forget(self, key: str): ...
    
    # Semantic (if vectorstore provided)
    async def add_document(self, text: str, metadata: dict = {}): ...
    async def search(self, query: str, k: int = 5): ...
    
    # Messages (conversation)
    async def add_message(self, message: Message): ...
    async def get_messages(self, limit: int = 50) -> list[Message]: ...
    
    # Scoping
    def scoped(self, namespace: str) -> Memory: ...
```

### Usage Patterns

```python
# Basic
memory = Memory()  # InMemoryStore, no vectorstore

# With persistence
memory = Memory(store=SQLiteStore("./data.db"))

# With semantic search (reuse VectorStore!)
from agenticflow.vectorstore import VectorStore
memory = Memory(
    store=SQLiteStore("./data.db"),
    vectorstore=VectorStore(backend="chroma"),
)

# Scoping (replaces UserMemory, TeamMemory, etc.)
user_mem = memory.scoped("user:alice")
team_mem = memory.scoped("team:research")
conv_mem = memory.scoped("conv:thread-123")
```

---

## Migration Path

### Phase 1: Create new unified Memory
- [x] core.py with Memory, Store, InMemoryStore
- [x] stores.py with SQLiteStore, PostgresStore, RedisStore
- [ ] Refactor to use VectorStore instead of Index

### Phase 2: Bridge old → new
- [ ] MemoryManager delegates to Memory internally
- [ ] TeamMemory/UserMemory/etc become thin wrappers
- [ ] Tests pass with no external changes

### Phase 3: Clean up
- [ ] Remove old abstractions (BaseMemory, MemoryBackend, etc.)
- [ ] Update agent/base.py to use Memory directly
- [ ] Simplify __init__.py exports

---

## Key Decisions

1. **Reuse VectorStore** - Don't rebuild semantic search
2. **Namespace > Separate Classes** - One Memory, different views
3. **Messages in Memory** - Conversation is just namespaced memory
4. **Backward Compat** - Old imports work via thin wrappers
