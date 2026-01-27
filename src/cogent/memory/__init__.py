"""Memory-first architecture for AgenticFlow.

Core principle: Memory is a first-class citizen.
Wire it to any entity (Agent, Team, Flow) and it changes behavior.

Example:
    ```python
    from cogent.memory import Memory

    # Basic - in-memory, no persistence
    memory = Memory()
    await memory.remember("key", "value")
    value = await memory.recall("key")

    # With persistence (SQLAlchemy 2.0 async)
    from cogent.memory import SQLAlchemyStore
    memory = Memory(store=SQLAlchemyStore("sqlite+aiosqlite:///./data.db"))

    # With semantic search (reuses existing VectorStore)
    from cogent.vectorstore import VectorStore
    memory = Memory(
        store=SQLAlchemyStore("sqlite+aiosqlite:///./data.db"),
        vectorstore=VectorStore(),
    )

    # Scoped views (replaces TeamMemory, UserMemory, etc.)
    user_mem = memory.scoped("user:alice")
    team_mem = memory.scoped("team:research")
    conv_mem = memory.scoped("conv:thread-123")

    # Share between entities
    agent_a = Agent(memory=memory)
    agent_b = Agent(memory=memory)  # Same memory = shared knowledge
    ```

Available stores:
    - InMemoryStore: Fast, no persistence (default)
    - SQLAlchemyStore: SQLAlchemy 2.0 async (SQLite, PostgreSQL, MySQL)
    - RedisStore: Distributed cache with native TTL
"""

from cogent.memory.core import (
    InMemoryStore,
    Memory,
    Store,
)
from cogent.memory.acc import (
    AgentCognitiveCompressor,
    BoundedMemoryState,
    MemoryItem,
    SemanticForgetGate,
)
from cogent.memory.cache import (
    CachedArtifact,
    SemanticCache,
    compute_context_hash,
)
from cogent.memory.stores import (
    RedisStore,
    SQLAlchemyStore,
)

__all__ = [
    # Core
    "Memory",
    # Protocol
    "Store",
    # Stores
    "InMemoryStore",
    "SQLAlchemyStore",
    "RedisStore",
    # Agent Cognitive Compressor (ACC) - Bounded memory control
    "AgentCognitiveCompressor",
    "BoundedMemoryState",
    "MemoryItem",
    "SemanticForgetGate",
    # Semantic caching - Pipeline-aware reasoning cache
    "SemanticCache",
    "CachedArtifact",
    "compute_context_hash",
]
