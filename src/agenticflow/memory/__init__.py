"""Memory module - minimal wrappers around LangChain/LangGraph memory.

This module provides thin wrappers around:

1. **Checkpointers** (Short-term Memory)
   - Thread-scoped conversation persistence
   - Ephemeral (MemorySaver) or Persistent (SQLite/Postgres)
   - Stores graph state at each step

2. **Stores** (Long-term Memory)
   - Cross-thread persistent storage
   - Namespaced by user_id, assistant_id, etc.
   - Optional semantic search with embeddings

3. **Vector Stores** (Semantic Memory)
   - Embedding-based similarity search
   - Document storage and retrieval
   - RAG support

Memory Types:
-------------
- **Ephemeral**: Lost on restart (in-memory)
- **Persistent**: Survives restarts (SQLite, Postgres, etc.)
- **Short-term**: Thread-scoped (within a conversation)
- **Long-term**: Cross-thread (across conversations)
- **Semantic**: Embedding-based retrieval

Example:
--------
```python
from agenticflow.memory import (
    # Short-term (checkpointers)
    create_checkpointer,
    memory_checkpointer,

    # Long-term (stores)
    create_store,
    memory_store,
    semantic_store,

    # Semantic (vector stores)
    create_vectorstore,
    memory_vectorstore,
)

# Short-term: thread-scoped conversation
checkpointer = memory_checkpointer()  # ephemeral
# checkpointer = sqlite_checkpointer("chat.db")  # persistent

# Long-term: cross-thread user memory
store = memory_store()  # ephemeral
# store = semantic_store(embeddings, dims=1536)  # with search

# Compile graph with both
graph = builder.compile(checkpointer=checkpointer, store=store)

# Invoke with thread_id (short-term) and user_id (long-term)
graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    {"configurable": {"thread_id": "conv-1", "user_id": "user-123"}}
)
```
"""

# Checkpointers (short-term memory)
from agenticflow.memory.checkpointer import (
    create_checkpointer,
    memory_checkpointer,
    sqlite_checkpointer,
    postgres_checkpointer,
    MemorySaver,
    BaseCheckpointSaver,
)

# Stores (long-term memory)
from agenticflow.memory.store import (
    create_store,
    memory_store,
    semantic_store,
    BaseStore,
    InMemoryStore,
)

# Vector stores (semantic memory)
from agenticflow.memory.vectorstore import (
    create_vectorstore,
    memory_vectorstore,
    faiss_vectorstore,
    chroma_vectorstore,
    VectorStore,
    InMemoryVectorStore,
    Document,
)

__all__ = [
    # Checkpointers (short-term)
    "create_checkpointer",
    "memory_checkpointer",
    "sqlite_checkpointer",
    "postgres_checkpointer",
    "MemorySaver",
    "BaseCheckpointSaver",
    # Stores (long-term)
    "create_store",
    "memory_store",
    "semantic_store",
    "BaseStore",
    "InMemoryStore",
    # Vector stores (semantic)
    "create_vectorstore",
    "memory_vectorstore",
    "faiss_vectorstore",
    "chroma_vectorstore",
    "VectorStore",
    "InMemoryVectorStore",
    "Document",
]
