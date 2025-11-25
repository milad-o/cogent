"""Store wrappers for long-term memory.

Thin wrappers around LangGraph stores that provide
cross-thread persistent memory with optional semantic search.

Long-term memory = Stores:
- Cross-thread (persists across conversations)
- Namespaced by user_id, assistant_id, etc.
- Supports semantic search with embeddings
- User preferences, learned facts, accumulated knowledge
"""

from __future__ import annotations

from typing import Any

from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore


def create_store(
    backend: str = "memory",
    *,
    index: dict[str, Any] | None = None,
    **kwargs: Any,
) -> BaseStore:
    """Create a store for long-term memory.

    Stores provide cross-thread persistent memory, namespaced
    by user_id or other identifiers.

    Args:
        backend: Storage backend type:
            - "memory": In-memory with optional semantic search
        index: Optional index config for semantic search:
            - embed: Embeddings instance or model name
            - dims: Embedding dimensions
            - fields: Fields to index (default: ["$"] for all)
        **kwargs: Backend-specific configuration

    Returns:
        A configured store instance.

    Examples:
        # Basic in-memory store
        >>> store = create_store("memory")

        # With semantic search
        >>> from langchain.embeddings import init_embeddings
        >>> embeddings = init_embeddings("openai:text-embedding-3-small")
        >>> store = create_store(
        ...     "memory",
        ...     index={"embed": embeddings, "dims": 1536}
        ... )

    Usage with LangGraph:
        >>> store = create_store("memory")
        >>> graph = builder.compile(checkpointer=checkpointer, store=store)

        # In a node, access via config:
        >>> def my_node(state, config, *, store: BaseStore):
        ...     # Put data (namespaced by user)
        ...     user_id = config["configurable"]["user_id"]
        ...     store.put((user_id, "preferences"), "theme", {"value": "dark"})
        ...
        ...     # Get data
        ...     items = store.search((user_id, "preferences"))
        ...     return state
    """
    if backend == "memory":
        return InMemoryStore(index=index, **kwargs)

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Supported: 'memory'"
        )


def memory_store(
    *,
    index: dict[str, Any] | None = None,
    **kwargs: Any,
) -> InMemoryStore:
    """Create an in-memory store with optional semantic search.

    Args:
        index: Optional index config for semantic search
        **kwargs: Additional configuration

    Returns:
        InMemoryStore instance.

    Example:
        # Basic store
        >>> store = memory_store()

        # With semantic search
        >>> from langchain.embeddings import init_embeddings
        >>> embeddings = init_embeddings("openai:text-embedding-3-small")
        >>> store = memory_store(index={"embed": embeddings, "dims": 1536})
    """
    return InMemoryStore(index=index, **kwargs)


def semantic_store(
    embeddings: Any,
    dims: int,
    fields: list[str] | None = None,
    **kwargs: Any,
) -> InMemoryStore:
    """Create a store with semantic search enabled.

    Args:
        embeddings: Embeddings instance or model string
        dims: Embedding dimensions
        fields: Fields to index (default: ["$"] for root)
        **kwargs: Additional configuration

    Returns:
        InMemoryStore with semantic search.

    Example:
        >>> from langchain.embeddings import init_embeddings
        >>> embeddings = init_embeddings("openai:text-embedding-3-small")
        >>> store = semantic_store(embeddings, dims=1536)
        >>>
        >>> # Store some memories
        >>> store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
        >>> store.put(("user_123", "memories"), "2", {"text": "I work as engineer"})
        >>>
        >>> # Semantic search
        >>> results = store.search(
        ...     ("user_123", "memories"),
        ...     query="What food do I like?",
        ...     limit=1
        ... )
    """
    index = {
        "embed": embeddings,
        "dims": dims,
        "fields": fields or ["$"],
    }
    return InMemoryStore(index=index, **kwargs)


# Re-export base types
__all__ = [
    "create_store",
    "memory_store",
    "semantic_store",
    "BaseStore",
    "InMemoryStore",
]
