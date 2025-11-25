"""Checkpointer wrappers for short-term memory.

Thin wrappers around LangGraph checkpointers that provide
thread-scoped conversation persistence (ephemeral or persistent).

Short-term memory = Checkpointers:
- Thread-scoped (each thread_id has its own conversation)
- Ephemeral (MemorySaver) or Persistent (SQLite/Postgres)
- Stores graph state at each step
- Enables conversation history within a thread
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver


def create_checkpointer(
    backend: str = "memory",
    **kwargs: Any,
) -> BaseCheckpointSaver:
    """Create a checkpointer for short-term memory.

    Checkpointers provide thread-scoped conversation persistence.
    Each thread_id maintains its own conversation state.

    Args:
        backend: Storage backend type:
            - "memory": In-memory (ephemeral, lost on restart)
            - "sqlite": SQLite file (persistent)
            - "postgres": PostgreSQL (persistent, production-ready)
        **kwargs: Backend-specific configuration

    Returns:
        A configured checkpointer instance.

    Examples:
        # Ephemeral (development/testing)
        >>> checkpointer = create_checkpointer("memory")

        # Persistent SQLite
        >>> checkpointer = create_checkpointer(
        ...     "sqlite",
        ...     conn_string="checkpoints.db"
        ... )

        # Production PostgreSQL
        >>> checkpointer = create_checkpointer(
        ...     "postgres",
        ...     conn_string="postgresql://user:pass@host/db"
        ... )

    Usage with LangGraph:
        >>> from langgraph.graph import StateGraph
        >>> checkpointer = create_checkpointer("memory")
        >>> graph = builder.compile(checkpointer=checkpointer)
        >>> graph.invoke(
        ...     {"messages": [HumanMessage(content="Hello")]},
        ...     {"configurable": {"thread_id": "conversation-123"}}
        ... )
    """
    if backend == "memory":
        return MemorySaver(**kwargs)

    elif backend == "sqlite":
        conn_string = kwargs.pop("conn_string", "checkpoints.sqlite")
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found]
            return SqliteSaver.from_conn_string(conn_string, **kwargs)
        except ImportError as e:
            raise ImportError(
                "SQLite checkpointer requires langgraph-checkpoint-sqlite. "
                "Install with: pip install langgraph-checkpoint-sqlite"
            ) from e

    elif backend == "postgres":
        conn_string = kwargs.pop("conn_string")
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import-not-found]
            return PostgresSaver.from_conn_string(conn_string, **kwargs)
        except ImportError as e:
            raise ImportError(
                "PostgreSQL checkpointer requires langgraph-checkpoint-postgres. "
                "Install with: pip install langgraph-checkpoint-postgres"
            ) from e

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            "Supported: 'memory', 'sqlite', 'postgres'"
        )


# Convenience aliases
def memory_checkpointer(**kwargs: Any) -> MemorySaver:
    """Create an in-memory (ephemeral) checkpointer."""
    return MemorySaver(**kwargs)


def sqlite_checkpointer(
    conn_string: str = "checkpoints.sqlite",
    **kwargs: Any,
) -> BaseCheckpointSaver:
    """Create a SQLite (persistent) checkpointer.

    Requires: pip install langgraph-checkpoint-sqlite
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found]
        return SqliteSaver.from_conn_string(conn_string, **kwargs)
    except ImportError as e:
        raise ImportError(
            "SQLite checkpointer requires langgraph-checkpoint-sqlite. "
            "Install with: pip install langgraph-checkpoint-sqlite"
        ) from e


def postgres_checkpointer(
    conn_string: str,
    **kwargs: Any,
) -> BaseCheckpointSaver:
    """Create a PostgreSQL (persistent) checkpointer.

    Requires: pip install langgraph-checkpoint-postgres
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import-not-found]
        return PostgresSaver.from_conn_string(conn_string, **kwargs)
    except ImportError as e:
        raise ImportError(
            "PostgreSQL checkpointer requires langgraph-checkpoint-postgres. "
            "Install with: pip install langgraph-checkpoint-postgres"
        ) from e
