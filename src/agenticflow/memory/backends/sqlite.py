"""SQLite storage backend for memory system.

Persistent backend that stores data in a SQLite database.
Supports ConversationMemory and UserMemory persistence.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

from agenticflow.memory.base import MemoryEntry, MemoryScope

if TYPE_CHECKING:
    from agenticflow.core.messages import Message


def _serialize_value(value: Any) -> str:
    """Serialize a value to JSON string."""
    return json.dumps(value)


def _deserialize_value(data: str) -> Any:
    """Deserialize a value from JSON string."""
    return json.loads(data)


def _serialize_message(message: Message) -> str:
    """Serialize a Message to JSON string."""
    return json.dumps({
        "role": message.role,
        "content": message.content,
        "name": getattr(message, "name", None),
        "tool_calls": getattr(message, "tool_calls", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
    })


def _deserialize_message(data: str) -> Message:
    """Deserialize a Message from JSON string."""
    from agenticflow.core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    
    obj = json.loads(data)
    role = obj["role"]
    content = obj["content"]
    
    if role == "system":
        return SystemMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(
            content=content,
            tool_calls=obj.get("tool_calls"),
        )
    elif role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=obj.get("tool_call_id", ""),
        )
    else:
        return HumanMessage(content=content)


@dataclass
class SQLiteBackend:
    """SQLite implementation of MemoryBackend.

    Stores data persistently in a SQLite database file.
    
    Attributes:
        db_path: Path to SQLite database file (default: memory.db).
        
    Example:
        backend = SQLiteBackend(db_path="./data/memory.db")
        await backend.set("name", "Alice", MemoryScope.USER, "user-123")
        name = await backend.get("name", MemoryScope.USER, "user-123")
    """

    db_path: str | Path = "memory.db"
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize database schema."""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._initialized = True

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    ttl_seconds INTEGER,
                    UNIQUE(scope, scope_id, key)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_scope 
                ON memory_entries(scope, scope_id)
            """)

    async def get(self, key: str, scope: MemoryScope, scope_id: str) -> Any | None:
        """Get a value by key within a scope."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT value, created_at, ttl_seconds 
                FROM memory_entries 
                WHERE scope = ? AND scope_id = ? AND key = ?
                """,
                (scope.value, scope_id, key),
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            # Check TTL
            if row["ttl_seconds"] is not None:
                created = datetime.fromisoformat(row["created_at"])
                if (datetime.now(UTC) - created).total_seconds() > row["ttl_seconds"]:
                    # Expired - delete and return None
                    conn.execute(
                        "DELETE FROM memory_entries WHERE scope = ? AND scope_id = ? AND key = ?",
                        (scope.value, scope_id, key),
                    )
                    return None
            
            return _deserialize_value(row["value"])

    async def set(
        self,
        key: str,
        value: Any,
        scope: MemoryScope,
        scope_id: str,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set a value by key within a scope."""
        now = datetime.now(UTC).isoformat()
        
        with self._get_connection() as conn:
            # Check if exists to preserve created_at
            cursor = conn.execute(
                "SELECT created_at FROM memory_entries WHERE scope = ? AND scope_id = ? AND key = ?",
                (scope.value, scope_id, key),
            )
            existing = cursor.fetchone()
            created_at = existing["created_at"] if existing else now
            
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_entries 
                (key, value, scope, scope_id, created_at, updated_at, metadata, ttl_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    _serialize_value(value),
                    scope.value,
                    scope_id,
                    created_at,
                    now,
                    json.dumps(metadata) if metadata else None,
                    ttl_seconds,
                ),
            )

    async def delete(self, key: str, scope: MemoryScope, scope_id: str) -> bool:
        """Delete a value by key."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memory_entries WHERE scope = ? AND scope_id = ? AND key = ?",
                (scope.value, scope_id, key),
            )
            return cursor.rowcount > 0

    async def list_keys(self, scope: MemoryScope, scope_id: str) -> list[str]:
        """List all keys within a scope."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT key FROM memory_entries WHERE scope = ? AND scope_id = ?",
                (scope.value, scope_id),
            )
            return [row["key"] for row in cursor.fetchall()]

    async def clear(self, scope: MemoryScope, scope_id: str) -> None:
        """Clear all entries within a scope."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM memory_entries WHERE scope = ? AND scope_id = ?",
                (scope.value, scope_id),
            )

    async def search(
        self,
        query: str,
        scope: MemoryScope,
        scope_id: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search for entries matching a query."""
        query_pattern = f"%{query}%"
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT key, value, scope, scope_id, created_at, updated_at, metadata, ttl_seconds
                FROM memory_entries 
                WHERE scope = ? AND scope_id = ? 
                AND (key LIKE ? OR value LIKE ?)
                LIMIT ?
                """,
                (scope.value, scope_id, query_pattern, query_pattern, limit),
            )
            
            results = []
            for row in cursor.fetchall():
                entry = MemoryEntry(
                    key=row["key"],
                    value=_deserialize_value(row["value"]),
                    scope=MemoryScope(row["scope"]),
                    scope_id=row["scope_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    ttl_seconds=row["ttl_seconds"],
                )
                results.append(entry)
            
            return results

    async def clear_all(self) -> None:
        """Clear all data. Use with caution."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM memory_entries")


@dataclass
class SQLiteConversationBackend:
    """SQLite implementation of ConversationBackend.

    Stores conversation history persistently in a SQLite database.
    
    Attributes:
        db_path: Path to SQLite database file (default: conversations.db).
        
    Example:
        backend = SQLiteConversationBackend(db_path="./data/conversations.db")
        await backend.add_message("thread-1", message)
        messages = await backend.get_messages("thread-1", limit=10)
    """

    db_path: str | Path = "conversations.db"
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize database schema."""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._initialized = True

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_thread 
                ON conversations(thread_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_thread_time 
                ON conversations(thread_id, created_at)
            """)

    async def add_message(
        self,
        thread_id: str,
        message: Message,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a message to conversation history."""
        now = datetime.now(UTC).isoformat()
        
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations (thread_id, message, metadata, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    thread_id,
                    _serialize_message(message),
                    json.dumps(metadata) if metadata else None,
                    now,
                ),
            )

    async def get_messages(
        self,
        thread_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        """Get messages from conversation history."""
        with self._get_connection() as conn:
            query = """
                SELECT message FROM conversations 
                WHERE thread_id = ?
                ORDER BY created_at ASC
            """
            params: list[Any] = [thread_id]
            
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])
            elif offset > 0:
                query += " LIMIT -1 OFFSET ?"
                params.append(offset)
            
            cursor = conn.execute(query, params)
            return [_deserialize_message(row["message"]) for row in cursor.fetchall()]

    async def get_recent_messages(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> list[Message]:
        """Get the most recent messages from a thread."""
        with self._get_connection() as conn:
            # Get total count first
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM conversations WHERE thread_id = ?",
                (thread_id,),
            )
            total = cursor.fetchone()["count"]
            
            offset = max(0, total - limit)
            
            cursor = conn.execute(
                """
                SELECT message FROM conversations 
                WHERE thread_id = ?
                ORDER BY created_at ASC
                LIMIT ? OFFSET ?
                """,
                (thread_id, limit, offset),
            )
            return [_deserialize_message(row["message"]) for row in cursor.fetchall()]

    async def clear_thread(self, thread_id: str) -> None:
        """Clear all messages in a thread."""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE thread_id = ?",
                (thread_id,),
            )

    async def get_message_count(self, thread_id: str) -> int:
        """Get the number of messages in a thread."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM conversations WHERE thread_id = ?",
                (thread_id,),
            )
            return cursor.fetchone()["count"]

    async def list_threads(self) -> list[str]:
        """List all thread IDs."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT thread_id FROM conversations"
            )
            return [row["thread_id"] for row in cursor.fetchall()]

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread entirely."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE thread_id = ?",
                (thread_id,),
            )
            return cursor.rowcount > 0

    async def clear_all(self) -> None:
        """Clear all threads. Use with caution."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM conversations")
