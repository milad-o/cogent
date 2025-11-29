"""Store implementations for Memory persistence.

Available stores:
- InMemoryStore: Fast, no persistence (default)
- SQLAlchemyStore: SQLAlchemy 2.0 async ORM (SQLite, PostgreSQL, etc.)
- RedisStore: Distributed cache with native TTL

All stores implement the Store protocol:
    - get(key) -> value | None
    - set(key, value, ttl=None)
    - delete(key) -> bool
    - keys(prefix="") -> list[str]
    - clear(prefix="")
    
Batch operations (optional, for performance):
    - get_many(keys) -> dict[str, Any]
    - set_many(items, ttl=None)
    - delete_many(keys) -> int
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from agenticflow.memory.core import InMemoryStore

__all__ = [
    "InMemoryStore",
    "SQLAlchemyStore",
    "RedisStore",
]


# =============================================================================
# SQLALCHEMY STORE - Async ORM with SQLite/PostgreSQL
# =============================================================================


class SQLAlchemyStore:
    """SQLAlchemy 2.0 async store.
    
    Supports SQLite, PostgreSQL, MySQL, and other SQLAlchemy-compatible databases.
    Uses async ORM for non-blocking operations with batch support.
    
    Example:
        ```python
        # SQLite (local file)
        store = SQLAlchemyStore("sqlite+aiosqlite:///./data.db")
        
        # PostgreSQL
        store = SQLAlchemyStore("postgresql+asyncpg://user:pass@localhost/db")
        
        # With custom table name
        store = SQLAlchemyStore(
            "sqlite+aiosqlite:///./data.db",
            table_name="agent_memory"
        )
        
        memory = Memory(store=store)
        ```
    """
    
    def __init__(
        self,
        url: str,
        *,
        table_name: str = "memory",
        echo: bool = False,
    ) -> None:
        """Initialize SQLAlchemy store.
        
        Args:
            url: SQLAlchemy async connection URL.
                - SQLite: "sqlite+aiosqlite:///./data.db"
                - PostgreSQL: "postgresql+asyncpg://user:pass@host/db"
            table_name: Table name for storage.
            echo: Echo SQL statements (for debugging).
        """
        self._url = url
        self._table_name = table_name
        self._echo = echo
        self._engine: Any = None
        self._sessionmaker: Any = None
        self._model: Any = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Explicitly initialize the store (creates tables).
        
        This is optional - the store auto-initializes on first use.
        """
        await self._ensure_initialized()
    
    async def _ensure_initialized(self) -> None:
        """Lazy initialization of SQLAlchemy engine and tables."""
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:
                return
            
            from sqlalchemy import DateTime, Index, String, Text
            from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
            from sqlalchemy.orm import DeclarativeBase, mapped_column, sessionmaker
            
            # Create engine with connection pooling
            self._engine = create_async_engine(
                self._url,
                echo=self._echo,
                pool_pre_ping=True,  # Verify connections before use
            )
            
            # Create async sessionmaker
            self._sessionmaker = sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            # Define model dynamically using SQLAlchemy 2.0 ORM
            # Using mapped_column without Mapped[] annotations because the model
            # is defined inside a function and SQLAlchemy can't resolve annotations
            table_name = self._table_name
            
            class Base(DeclarativeBase):
                pass
            
            class MemoryEntry(Base):
                __tablename__ = table_name
                
                key = mapped_column(String(512), primary_key=True)
                value = mapped_column(Text, nullable=False)
                expires_at = mapped_column(DateTime(timezone=True), nullable=True)
                
                __table_args__ = (
                    Index(f"idx_{table_name}_expires", "expires_at", postgresql_where=expires_at.isnot(None)),
                )
            
            self._model = MemoryEntry
            self._base = Base
            
            # Create tables
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self._initialized = True
    
    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        await self._ensure_initialized()
        
        from sqlalchemy import select
        
        async with self._sessionmaker() as session:
            stmt = select(self._model).where(self._model.key == key)
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()
            
            if not entry:
                return None
            
            # Check expiry
            if entry.expires_at and datetime.now(UTC) > entry.expires_at:
                await session.delete(entry)
                await session.commit()
                return None
            
            return json.loads(entry.value)
    
    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values by keys (batch operation)."""
        if not keys:
            return {}
        
        await self._ensure_initialized()
        
        from sqlalchemy import select
        
        now = datetime.now(UTC)
        result_dict: dict[str, Any] = {}
        expired_entries: list[Any] = []
        
        async with self._sessionmaker() as session:
            stmt = select(self._model).where(self._model.key.in_(keys))
            result = await session.execute(stmt)
            entries = result.scalars().all()
            
            for entry in entries:
                if entry.expires_at and now > entry.expires_at:
                    expired_entries.append(entry)
                else:
                    result_dict[entry.key] = json.loads(entry.value)
            
            # Batch delete expired entries
            if expired_entries:
                for entry in expired_entries:
                    await session.delete(entry)
                await session.commit()
        
        return result_dict
    
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value by key."""
        await self.set_many({key: value}, ttl=ttl)
    
    async def set_many(
        self,
        items: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Set multiple values (batch upsert)."""
        if not items:
            return
        
        await self._ensure_initialized()
        
        expires_at = None
        if ttl is not None:
            expires_at = datetime.now(UTC) + timedelta(seconds=ttl)
        
        values = [
            {"key": k, "value": json.dumps(v), "expires_at": expires_at}
            for k, v in items.items()
        ]
        
        async with self._sessionmaker() as session:
            # Use dialect-specific upsert
            if "postgresql" in self._url:
                from sqlalchemy.dialects.postgresql import insert as pg_insert
                
                stmt = pg_insert(self._model).values(values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["key"],
                    set_={
                        "value": stmt.excluded.value,
                        "expires_at": stmt.excluded.expires_at,
                    },
                )
                await session.execute(stmt)
                
            elif "sqlite" in self._url:
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert
                
                # SQLite supports batch upsert
                stmt = sqlite_insert(self._model).values(values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["key"],
                    set_={
                        "value": stmt.excluded.value,
                        "expires_at": stmt.excluded.expires_at,
                    },
                )
                await session.execute(stmt)
                
            else:
                # Fallback: merge each (slower but compatible)
                from sqlalchemy import select
                
                for item in values:
                    existing = await session.execute(
                        select(self._model).where(self._model.key == item["key"])
                    )
                    entry = existing.scalar_one_or_none()
                    
                    if entry:
                        entry.value = item["value"]
                        entry.expires_at = item["expires_at"]
                    else:
                        session.add(self._model(**item))
            
            await session.commit()
    
    async def delete(self, key: str) -> bool:
        """Delete by key."""
        count = await self.delete_many([key])
        return count > 0
    
    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys (batch operation). Returns count deleted."""
        if not keys:
            return 0
        
        await self._ensure_initialized()
        
        from sqlalchemy import delete
        
        async with self._sessionmaker() as session:
            stmt = delete(self._model).where(self._model.key.in_(keys))
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount
    
    async def keys(self, prefix: str = "") -> list[str]:
        """List keys matching prefix."""
        await self._ensure_initialized()
        
        from sqlalchemy import or_, select
        
        async with self._sessionmaker() as session:
            stmt = select(self._model.key)
            
            if prefix:
                stmt = stmt.where(self._model.key.like(f"{prefix}%"))
            
            # Exclude expired
            stmt = stmt.where(
                or_(
                    self._model.expires_at.is_(None),
                    self._model.expires_at > datetime.now(UTC),
                )
            )
            
            result = await session.execute(stmt)
            return [row[0] for row in result.fetchall()]
    
    async def clear(self, prefix: str = "") -> None:
        """Clear keys matching prefix."""
        await self._ensure_initialized()
        
        from sqlalchemy import delete
        
        async with self._sessionmaker() as session:
            if prefix:
                stmt = delete(self._model).where(self._model.key.like(f"{prefix}%"))
            else:
                stmt = delete(self._model)
            
            await session.execute(stmt)
            await session.commit()
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        await self._ensure_initialized()
        
        from sqlalchemy import and_, delete
        
        async with self._sessionmaker() as session:
            stmt = delete(self._model).where(
                and_(
                    self._model.expires_at.isnot(None),
                    self._model.expires_at < datetime.now(UTC),
                )
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount
    
    async def close(self) -> None:
        """Close the engine and dispose connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._initialized = False


# =============================================================================
# REDIS STORE - Distributed Cache with Pipelining
# =============================================================================


class RedisStore:
    """Redis-based distributed store.
    
    Uses native Redis TTL for expiration.
    Supports pipelining for efficient batch operations.
    
    Example:
        ```python
        store = RedisStore("redis://localhost:6379/0")
        
        # With custom prefix
        store = RedisStore(
            "redis://localhost:6379/0",
            prefix="myapp:memory:"
        )
        
        memory = Memory(store=store)
        ```
    """
    
    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "agenticflow:",
    ) -> None:
        """Initialize Redis store.
        
        Args:
            url: Redis connection URL.
            prefix: Key prefix for all operations.
        """
        self._url = url
        self._prefix = prefix
        self._client: Any = None
        self._init_lock = asyncio.Lock()
    
    async def _ensure_client(self) -> Any:
        """Get or create Redis client."""
        if self._client is None:
            async with self._init_lock:
                if self._client is None:
                    import redis.asyncio as redis
                    self._client = await redis.from_url(self._url)
        return self._client
    
    def _full_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self._prefix}{key}"
    
    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        client = await self._ensure_client()
        
        value = await client.get(self._full_key(key))
        if value is None:
            return None
        
        return json.loads(value)
    
    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values (uses MGET - single round trip)."""
        if not keys:
            return {}
        
        client = await self._ensure_client()
        
        full_keys = [self._full_key(k) for k in keys]
        values = await client.mget(full_keys)
        
        result = {}
        for key, value in zip(keys, values):
            if value is not None:
                result[key] = json.loads(value)
        
        return result
    
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value by key."""
        client = await self._ensure_client()
        
        serialized = json.dumps(value)
        
        if ttl is not None:
            await client.setex(self._full_key(key), ttl, serialized)
        else:
            await client.set(self._full_key(key), serialized)
    
    async def set_many(
        self,
        items: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Set multiple values (uses pipeline - single round trip)."""
        if not items:
            return
        
        client = await self._ensure_client()
        
        async with client.pipeline(transaction=True) as pipe:
            for key, value in items.items():
                serialized = json.dumps(value)
                full_key = self._full_key(key)
                
                if ttl is not None:
                    pipe.setex(full_key, ttl, serialized)
                else:
                    pipe.set(full_key, serialized)
            
            await pipe.execute()
    
    async def delete(self, key: str) -> bool:
        """Delete by key."""
        client = await self._ensure_client()
        result = await client.delete(self._full_key(key))
        return result > 0
    
    async def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys (single operation). Returns count deleted."""
        if not keys:
            return 0
        
        client = await self._ensure_client()
        full_keys = [self._full_key(k) for k in keys]
        return await client.delete(*full_keys)
    
    async def keys(self, prefix: str = "") -> list[str]:
        """List keys matching prefix."""
        client = await self._ensure_client()
        
        pattern = f"{self._prefix}{prefix}*"
        all_keys = await client.keys(pattern)
        
        # Strip global prefix
        prefix_len = len(self._prefix)
        return [
            k.decode()[prefix_len:] if isinstance(k, bytes) else k[prefix_len:]
            for k in all_keys
        ]
    
    async def clear(self, prefix: str = "") -> None:
        """Clear keys matching prefix (uses SCAN for large keyspaces)."""
        client = await self._ensure_client()
        
        pattern = f"{self._prefix}{prefix}*"
        
        # Use SCAN to avoid blocking on large keyspaces
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=1000)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break
    
    async def close(self) -> None:
        """Close Redis client."""
        if self._client:
            await self._client.close()
            self._client = None
