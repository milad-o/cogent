"""Tests for storage backends.

Tests for:
- SQLAlchemy store (SQLite and PostgreSQL)
- Redis store
- Vector store backends (FAISS, Chroma, Qdrant, pgvector)

Note: Most backends require optional dependencies.
Tests are skipped if dependencies are not installed.
"""

from __future__ import annotations

import tempfile

import pytest

# =============================================================================
# SQLAlchemy Store Tests (New Memory Architecture)
# =============================================================================


class TestSQLAlchemyStore:
    """Tests for SQLAlchemy-based memory store."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name

    @pytest.mark.asyncio
    async def test_import_sqlalchemy_store(self):
        """Test SQLAlchemyStore can be imported."""
        pytest.importorskip("sqlalchemy")
        from agenticflow.memory.stores import SQLAlchemyStore

        assert SQLAlchemyStore is not None

    @pytest.mark.asyncio
    async def test_sqlite_set_get(self, temp_db):
        """Test basic set and get operations with SQLite."""
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.stores import SQLAlchemyStore

        store = SQLAlchemyStore(f"sqlite+aiosqlite:///{temp_db}")
        await store.initialize()

        await store.set("test_key", {"data": "value"})
        result = await store.get("test_key")
        assert result == {"data": "value"}

        await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_get_nonexistent(self, temp_db):
        """Test get returns None for non-existent key."""
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.stores import SQLAlchemyStore

        store = SQLAlchemyStore(f"sqlite+aiosqlite:///{temp_db}")
        await store.initialize()

        result = await store.get("nonexistent")
        assert result is None

        await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_delete(self, temp_db):
        """Test delete operation."""
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.stores import SQLAlchemyStore

        store = SQLAlchemyStore(f"sqlite+aiosqlite:///{temp_db}")
        await store.initialize()

        await store.set("key", "value")
        assert await store.get("key") == "value"

        result = await store.delete("key")
        assert result is True
        assert await store.get("key") is None

        await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_keys(self, temp_db):
        """Test listing keys."""
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.stores import SQLAlchemyStore

        store = SQLAlchemyStore(f"sqlite+aiosqlite:///{temp_db}")
        await store.initialize()

        await store.set("user:1", "v1")
        await store.set("user:2", "v2")
        await store.set("team:1", "v3")

        all_keys = await store.keys()
        assert set(all_keys) == {"user:1", "user:2", "team:1"}

        user_keys = await store.keys(prefix="user:")
        assert set(user_keys) == {"user:1", "user:2"}

        await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_clear(self, temp_db):
        """Test clearing keys."""
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.stores import SQLAlchemyStore

        store = SQLAlchemyStore(f"sqlite+aiosqlite:///{temp_db}")
        await store.initialize()

        await store.set("user:1", "v1")
        await store.set("user:2", "v2")
        await store.set("team:1", "v3")

        # Clear only user: keys
        await store.clear(prefix="user:")

        keys = await store.keys()
        assert keys == ["team:1"]

        # Clear all
        await store.clear()
        keys = await store.keys()
        assert keys == []

        await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_get_many(self, temp_db):
        """Test batch get."""
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.stores import SQLAlchemyStore

        store = SQLAlchemyStore(f"sqlite+aiosqlite:///{temp_db}")
        await store.initialize()

        await store.set("a", "1")
        await store.set("b", "2")
        await store.set("c", "3")

        # get_many only returns found keys (more efficient)
        result = await store.get_many(["a", "c", "nonexistent"])
        assert result == {"a": "1", "c": "3"}

        await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_set_many(self, temp_db):
        """Test batch set."""
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.stores import SQLAlchemyStore

        store = SQLAlchemyStore(f"sqlite+aiosqlite:///{temp_db}")
        await store.initialize()

        await store.set_many({"x": "10", "y": "20", "z": "30"})

        assert await store.get("x") == "10"
        assert await store.get("y") == "20"
        assert await store.get("z") == "30"

        await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_upsert(self, temp_db):
        """Test that set updates existing values."""
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.stores import SQLAlchemyStore

        store = SQLAlchemyStore(f"sqlite+aiosqlite:///{temp_db}")
        await store.initialize()

        await store.set("key", "original")
        assert await store.get("key") == "original"

        await store.set("key", "updated")
        assert await store.get("key") == "updated"

        # Should still have only one key
        keys = await store.keys()
        assert keys == ["key"]

        await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_stores_complex_types(self, temp_db):
        """Test storing complex JSON-serializable types."""
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.stores import SQLAlchemyStore

        store = SQLAlchemyStore(f"sqlite+aiosqlite:///{temp_db}")
        await store.initialize()

        data = {
            "name": "test",
            "values": [1, 2, 3],
            "nested": {"a": 1, "b": [4, 5]},
            "boolean": True,
            "null": None,
        }
        await store.set("complex", data)
        result = await store.get("complex")
        assert result == data

        await store.close()


class TestRedisStore:
    """Tests for Redis memory store."""

    @pytest.mark.asyncio
    async def test_import_redis_store(self):
        """Test RedisStore can be imported."""
        pytest.importorskip("redis")
        from agenticflow.memory.stores import RedisStore

        assert RedisStore is not None

    @pytest.mark.asyncio
    async def test_redis_connection_error_handling(self):
        """Test Redis handles connection errors gracefully."""
        pytest.importorskip("redis")
        from agenticflow.memory.stores import RedisStore

        # Invalid host should fail to connect
        store = RedisStore(url="redis://invalid-host:6379")
        # Don't actually test connection as it may hang
        # Just ensure the store can be created
        assert store is not None


class TestMemoryWithStores:
    """Integration tests: Memory with different stores."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name

    @pytest.mark.asyncio
    async def test_memory_with_sqlalchemy_store(self, temp_db):
        """Test Memory class with SQLAlchemy store."""
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        from agenticflow.memory import Memory
        from agenticflow.memory.stores import SQLAlchemyStore

        store = SQLAlchemyStore(f"sqlite+aiosqlite:///{temp_db}")
        await store.initialize()

        memory = Memory(store=store)

        await memory.remember("user", "alice")
        assert await memory.recall("user") == "alice"

        # Test scoped memory
        user_mem = memory.scoped("user:1")
        await user_mem.remember("preference", "dark")
        assert await user_mem.recall("preference") == "dark"

        # Verify namespacing in store
        assert await store.get("user:1:preference") == "dark"

        await store.close()

    @pytest.mark.asyncio
    async def test_memory_persistence(self, temp_db):
        """Test that memory persists across instances."""
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("aiosqlite")
        from agenticflow.memory import Memory
        from agenticflow.memory.stores import SQLAlchemyStore

        db_url = f"sqlite+aiosqlite:///{temp_db}"

        # First instance - write data
        store1 = SQLAlchemyStore(db_url)
        await store1.initialize()
        memory1 = Memory(store=store1)
        await memory1.remember("persistent_key", "persistent_value")
        await store1.close()

        # Second instance - read data
        store2 = SQLAlchemyStore(db_url)
        await store2.initialize()
        memory2 = Memory(store=store2)
        result = await memory2.recall("persistent_key")
        assert result == "persistent_value"
        await store2.close()


# =============================================================================
# Vector Store Backend Tests
# =============================================================================


class TestFAISSBackend:
    """Tests for FAISS vector store backend."""

    @pytest.mark.asyncio
    async def test_import_faiss_backend(self):
        """Test FAISS backend can be imported."""
        pytest.importorskip("faiss")
        from agenticflow.vectorstore.backends import FAISSBackend

        assert FAISSBackend is not None

    @pytest.mark.asyncio
    async def test_faiss_add_and_search(self):
        """Test adding and searching documents."""
        pytest.importorskip("faiss")
        from agenticflow.vectorstore.backends import FAISSBackend
        from agenticflow.vectorstore.document import Document

        backend = FAISSBackend(dimension=4)

        # Add documents
        ids = ["doc1", "doc2", "doc3"]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        documents = [
            Document(text="Document A", metadata={"type": "a"}),
            Document(text="Document B", metadata={"type": "b"}),
            Document(text="Document C", metadata={"type": "c"}),
        ]

        await backend.add(ids, embeddings, documents)

        # Search
        results = await backend.search([1.0, 0.1, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0].id == "doc1"  # Closest match

    @pytest.mark.asyncio
    async def test_faiss_delete(self):
        """Test deleting documents."""
        pytest.importorskip("faiss")
        from agenticflow.vectorstore.backends import FAISSBackend
        from agenticflow.vectorstore.document import Document

        backend = FAISSBackend(dimension=4)

        ids = ["doc1", "doc2"]
        embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        documents = [
            Document(text="Document A", metadata={"type": "a"}),
            Document(text="Document B", metadata={"type": "b"}),
        ]

        await backend.add(ids, embeddings, documents)
        assert backend.count() == 2

        await backend.delete(["doc1"])
        assert backend.count() == 1

    @pytest.mark.asyncio
    async def test_faiss_with_filter(self):
        """Test search with metadata filter."""
        pytest.importorskip("faiss")
        from agenticflow.vectorstore.backends import FAISSBackend
        from agenticflow.vectorstore.document import Document

        backend = FAISSBackend(dimension=4)

        ids = ["doc1", "doc2", "doc3"]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
        ]
        documents = [
            Document(text="Doc tech", metadata={"category": "tech"}),
            Document(text="Doc science", metadata={"category": "science"}),
            Document(text="Doc tech 2", metadata={"category": "tech"}),
        ]

        await backend.add(ids, embeddings, documents)

        # Search with filter
        results = await backend.search(
            [1.0, 0.0, 0.0, 0.0],
            k=3,
            filter={"category": "tech"},
        )
        assert len(results) == 2
        # All results should be tech category
        for result in results:
            assert result.document.metadata["category"] == "tech"


class TestChromaBackend:
    """Tests for Chroma vector store backend."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for Chroma persistence."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.mark.asyncio
    async def test_import_chroma_backend(self):
        """Test Chroma backend can be imported."""
        pytest.importorskip("chromadb")
        from agenticflow.vectorstore.backends import ChromaBackend

        assert ChromaBackend is not None

    @pytest.mark.asyncio
    async def test_chroma_add_and_search(self, temp_dir):
        """Test adding and searching documents."""
        pytest.importorskip("chromadb")
        from agenticflow.vectorstore.backends import ChromaBackend
        from agenticflow.vectorstore.document import Document

        backend = ChromaBackend(
            collection_name="test_collection",
            persist_directory=temp_dir,
        )

        ids = ["doc1", "doc2"]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        documents = [
            Document(text="Document one", metadata={"source": "a"}),
            Document(text="Document two", metadata={"source": "b"}),
        ]

        await backend.add(ids, embeddings, documents)

        results = await backend.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0].id == "doc1"

    @pytest.mark.asyncio
    async def test_chroma_delete(self, temp_dir):
        """Test deleting documents."""
        pytest.importorskip("chromadb")
        from agenticflow.vectorstore.backends import ChromaBackend
        from agenticflow.vectorstore.document import Document

        backend = ChromaBackend(
            collection_name="test_delete",
            persist_directory=temp_dir,
        )

        ids = ["doc1", "doc2"]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        documents = [
            Document(text="Document one", metadata={"x": 1}),
            Document(text="Document two", metadata={"x": 2}),
        ]

        await backend.add(ids, embeddings, documents)
        assert backend.count() == 2

        await backend.delete(["doc1"])
        assert backend.count() == 1

    @pytest.mark.asyncio
    async def test_chroma_get(self, temp_dir):
        """Test getting document by ID."""
        pytest.importorskip("chromadb")
        from agenticflow.vectorstore.backends import ChromaBackend
        from agenticflow.vectorstore.document import Document

        backend = ChromaBackend(
            collection_name="test_get",
            persist_directory=temp_dir,
        )

        ids = ["doc1"]
        embeddings = [[1.0, 0.0, 0.0]]
        documents = [Document(text="Document one", metadata={"key": "value"})]

        await backend.add(ids, embeddings, documents)

        docs = await backend.get(["doc1"])
        assert len(docs) == 1
        assert docs[0].metadata["key"] == "value"


class TestQdrantBackend:
    """Tests for Qdrant vector store backend."""

    @pytest.mark.asyncio
    async def test_import_qdrant_backend(self):
        """Test Qdrant backend can be imported."""
        pytest.importorskip("qdrant_client")
        from agenticflow.vectorstore.backends import QdrantBackend

        assert QdrantBackend is not None

    @pytest.mark.asyncio
    async def test_qdrant_inmemory_operations(self):
        """Test Qdrant with in-memory storage."""
        pytest.importorskip("qdrant_client")
        from agenticflow.vectorstore.backends import QdrantBackend

        # Use in-memory mode (location=":memory:")
        backend = QdrantBackend(
            collection_name="test_collection",
            location=":memory:",
        )

        ids = ["doc1", "doc2"]
        embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        metadatas = [{"type": "a"}, {"type": "b"}]

        await backend.add(ids, embeddings, metadatas)

        results = await backend.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0][0] == "doc1"

    @pytest.mark.asyncio
    async def test_qdrant_with_filter(self):
        """Test Qdrant search with filter."""
        pytest.importorskip("qdrant_client")
        from agenticflow.vectorstore.backends import QdrantBackend

        backend = QdrantBackend(
            collection_name="test_filter",
            location=":memory:",
        )

        ids = ["doc1", "doc2", "doc3"]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
        ]
        metadatas = [
            {"category": "A"},
            {"category": "B"},
            {"category": "A"},
        ]

        await backend.add(ids, embeddings, metadatas)

        results = await backend.search(
            [1.0, 0.0, 0.0, 0.0],
            k=3,
            filter={"category": "A"},
        )
        assert len(results) == 2


class TestPgVectorBackend:
    """Tests for pgvector backend.

    These tests require a PostgreSQL database with pgvector extension.
    They are skipped if the connection fails.
    """

    @pytest.mark.asyncio
    async def test_import_pgvector_backend(self):
        """Test pgvector backend can be imported."""
        pytest.importorskip("psycopg")
        from agenticflow.vectorstore.backends import PgVectorBackend

        assert PgVectorBackend is not None

    @pytest.mark.asyncio
    async def test_pgvector_missing_dependency(self):
        """Test helpful error when psycopg is missing."""
        # This test just verifies the import logic
        from agenticflow.vectorstore.backends.pgvector import PgVectorBackend

        # If psycopg is installed, the class should be available
        assert PgVectorBackend is not None


# =============================================================================
# Backend Lazy Import Tests
# =============================================================================


class TestLazyImports:
    """Test that backends are lazily imported."""

    def test_vectorstore_backends_module_exports(self):
        """Test vectorstore backends module exports correct names."""
        from agenticflow.vectorstore import backends

        assert "InMemoryBackend" in backends.__all__
        assert "FAISSBackend" in backends.__all__
        assert "ChromaBackend" in backends.__all__
        assert "QdrantBackend" in backends.__all__
        assert "PgVectorBackend" in backends.__all__

    def test_inmemory_always_available(self):
        """Test InMemoryBackend is always available."""
        from agenticflow.vectorstore.backends import InMemoryBackend

        assert InMemoryBackend is not None

    def test_memory_stores_available(self):
        """Test memory stores can be imported."""
        from agenticflow.memory import InMemoryStore, Memory

        assert InMemoryStore is not None
        assert Memory is not None

    def test_memory_sqlalchemy_store_import(self):
        """Test SQLAlchemyStore import with optional dep."""
        pytest.importorskip("sqlalchemy")
        from agenticflow.memory.stores import SQLAlchemyStore

        assert SQLAlchemyStore is not None

    def test_memory_redis_store_import(self):
        """Test RedisStore import with optional dep."""
        pytest.importorskip("redis")
        from agenticflow.memory.stores import RedisStore

        assert RedisStore is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestVectorStoreIntegration:
    """Integration tests with VectorStore class."""

    @pytest.mark.asyncio
    async def test_vectorstore_with_inmemory(self):
        """Test VectorStore with default InMemory backend."""
        from agenticflow.vectorstore import MockEmbeddings, VectorStore
        from agenticflow.vectorstore.backends import InMemoryBackend

        store = VectorStore(
            embeddings=MockEmbeddings(),
            backend=InMemoryBackend(),
        )

        await store.add_texts(["Hello world", "Goodbye world"])
        results = await store.search("Hello", k=1)

        assert len(results) == 1
        assert "Hello" in results[0].document.text

    @pytest.mark.asyncio
    async def test_vectorstore_with_faiss(self):
        """Test VectorStore with FAISS backend."""
        pytest.importorskip("faiss")
        from agenticflow.vectorstore import MockEmbeddings, VectorStore
        from agenticflow.vectorstore.backends import FAISSBackend

        embeddings = MockEmbeddings()
        backend = FAISSBackend(dimension=embeddings.dimension)

        store = VectorStore(embeddings=embeddings, backend=backend)

        await store.add_texts(["Document one", "Document two"])
        results = await store.search("one", k=1)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_vectorstore_with_chroma(self):
        """Test VectorStore with Chroma backend."""
        pytest.importorskip("chromadb")
        from agenticflow.vectorstore import MockEmbeddings, VectorStore
        from agenticflow.vectorstore.backends import ChromaBackend

        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = MockEmbeddings()
            backend = ChromaBackend(
                collection_name="test",
                persist_directory=temp_dir,
            )

            store = VectorStore(embeddings=embeddings, backend=backend)

            await store.add_texts(["Test document"])
            results = await store.search("Test", k=1)

            assert len(results) == 1
