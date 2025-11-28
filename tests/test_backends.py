"""Tests for storage backends.

Tests for:
- SQLite memory backend
- Vector store backends (FAISS, Chroma, Qdrant, pgvector)

Note: Most backends require optional dependencies.
Tests are skipped if dependencies are not installed.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest


# =============================================================================
# SQLite Memory Backend Tests
# =============================================================================


class TestSQLiteBackend:
    """Tests for SQLite memory backend."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        # Cleanup happens automatically after test
    
    @pytest.mark.asyncio
    async def test_import_sqlite_backend(self):
        """Test SQLite backend can be imported."""
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.backends import SQLiteBackend, SQLiteConversationBackend
        assert SQLiteBackend is not None
        assert SQLiteConversationBackend is not None
    
    @pytest.mark.asyncio
    async def test_sqlite_backend_set_get(self, temp_db):
        """Test basic set and get operations."""
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.backends import SQLiteBackend
        
        backend = SQLiteBackend(db_path=temp_db)
        await backend.initialize()
        
        # Set a value
        await backend.set("test_key", {"data": "value"}, scope="user", scope_id="user1")
        
        # Get it back
        result = await backend.get("test_key", scope="user", scope_id="user1")
        assert result == {"data": "value"}
        
        await backend.close()
    
    @pytest.mark.asyncio
    async def test_sqlite_backend_scope_isolation(self, temp_db):
        """Test that different scopes are isolated."""
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.backends import SQLiteBackend
        
        backend = SQLiteBackend(db_path=temp_db)
        await backend.initialize()
        
        # Set same key in different scopes
        await backend.set("key", "user1_value", scope="user", scope_id="user1")
        await backend.set("key", "user2_value", scope="user", scope_id="user2")
        await backend.set("key", "team_value", scope="team", scope_id="team1")
        
        # Each should have its own value
        assert await backend.get("key", scope="user", scope_id="user1") == "user1_value"
        assert await backend.get("key", scope="user", scope_id="user2") == "user2_value"
        assert await backend.get("key", scope="team", scope_id="team1") == "team_value"
        
        await backend.close()
    
    @pytest.mark.asyncio
    async def test_sqlite_backend_delete(self, temp_db):
        """Test delete operation."""
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.backends import SQLiteBackend
        
        backend = SQLiteBackend(db_path=temp_db)
        await backend.initialize()
        
        await backend.set("key", "value", scope="user", scope_id="user1")
        assert await backend.get("key", scope="user", scope_id="user1") == "value"
        
        await backend.delete("key", scope="user", scope_id="user1")
        assert await backend.get("key", scope="user", scope_id="user1") is None
        
        await backend.close()
    
    @pytest.mark.asyncio
    async def test_sqlite_backend_list_keys(self, temp_db):
        """Test listing keys."""
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.backends import SQLiteBackend
        
        backend = SQLiteBackend(db_path=temp_db)
        await backend.initialize()
        
        await backend.set("key1", "v1", scope="user", scope_id="user1")
        await backend.set("key2", "v2", scope="user", scope_id="user1")
        await backend.set("key3", "v3", scope="user", scope_id="user2")
        
        keys = await backend.list_keys(scope="user", scope_id="user1")
        assert set(keys) == {"key1", "key2"}
        
        await backend.close()
    
    @pytest.mark.asyncio
    async def test_sqlite_backend_search(self, temp_db):
        """Test search functionality."""
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.backends import SQLiteBackend
        
        backend = SQLiteBackend(db_path=temp_db)
        await backend.initialize()
        
        await backend.set("pref_color", "blue", scope="user", scope_id="user1")
        await backend.set("pref_size", "large", scope="user", scope_id="user1")
        await backend.set("other_key", "value", scope="user", scope_id="user1")
        
        results = await backend.search("pref_", scope="user", scope_id="user1")
        assert len(results) == 2
        assert "pref_color" in results
        assert "pref_size" in results
        
        await backend.close()
    
    @pytest.mark.asyncio
    async def test_sqlite_conversation_backend(self, temp_db):
        """Test conversation message history."""
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.backends import SQLiteConversationBackend
        
        backend = SQLiteConversationBackend(db_path=temp_db)
        await backend.initialize()
        
        # Add messages
        await backend.add_message(
            {"role": "user", "content": "Hello"},
            conversation_id="conv1",
        )
        await backend.add_message(
            {"role": "assistant", "content": "Hi there!"},
            conversation_id="conv1",
        )
        
        # Get messages
        messages = await backend.get_messages(conversation_id="conv1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        
        await backend.close()
    
    @pytest.mark.asyncio
    async def test_sqlite_conversation_max_messages(self, temp_db):
        """Test message limit."""
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.backends import SQLiteConversationBackend
        
        backend = SQLiteConversationBackend(db_path=temp_db)
        await backend.initialize()
        
        # Add many messages
        for i in range(10):
            await backend.add_message(
                {"role": "user", "content": f"Message {i}"},
                conversation_id="conv1",
            )
        
        # Get limited messages
        messages = await backend.get_messages(conversation_id="conv1", max_messages=5)
        assert len(messages) == 5
        # Should be last 5 messages
        assert messages[0]["content"] == "Message 5"
        
        await backend.close()
    
    @pytest.mark.asyncio
    async def test_sqlite_conversation_clear(self, temp_db):
        """Test clearing conversation."""
        pytest.importorskip("aiosqlite")
        from agenticflow.memory.backends import SQLiteConversationBackend
        
        backend = SQLiteConversationBackend(db_path=temp_db)
        await backend.initialize()
        
        await backend.add_message(
            {"role": "user", "content": "Hello"},
            conversation_id="conv1",
        )
        
        await backend.clear(conversation_id="conv1")
        messages = await backend.get_messages(conversation_id="conv1")
        assert len(messages) == 0
        
        await backend.close()


# =============================================================================
# Vector Store Backend Tests
# =============================================================================


class TestFAISSBackend:
    """Tests for FAISS vector store backend."""
    
    @pytest.mark.asyncio
    async def test_import_faiss_backend(self):
        """Test FAISS backend can be imported."""
        faiss = pytest.importorskip("faiss")
        from agenticflow.vectorstore.backends import FAISSBackend
        assert FAISSBackend is not None
    
    @pytest.mark.asyncio
    async def test_faiss_add_and_search(self):
        """Test adding and searching documents."""
        pytest.importorskip("faiss")
        from agenticflow.vectorstore.backends import FAISSBackend
        
        backend = FAISSBackend(dimension=4)
        
        # Add documents
        ids = ["doc1", "doc2", "doc3"]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        metadatas = [{"type": "a"}, {"type": "b"}, {"type": "c"}]
        
        await backend.add(ids, embeddings, metadatas)
        
        # Search
        results = await backend.search([1.0, 0.1, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0][0] == "doc1"  # Closest match
    
    @pytest.mark.asyncio
    async def test_faiss_delete(self):
        """Test deleting documents."""
        pytest.importorskip("faiss")
        from agenticflow.vectorstore.backends import FAISSBackend
        
        backend = FAISSBackend(dimension=4)
        
        ids = ["doc1", "doc2"]
        embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        metadatas = [{"type": "a"}, {"type": "b"}]
        
        await backend.add(ids, embeddings, metadatas)
        assert await backend.count() == 2
        
        await backend.delete(["doc1"])
        assert await backend.count() == 1
    
    @pytest.mark.asyncio
    async def test_faiss_with_filter(self):
        """Test search with metadata filter."""
        pytest.importorskip("faiss")
        from agenticflow.vectorstore.backends import FAISSBackend
        
        backend = FAISSBackend(dimension=4)
        
        ids = ["doc1", "doc2", "doc3"]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
        ]
        metadatas = [
            {"category": "tech"},
            {"category": "science"},
            {"category": "tech"},
        ]
        
        await backend.add(ids, embeddings, metadatas)
        
        # Search with filter
        results = await backend.search(
            [1.0, 0.0, 0.0, 0.0],
            k=3,
            filter={"category": "tech"},
        )
        assert len(results) == 2
        # All results should be tech category
        for id_, score, meta in results:
            assert meta["category"] == "tech"


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
        
        backend = ChromaBackend(
            collection_name="test_collection",
            persist_directory=temp_dir,
        )
        
        ids = ["doc1", "doc2"]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        metadatas = [{"source": "a"}, {"source": "b"}]
        
        await backend.add(ids, embeddings, metadatas)
        
        results = await backend.search([1.0, 0.0, 0.0], k=1)
        assert len(results) == 1
        assert results[0][0] == "doc1"
    
    @pytest.mark.asyncio
    async def test_chroma_delete(self, temp_dir):
        """Test deleting documents."""
        pytest.importorskip("chromadb")
        from agenticflow.vectorstore.backends import ChromaBackend
        
        backend = ChromaBackend(
            collection_name="test_delete",
            persist_directory=temp_dir,
        )
        
        ids = ["doc1", "doc2"]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        metadatas = [{"x": 1}, {"x": 2}]
        
        await backend.add(ids, embeddings, metadatas)
        assert await backend.count() == 2
        
        await backend.delete(["doc1"])
        assert await backend.count() == 1
    
    @pytest.mark.asyncio
    async def test_chroma_get(self, temp_dir):
        """Test getting document by ID."""
        pytest.importorskip("chromadb")
        from agenticflow.vectorstore.backends import ChromaBackend
        
        backend = ChromaBackend(
            collection_name="test_get",
            persist_directory=temp_dir,
        )
        
        ids = ["doc1"]
        embeddings = [[1.0, 0.0, 0.0]]
        metadatas = [{"key": "value"}]
        
        await backend.add(ids, embeddings, metadatas)
        
        result = await backend.get("doc1")
        assert result is not None
        embedding, metadata = result
        assert metadata["key"] == "value"


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
    
    def test_backends_module_exports(self):
        """Test backends module exports correct names."""
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
    
    def test_memory_backends_module_exports(self):
        """Test memory backends module exports."""
        from agenticflow.memory import backends
        
        assert "InMemoryBackend" in backends.__all__
        assert "SQLiteBackend" in backends.__all__
        assert "SQLiteConversationBackend" in backends.__all__


# =============================================================================
# Integration Tests
# =============================================================================


class TestVectorStoreIntegration:
    """Integration tests with VectorStore class."""
    
    @pytest.mark.asyncio
    async def test_vectorstore_with_inmemory(self):
        """Test VectorStore with default InMemory backend."""
        from agenticflow.vectorstore import VectorStore, MockEmbeddings
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
        from agenticflow.vectorstore import VectorStore, MockEmbeddings
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
        from agenticflow.vectorstore import VectorStore, MockEmbeddings
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
