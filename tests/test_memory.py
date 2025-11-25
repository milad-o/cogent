"""Tests for the memory module - minimal LangChain/LangGraph wrappers."""

import pytest

from agenticflow.memory import (
    # Checkpointers
    create_checkpointer,
    memory_checkpointer,
    MemorySaver,
    BaseCheckpointSaver,
    # Stores
    create_store,
    memory_store,
    BaseStore,
    InMemoryStore,
    # Vector stores
    create_vectorstore,
    VectorStore,
    InMemoryVectorStore,
    Document,
)


class TestCheckpointers:
    """Tests for checkpointer factory functions."""

    def test_memory_checkpointer(self):
        """Test creating in-memory checkpointer."""
        checkpointer = memory_checkpointer()
        assert checkpointer is not None
        assert isinstance(checkpointer, MemorySaver)
        assert isinstance(checkpointer, BaseCheckpointSaver)

    def test_create_checkpointer_memory(self):
        """Test create_checkpointer with memory backend."""
        checkpointer = create_checkpointer("memory")
        assert isinstance(checkpointer, MemorySaver)

    def test_create_checkpointer_invalid_backend(self):
        """Test create_checkpointer with invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_checkpointer("invalid")

    def test_sqlite_checkpointer_requires_package(self):
        """Test that sqlite checkpointer requires optional package."""
        # This may succeed if package is installed, or fail with ImportError
        try:
            from agenticflow.memory import sqlite_checkpointer
            # If import succeeds, the function should work
            # (but we don't actually create it as it needs file access)
        except ImportError:
            pass  # Expected if package not installed

    def test_postgres_checkpointer_requires_package(self):
        """Test that postgres checkpointer requires optional package."""
        try:
            from agenticflow.memory import postgres_checkpointer
        except ImportError:
            pass  # Expected if package not installed


class TestStores:
    """Tests for store factory functions."""

    def test_memory_store(self):
        """Test creating in-memory store."""
        store = memory_store()
        assert store is not None
        assert isinstance(store, InMemoryStore)
        assert isinstance(store, BaseStore)

    def test_create_store_memory(self):
        """Test create_store with memory backend."""
        store = create_store("memory")
        assert isinstance(store, InMemoryStore)

    def test_create_store_invalid_backend(self):
        """Test create_store with invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_store("invalid")

    def test_store_put_and_get(self):
        """Test basic store put and get operations."""
        store = memory_store()

        # Put a value
        store.put(("user", "prefs"), "theme", {"value": "dark"})

        # Get it back
        items = list(store.search(("user", "prefs")))
        assert len(items) == 1
        assert items[0].value == {"value": "dark"}

    def test_store_namespacing(self):
        """Test that store namespaces work correctly."""
        store = memory_store()

        # Put in different namespaces
        store.put(("user1", "prefs"), "theme", {"value": "dark"})
        store.put(("user2", "prefs"), "theme", {"value": "light"})

        # Each namespace is isolated
        user1_items = list(store.search(("user1", "prefs")))
        user2_items = list(store.search(("user2", "prefs")))

        assert len(user1_items) == 1
        assert len(user2_items) == 1
        assert user1_items[0].value["value"] == "dark"
        assert user2_items[0].value["value"] == "light"

    def test_store_delete(self):
        """Test store delete operation."""
        store = memory_store()

        store.put(("ns",), "key1", {"data": "test"})
        items = list(store.search(("ns",)))
        assert len(items) == 1

        store.delete(("ns",), "key1")
        items = list(store.search(("ns",)))
        assert len(items) == 0


class TestVectorStores:
    """Tests for vector store factory functions."""

    def test_memory_vectorstore_requires_embeddings(self):
        """Test that memory vectorstore requires embeddings."""
        with pytest.raises(ValueError, match="embeddings required"):
            create_vectorstore("memory")

    def test_create_vectorstore_invalid_backend(self):
        """Test create_vectorstore with invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_vectorstore("invalid")

    def test_document_creation(self):
        """Test Document can be created."""
        doc = Document(page_content="Hello world", metadata={"source": "test"})
        assert doc.page_content == "Hello world"
        assert doc.metadata["source"] == "test"


class TestMemoryTypes:
    """Tests for understanding memory type concepts."""

    def test_short_term_concept(self):
        """Short-term = Checkpointer = thread-scoped."""
        # Create checkpointer (ephemeral short-term memory)
        checkpointer = memory_checkpointer()

        # Short-term memory is thread-scoped
        # Each thread_id has its own conversation state
        # This is managed by LangGraph when you compile with checkpointer
        assert checkpointer is not None

    def test_long_term_concept(self):
        """Long-term = Store = cross-thread, namespaced."""
        # Create store (long-term memory)
        store = memory_store()

        # Long-term memory persists across threads
        # Namespaced by user_id, assistant_id, etc.
        store.put(("user_123", "memories"), "fact_1", {"text": "User likes pizza"})
        store.put(("user_123", "memories"), "fact_2", {"text": "User is an engineer"})

        # Can retrieve across any thread
        memories = list(store.search(("user_123", "memories")))
        assert len(memories) == 2

    def test_ephemeral_vs_persistent(self):
        """Test ephemeral (memory) vs persistent (sqlite/postgres) concept."""
        # Ephemeral: lost on restart
        ephemeral_checkpointer = memory_checkpointer()
        ephemeral_store = memory_store()

        # Persistent requires optional packages:
        # - sqlite_checkpointer("file.db")  # survives restart
        # - postgres_checkpointer("conn_string")  # production

        assert ephemeral_checkpointer is not None
        assert ephemeral_store is not None
