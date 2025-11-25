"""Tests for the memory module."""

import pytest
from datetime import datetime

from agenticflow.memory import (
    MemoryManager,
    ShortTermMemory,
    LongTermMemory,
    SharedMemory,
    MemoryEntry,
    MemoryType,
    MemoryConfig,
)


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_create_memory_entry(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            content={"data": "test"},
            namespace="test-ns",
        )
        assert entry.content == {"data": "test"}
        assert entry.namespace == "test-ns"
        assert entry.id is not None

    def test_memory_entry_to_dict(self):
        """Test converting entry to dictionary."""
        entry = MemoryEntry(
            content="test-value",
            namespace="custom",
            metadata={"key": "value"},
        )
        data = entry.to_dict()
        assert data["content"] == "test-value"
        assert data["namespace"] == "custom"
        assert data["metadata"] == {"key": "value"}

    def test_memory_entry_from_dict(self):
        """Test creating entry from dictionary."""
        data = {
            "content": "test",
            "namespace": "ns1",
            "timestamp": "2024-01-01T00:00:00",
        }
        entry = MemoryEntry.from_dict(data)
        assert entry.content == "test"
        assert entry.namespace == "ns1"


class TestShortTermMemory:
    """Tests for ShortTermMemory."""

    @pytest.fixture
    def memory(self):
        """Create short-term memory instance."""
        config = MemoryConfig(max_entries=10, memory_type=MemoryType.SHORT_TERM)
        return ShortTermMemory(thread_id="test-thread", config=config)

    @pytest.mark.asyncio
    async def test_add_and_search(self, memory):
        """Test adding and searching entries."""
        entry = await memory.add("value1", metadata={"key": "k1"})
        assert entry.content == "value1"

        results = await memory.search()
        assert len(results) >= 1
        assert any(r.content == "value1" for r in results)

    @pytest.mark.asyncio
    async def test_thread_isolation(self, memory):
        """Test that different threads are isolated."""
        await memory.add("value1")
        memory.switch_thread("thread-2")
        await memory.add("value2")

        # Switch back to original thread
        memory.switch_thread("test-thread")
        results = await memory.search()
        assert any(r.content == "value1" for r in results)

    @pytest.mark.asyncio
    async def test_delete(self, memory):
        """Test deleting from memory."""
        entry = await memory.add("value1")
        deleted = await memory.delete(entry.id)
        assert deleted is True

        result = await memory.get(entry.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_clear(self, memory):
        """Test clearing memory."""
        await memory.add("value1")
        await memory.add("value2")
        count = await memory.clear()
        assert count == 2

        results = await memory.search()
        assert len(results) == 0


class TestLongTermMemory:
    """Tests for LongTermMemory."""

    @pytest.fixture
    def memory(self):
        """Create long-term memory instance."""
        config = MemoryConfig(memory_type=MemoryType.LONG_TERM)
        return LongTermMemory(config=config)

    @pytest.mark.asyncio
    async def test_add_and_get(self, memory):
        """Test adding and getting entries."""
        entry = await memory.add({"data": "test"}, namespace="test")
        assert entry.content == {"data": "test"}

        retrieved = await memory.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == {"data": "test"}

    @pytest.mark.asyncio
    async def test_search_by_namespace(self, memory):
        """Test searching by namespace."""
        await memory.add({"name": "Alice"}, namespace="users")
        await memory.add({"name": "Bob"}, namespace="users")
        await memory.add({"setting": "value"}, namespace="config")

        results = await memory.search(namespace="users")
        assert len(results) == 2


class TestSharedMemory:
    """Tests for SharedMemory."""

    @pytest.fixture
    def memory(self):
        """Create shared memory instance."""
        return SharedMemory()

    @pytest.mark.asyncio
    async def test_add_and_search(self, memory):
        """Test adding and searching shared memory."""
        entry = await memory.add("shared-value", namespace="shared")
        assert entry.content == "shared-value"

        results = await memory.search(namespace="shared")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_cross_agent_access(self, memory):
        """Test that shared memory is accessible across agents."""
        # Agent 1 adds
        entry = await memory.add("shared-data", metadata={"agent": "agent-1"})

        # Any agent can read
        result = await memory.get(entry.id)
        assert result is not None
        assert result.content == "shared-data"


class TestMemoryManager:
    """Tests for MemoryManager."""

    @pytest.fixture
    def manager(self):
        """Create memory manager."""
        return MemoryManager()

    @pytest.mark.asyncio
    async def test_short_term_operations(self, manager):
        """Test short-term memory through manager."""
        entry = await manager.short_term.add("value1")
        assert entry.content == "value1"

        results = await manager.short_term.search()
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_long_term_operations(self, manager):
        """Test long-term memory through manager."""
        entry = await manager.long_term.add("value1")
        assert entry.content == "value1"

    @pytest.mark.asyncio
    async def test_shared_operations(self, manager):
        """Test shared memory through manager."""
        entry = await manager.shared.add("value1")
        assert entry.content == "value1"
