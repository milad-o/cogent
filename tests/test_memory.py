"""Tests for the new Memory-first architecture."""

import pytest

from agenticflow.memory import Memory, InMemoryStore
from agenticflow.memory.tools import create_memory_tools, format_memory_context


class TestInMemoryStore:
    """Tests for InMemoryStore implementation."""

    @pytest.fixture
    def store(self) -> InMemoryStore:
        return InMemoryStore()

    @pytest.mark.asyncio
    async def test_set_and_get(self, store: InMemoryStore) -> None:
        await store.set("key1", "value1")
        result = await store.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, store: InMemoryStore) -> None:
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_default(self, store: InMemoryStore) -> None:
        result = await store.get("nonexistent", default="fallback")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_delete(self, store: InMemoryStore) -> None:
        await store.set("key1", "value1")
        await store.delete("key1")
        result = await store.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_keys(self, store: InMemoryStore) -> None:
        await store.set("a:1", "v1")
        await store.set("a:2", "v2")
        await store.set("b:1", "v3")

        all_keys = await store.keys()
        assert set(all_keys) == {"a:1", "a:2", "b:1"}

        a_keys = await store.keys(prefix="a:")
        assert set(a_keys) == {"a:1", "a:2"}

    @pytest.mark.asyncio
    async def test_clear(self, store: InMemoryStore) -> None:
        await store.set("key1", "value1")
        await store.set("key2", "value2")
        await store.clear()
        keys = await store.keys()
        assert keys == []

    @pytest.mark.asyncio
    async def test_get_many(self, store: InMemoryStore) -> None:
        await store.set("a", "1")
        await store.set("b", "2")
        await store.set("c", "3")

        result = await store.get_many(["a", "c", "nonexistent"])
        assert result == {"a": "1", "c": "3", "nonexistent": None}

    @pytest.mark.asyncio
    async def test_set_many(self, store: InMemoryStore) -> None:
        await store.set_many({"x": "10", "y": "20", "z": "30"})

        assert await store.get("x") == "10"
        assert await store.get("y") == "20"
        assert await store.get("z") == "30"

    @pytest.mark.asyncio
    async def test_stores_complex_types(self, store: InMemoryStore) -> None:
        data = {"name": "test", "values": [1, 2, 3], "nested": {"a": 1}}
        await store.set("complex", data)
        result = await store.get("complex")
        assert result == data


class TestMemory:
    """Tests for Memory class."""

    @pytest.fixture
    def memory(self) -> Memory:
        return Memory()

    @pytest.mark.asyncio
    async def test_remember_and_recall(self, memory: Memory) -> None:
        await memory.remember("fact1", "The sky is blue")
        result = await memory.recall("fact1")
        assert result == "The sky is blue"

    @pytest.mark.asyncio
    async def test_recall_nonexistent(self, memory: Memory) -> None:
        result = await memory.recall("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_forget(self, memory: Memory) -> None:
        await memory.remember("temp", "temporary data")
        await memory.forget("temp")
        result = await memory.recall("temp")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_keys(self, memory: Memory) -> None:
        await memory.remember("topic:a", "data a")
        await memory.remember("topic:b", "data b")
        await memory.remember("other", "other data")

        all_keys = await memory.list()
        assert set(all_keys) == {"topic:a", "topic:b", "other"}

        topic_keys = await memory.list(prefix="topic:")
        assert set(topic_keys) == {"topic:a", "topic:b"}

    @pytest.mark.asyncio
    async def test_clear(self, memory: Memory) -> None:
        await memory.remember("key1", "value1")
        await memory.remember("key2", "value2")
        await memory.clear()
        keys = await memory.list()
        assert keys == []

    @pytest.mark.asyncio
    async def test_batch_remember_and_recall(self, memory: Memory) -> None:
        await memory.remember_many({
            "user:name": "Alice",
            "user:age": "30",
            "user:city": "NYC",
        })

        result = await memory.recall_many(["user:name", "user:city", "missing"])
        assert result == {
            "user:name": "Alice",
            "user:city": "NYC",
            "missing": None,
        }


class TestScopedMemory:
    """Tests for scoped memory isolation."""

    @pytest.fixture
    def memory(self) -> Memory:
        return Memory()

    @pytest.mark.asyncio
    async def test_scoped_isolation(self, memory: Memory) -> None:
        alice = memory.scoped("user:alice")
        bob = memory.scoped("user:bob")

        await alice.remember("preference", "dark mode")
        await bob.remember("preference", "light mode")

        assert await alice.recall("preference") == "dark mode"
        assert await bob.recall("preference") == "light mode"

    @pytest.mark.asyncio
    async def test_scoped_list_only_own_keys(self, memory: Memory) -> None:
        alice = memory.scoped("user:alice")
        bob = memory.scoped("user:bob")

        await alice.remember("a", "1")
        await alice.remember("b", "2")
        await bob.remember("c", "3")

        alice_keys = await alice.list()
        assert set(alice_keys) == {"a", "b"}

        bob_keys = await bob.list()
        assert set(bob_keys) == {"c"}

    @pytest.mark.asyncio
    async def test_scoped_clear_only_own_namespace(self, memory: Memory) -> None:
        alice = memory.scoped("user:alice")
        bob = memory.scoped("user:bob")

        await alice.remember("data", "alice data")
        await bob.remember("data", "bob data")

        await alice.clear()

        assert await alice.recall("data") is None
        assert await bob.recall("data") == "bob data"

    @pytest.mark.asyncio
    async def test_nested_scoping(self, memory: Memory) -> None:
        team = memory.scoped("team:alpha")
        member = team.scoped("member:1")

        await member.remember("role", "developer")

        # Should be accessible via fully qualified key
        result = await memory.recall("team:alpha:member:1:role")
        assert result == "developer"

    @pytest.mark.asyncio
    async def test_batch_operations_on_scoped(self, memory: Memory) -> None:
        user = memory.scoped("user:1")

        await user.remember_many({
            "name": "Alice",
            "email": "alice@example.com",
        })

        result = await user.recall_many(["name", "email"])
        assert result == {"name": "Alice", "email": "alice@example.com"}


class TestMemoryWithCustomStore:
    """Tests for Memory with custom store."""

    @pytest.mark.asyncio
    async def test_shared_store_between_memories(self) -> None:
        shared_store = InMemoryStore()

        memory1 = Memory(store=shared_store, namespace="m1")
        memory2 = Memory(store=shared_store, namespace="m2")

        await memory1.remember("key", "from m1")
        await memory2.remember("key", "from m2")

        # Each sees their own namespaced data
        assert await memory1.recall("key") == "from m1"
        assert await memory2.recall("key") == "from m2"

        # But they share the underlying store
        assert await shared_store.get("m1:key") == "from m1"
        assert await shared_store.get("m2:key") == "from m2"


class TestMemoryTools:
    """Tests for memory tools integration."""

    @pytest.mark.asyncio
    async def test_create_memory_tools(self) -> None:
        memory = Memory()
        tools = create_memory_tools(memory)

        # Should return a list of tool definitions
        assert isinstance(tools, list)
        assert len(tools) >= 2  # At least remember and recall

    def test_format_memory_context_empty(self) -> None:
        result = format_memory_context({})
        assert result == ""

    def test_format_memory_context_with_data(self) -> None:
        memories = {
            "user_name": "Alice",
            "preference": "dark mode",
        }
        result = format_memory_context(memories)
        assert "user_name" in result
        assert "Alice" in result
        assert "preference" in result
        assert "dark mode" in result


class TestMemorySearch:
    """Tests for memory search functionality (when vectorstore is available)."""

    @pytest.fixture
    def memory(self) -> Memory:
        return Memory()

    @pytest.mark.asyncio
    async def test_search_without_vectorstore_raises_error(
        self, memory: Memory
    ) -> None:
        # Without a vectorstore configured, search should raise RuntimeError
        await memory.remember("topic", "some content")
        with pytest.raises(RuntimeError, match="No vectorstore configured"):
            await memory.search("content")


class TestAgenticMemory:
    """Tests for agentic memory functionality."""

    def test_agentic_false_by_default(self) -> None:
        memory = Memory()
        assert memory.agentic is False

    def test_agentic_true_when_set(self) -> None:
        memory = Memory(agentic=True)
        assert memory.agentic is True

    def test_tools_empty_when_not_agentic(self) -> None:
        memory = Memory()  # agentic=False
        assert memory.tools == []

    def test_tools_available_when_agentic(self) -> None:
        memory = Memory(agentic=True)
        tools = memory.tools
        
        assert len(tools) == 4
        tool_names = {t.name for t in tools}
        assert tool_names == {"remember", "recall", "forget", "search_memories"}

    def test_tools_cached(self) -> None:
        memory = Memory(agentic=True)
        tools1 = memory.tools
        tools2 = memory.tools
        
        # Should return same list instance (cached)
        assert tools1 is tools2

    @pytest.mark.asyncio
    async def test_agentic_tools_work(self) -> None:
        memory = Memory(agentic=True)
        tools = memory.tools
        
        # Find remember and recall tools
        remember = next(t for t in tools if t.name == "remember")
        recall = next(t for t in tools if t.name == "recall")
        
        # Use them
        result = await remember.func(key="name", value="Alice")
        assert "Remembered" in result
        
        result = await recall.func(key="name")
        assert "Alice" in result
