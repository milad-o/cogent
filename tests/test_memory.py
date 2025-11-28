"""Tests for the memory system."""

import pytest

from agenticflow.core.messages import HumanMessage, AIMessage
from agenticflow.memory import (
    ConversationMemory,
    UserMemory,
    TeamMemory,
    WorkingMemory,
    InMemoryBackend,
    InMemoryConversationBackend,
    MemoryScope,
    MemoryConfig,
    MemoryManager,
    create_memory,
    create_default_memory,
    create_memory_tools,
    get_memory_prompt_addition,
    format_memory_context,
)


# =============================================================================
# InMemoryBackend Tests
# =============================================================================


class TestInMemoryBackend:
    """Tests for InMemoryBackend."""

    @pytest.fixture
    def backend(self) -> InMemoryBackend:
        return InMemoryBackend()

    @pytest.mark.asyncio
    async def test_set_and_get(self, backend: InMemoryBackend) -> None:
        """Test basic set and get operations."""
        await backend.set("key1", "value1", MemoryScope.USER, "user-1")
        result = await backend.get("key1", MemoryScope.USER, "user-1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(
        self, backend: InMemoryBackend
    ) -> None:
        """Test getting a nonexistent key returns None."""
        result = await backend.get("nonexistent", MemoryScope.USER, "user-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_scope_isolation(self, backend: InMemoryBackend) -> None:
        """Test that different scopes are isolated."""
        await backend.set("key", "user_value", MemoryScope.USER, "id-1")
        await backend.set("key", "thread_value", MemoryScope.THREAD, "id-1")

        user_result = await backend.get("key", MemoryScope.USER, "id-1")
        thread_result = await backend.get("key", MemoryScope.THREAD, "id-1")

        assert user_result == "user_value"
        assert thread_result == "thread_value"

    @pytest.mark.asyncio
    async def test_scope_id_isolation(self, backend: InMemoryBackend) -> None:
        """Test that different scope IDs are isolated."""
        await backend.set("key", "user1_value", MemoryScope.USER, "user-1")
        await backend.set("key", "user2_value", MemoryScope.USER, "user-2")

        result1 = await backend.get("key", MemoryScope.USER, "user-1")
        result2 = await backend.get("key", MemoryScope.USER, "user-2")

        assert result1 == "user1_value"
        assert result2 == "user2_value"

    @pytest.mark.asyncio
    async def test_delete(self, backend: InMemoryBackend) -> None:
        """Test delete operation."""
        await backend.set("key", "value", MemoryScope.USER, "user-1")
        deleted = await backend.delete("key", MemoryScope.USER, "user-1")
        assert deleted is True

        result = await backend.get("key", MemoryScope.USER, "user-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, backend: InMemoryBackend) -> None:
        """Test deleting nonexistent key returns False."""
        deleted = await backend.delete("nonexistent", MemoryScope.USER, "user-1")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_keys(self, backend: InMemoryBackend) -> None:
        """Test listing keys."""
        await backend.set("key1", "v1", MemoryScope.USER, "user-1")
        await backend.set("key2", "v2", MemoryScope.USER, "user-1")
        await backend.set("key3", "v3", MemoryScope.USER, "user-2")  # Different user

        keys = await backend.list_keys(MemoryScope.USER, "user-1")
        assert set(keys) == {"key1", "key2"}

    @pytest.mark.asyncio
    async def test_clear(self, backend: InMemoryBackend) -> None:
        """Test clearing a scope."""
        await backend.set("key1", "v1", MemoryScope.USER, "user-1")
        await backend.set("key2", "v2", MemoryScope.USER, "user-1")
        await backend.clear(MemoryScope.USER, "user-1")

        keys = await backend.list_keys(MemoryScope.USER, "user-1")
        assert keys == []

    @pytest.mark.asyncio
    async def test_search(self, backend: InMemoryBackend) -> None:
        """Test searching entries."""
        await backend.set("name", "Alice", MemoryScope.USER, "user-1")
        await backend.set("preference", "dark mode", MemoryScope.USER, "user-1")
        await backend.set("language", "Python", MemoryScope.USER, "user-1")

        results = await backend.search("dark", MemoryScope.USER, "user-1")
        assert len(results) == 1
        assert results[0].value == "dark mode"

    @pytest.mark.asyncio
    async def test_complex_values(self, backend: InMemoryBackend) -> None:
        """Test storing complex values."""
        data = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        await backend.set("complex", data, MemoryScope.USER, "user-1")

        result = await backend.get("complex", MemoryScope.USER, "user-1")
        assert result == data


# =============================================================================
# InMemoryConversationBackend Tests
# =============================================================================


class TestInMemoryConversationBackend:
    """Tests for InMemoryConversationBackend."""

    @pytest.fixture
    def backend(self) -> InMemoryConversationBackend:
        return InMemoryConversationBackend()

    @pytest.mark.asyncio
    async def test_add_and_get_messages(
        self, backend: InMemoryConversationBackend
    ) -> None:
        """Test adding and retrieving messages."""
        msg1 = HumanMessage(content="Hello")
        msg2 = AIMessage(content="Hi there!")

        await backend.add_message("thread-1", msg1)
        await backend.add_message("thread-1", msg2)

        messages = await backend.get_messages("thread-1")
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_get_messages_with_limit(
        self, backend: InMemoryConversationBackend
    ) -> None:
        """Test getting messages with limit."""
        for i in range(5):
            await backend.add_message("thread-1", HumanMessage(content=f"Msg {i}"))

        messages = await backend.get_messages("thread-1", limit=3)
        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_get_recent_messages(
        self, backend: InMemoryConversationBackend
    ) -> None:
        """Test getting recent messages."""
        for i in range(5):
            await backend.add_message("thread-1", HumanMessage(content=f"Msg {i}"))

        messages = await backend.get_recent_messages("thread-1", limit=2)
        assert len(messages) == 2
        assert messages[0].content == "Msg 3"
        assert messages[1].content == "Msg 4"

    @pytest.mark.asyncio
    async def test_thread_isolation(
        self, backend: InMemoryConversationBackend
    ) -> None:
        """Test that threads are isolated."""
        await backend.add_message("thread-1", HumanMessage(content="Thread 1"))
        await backend.add_message("thread-2", HumanMessage(content="Thread 2"))

        msgs1 = await backend.get_messages("thread-1")
        msgs2 = await backend.get_messages("thread-2")

        assert len(msgs1) == 1
        assert len(msgs2) == 1
        assert msgs1[0].content == "Thread 1"
        assert msgs2[0].content == "Thread 2"

    @pytest.mark.asyncio
    async def test_clear_thread(self, backend: InMemoryConversationBackend) -> None:
        """Test clearing a thread."""
        await backend.add_message("thread-1", HumanMessage(content="Hello"))
        await backend.clear_thread("thread-1")

        messages = await backend.get_messages("thread-1")
        assert messages == []

    @pytest.mark.asyncio
    async def test_message_count(self, backend: InMemoryConversationBackend) -> None:
        """Test getting message count."""
        for i in range(3):
            await backend.add_message("thread-1", HumanMessage(content=f"Msg {i}"))

        count = await backend.get_message_count("thread-1")
        assert count == 3

    @pytest.mark.asyncio
    async def test_list_threads(self, backend: InMemoryConversationBackend) -> None:
        """Test listing threads."""
        await backend.add_message("thread-1", HumanMessage(content="T1"))
        await backend.add_message("thread-2", HumanMessage(content="T2"))

        threads = await backend.list_threads()
        assert set(threads) == {"thread-1", "thread-2"}


# =============================================================================
# ConversationMemory Tests
# =============================================================================


class TestConversationMemory:
    """Tests for ConversationMemory."""

    @pytest.fixture
    def memory(self) -> ConversationMemory:
        return ConversationMemory(max_messages=5)

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, memory: ConversationMemory) -> None:
        """Test basic message operations."""
        msg1 = HumanMessage(content="Hello")
        msg2 = AIMessage(content="Hi!")

        await memory.add_message("thread-1", msg1)
        await memory.add_message("thread-1", msg2)

        messages = await memory.get_messages("thread-1")
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi!"

    @pytest.mark.asyncio
    async def test_sliding_window(self, memory: ConversationMemory) -> None:
        """Test that old messages are trimmed."""
        for i in range(7):
            await memory.add_message("thread-1", HumanMessage(content=f"Msg {i}"))

        messages = await memory.get_messages("thread-1")
        assert len(messages) == 5
        # Should have the most recent 5 messages
        assert messages[0].content == "Msg 2"
        assert messages[-1].content == "Msg 6"

    @pytest.mark.asyncio
    async def test_get_recent_messages(self, memory: ConversationMemory) -> None:
        """Test getting recent messages."""
        for i in range(5):
            await memory.add_message("thread-1", HumanMessage(content=f"Msg {i}"))

        recent = await memory.get_recent_messages("thread-1", limit=2)
        assert len(recent) == 2
        assert recent[0].content == "Msg 3"
        assert recent[1].content == "Msg 4"

    @pytest.mark.asyncio
    async def test_clear_thread(self, memory: ConversationMemory) -> None:
        """Test clearing a thread."""
        await memory.add_message("thread-1", HumanMessage(content="Hello"))
        await memory.clear(thread_id="thread-1")

        messages = await memory.get_messages("thread-1")
        assert messages == []

    @pytest.mark.asyncio
    async def test_list_threads(self, memory: ConversationMemory) -> None:
        """Test listing threads."""
        await memory.add_message("thread-1", HumanMessage(content="T1"))
        await memory.add_message("thread-2", HumanMessage(content="T2"))

        threads = await memory.list_threads()
        assert set(threads) == {"thread-1", "thread-2"}


# =============================================================================
# UserMemory Tests
# =============================================================================


class TestUserMemory:
    """Tests for UserMemory."""

    @pytest.fixture
    def memory(self) -> UserMemory:
        return UserMemory()

    @pytest.mark.asyncio
    async def test_save_and_load(self, memory: UserMemory) -> None:
        """Test basic save and load."""
        await memory.save("name", "Alice", user_id="user-1")
        result = await memory.load("name", user_id="user-1")
        assert result == "Alice"

    @pytest.mark.asyncio
    async def test_user_isolation(self, memory: UserMemory) -> None:
        """Test that users are isolated."""
        await memory.save("name", "Alice", user_id="user-1")
        await memory.save("name", "Bob", user_id="user-2")

        name1 = await memory.load("name", user_id="user-1")
        name2 = await memory.load("name", user_id="user-2")

        assert name1 == "Alice"
        assert name2 == "Bob"

    @pytest.mark.asyncio
    async def test_delete(self, memory: UserMemory) -> None:
        """Test deleting a fact."""
        await memory.save("name", "Alice", user_id="user-1")
        deleted = await memory.delete("name", user_id="user-1")
        assert deleted is True

        result = await memory.load("name", user_id="user-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_facts(self, memory: UserMemory) -> None:
        """Test listing facts."""
        await memory.save("name", "Alice", user_id="user-1")
        await memory.save("language", "Python", user_id="user-1")

        facts = await memory.list_facts("user-1")
        assert set(facts) == {"name", "language"}

    @pytest.mark.asyncio
    async def test_get_all_facts(self, memory: UserMemory) -> None:
        """Test getting all facts."""
        await memory.save("name", "Alice", user_id="user-1")
        await memory.save("language", "Python", user_id="user-1")

        all_facts = await memory.get_all_facts("user-1")
        assert all_facts == {"name": "Alice", "language": "Python"}

    @pytest.mark.asyncio
    async def test_search(self, memory: UserMemory) -> None:
        """Test searching facts."""
        await memory.save("name", "Alice", user_id="user-1")
        await memory.save("preference", "dark mode", user_id="user-1")

        results = await memory.search("dark", user_id="user-1")
        assert len(results) == 1
        assert results[0].value == "dark mode"

    @pytest.mark.asyncio
    async def test_has_fact(self, memory: UserMemory) -> None:
        """Test checking if fact exists."""
        await memory.save("name", "Alice", user_id="user-1")

        assert await memory.has_fact("name", user_id="user-1") is True
        assert await memory.has_fact("age", user_id="user-1") is False

    @pytest.mark.asyncio
    async def test_update_with_merge(self, memory: UserMemory) -> None:
        """Test updating with merge."""
        await memory.save("prefs", {"theme": "dark"}, user_id="user-1")
        await memory.update("prefs", {"font": "mono"}, user_id="user-1", merge=True)

        result = await memory.load("prefs", user_id="user-1")
        assert result == {"theme": "dark", "font": "mono"}


# =============================================================================
# TeamMemory Tests
# =============================================================================


class TestTeamMemory:
    """Tests for TeamMemory."""

    @pytest.fixture
    def memory(self) -> TeamMemory:
        return TeamMemory(team_id="test-team")

    @pytest.mark.asyncio
    async def test_save_and_load(self, memory: TeamMemory) -> None:
        """Test basic save and load."""
        await memory.save("findings", {"topic": "AI"})
        result = await memory.load("findings")
        assert result == {"topic": "AI"}

    @pytest.mark.asyncio
    async def test_team_id(self, memory: TeamMemory) -> None:
        """Test team ID property."""
        assert memory.team_id == "test-team"

    @pytest.mark.asyncio
    async def test_team_isolation(self) -> None:
        """Test that different teams are isolated."""
        team1 = TeamMemory(team_id="team-1")
        team2 = TeamMemory(team_id="team-2")

        await team1.save("data", "team1_data")
        await team2.save("data", "team2_data")

        result1 = await team1.load("data")
        result2 = await team2.load("data")

        assert result1 == "team1_data"
        assert result2 == "team2_data"

    @pytest.mark.asyncio
    async def test_append_to_list(self, memory: TeamMemory) -> None:
        """Test appending to a list."""
        await memory.append_to_list("items", "item1")
        await memory.append_to_list("items", "item2")

        result = await memory.load("items")
        assert result == ["item1", "item2"]

    @pytest.mark.asyncio
    async def test_merge_dict(self, memory: TeamMemory) -> None:
        """Test merging dicts."""
        await memory.save("config", {"a": 1})
        await memory.merge_dict("config", {"b": 2})

        result = await memory.load("config")
        assert result == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_increment(self, memory: TeamMemory) -> None:
        """Test incrementing a counter."""
        result1 = await memory.increment("counter")
        result2 = await memory.increment("counter")
        result3 = await memory.increment("counter", 5)

        assert result1 == 1
        assert result2 == 2
        assert result3 == 7

    @pytest.mark.asyncio
    async def test_report_status(self, memory: TeamMemory) -> None:
        """Test reporting agent status."""
        await memory.report_status("researcher", "working")
        await memory.report_status("writer", "waiting")

        statuses = await memory.get_agent_statuses()
        assert statuses == {"researcher": "working", "writer": "waiting"}

    @pytest.mark.asyncio
    async def test_share_result(self, memory: TeamMemory) -> None:
        """Test sharing agent results."""
        await memory.share_result("researcher", {"findings": "AI trends"})
        await memory.share_result("writer", {"draft": "Article..."})

        results = await memory.get_agent_results()
        assert results == {
            "researcher": {"findings": "AI trends"},
            "writer": {"draft": "Article..."},
        }


# =============================================================================
# WorkingMemory Tests
# =============================================================================


class TestWorkingMemory:
    """Tests for WorkingMemory."""

    @pytest.fixture
    def memory(self) -> WorkingMemory:
        return WorkingMemory(execution_id="exec-1", auto_cleanup=False)

    @pytest.mark.asyncio
    async def test_save_and_load(self, memory: WorkingMemory) -> None:
        """Test basic save and load."""
        await memory.save("draft", "First draft...")
        result = await memory.load("draft")
        assert result == "First draft..."

    @pytest.mark.asyncio
    async def test_get_with_default(self, memory: WorkingMemory) -> None:
        """Test get with default."""
        result = await memory.get("nonexistent", "default_value")
        assert result == "default_value"

    @pytest.mark.asyncio
    async def test_execution_isolation(self) -> None:
        """Test that different executions are isolated."""
        exec1 = WorkingMemory(execution_id="exec-1", auto_cleanup=False)
        exec2 = WorkingMemory(execution_id="exec-2", auto_cleanup=False)

        await exec1.save("data", "exec1_data")
        await exec2.save("data", "exec2_data")

        result1 = await exec1.load("data")
        result2 = await exec2.load("data")

        assert result1 == "exec1_data"
        assert result2 == "exec2_data"

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self) -> None:
        """Test auto cleanup in context manager."""
        async with WorkingMemory(execution_id="exec-ctx") as working:
            await working.save("temp", "temporary data")
            assert await working.load("temp") == "temporary data"

        # After context exit, should be cleared
        # Note: We can't really test this since the memory is scoped
        # and cleared on exit, but we can verify no errors occur

    @pytest.mark.asyncio
    async def test_append(self, memory: WorkingMemory) -> None:
        """Test appending to a list."""
        await memory.append("log", "step 1")
        await memory.append("log", "step 2")

        result = await memory.load("log")
        assert result == ["step 1", "step 2"]

    @pytest.mark.asyncio
    async def test_update(self, memory: WorkingMemory) -> None:
        """Test updating a dict."""
        await memory.save("state", {"phase": "init"})
        await memory.update("state", {"progress": 50})

        result = await memory.load("state")
        assert result == {"phase": "init", "progress": 50}

    @pytest.mark.asyncio
    async def test_set_if_absent(self, memory: WorkingMemory) -> None:
        """Test set if absent."""
        result1 = await memory.set_if_absent("key", "value1")
        result2 = await memory.set_if_absent("key", "value2")

        assert result1 is True
        assert result2 is False
        assert await memory.load("key") == "value1"

    @pytest.mark.asyncio
    async def test_type_safe_getters(self, memory: WorkingMemory) -> None:
        """Test type-safe getters."""
        await memory.save("count", 42)
        await memory.save("name", "test")
        await memory.save("items", [1, 2, 3])
        await memory.save("config", {"key": "value"})

        assert await memory.get_int("count") == 42
        assert await memory.get_str("name") == "test"
        assert await memory.get_list("items") == [1, 2, 3]
        assert await memory.get_dict("config") == {"key": "value"}

        # Test defaults
        assert await memory.get_int("missing") == 0
        assert await memory.get_str("missing") == ""
        assert await memory.get_list("missing") == []
        assert await memory.get_dict("missing") == {}


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateDefaultMemory:
    """Tests for create_default_memory factory."""

    def test_creates_conversation_memory(self) -> None:
        """Test that default memory is ConversationMemory."""
        memory = create_default_memory()
        assert isinstance(memory, ConversationMemory)


# =============================================================================
# MemoryManager Tests
# =============================================================================


class TestMemoryManager:
    """Tests for MemoryManager."""

    def test_default_creation(self) -> None:
        """Test default MemoryManager creation."""
        manager = MemoryManager()
        assert manager.has_short_term is True
        assert manager.has_long_term is False
        assert manager.has_team is False
        assert manager.has_working is False

    def test_full_creation(self) -> None:
        """Test MemoryManager with all features."""
        manager = MemoryManager.full()
        assert manager.has_short_term is True
        assert manager.has_long_term is True
        assert manager.has_team is True
        assert manager.has_working is True

    def test_custom_creation(self) -> None:
        """Test custom MemoryManager creation."""
        manager = MemoryManager(
            short_term=True,
            long_term=True,
            team=False,
            working=False,
            max_messages=100,
        )
        assert manager.has_short_term is True
        assert manager.has_long_term is True
        assert manager.has_team is False
        assert manager.has_working is False
        assert manager.config.max_messages == 100

    @pytest.mark.asyncio
    async def test_conversation_operations(self) -> None:
        """Test conversation memory operations via manager."""
        manager = MemoryManager(short_term=True)

        msg1 = HumanMessage(content="Hello")
        msg2 = AIMessage(content="Hi!")

        await manager.add_message("thread-1", msg1)
        await manager.add_message("thread-1", msg2)

        messages = await manager.get_messages("thread-1")
        assert len(messages) == 2
        assert messages[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_long_term_operations(self) -> None:
        """Test long-term memory operations via manager."""
        manager = MemoryManager(long_term=True)

        await manager.remember("name", "Alice", user_id="user-1")
        await manager.remember("color", "blue", user_id="user-1")

        name = await manager.recall("name", user_id="user-1")
        assert name == "Alice"

        facts = await manager.get_user_facts("user-1")
        assert facts == {"name": "Alice", "color": "blue"}

    @pytest.mark.asyncio
    async def test_forget_operation(self) -> None:
        """Test forgetting a fact."""
        manager = MemoryManager(long_term=True)

        await manager.remember("secret", "hidden", user_id="user-1")
        assert await manager.recall("secret", user_id="user-1") == "hidden"

        success = await manager.forget("secret", user_id="user-1")
        assert success is True
        assert await manager.recall("secret", user_id="user-1") is None

    @pytest.mark.asyncio
    async def test_context_for_prompt(self) -> None:
        """Test generating context for prompts."""
        manager = MemoryManager(short_term=True, long_term=True)

        # Add some data
        await manager.remember("preference", "dark mode", user_id="user-1")
        await manager.add_message("thread-1", HumanMessage(content="Hello"))

        context = await manager.get_context_for_prompt(
            thread_id="thread-1",
            user_id="user-1",
        )

        assert "preference: dark mode" in context
        assert "Conversation context" in context


class TestCreateMemory:
    """Tests for create_memory factory function."""

    def test_none_returns_none(self) -> None:
        """Test that None returns None."""
        assert create_memory(None) is None
        assert create_memory(False) is None

    def test_true_returns_default(self) -> None:
        """Test that True returns default manager."""
        manager = create_memory(True)
        assert isinstance(manager, MemoryManager)
        assert manager.has_short_term is True

    def test_manager_passthrough(self) -> None:
        """Test that MemoryManager passes through."""
        original = MemoryManager(long_term=True)
        result = create_memory(original)
        assert result is original

    def test_config_creates_manager(self) -> None:
        """Test that MemoryConfig creates manager."""
        config = MemoryConfig(short_term=True, long_term=True)
        manager = create_memory(config)
        assert isinstance(manager, MemoryManager)
        assert manager.has_long_term is True

    def test_dict_creates_manager(self) -> None:
        """Test that dict creates manager."""
        manager = create_memory({"short_term": True, "long_term": True})
        assert isinstance(manager, MemoryManager)
        assert manager.has_long_term is True


class TestMemoryTools:
    """Tests for memory tools."""

    @pytest.mark.asyncio
    async def test_create_tools_with_long_term(self) -> None:
        """Test creating tools when long-term is enabled."""
        manager = MemoryManager(long_term=True)
        tools = create_memory_tools(manager, user_id="user-1")
        
        # Should have 4 tools: remember, recall, forget, search_memories
        assert len(tools) == 4
        
        tool_names = [t.name for t in tools]
        assert "remember" in tool_names
        assert "recall" in tool_names
        assert "forget" in tool_names
        assert "search_memories" in tool_names

    def test_no_tools_without_long_term(self) -> None:
        """Test that no tools are created without long-term memory."""
        manager = MemoryManager(short_term=True, long_term=False)
        tools = create_memory_tools(manager)
        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_save_and_recall_tools(self) -> None:
        """Test save and recall tool execution."""
        manager = MemoryManager(long_term=True)
        tools = create_memory_tools(manager, user_id="user-1")
        
        remember_tool = next(t for t in tools if t.name == "remember")
        recall_tool = next(t for t in tools if t.name == "recall")
        
        # Save
        result = await remember_tool.ainvoke({"key": "name", "value": "Bob"})
        assert "Remembered" in result
        
        # Recall
        result = await recall_tool.ainvoke({"key": "name"})
        assert "Bob" in result


class TestMemoryPrompts:
    """Tests for memory prompt utilities."""

    def test_prompt_addition_with_tools(self) -> None:
        """Test getting prompt addition with tools."""
        prompt = get_memory_prompt_addition(has_tools=True)
        assert "remember" in prompt
        assert "recall" in prompt

    def test_prompt_addition_without_tools(self) -> None:
        """Test getting prompt addition without tools."""
        prompt = get_memory_prompt_addition(has_tools=False)
        assert prompt == ""

    def test_format_memory_context(self) -> None:
        """Test formatting memory context."""
        context = format_memory_context({"name": "Alice", "preference": "dark mode"})
        assert "User Context" in context
        assert "name: Alice" in context

    def test_format_empty_context(self) -> None:
        """Test formatting empty context."""
        assert format_memory_context("") == ""
        assert format_memory_context({}) == ""


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryIntegration:
    """Integration tests for memory system."""

    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self) -> None:
        """Test a multi-agent workflow with team memory."""
        team = TeamMemory(team_id="content-team")

        # Researcher stores findings
        await team.report_status("researcher", "working")
        await team.share_result(
            "researcher",
            {
                "topic": "AI in 2024",
                "sources": ["arxiv", "nature"],
                "key_points": ["LLMs", "Agents", "RAG"],
            },
        )
        await team.report_status("researcher", "done")

        # Writer reads findings and creates draft
        findings = await team.get_agent_results()
        assert "researcher" in findings

        await team.report_status("writer", "working")
        await team.share_result(
            "writer",
            {
                "draft": "AI has evolved significantly...",
                "word_count": 500,
            },
        )
        await team.report_status("writer", "done")

        # Verify all statuses
        statuses = await team.get_agent_statuses()
        assert statuses == {"researcher": "done", "writer": "done"}

    @pytest.mark.asyncio
    async def test_user_across_conversations(self) -> None:
        """Test user memory persisting across conversations."""
        user_mem = UserMemory()
        conv_mem = ConversationMemory()

        # First conversation
        await conv_mem.add_message(
            "thread-1", HumanMessage(content="My name is Alice")
        )
        await user_mem.save("name", "Alice", user_id="user-123")

        # Second conversation (different thread)
        await conv_mem.add_message("thread-2", HumanMessage(content="What's my name?"))

        # User memory still has the name
        name = await user_mem.load("name", user_id="user-123")
        assert name == "Alice"

        # But conversation history is separate
        thread1_msgs = await conv_mem.get_messages("thread-1")
        thread2_msgs = await conv_mem.get_messages("thread-2")
        assert len(thread1_msgs) == 1
        assert len(thread2_msgs) == 1


class TestMemoryManagerAddMessages:
    """Tests for MemoryManager.add_messages method."""

    @pytest.mark.asyncio
    async def test_add_messages_single(self) -> None:
        """Test adding a single message via add_messages."""
        manager = MemoryManager()
        
        msg = HumanMessage(content="Hello")
        await manager.add_messages("thread-1", [msg])
        
        messages = await manager.get_messages("thread-1")
        assert len(messages) == 1
        assert messages[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_add_messages_multiple(self) -> None:
        """Test adding multiple messages via add_messages."""
        manager = MemoryManager()
        
        msgs = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
        ]
        await manager.add_messages("thread-1", msgs)
        
        messages = await manager.get_messages("thread-1")
        assert len(messages) == 3
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there!"
        assert messages[2].content == "How are you?"

    @pytest.mark.asyncio
    async def test_add_messages_with_metadata(self) -> None:
        """Test adding messages with metadata."""
        manager = MemoryManager()
        
        msgs = [HumanMessage(content="Test")]
        await manager.add_messages("thread-1", msgs, metadata={"agent": "test"})
        
        messages = await manager.get_messages("thread-1")
        assert len(messages) == 1
