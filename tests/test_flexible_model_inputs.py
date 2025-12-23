"""
Tests for flexible model input handling.

Verifies that models accept various input types:
- Strings
- List of dicts
- List of message objects
- Mixed lists
"""

import pytest
from agenticflow.core.messages import HumanMessage, SystemMessage, AIMessage
from agenticflow.models.mock import MockChatModel
from agenticflow.models.base import normalize_input, convert_messages


class TestNormalizeInput:
    """Tests for normalize_input helper function."""

    def test_string_input(self) -> None:
        """String should be converted to user message."""
        result = normalize_input("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_list_input_passthrough(self) -> None:
        """List input should pass through unchanged."""
        messages = [{"role": "user", "content": "Hello"}]
        result = normalize_input(messages)
        assert result == messages

    def test_empty_string(self) -> None:
        """Empty string should become empty user message."""
        result = normalize_input("")
        assert result == [{"role": "user", "content": ""}]


class TestConvertMessages:
    """Tests for convert_messages helper function."""

    def test_dict_messages(self) -> None:
        """Dict messages should pass through."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = convert_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_message_objects(self) -> None:
        """Message objects should convert to dicts."""
        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
        ]
        result = convert_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello"

    def test_mixed_list(self) -> None:
        """Mixed dicts and objects should work."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            HumanMessage(content="Hello"),
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = convert_messages(messages)
        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_ai_message_with_tool_calls(self) -> None:
        """AIMessage with tool_calls should convert correctly."""
        messages = [
            AIMessage(
                content="Let me search for that",
                tool_calls=[
                    {"id": "call_1", "name": "search", "args": {"query": "test"}},
                ],
            ),
        ]
        result = convert_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1

    def test_none_content(self) -> None:
        """None content should become empty string."""
        messages = [{"role": "user", "content": None}]
        result = convert_messages(messages)
        assert result[0]["content"] == ""

    def test_list_content_concatenation(self) -> None:
        """List content should be joined into string."""
        messages = [{"role": "user", "content": ["Hello", "World"]}]
        result = convert_messages(messages)
        assert result[0]["content"] == "Hello World"

    def test_dict_content_json(self) -> None:
        """Dict content should be JSON stringified."""
        messages = [{"role": "user", "content": {"key": "value"}}]
        result = convert_messages(messages)
        assert '"key"' in result[0]["content"]
        assert '"value"' in result[0]["content"]

    def test_multimodal_content_preserved(self) -> None:
        """Multimodal content (OpenAI vision format) should be preserved."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                ],
            }
        ]
        result = convert_messages(messages)
        # Multimodal content should be preserved as-is
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"


class TestMockModelFlexibleInputs:
    """Tests for MockChatModel with flexible inputs."""

    @pytest.mark.asyncio
    async def test_string_input(self) -> None:
        """Model should accept string input."""
        model = MockChatModel(responses=["Hello!"])
        response = await model.ainvoke("Hi there")
        assert response.content == "Hello!"

    @pytest.mark.asyncio
    async def test_dict_list_input(self) -> None:
        """Model should accept list of dicts."""
        model = MockChatModel(responses=["Response"])
        response = await model.ainvoke([{"role": "user", "content": "Hello"}])
        assert response.content == "Response"

    @pytest.mark.asyncio
    async def test_message_object_input(self) -> None:
        """Model should accept message objects."""
        model = MockChatModel(responses=["Response"])
        response = await model.ainvoke([HumanMessage(content="Hello")])
        assert response.content == "Response"

    @pytest.mark.asyncio
    async def test_mixed_input(self) -> None:
        """Model should accept mixed list."""
        model = MockChatModel(responses=["Response"])
        response = await model.ainvoke([
            SystemMessage(content="You are helpful"),
            {"role": "user", "content": "Hello"},
        ])
        assert response.content == "Response"

    def test_sync_string_input(self) -> None:
        """Sync invoke should also accept string."""
        model = MockChatModel(responses=["Hello!"])
        response = model.invoke("Hi there")
        assert response.content == "Hello!"

    def test_sync_message_object_input(self) -> None:
        """Sync invoke should accept message objects."""
        model = MockChatModel(responses=["Response"])
        response = model.invoke([
            SystemMessage(content="System"),
            HumanMessage(content="User"),
        ])
        assert response.content == "Response"


class TestToolMessagesHandling:
    """Tests for tool message handling in convert_messages."""

    def test_tool_calls_sanitization(self) -> None:
        """Tool calls should get proper IDs assigned."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me search",
                "tool_calls": [
                    {"name": "search", "args": {"query": "test"}},  # No ID
                ],
            },
        ]
        result = convert_messages(messages)
        # Should have assigned an ID
        assert result[0]["tool_calls"][0]["id"].startswith("call_")

    def test_orphan_tool_message_dropped(self) -> None:
        """Tool message without preceding assistant tool_calls should be dropped."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "content": "Result", "tool_call_id": "call_1"},
        ]
        result = convert_messages(messages)
        # Tool message should be dropped since no preceding assistant with tool_calls
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_tool_message_pairing(self) -> None:
        """Tool messages should be paired with assistant tool_calls."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "search", "args": {"query": "test"}},
                ],
            },
            {"role": "tool", "content": "Result", "tool_call_id": "call_1"},
        ]
        result = convert_messages(messages)
        assert len(result) == 2
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_1"

    def test_tool_message_inferred_id(self) -> None:
        """Tool message without ID should infer from preceding tool_calls."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "search", "args": {"query": "test"}},
                ],
            },
            {"role": "tool", "content": "Result"},  # No tool_call_id
        ]
        result = convert_messages(messages)
        assert len(result) == 2
        assert result[1]["tool_call_id"] == "call_1"

    def test_multiple_tool_results(self) -> None:
        """Multiple tool results should be paired in order."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "search", "args": {}},
                    {"id": "call_2", "name": "calculate", "args": {}},
                ],
            },
            {"role": "tool", "content": "Result 1", "tool_call_id": "call_1"},
            {"role": "tool", "content": "Result 2", "tool_call_id": "call_2"},
        ]
        result = convert_messages(messages)
        assert len(result) == 3
        assert result[1]["tool_call_id"] == "call_1"
        assert result[2]["tool_call_id"] == "call_2"
