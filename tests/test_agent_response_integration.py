"""Tests for Agent integration with Response protocol (Phase 6.2)."""

import pytest

from cogent.agent import Agent
from cogent.core import Response
from cogent.models.mock import MockChatModel


class TestAgentResponseIntegration:
    """Tests for Agent methods returning Response[T]."""

    @pytest.mark.asyncio
    async def test_agent_think_returns_response(self):
        """Test that think() returns Response[str]."""
        agent = Agent(
            name="TestAgent",
            model=MockChatModel(responses=["Hello from agent"]),
        )

        response = await agent.think("What should I do?")

        assert isinstance(response, Response)
        assert isinstance(response.content, str)
        assert response.content == "Hello from agent"
        assert response.success is True
        assert response.error is None

    @pytest.mark.asyncio
    async def test_agent_think_response_metadata(self):
        """Test that think() response has correct metadata."""
        agent = Agent(
            name="TestAgent",
            model=MockChatModel(responses=["Response"]),
        )

        response = await agent.think("Test", correlation_id="corr-123")

        assert response.metadata.agent == "TestAgent"
        assert response.metadata.model is not None
        assert response.metadata.duration > 0
        assert response.metadata.correlation_id == "corr-123"

    @pytest.mark.asyncio
    async def test_agent_run_returns_response(self):
        """Test that run() returns Response[Any]."""
        agent = Agent(
            name="TaskAgent",
            model=MockChatModel(responses=["Task completed"]),
        )

        response = await agent.run("Complete this task")

        assert isinstance(response, Response)
        assert response.content is not None
        assert response.success is True
        assert response.error is None

    @pytest.mark.asyncio
    async def test_agent_run_response_metadata(self):
        """Test that run() response has correct metadata."""
        agent = Agent(
            name="TaskAgent",
            model=MockChatModel(responses=["Done"]),
        )

        response = await agent.run("Task")

        assert response.metadata.agent == "TaskAgent"
        assert response.metadata.model is not None
        assert response.metadata.duration > 0

    @pytest.mark.asyncio
    async def test_response_unwrap_success(self):
        """Test unwrapping successful response."""
        agent = Agent(
            name="TestAgent",
            model=MockChatModel(responses=["Result"]),
        )

        response = await agent.think("Test")
        result = response.unwrap()

        assert result == "Result"

    @pytest.mark.asyncio
    async def test_response_content_access(self):
        """Test accessing response content directly."""
        agent = Agent(
            name="TestAgent",
            model=MockChatModel(responses=["Content"]),
        )

        response = await agent.think("Test")

        # Can access content directly
        assert response.content == "Content"

        # Or use unwrap()
        assert response.unwrap() == "Content"

    @pytest.mark.asyncio
    async def test_response_to_dict(self):
        """Test serializing response to dict."""
        agent = Agent(
            name="TestAgent",
            model=MockChatModel(responses=["Result"]),
        )

        response = await agent.think("Test")
        data = response.to_dict()

        assert data["content"] == "Result"
        assert data["metadata"]["agent"] == "TestAgent"
        assert data["success"] is True
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_response_includes_messages(self):
        """Test that response includes conversation messages."""
        agent = Agent(
            name="TestAgent",
            model=MockChatModel(responses=["Hello world"]),
        )

        response = await agent.think("Say hello")

        # Should have system prompt, user message, and AI response
        assert len(response.messages) >= 2

        # Check message types
        has_user_msg = any(
            hasattr(m, "role") and m.role == "user" for m in response.messages
        )
        has_ai_msg = any(
            hasattr(m, "role") and m.role == "assistant" for m in response.messages
        )

        assert has_user_msg, "Should have user message"
        assert has_ai_msg, "Should have AI message"

    @pytest.mark.asyncio
    async def test_response_messages_in_to_dict(self):
        """Test that messages are included in serialization."""
        agent = Agent(
            name="TestAgent",
            model=MockChatModel(responses=["Response"]),
        )

        response = await agent.think("Test")
        data = response.to_dict()

        assert "messages" in data
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) >= 2
