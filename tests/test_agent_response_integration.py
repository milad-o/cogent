"""Tests for Agent integration with Response protocol (Phase 6.2)."""

import pytest

from agenticflow.agent import Agent
from agenticflow.core import Response
from agenticflow.models.mock import MockChatModel


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
    async def test_response_to_event(self):
        """Test converting response to event."""
        agent = Agent(
            name="TestAgent",
            model=MockChatModel(responses=["Result"]),
        )

        response = await agent.think("Test", correlation_id="corr-456")
        event = response.to_event("task.done", "TestAgent")

        assert event.name == "task.done"
        assert event.source == "TestAgent"
        assert event.data["content"] == "Result"
        assert event.data["success"] is True
        assert event.correlation_id == "corr-456"
