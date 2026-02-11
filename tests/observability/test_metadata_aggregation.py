"""Tests for subagent metadata aggregation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogent.agent.base import Agent
from cogent.core.response import Response, ResponseMetadata, TokenUsage
from cogent.models import BaseChatModel


def create_mock_model():
    """Create a mock model for testing."""
    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.model = "mock-model"
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    return mock_model


@pytest.mark.asyncio
async def test_metadata_aggregation_no_subagents():
    """Test that metadata works correctly without subagents."""
    model = create_mock_model()

    agent = Agent(
        name="simple_agent",
        model=model,
        system_prompt="You are a helpful assistant",
    )

    # Mock executor to return a simple result
    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25
        mock_executor.execute = AsyncMock(return_value="Simple response")
        mock_executor._last_messages = []
        mock_executor_factory.return_value = mock_executor

        response = await agent.run("Test task")

        assert response.content == "Simple response"
        assert response.subagent_responses is None
        assert response.metadata.delegation_chain is None


@pytest.mark.asyncio
async def test_metadata_aggregation_with_subagents():
    """Test that tokens are aggregated from coordinator + subagents."""
    model = create_mock_model()

    # Create mock subagent
    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type(
                "obj", (object,), {"description": f"{name} specialist"}
            )()

        async def run(self, task, context=None):
            return Response(
                content=f"Result from {self.name}",
                metadata=ResponseMetadata(
                    agent=self.name,
                    model="mock-model",
                    tokens=TokenUsage(
                        prompt_tokens=50, completion_tokens=100, total_tokens=150
                    ),
                    duration=1.0,
                ),
            )

    analyst = MockSubagent("analyst")
    researcher = MockSubagent("researcher")

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst, "researcher": researcher},
    )

    # Mock executor that simulates calling both subagents
    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25

        async def mock_execute(task, context):
            # Simulate executor calling both subagents
            if hasattr(coordinator, "_subagent_registry"):
                await coordinator._subagent_registry.execute(
                    "analyst", "Analyze data", context
                )
                await coordinator._subagent_registry.execute(
                    "researcher", "Research topic", context
                )
            return "Coordinated result"

        mock_executor.execute = AsyncMock(side_effect=mock_execute)

        # Mock coordinator tokens (from _aggregate_tokens_from_messages)
        from cogent.core.messages import AIMessage, MessageMetadata

        coordinator_msg = AIMessage(
            content="Coordinated result",
            metadata=MessageMetadata(
                tokens=TokenUsage(
                    prompt_tokens=30, completion_tokens=70, total_tokens=100
                )
            ),
        )
        mock_executor._last_messages = [coordinator_msg]

        mock_executor_factory.return_value = mock_executor

        response = await coordinator.run("Complex task")

        # Verify response content
        assert response.content == "Coordinated result"

        # Verify subagent responses are attached
        assert response.subagent_responses is not None
        assert len(response.subagent_responses) == 2

        # Verify delegation chain
        assert response.metadata.delegation_chain is not None
        assert len(response.metadata.delegation_chain) == 2

        # Each subagent should be in delegation chain
        agent_names = [d["agent"] for d in response.metadata.delegation_chain]
        assert "analyst" in agent_names
        assert "researcher" in agent_names

        # Verify token aggregation
        # Coordinator: 30 + 70 = 100
        # Analyst: 50 + 100 = 150
        # Researcher: 50 + 100 = 150
        # Total: 100 + 150 + 150 = 400
        assert response.metadata.tokens.total_tokens == 400
        assert response.metadata.tokens.prompt_tokens == 130  # 30 + 50 + 50
        assert response.metadata.tokens.completion_tokens == 270  # 70 + 100 + 100


@pytest.mark.asyncio
async def test_subagent_registry_cleared_before_run():
    """Test that subagent registry is cleared before each run."""
    model = create_mock_model()

    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type(
                "obj", (object,), {"description": f"{name} specialist"}
            )()

        async def run(self, task, context=None):
            return Response(
                content=f"Result from {self.name}",
                metadata=ResponseMetadata(
                    agent=self.name,
                    model="mock-model",
                    tokens=TokenUsage(
                        prompt_tokens=10, completion_tokens=20, total_tokens=30
                    ),
                    duration=0.5,
                ),
            )

    analyst = MockSubagent("analyst")

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )

    # Mock executor
    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25

        async def mock_execute_first(task, context):
            if hasattr(coordinator, "_subagent_registry"):
                await coordinator._subagent_registry.execute(
                    "analyst", "First task", context
                )
            return "First result"

        async def mock_execute_second(task, context):
            # Don't call subagent on second run
            return "Second result"

        # Mock coordinator tokens
        from cogent.core.messages import AIMessage, MessageMetadata

        # First run
        mock_executor.execute = AsyncMock(side_effect=mock_execute_first)
        mock_executor._last_messages = [
            AIMessage(
                content="First result",
                metadata=MessageMetadata(
                    tokens=TokenUsage(
                        prompt_tokens=5, completion_tokens=10, total_tokens=15
                    )
                ),
            )
        ]
        mock_executor_factory.return_value = mock_executor

        first_response = await coordinator.run("First task")

        # Should have 1 subagent response
        assert first_response.subagent_responses is not None
        assert len(first_response.subagent_responses) == 1
        assert first_response.metadata.tokens.total_tokens == 45  # 15 + 30

        # Second run - change executor behavior
        mock_executor.execute = AsyncMock(side_effect=mock_execute_second)
        mock_executor._last_messages = [
            AIMessage(
                content="Second result",
                metadata=MessageMetadata(
                    tokens=TokenUsage(
                        prompt_tokens=5, completion_tokens=10, total_tokens=15
                    )
                ),
            )
        ]

        second_response = await coordinator.run("Second task")

        # Should have NO subagent responses (cleared from previous run)
        assert second_response.subagent_responses is None
        assert (
            second_response.metadata.tokens.total_tokens == 15
        )  # Only coordinator tokens


@pytest.mark.asyncio
async def test_delegation_chain_metadata():
    """Test that delegation chain contains correct metadata."""
    model = create_mock_model()

    class MockSubagent:
        def __init__(self, name, model_name="mock-model"):
            self.name = name
            self.model_name = model_name
            self.config = type(
                "obj", (object,), {"description": f"{name} specialist"}
            )()

        async def run(self, task, context=None):
            return Response(
                content=f"Result from {self.name}",
                metadata=ResponseMetadata(
                    agent=self.name,
                    model=self.model_name,
                    tokens=TokenUsage(
                        prompt_tokens=25, completion_tokens=75, total_tokens=100
                    ),
                    duration=2.5,
                ),
            )

    analyst = MockSubagent("analyst", model_name="gpt-4")

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )

    # Mock executor
    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25

        async def mock_execute(task, context):
            if hasattr(coordinator, "_subagent_registry"):
                await coordinator._subagent_registry.execute(
                    "analyst", "Analyze", context
                )
            return "Result"

        mock_executor.execute = AsyncMock(side_effect=mock_execute)
        mock_executor._last_messages = []
        mock_executor_factory.return_value = mock_executor

        response = await coordinator.run("Task")

        # Verify delegation chain structure
        assert response.metadata.delegation_chain is not None
        assert len(response.metadata.delegation_chain) == 1

        chain_entry = response.metadata.delegation_chain[0]
        assert chain_entry["agent"] == "analyst"
        assert chain_entry["model"] == "gpt-4"
        assert chain_entry["tokens"] == 100
        assert chain_entry["duration"] == 2.5


@pytest.mark.asyncio
async def test_no_delegation_chain_without_subagent_execution():
    """Test that delegation_chain is None when no subagents are executed."""
    model = create_mock_model()

    class MockSubagent:
        def __init__(self, name):
            self.name = name
            self.config = type(
                "obj", (object,), {"description": f"{name} specialist"}
            )()

        async def run(self, task, context=None):
            return Response(
                content="Result",
                metadata=ResponseMetadata(agent=self.name, model="mock"),
            )

    analyst = MockSubagent("analyst")

    coordinator = Agent(
        name="coordinator",
        model=model,
        subagents={"analyst": analyst},
    )

    # Mock executor that doesn't call subagent
    with patch("cogent.executors.create_executor") as mock_executor_factory:
        mock_executor = MagicMock()
        mock_executor.max_iterations = 25
        mock_executor.execute = AsyncMock(return_value="Direct result")
        mock_executor._last_messages = []
        mock_executor_factory.return_value = mock_executor

        response = await coordinator.run("Simple task")

        # Should have no delegation chain or subagent responses
        assert response.metadata.delegation_chain is None
        assert response.subagent_responses is None
