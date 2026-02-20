"""
Tests for executors.

Tests the execution strategies including:
- NativeExecutor (parallel execution)
- SequentialExecutor (sequential execution)
- TreeSearchExecutor (REMOVED - was part of multi-agent orchestration)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from cogent.agent import Agent, AgentConfig
from cogent.agent.taskboard import TaskBoard
from cogent.executors import (
    ExecutionStrategy,
    NativeExecutor,
    SequentialExecutor,
    create_executor,
)


class TestBaseExecutor:
    """Tests for BaseExecutor functionality."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock(spec=Agent)
        agent.name = "test-agent"
        agent.config = AgentConfig(name="test-agent")
        agent._taskboard = TaskBoard()
        agent.think = AsyncMock(return_value="Test response")
        agent.act = AsyncMock(return_value="Tool result")
        agent._get_tool = MagicMock(return_value=None)
        agent.get_tool_descriptions = MagicMock(return_value="tool1: Does something")
        agent.all_tools = []
        agent.model = MagicMock()
        return agent

    def test_executor_creation(self, mock_agent: MagicMock) -> None:
        """Test creating an executor."""
        executor = NativeExecutor(mock_agent)
        assert executor.agent == mock_agent
        assert executor.max_iterations == 25

    def test_executor_custom_iterations(self, mock_agent: MagicMock) -> None:
        """Test executor with custom max iterations."""
        executor = NativeExecutor(mock_agent)
        executor.max_iterations = 5
        assert executor.max_iterations == 5


class TestNativeExecutor:
    """Tests for NativeExecutor."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock(spec=Agent)
        agent.name = "test-agent"
        agent.config = AgentConfig(name="test-agent")
        agent._taskboard = TaskBoard()
        agent.think = AsyncMock()
        agent.act = AsyncMock()
        agent._get_tool = MagicMock(return_value=None)
        agent.get_tool_descriptions = MagicMock(return_value="search: Search for info")
        agent.all_tools = []
        agent.model = MagicMock()
        agent.model.bind_tools = MagicMock(return_value=agent.model)
        return agent

    def test_native_executor_creation(self, mock_agent: MagicMock) -> None:
        """Test creating NativeExecutor."""
        executor = NativeExecutor(mock_agent)
        assert executor.agent == mock_agent
        assert executor.max_iterations == 25


class TestSequentialExecutor:
    """Tests for SequentialExecutor."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock(spec=Agent)
        agent.name = "test-agent"
        agent.config = AgentConfig(name="test-agent")
        agent._taskboard = TaskBoard()
        agent.think = AsyncMock()
        agent.act = AsyncMock()
        agent._get_tool = MagicMock(return_value=None)
        agent.get_tool_descriptions = MagicMock(return_value="search: Search for info")
        agent.all_tools = []
        agent.model = MagicMock()
        agent.model.bind_tools = MagicMock(return_value=agent.model)
        return agent

    def test_sequential_executor_creation(self, mock_agent: MagicMock) -> None:
        """Test creating SequentialExecutor."""
        executor = SequentialExecutor(mock_agent)
        assert executor.agent == mock_agent
        assert executor.max_iterations == 25


class TestCreateExecutor:
    """Tests for executor factory."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock(spec=Agent)
        agent.name = "test-agent"
        agent.config = AgentConfig(name="test-agent")
        agent.all_tools = []
        agent.model = MagicMock()
        agent.model.bind_tools = MagicMock(return_value=agent.model)
        return agent

    def test_create_native(self, mock_agent: MagicMock) -> None:
        """Test creating Native executor."""
        executor = create_executor(mock_agent, ExecutionStrategy.NATIVE)
        assert isinstance(executor, NativeExecutor)

    def test_create_sequential(self, mock_agent: MagicMock) -> None:
        """Test creating Sequential executor."""
        executor = create_executor(mock_agent, ExecutionStrategy.SEQUENTIAL)
        assert isinstance(executor, SequentialExecutor)

    # def test_create_tre e_search(self, mock_agent: MagicMock) -> None:
    #     """Test creating TreeSearch executor."""
    #     executor = create_executor(mock_agent, ExecutionStrategy.TREE_SEARCH)
    #     assert isinstance(executor, TreeSearchExecutor)

    def test_default_is_native(self, mock_agent: MagicMock) -> None:
        """Test default strategy is NATIVE."""
        executor = create_executor(mock_agent)
        assert isinstance(executor, NativeExecutor)
