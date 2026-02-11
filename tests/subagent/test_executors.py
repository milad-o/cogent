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
    ExecutionPlan,
    ExecutionStrategy,
    NativeExecutor,
    # NodeState,  # REMOVED - was part of TreeSearch multi-agent orchestration
    # SearchNode,  # REMOVED - was part of TreeSearch multi-agent orchestration
    SequentialExecutor,
    ToolCall,
    # TreeSearchExecutor,  # REMOVED - was part of TreeSearch multi-agent orchestration
    create_executor,
)


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_create_tool_call(self) -> None:
        """Test creating a basic tool call."""
        call = ToolCall(
            id="call_0",
            tool_name="search",
            args={"query": "test"},
        )
        assert call.id == "call_0"
        assert call.tool_name == "search"
        assert call.args == {"query": "test"}
        assert call.status == "pending"

    def test_is_ready_no_deps(self) -> None:
        """Test call is ready when no dependencies."""
        call = ToolCall(id="call_0", tool_name="search", args={})
        assert call.is_ready(set()) is True

    def test_is_ready_with_deps(self) -> None:
        """Test call readiness with dependencies."""
        call = ToolCall(
            id="call_1",
            tool_name="process",
            args={},
            depends_on=["call_0"],
        )
        assert call.is_ready(set()) is False
        assert call.is_ready({"call_0"}) is True

    def test_is_ready_multiple_deps(self) -> None:
        """Test readiness with multiple dependencies."""
        call = ToolCall(
            id="call_2",
            tool_name="combine",
            args={},
            depends_on=["call_0", "call_1"],
        )
        assert call.is_ready({"call_0"}) is False
        assert call.is_ready({"call_0", "call_1"}) is True


class TestExecutionPlan:
    """Tests for ExecutionPlan dataclass."""

    def test_create_empty_plan(self) -> None:
        """Test creating an empty plan."""
        plan = ExecutionPlan()
        assert len(plan) == 0
        assert bool(plan) is False

    def test_add_call(self) -> None:
        """Test adding calls to plan."""
        plan = ExecutionPlan()
        id1 = plan.add_call("search", {"query": "test"})
        id2 = plan.add_call("process", {"data": "$call_0"}, depends_on=[id1])

        assert len(plan) == 2
        assert id1 == "call_0"
        assert id2 == "call_1"
        assert plan.calls[1].depends_on == ["call_0"]

    def test_get_ready_calls(self) -> None:
        """Test getting ready calls."""
        plan = ExecutionPlan()
        plan.add_call("search_a", {"q": "A"})
        plan.add_call("search_b", {"q": "B"})
        plan.add_call("combine", {}, depends_on=["call_0", "call_1"])

        ready = plan.get_ready_calls(set())
        assert len(ready) == 2
        assert ready[0].tool_name == "search_a"
        assert ready[1].tool_name == "search_b"

    def test_get_execution_order(self) -> None:
        """Test execution wave calculation."""
        plan = ExecutionPlan()
        plan.add_call("search_a", {})  # call_0, no deps
        plan.add_call("search_b", {})  # call_1, no deps
        plan.add_call("combine", {}, depends_on=["call_0", "call_1"])  # call_2

        waves = plan.get_execution_order()

        assert len(waves) == 2
        assert set(waves[0]) == {"call_0", "call_1"}  # Parallel
        assert waves[1] == ["call_2"]  # Sequential


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
