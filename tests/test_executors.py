"""
Tests for executors.

Tests the execution strategies including:
- NativeExecutor (parallel execution)
- SequentialExecutor (sequential execution)
- TreeSearchExecutor (LATS-style tree search)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from agenticflow.executors import (
    ExecutionStrategy,
    ExecutionPlan,
    ToolCall,
    BaseExecutor,
    NativeExecutor,
    SequentialExecutor,
    TreeSearchExecutor,
    SearchNode,
    NodeState,
    create_executor,
)
from agenticflow.agent import Agent, AgentConfig
from agenticflow.agent.scratchpad import Scratchpad


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
        agent.scratchpad = Scratchpad()
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
        assert executor.max_iterations == 10

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
        agent.scratchpad = Scratchpad()
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
        assert executor.max_iterations == 10


class TestSequentialExecutor:
    """Tests for SequentialExecutor."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock(spec=Agent)
        agent.name = "test-agent"
        agent.config = AgentConfig(name="test-agent")
        agent.scratchpad = Scratchpad()
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
        assert executor.max_iterations == 10


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

    def test_create_tree_search(self, mock_agent: MagicMock) -> None:
        """Test creating TreeSearch executor."""
        executor = create_executor(mock_agent, ExecutionStrategy.TREE_SEARCH)
        assert isinstance(executor, TreeSearchExecutor)

    def test_default_is_native(self, mock_agent: MagicMock) -> None:
        """Test default strategy is NATIVE."""
        executor = create_executor(mock_agent)
        assert isinstance(executor, NativeExecutor)


class TestSearchNode:
    """Tests for SearchNode dataclass."""

    def test_create_node(self) -> None:
        """Test creating a search node."""
        node = SearchNode(id="node_1", action={"type": "tool_call", "tool": "search"})
        assert node.id == "node_1"
        assert node.state == NodeState.PENDING
        assert node.value == 0.5
        assert node.visits == 0

    def test_is_leaf(self) -> None:
        """Test leaf node detection."""
        node = SearchNode(id="node_1")
        assert node.is_leaf() is True
        
        child = SearchNode(id="node_2", parent=node)
        node.children.append(child)
        assert node.is_leaf() is False

    def test_is_terminal(self) -> None:
        """Test terminal node detection."""
        node = SearchNode(id="node_1")
        assert node.is_terminal() is False
        
        node.state = NodeState.SUCCESS
        assert node.is_terminal() is True
        
        node.state = NodeState.FAILED
        assert node.is_terminal() is True

    def test_ucb1_score_unvisited(self) -> None:
        """Test UCB1 score for unvisited node."""
        node = SearchNode(id="node_1")
        assert node.ucb1_score() == float("inf")

    def test_ucb1_score_visited(self) -> None:
        """Test UCB1 score for visited node."""
        parent = SearchNode(id="parent")
        parent.visits = 10
        
        node = SearchNode(id="node_1", parent=parent, value=0.7)
        node.visits = 5
        
        score = node.ucb1_score()
        # Should be value + exploration bonus
        assert score > 0.7  # Greater than just exploitation
        assert score < 2.0  # But reasonable

    def test_backpropagate(self) -> None:
        """Test value backpropagation."""
        root = SearchNode(id="root")
        child = SearchNode(id="child", parent=root)
        grandchild = SearchNode(id="grandchild", parent=child)
        
        # Backpropagate a value
        grandchild.backpropagate(0.8)
        
        assert grandchild.visits == 1
        assert child.visits == 1
        assert root.visits == 1
        
        # Values should be updated
        assert grandchild.value == 0.8  # First visit, takes the value directly

    def test_get_path(self) -> None:
        """Test path from root to node."""
        root = SearchNode(id="root")
        child = SearchNode(id="child", parent=root)
        grandchild = SearchNode(id="grandchild", parent=child)
        
        path = grandchild.get_path()
        assert len(path) == 3
        assert path[0].id == "root"
        assert path[1].id == "child"
        assert path[2].id == "grandchild"

    def test_to_dict(self) -> None:
        """Test node serialization."""
        node = SearchNode(
            id="node_1",
            action={"type": "tool_call"},
            observation="Result",
            state=NodeState.SUCCESS,
            value=0.9,
            visits=3,
        )
        
        data = node.to_dict()
        assert data["id"] == "node_1"
        assert data["state"] == "success"
        assert data["value"] == 0.9
        assert data["visits"] == 3


class TestTreeSearchExecutor:
    """Tests for TreeSearchExecutor (LATS)."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock(spec=Agent)
        agent.name = "test-agent"
        agent.config = AgentConfig(name="test-agent")
        agent.scratchpad = Scratchpad()
        agent.think = AsyncMock()
        agent.act = AsyncMock()
        agent._get_tool = MagicMock(return_value=None)
        agent.get_tool_descriptions = MagicMock(return_value="search: Search for info")
        return agent

    def test_executor_creation(self, mock_agent: MagicMock) -> None:
        """Test creating TreeSearchExecutor."""
        executor = TreeSearchExecutor(mock_agent)
        assert executor.max_depth == 5
        assert executor.num_candidates == 3
        assert executor.enable_reflection is True

    def test_executor_custom_params(self, mock_agent: MagicMock) -> None:
        """Test TreeSearchExecutor with custom parameters."""
        executor = TreeSearchExecutor(mock_agent)
        executor.max_depth = 3
        executor.num_candidates = 5
        executor.exploration_weight = 2.0
        
        assert executor.max_depth == 3
        assert executor.num_candidates == 5
        assert executor.exploration_weight == 2.0

    def test_parse_candidate_actions(self, mock_agent: MagicMock) -> None:
        """Test parsing candidate actions from LLM response."""
        executor = TreeSearchExecutor(mock_agent)
        
        response = """
ACTION 1:
First approach reasoning
TOOL: search({"query": "test"})

ACTION 2:
Second approach reasoning
FINAL ANSWER: The answer is 42

ACTION 3:
Third approach
TOOL: process({"data": "input"})
"""
        
        actions = executor._parse_candidate_actions(response)
        
        assert len(actions) == 3
        
        # First action is tool call
        assert actions[0]["type"] == "tool_call"
        assert actions[0]["tool"] == "search"
        
        # Second action is final answer
        assert actions[1]["type"] == "final_answer"
        assert "42" in actions[1]["answer"]
        
        # Third action is tool call
        assert actions[2]["type"] == "tool_call"
        assert actions[2]["tool"] == "process"

    def test_select_ucb1(self, mock_agent: MagicMock) -> None:
        """Test UCB1-based node selection."""
        executor = TreeSearchExecutor(mock_agent)
        
        # Create a simple tree
        root = SearchNode(id="root", state=NodeState.EXPANDED)
        root.visits = 10
        
        child1 = SearchNode(id="child1", parent=root, value=0.8)
        child1.visits = 5
        
        child2 = SearchNode(id="child2", parent=root, value=0.3)
        child2.visits = 1  # Less visited, higher exploration bonus
        
        root.children = [child1, child2]
        
        # Selection should favor less-visited node due to exploration
        selected = executor._select(root)
        
        # Should select one of the children (both are leaves)
        assert selected in [child1, child2]

    def test_format_path(self, mock_agent: MagicMock) -> None:
        """Test path formatting for prompts."""
        executor = TreeSearchExecutor(mock_agent)
        
        root = SearchNode(id="root", action={"type": "root", "task": "Test"})
        child = SearchNode(
            id="child",
            parent=root,
            action={"type": "tool_call", "tool": "search", "args": {"query": "test"}},
            observation="Found result",
            depth=1,
        )
        
        path = child.get_path()
        formatted = executor._format_path(path)
        
        assert "search" in formatted
        assert "Found result" in formatted
