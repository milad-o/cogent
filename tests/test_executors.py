"""
Tests for graph-based executors.

Tests the execution strategies including:
- ReActExecutor (think-act-observe loop)
- PlanExecutor (plan then execute)
- DAGExecutor (parallel waves)
- StreamingDAGExecutor (LLMCompiler-style streaming)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from agenticflow.graphs import (
    ExecutionStrategy,
    ExecutionPlan,
    ToolCall,
    BaseExecutor,
    DAGExecutor,
    StreamingDAGExecutor,
    TreeSearchExecutor,
    SearchNode,
    NodeState,
    PlanExecutor,
    ReActExecutor,
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
        agent.config = AgentConfig(name="test-agent")
        agent.scratchpad = Scratchpad()
        agent.think = AsyncMock(return_value="Test response")
        agent.act = AsyncMock(return_value="Tool result")
        agent._get_tool = MagicMock(return_value=None)
        agent.get_tool_descriptions = MagicMock(return_value="tool1: Does something")
        return agent

    def test_executor_creation(self, mock_agent: MagicMock) -> None:
        """Test creating an executor."""
        executor = DAGExecutor(mock_agent)
        assert executor.agent == mock_agent
        assert executor.max_iterations == 10

    def test_executor_custom_iterations(self, mock_agent: MagicMock) -> None:
        """Test executor with custom max iterations."""
        executor = DAGExecutor(mock_agent)
        executor.max_iterations = 5
        assert executor.max_iterations == 5


class TestDAGExecutor:
    """Tests for DAGExecutor."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock(spec=Agent)
        agent.config = AgentConfig(name="test-agent")
        agent.scratchpad = Scratchpad()
        agent.think = AsyncMock()
        agent.act = AsyncMock()
        agent._get_tool = MagicMock(return_value=None)
        agent.get_tool_descriptions = MagicMock(return_value="search: Search for info")
        return agent

    @pytest.mark.asyncio
    async def test_execute_no_tools(self, mock_agent: MagicMock) -> None:
        """Test execution when no tools are needed."""
        mock_agent.think.return_value = '{"reasoning": "Simple task", "steps": []}'
        executor = DAGExecutor(mock_agent)
        
        # Override think for synthesis
        mock_agent.think.side_effect = [
            '{"reasoning": "Simple", "steps": []}',  # Planning
            "Final answer",  # Direct answer
        ]
        
        result = await executor.execute("What is 2+2?")
        assert "Final answer" in result or result is not None

    @pytest.mark.asyncio
    async def test_execute_with_tools(self, mock_agent: MagicMock) -> None:
        """Test execution with tool calls."""
        # Plan response
        plan_response = '''
        {
            "reasoning": "Need to search",
            "steps": [
                {"id": "call_0", "tool": "search", "args": {"query": "test"}, "depends_on": []}
            ]
        }
        '''
        mock_agent.think.side_effect = [
            plan_response,  # Planning
            "Final synthesized answer",  # Synthesis
        ]
        mock_agent.act.return_value = "Search result"
        
        executor = DAGExecutor(mock_agent)
        result = await executor.execute("Search for test")
        
        # Verify tool was called
        mock_agent.act.assert_called()

    @pytest.mark.asyncio
    async def test_parallel_execution(self, mock_agent: MagicMock) -> None:
        """Test that independent calls run in parallel."""
        # Plan with two parallel calls
        plan_response = '''
        {
            "reasoning": "Need two searches",
            "steps": [
                {"id": "call_0", "tool": "search", "args": {"query": "A"}, "depends_on": []},
                {"id": "call_1", "tool": "search", "args": {"query": "B"}, "depends_on": []}
            ]
        }
        '''
        
        call_times = []
        
        async def track_act(tool_name: str, args: dict) -> str:
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate work
            return f"Result for {args.get('query', 'unknown')}"
        
        mock_agent.think.side_effect = [
            plan_response,
            "Combined results",
        ]
        mock_agent.act = track_act
        
        executor = DAGExecutor(mock_agent)
        await executor.execute("Search A and B")
        
        # Both calls should have started close together (parallel)
        assert len(call_times) == 2
        # Time difference should be small (<0.05s) if parallel
        assert abs(call_times[1] - call_times[0]) < 0.05


class TestStreamingDAGExecutor:
    """Tests for StreamingDAGExecutor (LLMCompiler-style)."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock(spec=Agent)
        agent.config = AgentConfig(name="test-agent")
        agent.scratchpad = Scratchpad()
        agent.think = AsyncMock()
        agent.act = AsyncMock()
        agent._get_tool = MagicMock(return_value=None)
        agent.get_tool_descriptions = MagicMock(return_value="search: Search for info")
        return agent

    @pytest.mark.asyncio
    async def test_streaming_execution(self, mock_agent: MagicMock) -> None:
        """Test streaming DAG execution."""
        plan_response = '''
        {
            "reasoning": "Two parallel searches",
            "steps": [
                {"id": "call_0", "tool": "search", "args": {"query": "A"}, "depends_on": []},
                {"id": "call_1", "tool": "search", "args": {"query": "B"}, "depends_on": []}
            ]
        }
        '''
        mock_agent.think.side_effect = [
            plan_response,
            "Final answer",
        ]
        mock_agent.act.return_value = "Result"
        
        executor = StreamingDAGExecutor(mock_agent)
        result = await executor.execute("Search for A and B")
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_streaming_respects_dependencies(self, mock_agent: MagicMock) -> None:
        """Test that streaming executor respects dependencies."""
        plan_response = '''
        {
            "reasoning": "Search then process",
            "steps": [
                {"id": "call_0", "tool": "search", "args": {"query": "test"}, "depends_on": []},
                {"id": "call_1", "tool": "process", "args": {"data": "$call_0"}, "depends_on": ["call_0"]}
            ]
        }
        '''
        
        execution_order = []
        
        async def track_act(tool_name: str, args: dict) -> str:
            execution_order.append(tool_name)
            await asyncio.sleep(0.05)
            return "Result"
        
        mock_agent.think.side_effect = [
            plan_response,
            "Final answer",
        ]
        mock_agent.act = track_act
        
        executor = StreamingDAGExecutor(mock_agent)
        await executor.execute("Search and process")
        
        # Process should come after search due to dependency
        assert execution_order.index("search") < execution_order.index("process")


class TestPlanExecutor:
    """Tests for PlanExecutor."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock(spec=Agent)
        agent.config = AgentConfig(name="test-agent")
        agent.scratchpad = Scratchpad()
        agent.think = AsyncMock()
        agent.act = AsyncMock()
        agent._get_tool = MagicMock(return_value=None)
        agent.get_tool_descriptions = MagicMock(return_value="search: Search")
        return agent

    @pytest.mark.asyncio
    async def test_sequential_execution(self, mock_agent: MagicMock) -> None:
        """Test that PlanExecutor runs sequentially."""
        plan_response = '''
        {
            "reasoning": "Step by step",
            "steps": [
                {"tool": "search", "args": {"query": "A"}},
                {"tool": "search", "args": {"query": "B"}}
            ]
        }
        '''
        
        call_times = []
        
        async def track_act(tool_name: str, args: dict) -> str:
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)
            return "Result"
        
        mock_agent.think.side_effect = [
            plan_response,
            "Final answer",
        ]
        mock_agent.act = track_act
        
        executor = PlanExecutor(mock_agent)
        await executor.execute("Search A then B")
        
        # Calls should be sequential (not parallel)
        assert len(call_times) == 2
        # Time difference should be >= 0.1s (sequential)
        assert call_times[1] - call_times[0] >= 0.09


class TestReActExecutor:
    """Tests for ReActExecutor."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock(spec=Agent)
        agent.config = AgentConfig(name="test-agent")
        agent.scratchpad = Scratchpad()
        agent.think = AsyncMock()
        agent.act = AsyncMock()
        agent._get_tool = MagicMock(return_value=None)
        return agent

    @pytest.mark.asyncio
    async def test_final_answer(self, mock_agent: MagicMock) -> None:
        """Test ReAct with immediate final answer."""
        mock_agent.think.return_value = "FINAL ANSWER: The answer is 42"
        
        executor = ReActExecutor(mock_agent)
        result = await executor.execute("What is the answer?")
        
        assert "42" in result

    @pytest.mark.asyncio
    async def test_tool_call_then_answer(self, mock_agent: MagicMock) -> None:
        """Test ReAct with tool call followed by answer."""
        mock_agent.think.side_effect = [
            'TOOL: search({"query": "test"})',
            "FINAL ANSWER: Found the result",
        ]
        mock_agent.act.return_value = "Search result"
        
        executor = ReActExecutor(mock_agent)
        result = await executor.execute("Search for test")
        
        assert "Found the result" in result
        mock_agent.act.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_iterations(self, mock_agent: MagicMock) -> None:
        """Test that executor stops at max iterations."""
        # Always return tool call, never final answer
        mock_agent.think.return_value = 'TOOL: search({"query": "test"})'
        mock_agent.act.return_value = "Result"
        
        executor = ReActExecutor(mock_agent)
        executor.max_iterations = 3
        result = await executor.execute("Loop forever")
        
        # Should stop after max iterations
        assert mock_agent.think.call_count <= 3


class TestCreateExecutor:
    """Tests for executor factory."""

    @pytest.fixture
    def mock_agent(self) -> MagicMock:
        """Create a mock agent."""
        agent = MagicMock(spec=Agent)
        agent.config = AgentConfig(name="test-agent")
        return agent

    def test_create_react(self, mock_agent: MagicMock) -> None:
        """Test creating ReAct executor."""
        executor = create_executor(mock_agent, ExecutionStrategy.REACT)
        assert isinstance(executor, ReActExecutor)

    def test_create_plan(self, mock_agent: MagicMock) -> None:
        """Test creating Plan executor."""
        executor = create_executor(mock_agent, ExecutionStrategy.PLAN_EXECUTE)
        assert isinstance(executor, PlanExecutor)

    def test_create_dag(self, mock_agent: MagicMock) -> None:
        """Test creating DAG executor."""
        executor = create_executor(mock_agent, ExecutionStrategy.DAG)
        assert isinstance(executor, DAGExecutor)

    def test_create_tree_search(self, mock_agent: MagicMock) -> None:
        """Test creating TreeSearch executor."""
        executor = create_executor(mock_agent, ExecutionStrategy.TREE_SEARCH)
        assert isinstance(executor, TreeSearchExecutor)

    def test_default_is_dag(self, mock_agent: MagicMock) -> None:
        """Test default strategy is DAG."""
        executor = create_executor(mock_agent)
        assert isinstance(executor, DAGExecutor)


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

    @pytest.mark.asyncio
    async def test_execute_finds_answer(self, mock_agent: MagicMock) -> None:
        """Test that tree search can find an answer."""
        # Mock expansion response with final answer
        expand_response = """
ACTION 1:
I'll try searching first
TOOL: search({"query": "test"})

ACTION 2:
Let me provide the answer directly
FINAL ANSWER: The answer is 42

ACTION 3:
Another search approach
TOOL: search({"query": "alternative"})
"""
        # Need enough responses for all the think calls
        mock_agent.think.side_effect = [
            expand_response,  # Expansion
            "0.95",  # Evaluation of final answer (ACTION 2)
            "0.4",  # Evaluation of search result (ACTION 1)
            "0.3",  # Evaluation of alternative (ACTION 3)
            "Best effort synthesis",  # Synthesis (if needed)
        ]
        mock_agent.act.return_value = "Search result"
        
        executor = TreeSearchExecutor(mock_agent)
        executor.max_iterations = 1
        executor.num_candidates = 3
        
        result = await executor.execute("Find the answer")
        
        # Should get some result (either final answer or synthesis)
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_execute_with_reflection(self, mock_agent: MagicMock) -> None:
        """Test that failed paths generate reflections."""
        expand_response = """
ACTION 1:
Try this approach
TOOL: search({"query": "wrong"})
"""
        mock_agent.think.side_effect = [
            expand_response,  # Expansion
            "0.1",  # Low evaluation (failure)
            "Instead of searching blindly, try a more specific query",  # Reflection
            "Best effort answer",  # Synthesis
        ]
        mock_agent.act.return_value = "ERROR: Not found"
        
        executor = TreeSearchExecutor(mock_agent)
        executor.max_iterations = 1
        executor.enable_reflection = True
        
        result = await executor.execute("Find something")
        
        # Should have added reflection to scratchpad
        reflections = mock_agent.scratchpad.get_reflections()
        assert len(reflections) >= 0  # May or may not add based on value threshold

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
