"""Tests for the graph module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agenticflow.graph import (
    GraphBuilder,
    NodeConfig,
    EdgeConfig,
    AgentGraphState,
    create_state_schema,
    merge_states,
    AgentNode,
    ToolNode,
    RouterNode,
    HumanNode,
    Handoff,
    HandoffType,
    create_handoff,
    GraphRunner,
    RunConfig,
    StreamMode,
)
from agenticflow.graph.builder import EdgeType
from agenticflow.graph.handoff import HandoffBuilder
from agenticflow.agents import Agent, AgentConfig
from agenticflow.core.enums import AgentRole


class TestAgentGraphState:
    """Tests for AgentGraphState."""

    def test_default_state(self):
        """Test default state values."""
        state = AgentGraphState()

        assert state.messages == []
        assert state.context == {}
        assert state.current_agent is None
        assert state.iteration == 0
        assert state.completed is False

    def test_state_to_dict(self):
        """Test converting state to dictionary."""
        state = AgentGraphState(
            task="test task",
            current_agent="agent-1",
            iteration=5,
        )

        data = state.to_dict()
        assert data["task"] == "test task"
        assert data["current_agent"] == "agent-1"
        assert data["iteration"] == 5

    def test_state_from_dict(self):
        """Test creating state from dictionary."""
        data = {
            "task": "test task",
            "current_agent": "agent-1",
            "iteration": 3,
        }

        state = AgentGraphState.from_dict(data)
        assert state.task == "test task"
        assert state.current_agent == "agent-1"
        assert state.iteration == 3


class TestCreateStateSchema:
    """Tests for create_state_schema."""

    def test_create_with_extra_fields(self):
        """Test creating schema with extra fields."""
        CustomState = create_state_schema(
            extra_fields={
                "sentiment": (str, "neutral"),
                "confidence": (float, 0.0),
            }
        )

        # Should be a new class
        assert CustomState is not AgentGraphState


class TestMergeStates:
    """Tests for merge_states."""

    def test_merge_overwrites(self):
        """Test that merge overwrites non-annotated fields."""
        base = {"task": "old task", "iteration": 1}
        update = {"task": "new task"}

        result = merge_states(base, update)
        assert result["task"] == "new task"
        assert result["iteration"] == 1

    def test_merge_adds_new_keys(self):
        """Test that merge adds new keys."""
        base = {"a": 1}
        update = {"b": 2}

        result = merge_states(base, update)
        assert result["a"] == 1
        assert result["b"] == 2


class TestGraphBuilder:
    """Tests for GraphBuilder."""

    def test_add_node(self):
        """Test adding nodes."""
        builder = GraphBuilder("test-graph")

        def node_fn(state):
            return state

        builder.add_node("node1", node_fn)

        assert "node1" in builder._nodes
        assert builder._nodes["node1"].func == node_fn

    def test_add_edge(self):
        """Test adding edges."""
        builder = GraphBuilder("test-graph")
        builder.add_node("a", lambda s: s)
        builder.add_node("b", lambda s: s)
        builder.add_edge("a", "b")

        assert len(builder._edges) == 1
        assert builder._edges[0].source == "a"
        assert builder._edges[0].target == "b"

    def test_add_conditional_edge(self):
        """Test adding conditional edges."""
        builder = GraphBuilder("test-graph")
        builder.add_node("router", lambda s: s)
        builder.add_node("a", lambda s: s)
        builder.add_node("b", lambda s: s)

        def route(state):
            return "a" if state.get("go_a") else "b"

        builder.add_conditional_edge("router", route, {"a": "a", "b": "b"})

        assert len(builder._edges) == 1
        assert builder._edges[0].edge_type == EdgeType.CONDITIONAL

    def test_set_entry(self):
        """Test setting entry point."""
        builder = GraphBuilder("test-graph")
        builder.add_node("start", lambda s: s)
        builder.set_entry("start")

        assert builder._entry_point == "start"

    def test_visualize(self):
        """Test DOT visualization."""
        builder = GraphBuilder("test-graph")
        builder.add_node("a", lambda s: s)
        builder.add_node("b", lambda s: s)
        builder.add_edge("a", "b")

        dot = builder.visualize()

        assert "digraph G" in dot
        assert '"a"' in dot
        assert '"b"' in dot
        assert '"a" -> "b"' in dot

    def test_empty_graph_raises(self):
        """Test compiling empty graph raises error."""
        builder = GraphBuilder("test")

        with pytest.raises(ValueError, match="no nodes"):
            builder.compile()


class TestHandoff:
    """Tests for Handoff."""

    def test_direct_handoff(self):
        """Test creating direct handoff."""
        handoff = Handoff(
            target="reviewer",
            message="Please review this",
        )

        assert handoff.target == "reviewer"
        assert handoff.handoff_type == HandoffType.DIRECT

    def test_handoff_to_command(self):
        """Test converting handoff to Command."""
        handoff = Handoff(
            target="reviewer",
            state_update={"draft": "content"},
        )

        command = handoff.to_command()
        assert command.goto == "reviewer"

    def test_create_handoff_helper(self):
        """Test create_handoff helper."""
        command = create_handoff(
            "reviewer",
            message="Review this",
            state_update={"priority": "high"},
        )

        assert command.goto == "reviewer"


class TestHandoffBuilder:
    """Tests for HandoffBuilder."""

    def test_fluent_builder(self):
        """Test fluent handoff builder."""
        handoff = (
            HandoffBuilder("reviewer")
            .with_message("Please review")
            .with_context(draft="content")
            .with_priority("high")
            .build()
        )

        assert handoff.target == "reviewer"
        assert handoff.message == "Please review"
        assert handoff.metadata["priority"] == "high"

    def test_builder_to_command(self):
        """Test builder directly to command."""
        command = (
            HandoffBuilder("agent")
            .with_message("test")
            .to_command()
        )

        assert command.goto == "agent"


class TestAgentNode:
    """Tests for AgentNode."""

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        config = AgentConfig(name="test-agent", role=AgentRole.WORKER)
        agent = MagicMock(spec=Agent)
        agent.config = config
        agent.think = AsyncMock(return_value="Agent thought")
        return agent

    @pytest.mark.asyncio
    async def test_agent_node_execution(self, mock_agent):
        """Test agent node executes agent."""
        node = AgentNode("test", mock_agent)

        state = {"task": "Do something", "context": {}}
        result = await node(state)

        assert "messages" in result
        assert result["current_agent"] == "test"
        mock_agent.think.assert_called_once()


class TestRouterNode:
    """Tests for RouterNode."""

    @pytest.mark.asyncio
    async def test_router_routes(self):
        """Test router node routes based on state."""
        def route_fn(state):
            return "agent_a" if state.get("use_a") else "agent_b"

        node = RouterNode("router", route_fn)

        result = await node({"use_a": True})
        assert result["next_agent"] == "agent_a"

        result = await node({"use_a": False})
        assert result["next_agent"] == "agent_b"


class TestRunConfig:
    """Tests for RunConfig."""

    def test_default_config(self):
        """Test default run configuration."""
        config = RunConfig()

        assert config.thread_id is None
        assert config.recursion_limit == 50
        assert config.stream_mode == StreamMode.VALUES

    def test_custom_config(self):
        """Test custom run configuration."""
        config = RunConfig(
            thread_id="my-thread",
            recursion_limit=100,
            stream_mode=StreamMode.UPDATES,
        )

        assert config.thread_id == "my-thread"
        assert config.recursion_limit == 100
        assert config.stream_mode == StreamMode.UPDATES
