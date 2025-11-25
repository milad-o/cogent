"""Tests for the topologies module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agenticflow.topologies import (
    TopologyConfig,
    TopologyState,
    TopologyFactory,
    TopologyType,
    SupervisorTopology,
    MeshTopology,
    PipelineTopology,
    HierarchicalTopology,
)
from agenticflow.topologies.base import HandoffStrategy
from agenticflow.agents import Agent, AgentConfig
from agenticflow.core.enums import AgentRole


class TestTopologyConfig:
    """Tests for TopologyConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TopologyConfig(name="test-topology")

        assert config.name == "test-topology"
        assert config.max_iterations == 100
        assert config.handoff_strategy == HandoffStrategy.AUTOMATIC
        assert config.enable_checkpointing is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = TopologyConfig(
            name="custom",
            description="A custom topology",
            max_iterations=50,
            handoff_strategy=HandoffStrategy.COMMAND,
        )

        assert config.name == "custom"
        assert config.max_iterations == 50
        assert config.handoff_strategy == HandoffStrategy.COMMAND


class TestTopologyState:
    """Tests for TopologyState."""

    def test_default_state(self):
        """Test default state values."""
        state = TopologyState()

        assert state.messages == []
        assert state.current_agent is None
        assert state.task == ""
        assert state.iteration == 0
        assert state.completed is False

    def test_state_to_dict(self):
        """Test converting state to dictionary."""
        state = TopologyState(
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
            "completed": True,
        }

        state = TopologyState.from_dict(data)
        assert state.task == "test task"
        assert state.current_agent == "agent-1"
        assert state.iteration == 3
        assert state.completed is True


class TestTopologyFactory:
    """Tests for TopologyFactory."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents."""
        agents = []
        for name in ["supervisor", "worker1", "worker2"]:
            config = AgentConfig(name=name, role=AgentRole.WORKER)
            agent = MagicMock(spec=Agent)
            agent.config = config
            agents.append(agent)
        return agents

    def test_create_supervisor_topology(self, mock_agents):
        """Test creating supervisor topology."""
        topology = TopologyFactory.create(
            TopologyType.SUPERVISOR,
            "test-supervisor",
            mock_agents,
            supervisor_name="supervisor",
        )

        assert isinstance(topology, SupervisorTopology)
        assert topology.config.name == "test-supervisor"
        assert topology.supervisor_name == "supervisor"

    def test_create_mesh_topology(self, mock_agents):
        """Test creating mesh topology."""
        topology = TopologyFactory.create(
            TopologyType.MESH,
            "test-mesh",
            mock_agents,
        )

        assert isinstance(topology, MeshTopology)
        assert len(topology.agents) == 3

    def test_create_pipeline_topology(self, mock_agents):
        """Test creating pipeline topology."""
        topology = TopologyFactory.create(
            TopologyType.PIPELINE,
            "test-pipeline",
            mock_agents,
            stages=["supervisor", "worker1", "worker2"],
        )

        assert isinstance(topology, PipelineTopology)
        assert topology.stages == ["supervisor", "worker1", "worker2"]

    def test_available_types(self):
        """Test getting available topology types."""
        types = TopologyFactory.available_types()

        assert TopologyType.SUPERVISOR in types
        assert TopologyType.MESH in types
        assert TopologyType.PIPELINE in types
        assert TopologyType.HIERARCHICAL in types

    def test_quick_supervisor(self, mock_agents):
        """Test quick supervisor helper."""
        supervisor = mock_agents[0]
        workers = mock_agents[1:]

        topology = TopologyFactory.quick_supervisor(
            "quick-test",
            supervisor,
            workers,
        )

        assert isinstance(topology, SupervisorTopology)
        assert topology.supervisor_name == "supervisor"

    def test_quick_pipeline(self, mock_agents):
        """Test quick pipeline helper."""
        topology = TopologyFactory.quick_pipeline(
            "quick-pipeline",
            mock_agents,
        )

        assert isinstance(topology, PipelineTopology)


class TestSupervisorTopology:
    """Tests for SupervisorTopology."""

    @pytest.fixture
    def topology(self):
        """Create supervisor topology with mocks."""
        agents = []
        for name in ["supervisor", "researcher", "writer"]:
            config = AgentConfig(name=name, role=AgentRole.WORKER)
            agent = MagicMock(spec=Agent)
            agent.config = config
            agent.think = AsyncMock(return_value="Test thought")
            agents.append(agent)

        return SupervisorTopology(
            config=TopologyConfig(name="test"),
            agents=agents,
            supervisor_name="supervisor",
        )

    def test_supervisor_identified(self, topology):
        """Test supervisor is correctly identified."""
        assert topology.supervisor_name == "supervisor"
        assert "researcher" in topology.worker_names
        assert "writer" in topology.worker_names
        assert "supervisor" not in topology.worker_names

    def test_parse_supervisor_decision_finish(self, topology):
        """Test parsing finish decision."""
        thought = "The task is now complete. We can finish here."
        decision = topology._parse_supervisor_decision(thought)
        assert decision == "FINISH"

    def test_parse_supervisor_decision_worker(self, topology):
        """Test parsing worker delegation."""
        thought = "Let's delegate this to the researcher for more info."
        decision = topology._parse_supervisor_decision(thought)
        assert decision == "researcher"


class TestPipelineTopology:
    """Tests for PipelineTopology."""

    @pytest.fixture
    def topology(self):
        """Create pipeline topology with mocks."""
        agents = []
        for name in ["extractor", "transformer", "loader"]:
            config = AgentConfig(name=name, role=AgentRole.WORKER)
            agent = MagicMock(spec=Agent)
            agent.config = config
            agent.think = AsyncMock(return_value="Test output")
            agents.append(agent)

        return PipelineTopology(
            config=TopologyConfig(name="etl-pipeline"),
            agents=agents,
            stages=["extractor", "transformer", "loader"],
        )

    def test_stages_defined(self, topology):
        """Test stages are defined correctly."""
        assert topology.stages == ["extractor", "transformer", "loader"]

    def test_invalid_stage_raises(self):
        """Test invalid stage name raises error."""
        agents = []
        for name in ["a", "b"]:
            config = AgentConfig(name=name, role=AgentRole.WORKER)
            agent = MagicMock(spec=Agent)
            agent.config = config
            agents.append(agent)

        with pytest.raises(ValueError, match="not in agents"):
            PipelineTopology(
                config=TopologyConfig(name="test"),
                agents=agents,
                stages=["a", "nonexistent"],
            )


class TestHierarchicalTopology:
    """Tests for HierarchicalTopology."""

    @pytest.fixture
    def topology(self):
        """Create hierarchical topology with mocks."""
        agents = []
        for name in ["ceo", "eng_lead", "dev1", "dev2"]:
            config = AgentConfig(name=name, role=AgentRole.WORKER)
            agent = MagicMock(spec=Agent)
            agent.config = config
            agent.think = AsyncMock(return_value="Test output")
            agents.append(agent)

        return HierarchicalTopology(
            config=TopologyConfig(name="org-chart"),
            agents=agents,
            hierarchy={
                "ceo": ["eng_lead"],
                "eng_lead": ["dev1", "dev2"],
            },
            root="ceo",
        )

    def test_hierarchy_structure(self, topology):
        """Test hierarchy is structured correctly."""
        assert topology.root == "ceo"
        assert "ceo" in topology.hierarchy
        assert "eng_lead" in topology.hierarchy["ceo"]

    def test_get_leaves(self, topology):
        """Test getting leaf nodes."""
        leaves = topology._get_leaves("ceo")
        assert "dev1" in leaves
        assert "dev2" in leaves
        assert "eng_lead" not in leaves

    def test_invalid_root_raises(self):
        """Test invalid root raises error."""
        config = AgentConfig(name="a", role=AgentRole.WORKER)
        agent = MagicMock(spec=Agent)
        agent.config = config

        with pytest.raises(ValueError, match="not in agents"):
            HierarchicalTopology(
                config=TopologyConfig(name="test"),
                agents=[agent],
                hierarchy={"a": []},
                root="nonexistent",
            )

    def test_requires_root_with_hierarchy(self):
        """Test root is required when hierarchy is provided."""
        config = AgentConfig(name="a", role=AgentRole.WORKER)
        agent = MagicMock(spec=Agent)
        agent.config = config

        with pytest.raises(ValueError, match="root must be specified"):
            HierarchicalTopology(
                config=TopologyConfig(name="test"),
                agents=[agent],
                hierarchy={"a": []},
            )
