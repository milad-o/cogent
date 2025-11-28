"""Tests for the native topologies module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agenticflow.topologies import (
    AgentConfig,
    BaseTopology,
    TopologyResult,
    TopologyType,
    Supervisor,
    Pipeline,
    Mesh,
    Hierarchical,
    supervisor,
    pipeline,
    mesh,
)


# ==================== Fixtures ====================


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    def _create(name: str, role: str | None = None):
        agent = MagicMock()
        agent.name = name
        agent.run = AsyncMock(return_value=f"Output from {name}")
        return agent
    return _create


@pytest.fixture
def mock_agents(mock_agent):
    """Create a set of mock agents."""
    return {
        "researcher": mock_agent("researcher", "research"),
        "writer": mock_agent("writer", "writing"),
        "editor": mock_agent("editor", "editing"),
    }


# ==================== Core Classes Tests ====================


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_creation(self, mock_agent):
        """Test basic creation."""
        agent = mock_agent("test")
        config = AgentConfig(agent=agent, name="test", role="tester")

        assert config.agent is agent
        assert config.name == "test"
        assert config.role == "tester"

    def test_auto_name_from_agent(self, mock_agent):
        """Test name is taken from agent if not provided."""
        agent = mock_agent("my_agent")
        config = AgentConfig(agent=agent)

        assert config.name == "my_agent"


class TestTopologyResult:
    """Tests for TopologyResult."""

    def test_creation(self):
        """Test basic creation."""
        result = TopologyResult(
            output="Final output",
            agent_outputs={"a": "output a", "b": "output b"},
            execution_order=["a", "b"],
        )

        assert result.output == "Final output"
        assert result.agent_outputs == {"a": "output a", "b": "output b"}
        assert result.execution_order == ["a", "b"]
        assert result.rounds == 1

    def test_success_property(self):
        """Test success property."""
        result = TopologyResult(output="Success")
        assert result.success is True

        empty_result = TopologyResult(output="")
        assert empty_result.success is False


class TestTopologyType:
    """Tests for TopologyType enum."""

    def test_values(self):
        """Test enum values."""
        assert TopologyType.SUPERVISOR.value == "supervisor"
        assert TopologyType.PIPELINE.value == "pipeline"
        assert TopologyType.MESH.value == "mesh"
        assert TopologyType.HIERARCHICAL.value == "hierarchical"


# ==================== Supervisor Tests ====================


class TestSupervisor:
    """Tests for Supervisor topology."""

    @pytest.fixture
    def supervisor_topology(self, mock_agents):
        """Create a supervisor topology."""
        return Supervisor(
            coordinator=AgentConfig(
                agent=mock_agents["researcher"],
                role="coordinator",
            ),
            workers=[
                AgentConfig(agent=mock_agents["writer"], role="content"),
                AgentConfig(agent=mock_agents["editor"], role="editing"),
            ],
        )

    def test_creation(self, supervisor_topology):
        """Test supervisor topology creation."""
        assert supervisor_topology.topology_type == TopologyType.SUPERVISOR
        assert supervisor_topology.coordinator.name == "researcher"
        assert len(supervisor_topology.workers) == 2

    def test_get_agents(self, supervisor_topology):
        """Test get_agents returns all agents."""
        agents = supervisor_topology.get_agents()
        names = [a.name for a in agents]

        assert "researcher" in names
        assert "writer" in names
        assert "editor" in names

    @pytest.mark.asyncio
    async def test_run(self, supervisor_topology):
        """Test supervisor run execution."""
        result = await supervisor_topology.run("Test task")

        assert isinstance(result, TopologyResult)
        assert result.output  # Has final output
        assert "researcher" in result.execution_order[0]  # Coordinator starts

    def test_parallel_flag(self, mock_agents):
        """Test parallel flag."""
        parallel = Supervisor(
            coordinator=AgentConfig(agent=mock_agents["researcher"]),
            workers=[AgentConfig(agent=mock_agents["writer"])],
            parallel=True,
        )
        assert parallel.parallel is True

        sequential = Supervisor(
            coordinator=AgentConfig(agent=mock_agents["researcher"]),
            workers=[AgentConfig(agent=mock_agents["writer"])],
            parallel=False,
        )
        assert sequential.parallel is False


# ==================== Pipeline Tests ====================


class TestPipeline:
    """Tests for Pipeline topology."""

    @pytest.fixture
    def pipeline_topology(self, mock_agents):
        """Create a pipeline topology."""
        return Pipeline(
            stages=[
                AgentConfig(agent=mock_agents["researcher"], role="research"),
                AgentConfig(agent=mock_agents["writer"], role="draft"),
                AgentConfig(agent=mock_agents["editor"], role="polish"),
            ]
        )

    def test_creation(self, pipeline_topology):
        """Test pipeline topology creation."""
        assert pipeline_topology.topology_type == TopologyType.PIPELINE
        assert len(pipeline_topology.stages) == 3

    def test_get_agents(self, pipeline_topology):
        """Test get_agents returns all stages."""
        agents = pipeline_topology.get_agents()
        assert len(agents) == 3

    @pytest.mark.asyncio
    async def test_run(self, pipeline_topology):
        """Test pipeline run execution."""
        result = await pipeline_topology.run("Test task")

        assert isinstance(result, TopologyResult)
        assert result.output  # Has final output
        # Should execute in order
        assert result.execution_order == ["researcher", "writer", "editor"]

    @pytest.mark.asyncio
    async def test_sequential_execution(self, mock_agents):
        """Test that stages run sequentially."""
        call_order = []

        mock_agents["researcher"].run = AsyncMock(
            side_effect=lambda _: (call_order.append("researcher"), "Output from researcher")[1]
        )
        mock_agents["writer"].run = AsyncMock(
            side_effect=lambda _: (call_order.append("writer"), "Output from writer")[1]
        )

        topo = Pipeline(
            stages=[
                AgentConfig(agent=mock_agents["researcher"]),
                AgentConfig(agent=mock_agents["writer"]),
            ]
        )

        await topo.run("Test")
        assert "researcher" in call_order
        assert "writer" in call_order


# ==================== Mesh Tests ====================


class TestMesh:
    """Tests for Mesh topology."""

    @pytest.fixture
    def mesh_topology(self, mock_agents):
        """Create a mesh topology."""
        return Mesh(
            agents=[
                AgentConfig(agent=mock_agents["researcher"], role="analyst1"),
                AgentConfig(agent=mock_agents["writer"], role="analyst2"),
            ],
            max_rounds=2,
        )

    def test_creation(self, mesh_topology):
        """Test mesh topology creation."""
        assert mesh_topology.topology_type == TopologyType.MESH
        assert len(mesh_topology.agents) == 2
        assert mesh_topology.max_rounds == 2

    def test_get_agents(self, mesh_topology):
        """Test get_agents returns all agents."""
        agents = mesh_topology.get_agents()
        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_run(self, mesh_topology):
        """Test mesh run execution."""
        result = await mesh_topology.run("Test task")

        assert isinstance(result, TopologyResult)
        assert result.output  # Has final output
        assert result.rounds == 2  # Used max_rounds

    def test_with_synthesizer(self, mock_agents, mock_agent):
        """Test mesh with dedicated synthesizer."""
        synth = mock_agent("synthesizer")
        topo = Mesh(
            agents=[
                AgentConfig(agent=mock_agents["researcher"]),
                AgentConfig(agent=mock_agents["writer"]),
            ],
            synthesizer=AgentConfig(agent=synth),
        )

        assert topo.synthesizer is not None
        assert topo.synthesizer.name == "synthesizer"


# ==================== Hierarchical Tests ====================


class TestHierarchical:
    """Tests for Hierarchical topology."""

    @pytest.fixture
    def hierarchical_topology(self, mock_agents, mock_agent):
        """Create a hierarchical topology."""
        ceo = mock_agent("ceo")
        return Hierarchical(
            root=AgentConfig(agent=ceo, role="executive"),
            structure={
                "ceo": [
                    AgentConfig(agent=mock_agents["researcher"], role="lead"),
                    AgentConfig(agent=mock_agents["writer"], role="lead"),
                ],
            },
        )

    def test_creation(self, hierarchical_topology):
        """Test hierarchical topology creation."""
        assert hierarchical_topology.topology_type == TopologyType.HIERARCHICAL
        assert hierarchical_topology.root.name == "ceo"
        assert "ceo" in hierarchical_topology.structure

    def test_get_agents(self, hierarchical_topology):
        """Test get_agents returns all agents."""
        agents = hierarchical_topology.get_agents()
        names = [a.name for a in agents]

        assert "ceo" in names
        assert "researcher" in names
        assert "writer" in names

    @pytest.mark.asyncio
    async def test_run(self, hierarchical_topology):
        """Test hierarchical run execution."""
        result = await hierarchical_topology.run("Test task")

        assert isinstance(result, TopologyResult)
        assert result.output  # Has final output


# ==================== Convenience Functions Tests ====================


class TestConvenienceFunctions:
    """Tests for convenience factory functions."""

    def test_supervisor_function(self, mock_agents):
        """Test supervisor() convenience function."""
        topo = supervisor(
            coordinator=mock_agents["researcher"],
            workers=[mock_agents["writer"], mock_agents["editor"]],
            parallel=False,
        )

        assert isinstance(topo, Supervisor)
        assert topo.coordinator.name == "researcher"
        assert len(topo.workers) == 2
        assert topo.parallel is False

    def test_pipeline_function(self, mock_agents):
        """Test pipeline() convenience function."""
        topo = pipeline(
            stages=[
                mock_agents["researcher"],
                mock_agents["writer"],
                mock_agents["editor"],
            ],
            roles=["research", "draft", "polish"],
        )

        assert isinstance(topo, Pipeline)
        assert len(topo.stages) == 3
        assert topo.stages[0].role == "research"
        assert topo.stages[1].role == "draft"
        assert topo.stages[2].role == "polish"

    def test_mesh_function(self, mock_agents):
        """Test mesh() convenience function."""
        topo = mesh(
            agents=[mock_agents["researcher"], mock_agents["writer"]],
            max_rounds=5,
            roles=["analyst1", "analyst2"],
        )

        assert isinstance(topo, Mesh)
        assert len(topo.agents) == 2
        assert topo.max_rounds == 5
        assert topo.agents[0].role == "analyst1"


# ==================== Integration Tests ====================


class TestTopologyIntegration:
    """Integration tests for topologies."""

    @pytest.mark.asyncio
    async def test_pipeline_passes_output(self, mock_agent):
        """Test that pipeline passes output between stages."""
        outputs = []

        async def stage1_run(prompt):
            outputs.append(("stage1", prompt))
            return "Stage 1 output"

        async def stage2_run(prompt):
            outputs.append(("stage2", prompt))
            return "Stage 2 output"

        agent1 = mock_agent("stage1")
        agent1.run = AsyncMock(side_effect=stage1_run)

        agent2 = mock_agent("stage2")
        agent2.run = AsyncMock(side_effect=stage2_run)

        topo = Pipeline(
            stages=[
                AgentConfig(agent=agent1),
                AgentConfig(agent=agent2),
            ]
        )

        result = await topo.run("Initial task")

        # Stage 2 should have received context from stage 1
        assert len(outputs) == 2
        assert "Stage 1 output" in outputs[1][1]  # Stage 2 saw stage 1's output

    @pytest.mark.asyncio
    async def test_stream_yields_events(self, mock_agents):
        """Test that stream yields status events."""
        topo = Pipeline(
            stages=[AgentConfig(agent=mock_agents["researcher"])]
        )

        events = []
        async for event in topo.stream("Test task"):
            events.append(event)

        assert len(events) >= 1
        # Should have status and output events
        event_types = [e["type"] for e in events]
        assert "status" in event_types or "output" in event_types


# ==================== TeamMemory Integration Tests ====================


class TestTeamMemoryIntegration:
    """Tests for TeamMemory integration with topologies."""

    @pytest.mark.asyncio
    async def test_supervisor_with_team_memory(self, mock_agents):
        """Test that Supervisor uses TeamMemory correctly."""
        from agenticflow.memory import TeamMemory

        team_memory = TeamMemory(team_id="test-supervisor")

        topo = Supervisor(
            coordinator=AgentConfig(agent=mock_agents["researcher"], role="coordinator"),
            workers=[
                AgentConfig(agent=mock_agents["writer"], role="writer"),
                AgentConfig(agent=mock_agents["editor"], role="editor"),
            ],
        )

        await topo.run("Write an article", team_memory=team_memory)

        # Check that statuses were reported
        statuses = await team_memory.get_agent_statuses()
        assert "researcher" in statuses
        assert statuses["researcher"] == "done"
        assert "writer" in statuses
        assert statuses["writer"] == "done"
        assert "editor" in statuses
        assert statuses["editor"] == "done"

        # Check that results were shared
        results = await team_memory.get_agent_results()
        assert "researcher" in results
        # Coordinator's final result is the synthesis
        assert "synthesis" in results["researcher"]
        assert results["researcher"]["final"] is True

    @pytest.mark.asyncio
    async def test_pipeline_with_team_memory(self, mock_agents):
        """Test that Pipeline uses TeamMemory correctly."""
        from agenticflow.memory import TeamMemory

        team_memory = TeamMemory(team_id="test-pipeline")

        topo = Pipeline(
            stages=[
                AgentConfig(agent=mock_agents["researcher"], role="research"),
                AgentConfig(agent=mock_agents["writer"], role="draft"),
                AgentConfig(agent=mock_agents["editor"], role="polish"),
            ]
        )

        await topo.run("Create content", team_memory=team_memory)

        # Check statuses
        statuses = await team_memory.get_agent_statuses()
        assert all(status == "done" for status in statuses.values())

        # Check results
        results = await team_memory.get_agent_results()
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_mesh_with_team_memory(self, mock_agents):
        """Test that Mesh uses TeamMemory correctly."""
        from agenticflow.memory import TeamMemory

        team_memory = TeamMemory(team_id="test-mesh")

        topo = Mesh(
            agents=[
                AgentConfig(agent=mock_agents["researcher"], role="analyst1"),
                AgentConfig(agent=mock_agents["writer"], role="analyst2"),
            ],
            max_rounds=2,
        )

        await topo.run("Analyze topic", team_memory=team_memory)

        # Check statuses - all agents should be done
        statuses = await team_memory.get_agent_statuses()
        assert "researcher" in statuses
        # First agent synthesizes by default
        assert statuses["researcher"] == "done"

        # Check results
        results = await team_memory.get_agent_results()
        assert "researcher" in results
        # Final synthesis should be stored
        assert "synthesis" in results["researcher"] or "output" in results["researcher"]

    @pytest.mark.asyncio
    async def test_hierarchical_with_team_memory(self, mock_agent):
        """Test that Hierarchical uses TeamMemory correctly."""
        from agenticflow.memory import TeamMemory

        team_memory = TeamMemory(team_id="test-hierarchical")

        root = mock_agent("manager")
        worker1 = mock_agent("worker1")
        worker2 = mock_agent("worker2")

        topo = Hierarchical(
            root=AgentConfig(agent=root, role="manager"),
            structure={
                "manager": [
                    AgentConfig(agent=worker1, role="developer"),
                    AgentConfig(agent=worker2, role="tester"),
                ]
            },
        )

        await topo.run("Build feature", team_memory=team_memory)

        # Check statuses
        statuses = await team_memory.get_agent_statuses()
        assert "manager" in statuses
        assert statuses["manager"] == "done"
        assert "worker1" in statuses
        assert "worker2" in statuses

    @pytest.mark.asyncio
    async def test_topology_without_team_memory(self, mock_agents):
        """Test that topologies still work without TeamMemory."""
        topo = Pipeline(
            stages=[
                AgentConfig(agent=mock_agents["researcher"]),
                AgentConfig(agent=mock_agents["writer"]),
            ]
        )

        # Should work fine without team_memory
        result = await topo.run("Test task")
        assert result.success
        assert len(result.agent_outputs) == 2

    @pytest.mark.asyncio
    async def test_stream_with_team_memory(self, mock_agents):
        """Test that stream works with TeamMemory."""
        from agenticflow.memory import TeamMemory

        team_memory = TeamMemory(team_id="test-stream")

        topo = Pipeline(
            stages=[AgentConfig(agent=mock_agents["researcher"])]
        )

        events = []
        async for event in topo.stream("Test task", team_memory=team_memory):
            events.append(event)

        assert len(events) >= 1
        # Check memory was updated
        statuses = await team_memory.get_agent_statuses()
        assert "researcher" in statuses
