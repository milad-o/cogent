"""Tests for flow pattern helpers."""

import pytest
from unittest.mock import MagicMock

from agenticflow.flow.patterns import pipeline, supervisor, mesh
from agenticflow.flow.patterns import chain, coordinator, collaborative, brainstorm
from agenticflow.flow.core import Flow


def make_mock_agent(name: str) -> MagicMock:
    """Create a mock Agent for testing."""
    agent = MagicMock()
    agent.name = name
    agent.__class__.__name__ = "Agent"
    return agent


class TestPipelinePattern:
    """Tests for pipeline pattern."""

    def test_pipeline_creates_flow(self) -> None:
        """pipeline() returns a Flow."""
        agents = [make_mock_agent(f"agent_{i}") for i in range(3)]
        flow = pipeline(agents)

        assert isinstance(flow, Flow)

    def test_pipeline_requires_stages(self) -> None:
        """pipeline() requires at least 1 stage."""
        with pytest.raises(ValueError, match="at least one stage"):
            pipeline([])

    def test_chain_calls_pipeline(self) -> None:
        """chain() wraps pipeline()."""
        agents = [make_mock_agent(f"agent_{i}") for i in range(3)]
        flow = chain(agents)
        assert isinstance(flow, Flow)


class TestSupervisorPattern:
    """Tests for supervisor pattern."""

    def test_supervisor_creates_flow(self) -> None:
        """supervisor() returns a Flow."""
        coord = make_mock_agent("coordinator")
        workers = [make_mock_agent(f"worker_{i}") for i in range(3)]

        flow = supervisor(coord, workers=workers)

        assert isinstance(flow, Flow)

    def test_supervisor_requires_workers(self) -> None:
        """supervisor() requires at least 1 worker."""
        coord = make_mock_agent("coordinator")
        with pytest.raises(ValueError, match="at least one worker"):
            supervisor(coord, workers=[])

    def test_coordinator_calls_supervisor(self) -> None:
        """coordinator() wraps supervisor()."""
        coord = make_mock_agent("coordinator")
        workers = [make_mock_agent(f"worker_{i}") for i in range(3)]
        flow = coordinator(coord, workers=workers)
        assert isinstance(flow, Flow)


class TestMeshPattern:
    """Tests for mesh pattern."""

    def test_mesh_creates_flow(self) -> None:
        """mesh() returns a Flow."""
        agents = [make_mock_agent(f"agent_{i}") for i in range(3)]
        flow = mesh(agents)

        assert isinstance(flow, Flow)

    def test_mesh_requires_multiple_agents(self) -> None:
        """mesh() requires at least 2 agents."""
        with pytest.raises(ValueError, match="at least 2"):
            mesh([make_mock_agent("single")])

    def test_mesh_max_rounds(self) -> None:
        """mesh() accepts max_rounds parameter."""
        agents = [make_mock_agent(f"agent_{i}") for i in range(2)]
        flow = mesh(agents, max_rounds=5)

        assert isinstance(flow, Flow)

    def test_collaborative_calls_mesh(self) -> None:
        """collaborative() wraps mesh()."""
        agents = [make_mock_agent(f"agent_{i}") for i in range(2)]
        flow = collaborative(agents)
        assert isinstance(flow, Flow)

    def test_brainstorm_calls_mesh(self) -> None:
        """brainstorm() wraps mesh()."""
        agents = [make_mock_agent(f"agent_{i}") for i in range(2)]
        flow = brainstorm(agents)
        assert isinstance(flow, Flow)
