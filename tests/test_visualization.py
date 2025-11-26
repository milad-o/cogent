"""Tests for visualization module."""

import pytest

from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    EventBus,
    TopologyFactory,
    TopologyType,
)
from agenticflow.visualization import (
    AgentDiagram,
    MermaidConfig,
    MermaidDirection,
    MermaidRenderer,
    MermaidTheme,
    TopologyDiagram,
)


@pytest.fixture
def event_bus() -> EventBus:
    """Create event bus for tests."""
    return EventBus()


@pytest.fixture
def sample_agent(event_bus: EventBus) -> Agent:
    """Create a sample agent with tools."""
    return Agent(
        config=AgentConfig(
            name="TestAgent",
            role=AgentRole.WORKER,
            description="A test agent",
            tools=["web_search", "doc_summarizer", "text_writer"],
        ),
        event_bus=event_bus,
    )


@pytest.fixture
def sample_topology(event_bus: EventBus) -> TopologyFactory:
    """Create a sample supervisor topology."""
    supervisor = Agent(
        config=AgentConfig(
            name="Supervisor",
            role=AgentRole.SUPERVISOR,
            tools=["coordinate"],
        ),
        event_bus=event_bus,
    )
    worker1 = Agent(
        config=AgentConfig(
            name="Worker1",
            role=AgentRole.WORKER,
            tools=["task_a"],
        ),
        event_bus=event_bus,
    )
    worker2 = Agent(
        config=AgentConfig(
            name="Worker2",
            role=AgentRole.WORKER,
            tools=["task_b"],
        ),
        event_bus=event_bus,
    )
    return TopologyFactory.create(
        TopologyType.SUPERVISOR,
        "test-team",
        agents=[supervisor, worker1, worker2],
        supervisor_name="Supervisor",
    )


class TestMermaidConfig:
    """Tests for MermaidConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MermaidConfig()
        assert config.title == ""
        assert config.theme == MermaidTheme.DEFAULT
        assert config.direction == MermaidDirection.TOP_DOWN
        assert config.show_tools is True
        assert config.show_roles is True
        assert config.show_config is False

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = MermaidConfig(
            title="My Diagram",
            theme=MermaidTheme.FOREST,
            direction=MermaidDirection.LEFT_RIGHT,
            show_tools=False,
        )
        assert config.title == "My Diagram"
        assert config.theme == MermaidTheme.FOREST
        assert config.direction == MermaidDirection.LEFT_RIGHT
        assert config.show_tools is False

    def test_frontmatter_generation(self) -> None:
        """Test YAML frontmatter generation."""
        config = MermaidConfig(
            title="Test",
            theme=MermaidTheme.DARK,
        )
        frontmatter = config.to_frontmatter()
        assert "---" in frontmatter
        assert "title: Test" in frontmatter
        assert "theme: dark" in frontmatter


class TestMermaidTheme:
    """Tests for MermaidTheme enum."""

    def test_theme_values(self) -> None:
        """Test theme enum values."""
        assert MermaidTheme.DEFAULT.value == "default"
        assert MermaidTheme.DARK.value == "dark"
        assert MermaidTheme.FOREST.value == "forest"
        assert MermaidTheme.NEUTRAL.value == "neutral"
        assert MermaidTheme.BASE.value == "base"


class TestMermaidDirection:
    """Tests for MermaidDirection enum."""

    def test_direction_values(self) -> None:
        """Test direction enum values."""
        assert MermaidDirection.TOP_DOWN.value == "TD"
        assert MermaidDirection.TOP_BOTTOM.value == "TB"
        assert MermaidDirection.BOTTOM_TOP.value == "BT"
        assert MermaidDirection.LEFT_RIGHT.value == "LR"
        assert MermaidDirection.RIGHT_LEFT.value == "RL"


class TestAgentDiagram:
    """Tests for AgentDiagram class."""

    def test_basic_diagram(self, sample_agent: Agent) -> None:
        """Test basic agent diagram generation."""
        diagram = AgentDiagram(sample_agent)
        mermaid = diagram.to_mermaid()

        # Check basic structure
        assert "flowchart" in mermaid
        assert "TestAgent" in mermaid
        assert "worker" in mermaid

    def test_diagram_with_tools(self, sample_agent: Agent) -> None:
        """Test agent diagram shows tools."""
        config = MermaidConfig(show_tools=True)
        diagram = AgentDiagram(sample_agent, config=config)
        mermaid = diagram.to_mermaid()

        # New compact format shows tools inline in the node label
        assert "web_search" in mermaid
        assert "doc_summarizer" in mermaid
        assert "text_writer" in mermaid
        # Tools are shown inline, not as separate nodes
        assert "<small>" in mermaid

    def test_diagram_without_tools(self, sample_agent: Agent) -> None:
        """Test agent diagram hides tools when configured."""
        config = MermaidConfig(show_tools=False)
        diagram = AgentDiagram(sample_agent, config=config)
        mermaid = diagram.to_mermaid()

        # Tool names should not appear in tool subgraph
        assert "web_search" not in mermaid  # Use specific tool name
        assert "subgraph" not in mermaid

    def test_diagram_with_config(self, sample_agent: Agent) -> None:
        """Test agent diagram shows config details."""
        config = MermaidConfig(show_config=True)
        diagram = AgentDiagram(sample_agent, config=config)
        mermaid = diagram.to_mermaid()

        # Config section should be present (shows model info if model is set)
        assert ":::config" in mermaid
        # Without a model set, shows "no model"
        assert "no model" in mermaid

    def test_diagram_frontmatter(self, sample_agent: Agent) -> None:
        """Test YAML frontmatter in diagram."""
        config = MermaidConfig(title="Test Agent", theme=MermaidTheme.FOREST)
        diagram = AgentDiagram(sample_agent, config=config)
        mermaid = diagram.to_mermaid()

        assert "---" in mermaid
        assert "title: Test Agent" in mermaid
        assert "theme: forest" in mermaid

    def test_diagram_direction(self, sample_agent: Agent) -> None:
        """Test diagram direction configuration."""
        config = MermaidConfig(direction=MermaidDirection.LEFT_RIGHT)
        diagram = AgentDiagram(sample_agent, config=config)
        mermaid = diagram.to_mermaid()

        assert "flowchart LR" in mermaid

    def test_svg_url_generation(self, sample_agent: Agent) -> None:
        """Test SVG URL generation."""
        diagram = AgentDiagram(sample_agent)
        url = diagram.get_svg_url()

        assert url.startswith("https://mermaid.ink/svg/")

    def test_png_url_generation(self, sample_agent: Agent) -> None:
        """Test PNG URL generation."""
        diagram = AgentDiagram(sample_agent)
        url = diagram.get_png_url()

        assert url.startswith("https://mermaid.ink/img/")

    def test_html_output(self, sample_agent: Agent) -> None:
        """Test HTML output generation."""
        diagram = AgentDiagram(sample_agent)
        html = diagram.to_html()

        assert '<div class="mermaid">' in html
        assert "mermaid" in html
        assert "script" in html

    def test_repr_html(self, sample_agent: Agent) -> None:
        """Test IPython HTML representation."""
        diagram = AgentDiagram(sample_agent)
        html = diagram._repr_html_()

        assert isinstance(html, str)
        assert "mermaid" in html


class TestTopologyDiagram:
    """Tests for TopologyDiagram class."""

    def test_supervisor_topology_diagram(self, sample_topology) -> None:
        """Test supervisor topology diagram."""
        diagram = TopologyDiagram(sample_topology)
        mermaid = diagram.to_mermaid()

        # Check structure
        assert "flowchart" in mermaid
        assert "Supervisor" in mermaid
        assert "Worker1" in mermaid
        assert "Worker2" in mermaid

        # Check connection to Workers subgraph
        assert "-->" in mermaid

    def test_topology_with_tools(self, sample_topology) -> None:
        """Test topology diagram structure."""
        config = MermaidConfig(show_tools=True)
        diagram = TopologyDiagram(sample_topology, config=config)
        mermaid = diagram.to_mermaid()

        # Clean diagram shows agents grouped
        assert "subgraph" in mermaid or "-->" in mermaid

    def test_topology_without_tools(self, sample_topology) -> None:
        """Test topology diagram hides tools."""
        config = MermaidConfig(show_tools=False)
        diagram = TopologyDiagram(sample_topology, config=config)
        mermaid = diagram.to_mermaid()

        # Tools should not appear as subgraphs
        assert "🛠️" not in mermaid

    def test_pipeline_topology(self, event_bus: EventBus) -> None:
        """Test pipeline topology diagram."""
        agents = [
            Agent(
                config=AgentConfig(name=f"Stage{i}", role=AgentRole.WORKER),
                event_bus=event_bus,
            )
            for i in range(3)
        ]
        topology = TopologyFactory.create(
            TopologyType.PIPELINE,
            "pipeline",
            agents=agents,
        )
        diagram = TopologyDiagram(topology)
        mermaid = diagram.to_mermaid()

        # Check sequential connections
        assert "-->" in mermaid  # Arrow for pipeline
        assert "Stage0" in mermaid
        assert "Stage1" in mermaid
        assert "Stage2" in mermaid

    def test_mesh_topology(self, event_bus: EventBus) -> None:
        """Test mesh topology diagram."""
        agents = [
            Agent(
                config=AgentConfig(name=f"Node{i}", role=AgentRole.WORKER),
                event_bus=event_bus,
            )
            for i in range(3)
        ]
        topology = TopologyFactory.create(
            TopologyType.MESH,
            "mesh",
            agents=agents,
        )
        diagram = TopologyDiagram(topology)
        mermaid = diagram.to_mermaid()

        # Check undirected connections (peer-to-peer)
        assert "---" in mermaid
        assert "Node0" in mermaid
        assert "Node1" in mermaid
        assert "Node2" in mermaid

    def test_topology_frontmatter(self, sample_topology) -> None:
        """Test topology diagram frontmatter."""
        config = MermaidConfig(title="My Team", theme=MermaidTheme.DARK)
        diagram = TopologyDiagram(sample_topology, config=config)
        mermaid = diagram.to_mermaid()

        assert "---" in mermaid
        assert "title: My Team" in mermaid
        assert "theme: dark" in mermaid


class TestMermaidRenderer:
    """Tests for MermaidRenderer class."""

    def test_to_url_svg(self) -> None:
        """Test SVG URL generation."""
        code = "flowchart TD\n    A --> B"
        url = MermaidRenderer.to_svg_url(code)

        assert url.startswith("https://mermaid.ink/svg/")

    def test_to_url_png(self) -> None:
        """Test PNG URL generation."""
        code = "flowchart TD\n    A --> B"
        url = MermaidRenderer.to_png_url(code)

        assert url.startswith("https://mermaid.ink/img/")

    def test_to_html(self) -> None:
        """Test HTML generation."""
        code = "flowchart TD\n    A --> B"
        html = MermaidRenderer.to_html(code)

        assert '<div class="mermaid">' in html
        assert "mermaid" in html


class TestAgentDrawMermaid:
    """Tests for Agent.draw_mermaid() method."""

    def test_draw_mermaid_basic(self, sample_agent: Agent) -> None:
        """Test basic draw_mermaid call."""
        mermaid = sample_agent.draw_mermaid()

        assert "flowchart" in mermaid
        assert "TestAgent" in mermaid

    def test_draw_mermaid_with_options(self, sample_agent: Agent) -> None:
        """Test draw_mermaid with options."""
        mermaid = sample_agent.draw_mermaid(
            theme="forest",
            direction="LR",
            title="My Agent",
            show_tools=True,
            show_config=True,
        )

        assert "theme: forest" in mermaid
        assert "flowchart LR" in mermaid
        assert "title: My Agent" in mermaid
        assert "web_search" in mermaid


class TestTopologyDrawMermaid:
    """Tests for BaseTopology.draw_mermaid() method."""

    def test_draw_mermaid_basic(self, sample_topology) -> None:
        """Test basic draw_mermaid call."""
        mermaid = sample_topology.draw_mermaid()

        assert "flowchart" in mermaid
        assert "Supervisor" in mermaid

    def test_draw_mermaid_with_options(self, sample_topology) -> None:
        """Test draw_mermaid with options."""
        mermaid = sample_topology.draw_mermaid(
            theme="dark",
            direction="LR",
            title="My Team",
            show_tools=False,
        )

        assert "theme: dark" in mermaid
        assert "flowchart LR" in mermaid
        assert "title: My Team" in mermaid


class TestClassDefinitions:
    """Tests for role-based styling."""

    def test_role_colors_defined(self, sample_agent: Agent) -> None:
        """Test that role-based colors are defined."""
        diagram = AgentDiagram(sample_agent)
        mermaid = diagram.to_mermaid()

        # Check compact class definitions (only check for roles that exist)
        assert "classDef" in mermaid
        assert "classDef work" in mermaid
        assert "classDef tool" in mermaid

    def test_role_class_assignment(self, sample_agent: Agent) -> None:
        """Test that role class is assigned to agent node."""
        diagram = AgentDiagram(sample_agent)
        mermaid = diagram.to_mermaid()

        # Agent should have worker class (compact: work)
        assert ":::work" in mermaid


class TestEdgeCases:
    """Tests for edge cases."""

    def test_agent_no_tools(self, event_bus: EventBus) -> None:
        """Test agent with no tools."""
        agent = Agent(
            config=AgentConfig(
                name="NoToolsAgent",
                role=AgentRole.WORKER,
            ),
            event_bus=event_bus,
        )
        diagram = AgentDiagram(agent)
        mermaid = diagram.to_mermaid()

        # Should not have tools subgraph
        assert "subgraph" not in mermaid
        assert "NoToolsAgent" in mermaid

    def test_agent_special_characters_in_name(self, event_bus: EventBus) -> None:
        """Test agent with special characters in name."""
        agent = Agent(
            config=AgentConfig(
                name="Agent (v2.0)",
                role=AgentRole.WORKER,
            ),
            event_bus=event_bus,
        )
        diagram = AgentDiagram(agent)
        mermaid = diagram.to_mermaid()

        # Should escape special characters
        assert "Agent" in mermaid
        # Parentheses should be escaped
        assert "&#40;" in mermaid or "(" not in mermaid.split("[")[1].split("]")[0]

    def test_single_agent_topology(self, event_bus: EventBus) -> None:
        """Test topology with single agent."""
        agent = Agent(
            config=AgentConfig(
                name="Solo",
                role=AgentRole.WORKER,
            ),
            event_bus=event_bus,
        )
        topology = TopologyFactory.create(
            TopologyType.SUPERVISOR,
            "solo-team",
            agents=[agent],
            supervisor_name="Solo",
        )
        diagram = TopologyDiagram(topology)
        mermaid = diagram.to_mermaid()

        assert "Solo" in mermaid
        assert "flowchart" in mermaid
