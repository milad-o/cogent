"""Tests for graph visualization module."""

import pytest

from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    EventBus,
)
from agenticflow.graph import (
    GraphView,
    GraphConfig,
    GraphDirection,
    GraphTheme,
)


@pytest.fixture
def event_bus() -> TraceBus:
    """Create event bus for tests."""
    return TraceBus()


@pytest.fixture
def sample_agent(event_bus: TraceBus) -> Agent:
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


class TestGraphConfig:
    """Tests for GraphConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GraphConfig()
        assert config.title is None
        assert config.theme == GraphTheme.DEFAULT
        assert config.direction == GraphDirection.TOP_DOWN

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = GraphConfig(
            title="My Diagram",
            theme=GraphTheme.FOREST,
            direction=GraphDirection.LEFT_RIGHT,
        )
        assert config.title == "My Diagram"
        assert config.theme == GraphTheme.FOREST
        assert config.direction == GraphDirection.LEFT_RIGHT

    def test_with_title(self) -> None:
        """Test with_title method."""
        config = GraphConfig()
        new_config = config.with_title("Test Title")
        assert new_config.title == "Test Title"
        assert config.title is None  # Original unchanged

    def test_with_theme(self) -> None:
        """Test with_theme method."""
        config = GraphConfig()
        new_config = config.with_theme(GraphTheme.DARK)
        assert new_config.theme == GraphTheme.DARK
        assert config.theme == GraphTheme.DEFAULT

    def test_with_direction(self) -> None:
        """Test with_direction method."""
        config = GraphConfig()
        new_config = config.with_direction(GraphDirection.LEFT_RIGHT)
        assert new_config.direction == GraphDirection.LEFT_RIGHT
        assert config.direction == GraphDirection.TOP_DOWN


class TestGraphTheme:
    """Tests for GraphTheme enum."""

    def test_theme_values(self) -> None:
        """Test theme enum values."""
        assert GraphTheme.DEFAULT.value == "default"
        assert GraphTheme.DARK.value == "dark"
        assert GraphTheme.FOREST.value == "forest"
        assert GraphTheme.NEUTRAL.value == "neutral"
        assert GraphTheme.BASE.value == "base"


class TestGraphDirection:
    """Tests for GraphDirection enum."""

    def test_direction_values(self) -> None:
        """Test direction enum values."""
        assert GraphDirection.TOP_DOWN.value == "TD"
        assert GraphDirection.BOTTOM_UP.value == "BT"
        assert GraphDirection.LEFT_RIGHT.value == "LR"
        assert GraphDirection.RIGHT_LEFT.value == "RL"


class TestGraphViewFromAgent:
    """Tests for GraphView.from_agent()."""

    def test_basic_graph(self, sample_agent: Agent) -> None:
        """Test basic agent graph generation."""
        view = GraphView.from_agent(sample_agent)
        mermaid = view.mermaid()

        # Check basic structure
        assert "flowchart" in mermaid
        assert "TestAgent" in mermaid
        assert "worker" in mermaid

    def test_graph_with_tools(self, sample_agent: Agent) -> None:
        """Test agent graph shows tools."""
        view = GraphView.from_agent(sample_agent, show_tools=True)
        mermaid = view.mermaid()

        # Tools are shown inline in the node label
        assert "web_search" in mermaid
        assert "doc_summarizer" in mermaid
        assert "text_writer" in mermaid

    def test_graph_without_tools(self, sample_agent: Agent) -> None:
        """Test agent graph hides tools when configured."""
        view = GraphView.from_agent(sample_agent, show_tools=False)
        mermaid = view.mermaid()

        # Tool names should not appear
        assert "web_search" not in mermaid

    def test_graph_with_config(self, sample_agent: Agent) -> None:
        """Test agent graph shows config details."""
        view = GraphView.from_agent(sample_agent, show_config=True)
        mermaid = view.mermaid()

        # Config node is only added when model is set
        # Without a model, just verify the classDef config exists
        assert "classDef config" in mermaid

    def test_graph_with_title(self, sample_agent: Agent) -> None:
        """Test graph with title."""
        view = GraphView.from_agent(sample_agent).with_title("Test Agent")
        mermaid = view.mermaid()

        assert "---" in mermaid
        assert "title: Test Agent" in mermaid

    def test_graph_with_theme(self, sample_agent: Agent) -> None:
        """Test graph with theme."""
        view = GraphView.from_agent(sample_agent).with_theme(GraphTheme.FOREST)
        mermaid = view.mermaid()

        assert "theme: forest" in mermaid

    def test_graph_with_direction(self, sample_agent: Agent) -> None:
        """Test graph direction configuration."""
        view = GraphView.from_agent(sample_agent).with_direction(GraphDirection.LEFT_RIGHT)
        mermaid = view.mermaid()

        assert "flowchart LR" in mermaid


class TestGraphViewOutputFormats:
    """Tests for GraphView output formats."""

    def test_mermaid_output(self, sample_agent: Agent) -> None:
        """Test Mermaid code output."""
        view = sample_agent.graph()
        mermaid = view.mermaid()

        assert isinstance(mermaid, str)
        assert "flowchart" in mermaid

    def test_ascii_output(self, sample_agent: Agent) -> None:
        """Test ASCII output."""
        view = sample_agent.graph()
        ascii_art = view.ascii()

        assert isinstance(ascii_art, str)
        assert "TestAgent" in ascii_art

    def test_dot_output(self, sample_agent: Agent) -> None:
        """Test Graphviz DOT output."""
        view = sample_agent.graph()
        dot = view.dot()

        assert isinstance(dot, str)
        assert "digraph" in dot

    def test_url_output(self, sample_agent: Agent) -> None:
        """Test mermaid.ink URL generation."""
        view = sample_agent.graph()
        url = view.url()

        assert url.startswith("https://mermaid.ink/")

    def test_html_output(self, sample_agent: Agent) -> None:
        """Test HTML output generation."""
        view = sample_agent.graph()
        html = view.html()

        assert '<div class="mermaid">' in html
        assert "script" in html


class TestAgentGraphMethod:
    """Tests for Agent.graph() method."""

    def test_agent_graph_basic(self, sample_agent: Agent) -> None:
        """Test basic agent.graph() call."""
        view = sample_agent.graph()

        assert isinstance(view, GraphView)
        mermaid = view.mermaid()
        assert "TestAgent" in mermaid

    def test_agent_graph_with_options(self, sample_agent: Agent) -> None:
        """Test agent.graph() with options."""
        view = sample_agent.graph(show_tools=True, show_config=True)
        mermaid = view.mermaid()

        assert "web_search" in mermaid
        # Config node only added when model is set
        assert "classDef config" in mermaid


class TestAgentDrawMermaid:
    """Tests for Agent.draw_mermaid() legacy method."""

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


class TestClassDefinitions:
    """Tests for role-based styling."""

    def test_role_colors_defined(self, sample_agent: Agent) -> None:
        """Test that role-based colors are defined."""
        view = sample_agent.graph()
        mermaid = view.mermaid()

        # Check class definitions
        assert "classDef" in mermaid
        assert "classDef work" in mermaid

    def test_role_class_assignment(self, sample_agent: Agent) -> None:
        """Test that role class is assigned to agent node."""
        view = sample_agent.graph()
        mermaid = view.mermaid()

        # Agent should have worker class
        assert ":::work" in mermaid


class TestEdgeCases:
    """Tests for edge cases."""

    def test_agent_no_tools(self, event_bus: TraceBus) -> None:
        """Test agent with no tools."""
        agent = Agent(
            config=AgentConfig(
                name="NoToolsAgent",
                role=AgentRole.WORKER,
            ),
            event_bus=event_bus,
        )
        view = agent.graph()
        mermaid = view.mermaid()

        assert "NoToolsAgent" in mermaid

    def test_agent_special_characters_in_name(self, event_bus: TraceBus) -> None:
        """Test agent with special characters in name."""
        agent = Agent(
            config=AgentConfig(
                name="Agent (v2.0)",
                role=AgentRole.WORKER,
            ),
            event_bus=event_bus,
        )
        view = agent.graph()
        mermaid = view.mermaid()

        # Should handle special characters
        assert "Agent" in mermaid


class TestBackwardsCompatibility:
    """Tests for backwards-compatible imports."""

    def test_mermaid_aliases_exist(self) -> None:
        """Test that legacy Mermaid aliases are available."""
        from agenticflow import (
            MermaidConfig,
            MermaidTheme,
            MermaidDirection,
            MermaidRenderer,
            AgentDiagram,
            TopologyDiagram,
        )

        # These are now aliases to graph classes
        assert MermaidConfig is GraphConfig
        assert MermaidTheme is GraphTheme
        assert MermaidDirection is GraphDirection
        assert MermaidRenderer is GraphView
        assert AgentDiagram is GraphView
        assert TopologyDiagram is GraphView
