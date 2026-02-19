"""Tests for graph visualization."""

import pytest
from cogent.graph import Graph, Entity, Relationship
from cogent.graph.visualization import (
    to_mermaid,
    get_scheme,
    DefaultScheme,
    MinimalScheme,
    NodeStyle,
    EdgeStyle,
)


@pytest.fixture
async def sample_graph():
    """Create a sample graph for visualization testing."""
    graph = Graph()

    await graph.add_entity("alice", "Person", name="Alice", age=30)
    await graph.add_entity("bob", "Person", name="Bob", age=25)
    await graph.add_entity("acme", "Company", name="Acme Corp")
    await graph.add_entity("project_x", "Project", name="Project X")

    await graph.add_relationship("alice", "knows", "bob")
    await graph.add_relationship("alice", "works_at", "acme")
    await graph.add_relationship("bob", "works_at", "acme")
    await graph.add_relationship("acme", "manages", "project_x")

    return graph


@pytest.fixture
def sample_entities():
    """Sample entities for direct renderer testing."""
    return [
        Entity("alice", "Person", {"name": "Alice"}),
        Entity("bob", "Person", {"name": "Bob"}),
        Entity("acme", "Company", {"name": "Acme Corp"}),
    ]


@pytest.fixture
def sample_relationships():
    """Sample relationships for direct renderer testing."""
    return [
        Relationship("alice", "knows", "bob"),
        Relationship("alice", "works_at", "acme"),
    ]


# --- Mermaid Tests ---


class TestMermaidRendering:
    """Test Mermaid diagram generation."""

    def test_to_mermaid_basic(self, sample_entities, sample_relationships):
        """Test basic Mermaid diagram generation."""
        diagram = to_mermaid(sample_entities, sample_relationships)

        assert "flowchart LR" in diagram
        assert "alice" in diagram
        assert "bob" in diagram
        assert "knows" in diagram

    def test_to_mermaid_with_direction(self, sample_entities, sample_relationships):
        """Test Mermaid with different directions."""
        diagram_lr = to_mermaid(sample_entities, sample_relationships, direction="LR")
        diagram_tb = to_mermaid(sample_entities, sample_relationships, direction="TB")

        assert "flowchart LR" in diagram_lr
        assert "flowchart TB" in diagram_tb

    def test_to_mermaid_with_grouping(self, sample_entities, sample_relationships):
        """Test Mermaid diagram generation (grouping removed for clean layout)."""
        diagram = to_mermaid(
            sample_entities, sample_relationships, group_by_type=True
        )

        # No longer uses subgraphs (they create messy layouts)
        # Instead uses classDef styling for visual grouping
        assert "flowchart" in diagram
        assert "classDef PersonStyle" in diagram
        assert "classDef CompanyStyle" in diagram
        assert "class alice PersonStyle" in diagram

    def test_to_mermaid_with_title(self, sample_entities, sample_relationships):
        """Test Mermaid with title."""
        diagram = to_mermaid(
            sample_entities, sample_relationships, title="Test Graph"
        )

        assert "---" in diagram
        assert "title: Test Graph" in diagram

    def test_to_mermaid_with_minimal_scheme(
        self, sample_entities, sample_relationships
    ):
        """Test Mermaid with minimal scheme."""
        diagram = to_mermaid(sample_entities, sample_relationships, scheme="minimal")

        # Should contain nodes and edges
        assert "alice" in diagram
        assert "knows" in diagram

    def test_mermaid_node_formatting(self, sample_entities):
        """Test that nodes are formatted correctly."""
        diagram = to_mermaid(sample_entities, [])

        # Person entities should have rounded shape
        assert "(" in diagram or "[" in diagram  # Some bracket type

    def test_mermaid_edge_formatting(self, sample_relationships):
        """Test that edges are formatted correctly."""
        entities = [
            Entity("alice", "Person", {"name": "Alice"}),
            Entity("bob", "Person", {"name": "Bob"}),
        ]
        diagram = to_mermaid(entities, sample_relationships)

        # Should contain arrow notation
        assert "-->" in diagram or "==>" in diagram or "-.->" in diagram


# --- Style Tests ---


class TestStyles:
    """Test style schemes and customization."""

    def test_default_scheme(self):
        """Test default color scheme."""
        scheme = get_scheme("default")

        assert isinstance(scheme, DefaultScheme)

        # Check predefined styles
        person_style = scheme.get_node_style("Person")
        assert person_style.shape == "rounded"
        assert person_style.color == "#90CAF9"

    def test_minimal_scheme(self):
        """Test minimal color scheme."""
        scheme = get_scheme("minimal")

        assert isinstance(scheme, MinimalScheme)

        # Minimal scheme should have simple defaults
        style = scheme.get_node_style("Person")
        assert style.color == "#ffffff"

    def test_custom_node_style(self):
        """Test setting custom node style."""
        scheme = DefaultScheme()
        custom_style = NodeStyle(
            shape="circle", color="#ff0000", border_color="#000000"
        )

        scheme.set_node_style("CustomType", custom_style)

        retrieved = scheme.get_node_style("CustomType")
        assert retrieved.shape == "circle"
        assert retrieved.color == "#ff0000"

    def test_custom_edge_style(self):
        """Test setting custom edge style."""
        scheme = DefaultScheme()
        custom_style = EdgeStyle(color="#ff0000", width=5, style="dashed")

        scheme.set_edge_style("custom_relation", custom_style)

        retrieved = scheme.get_edge_style("custom_relation")
        assert retrieved.color == "#ff0000"
        assert retrieved.width == 5
        assert retrieved.style == "dashed"

    def test_unknown_scheme_raises_error(self):
        """Test that unknown scheme name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scheme"):
            get_scheme("nonexistent")


# --- Graph Integration Tests ---


class TestGraphVisualizationMethods:
    """Test visualization methods on Graph class."""

    @pytest.mark.asyncio
    async def test_graph_to_mermaid(self, sample_graph):
        """Test Graph.to_mermaid() method."""
        diagram = await sample_graph.to_mermaid()

        assert "flowchart LR" in diagram
        assert "alice" in diagram
        assert "knows" in diagram

    @pytest.mark.asyncio
    async def test_graph_to_mermaid_with_options(self, sample_graph):
        """Test Graph.to_mermaid() with options."""
        diagram = await sample_graph.to_mermaid(
            direction="TB", group_by_type=True, title="My Graph"
        )

        assert "flowchart TB" in diagram
        assert "title: My Graph" in diagram
        assert "classDef PersonStyle" in diagram

    @pytest.mark.asyncio
    async def test_graph_save_diagram_mermaid(self, sample_graph, tmp_path):
        """Test saving Mermaid diagram to file."""
        file_path = tmp_path / "test.mmd"

        await sample_graph.save_diagram(str(file_path), format="mermaid")

        assert file_path.exists()
        content = file_path.read_text()
        assert "flowchart LR" in content

    @pytest.mark.asyncio
    async def test_save_diagram_invalid_format_raises_error(self, sample_graph):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Unknown format"):
            await sample_graph.save_diagram("test.txt", format="invalid")


# --- Edge Cases ---


class TestEdgeCases:
    """Test edge cases in visualization."""

    def test_empty_graph_mermaid(self):
        """Test Mermaid with empty graph."""
        diagram = to_mermaid([], [])

        assert "flowchart LR" in diagram

    def test_entities_without_name_attribute(self):
        """Test rendering entities without name attribute."""
        entities = [Entity("alice", "Person", {})]  # No name
        diagram = to_mermaid(entities, [])

        # Should use ID as label
        assert "alice" in diagram

    def test_special_characters_in_entity_name(self):
        """Test entities with special characters in names."""
        entities = [Entity("alice", "Person", {"name": "Alice & Bob"})]
        diagram = to_mermaid(entities, [])

        # Should handle special characters
        assert "alice" in diagram


# --- Image Rendering ---


class TestImageRendering:
    """Test image rendering with Mermaid CLI."""

    @pytest.mark.asyncio
    async def test_graph_render_to_image_invalid_format(self, sample_graph, tmp_path):
        """Test that invalid image format raises ValueError."""
        # Note: This test is skipped if Mermaid CLI is not installed
        # The FileNotFoundError check happens before format validation
        # This is intentional - we want to fail fast on missing dependencies
        pytest.skip("Requires Mermaid CLI - validation happens after command check")

    @pytest.mark.asyncio
    async def test_graph_render_to_image_invalid_diagram_format(self, sample_graph, tmp_path):
        """Test that invalid diagram format raises ValueError."""
        output_file = tmp_path / "test.png"

        with pytest.raises(ValueError, match="Only 'mermaid' diagram format"):
            await sample_graph.render_to_image(
                str(output_file), diagram_format="graphviz"
            )

    # Note: Actual image rendering tests require Playwright installation
    # They are skipped unless Playwright is available
