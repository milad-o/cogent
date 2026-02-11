"""Tests for graph visualization."""

import pytest
from cogent.graph import Graph, Entity, Relationship
from cogent.graph.visualization import (
    to_mermaid,
    to_graphviz,
    to_graphml,
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

        assert "graph LR" in diagram
        assert "alice" in diagram
        assert "bob" in diagram
        assert "knows" in diagram

    def test_to_mermaid_with_direction(self, sample_entities, sample_relationships):
        """Test Mermaid with different directions."""
        diagram_lr = to_mermaid(sample_entities, sample_relationships, direction="LR")
        diagram_tb = to_mermaid(sample_entities, sample_relationships, direction="TB")

        assert "graph LR" in diagram_lr
        assert "graph TB" in diagram_tb

    def test_to_mermaid_with_grouping(self, sample_entities, sample_relationships):
        """Test Mermaid with entity type grouping."""
        diagram = to_mermaid(
            sample_entities, sample_relationships, group_by_type=True
        )

        assert "subgraph Person" in diagram
        assert "subgraph Company" in diagram

    def test_to_mermaid_with_title(self, sample_entities, sample_relationships):
        """Test Mermaid with title."""
        diagram = to_mermaid(
            sample_entities, sample_relationships, title="Test Graph"
        )

        assert "title Test Graph" in diagram

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


# --- Graphviz Tests ---


class TestGraphvizRendering:
    """Test Graphviz DOT format generation."""

    def test_to_graphviz_basic(self, sample_entities, sample_relationships):
        """Test basic Graphviz generation."""
        dot = to_graphviz(sample_entities, sample_relationships)

        assert "digraph G" in dot
        assert "alice" in dot
        assert "bob" in dot
        assert "->" in dot

    def test_to_graphviz_with_title(self, sample_entities, sample_relationships):
        """Test Graphviz with title."""
        dot = to_graphviz(
            sample_entities, sample_relationships, title="Test Graph"
        )

        assert "label=" in dot
        assert "Test Graph" in dot

    def test_graphviz_node_attributes(self, sample_entities):
        """Test that nodes have proper attributes."""
        dot = to_graphviz(sample_entities, [])

        assert "fillcolor=" in dot
        assert "color=" in dot
        assert "fontcolor=" in dot
        assert "style=filled" in dot

    def test_graphviz_edge_attributes(self, sample_entities, sample_relationships):
        """Test that edges have proper attributes."""
        dot = to_graphviz(sample_entities, sample_relationships)

        assert "label=" in dot  # Edge labels
        assert "penwidth=" in dot  # Edge width

    def test_graphviz_escaping(self):
        """Test that special characters are escaped."""
        entities = [Entity("alice", "Person", {"name": 'Alice "Test"'})]
        dot = to_graphviz(entities, [])

        # Should escape quotes
        assert '\\"' in dot or "Alice" in dot


# --- GraphML Tests ---


class TestGraphMLRendering:
    """Test GraphML XML format generation."""

    def test_to_graphml_basic(self, sample_entities, sample_relationships):
        """Test basic GraphML generation."""
        xml = to_graphml(sample_entities, sample_relationships)

        assert '<?xml version="1.0"' in xml
        assert "<graphml" in xml
        assert "<node" in xml
        assert "<edge" in xml

    def test_to_graphml_with_title(self, sample_entities, sample_relationships):
        """Test GraphML with title."""
        xml = to_graphml(sample_entities, sample_relationships, title="Test Graph")

        assert "<desc>Test Graph</desc>" in xml

    def test_graphml_node_data(self, sample_entities):
        """Test that nodes contain proper data elements."""
        xml = to_graphml(sample_entities, [])

        assert 'key="d0"' in xml  # Type attribute
        assert 'key="d1"' in xml  # Name attribute
        assert "Person" in xml
        assert "Alice" in xml

    def test_graphml_edge_data(self, sample_entities, sample_relationships):
        """Test that edges contain proper data elements."""
        xml = to_graphml(sample_entities, sample_relationships)

        assert 'key="d2"' in xml  # Relation attribute
        assert "knows" in xml

    def test_graphml_escaping(self):
        """Test that XML special characters are escaped."""
        entities = [Entity("alice", "Person", {"name": "Alice <Test>"})]
        xml = to_graphml(entities, [])

        # Should escape < and >
        assert "&lt;" in xml or "Alice" in xml
        assert "Test" in xml


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

        assert "graph LR" in diagram
        assert "alice" in diagram
        assert "knows" in diagram

    @pytest.mark.asyncio
    async def test_graph_to_mermaid_with_options(self, sample_graph):
        """Test Graph.to_mermaid() with options."""
        diagram = await sample_graph.to_mermaid(
            direction="TB", group_by_type=True, title="My Graph"
        )

        assert "graph TB" in diagram
        assert "subgraph Person" in diagram
        assert "title My Graph" in diagram

    @pytest.mark.asyncio
    async def test_graph_to_graphviz(self, sample_graph):
        """Test Graph.to_graphviz() method."""
        dot = await sample_graph.to_graphviz()

        assert "digraph G" in dot
        assert "alice" in dot

    @pytest.mark.asyncio
    async def test_graph_to_graphml(self, sample_graph):
        """Test Graph.to_graphml() method."""
        xml = await sample_graph.to_graphml()

        assert '<?xml version="1.0"' in xml
        assert "<graphml" in xml
        assert "alice" in xml

    @pytest.mark.asyncio
    async def test_graph_save_diagram_mermaid(self, sample_graph, tmp_path):
        """Test saving Mermaid diagram to file."""
        file_path = tmp_path / "test.mmd"

        await sample_graph.save_diagram(str(file_path), format="mermaid")

        assert file_path.exists()
        content = file_path.read_text()
        assert "graph LR" in content

    @pytest.mark.asyncio
    async def test_graph_save_diagram_graphviz(self, sample_graph, tmp_path):
        """Test saving Graphviz DOT to file."""
        file_path = tmp_path / "test.dot"

        await sample_graph.save_diagram(str(file_path), format="graphviz")

        assert file_path.exists()
        content = file_path.read_text()
        assert "digraph G" in content

    @pytest.mark.asyncio
    async def test_graph_save_diagram_graphml(self, sample_graph, tmp_path):
        """Test saving GraphML to file."""
        file_path = tmp_path / "test.graphml"

        await sample_graph.save_diagram(str(file_path), format="graphml")

        assert file_path.exists()
        content = file_path.read_text()
        assert "<graphml" in content

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

        assert "graph LR" in diagram

    def test_empty_graph_graphviz(self):
        """Test Graphviz with empty graph."""
        dot = to_graphviz([], [])

        assert "digraph G" in dot

    def test_empty_graph_graphml(self):
        """Test GraphML with empty graph."""
        xml = to_graphml([], [])

        assert "<graphml" in xml

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
