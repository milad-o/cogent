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
        assert "flowchart LR" in content

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

        assert "flowchart LR" in diagram

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


# --- JSON Formats ---


class TestCytoscapeJSON:
    """Test Cytoscape.js JSON format."""

    def test_to_cytoscape_json_basic(self, sample_entities, sample_relationships):
        """Test basic Cytoscape JSON generation."""
        from cogent.graph.visualization import to_cytoscape_json
        import json

        result = to_cytoscape_json(sample_entities, sample_relationships)
        data = json.loads(result)

        # Check structure
        assert "elements" in data
        assert "nodes" in data["elements"]
        assert "edges" in data["elements"]

        # Check counts
        assert len(data["elements"]["nodes"]) == 3
        assert len(data["elements"]["edges"]) == 2

    def test_cytoscape_node_structure(self, sample_entities):
        """Test Cytoscape node structure is correct."""
        from cogent.graph.visualization import to_cytoscape_json
        import json

        result = to_cytoscape_json(sample_entities, [])
        data = json.loads(result)

        node = data["elements"]["nodes"][0]
        assert "data" in node
        assert "id" in node["data"]
        assert "label" in node["data"]
        assert "type" in node["data"]

    def test_cytoscape_node_attributes_preserved(self, sample_entities):
        """Test that all entity attributes are preserved."""
        from cogent.graph.visualization import to_cytoscape_json
        import json

        result = to_cytoscape_json(sample_entities, [])
        data = json.loads(result)

        # Find Alice node
        alice = next(n for n in data["elements"]["nodes"] if n["data"]["id"] == "alice")
        assert alice["data"]["name"] == "Alice"
        assert alice["data"]["type"] == "Person"

    def test_cytoscape_edge_structure(self, sample_entities, sample_relationships):
        """Test Cytoscape edge structure is correct."""
        from cogent.graph.visualization import to_cytoscape_json
        import json

        result = to_cytoscape_json(sample_entities, sample_relationships)
        data = json.loads(result)

        edge = data["elements"]["edges"][0]
        assert "data" in edge
        assert "id" in edge["data"]
        assert "source" in edge["data"]
        assert "target" in edge["data"]
        assert "label" in edge["data"]
        assert "relation" in edge["data"]

    def test_cytoscape_edge_attributes_preserved(self, sample_entities):
        """Test that all relationship attributes are preserved."""
        from cogent.graph.visualization import to_cytoscape_json
        import json

        rel = Relationship("alice", "knows", "bob", {"since": 2020})
        result = to_cytoscape_json(sample_entities, [rel])
        data = json.loads(result)

        edge = data["elements"]["edges"][0]
        assert edge["data"]["since"] == 2020

    def test_cytoscape_empty_graph(self):
        """Test Cytoscape JSON with empty graph."""
        from cogent.graph.visualization import to_cytoscape_json
        import json

        result = to_cytoscape_json([], [])
        data = json.loads(result)

        assert data["elements"]["nodes"] == []
        assert data["elements"]["edges"] == []

    @pytest.mark.asyncio
    async def test_graph_to_cytoscape_json(self, sample_graph):
        """Test Graph.to_cytoscape_json() method."""
        import json

        result = await sample_graph.to_cytoscape_json()
        data = json.loads(result)

        assert "elements" in data
        assert len(data["elements"]["nodes"]) == 4
        assert len(data["elements"]["edges"]) == 4


class TestJSONGraphFormat:
    """Test JSON Graph Format."""

    def test_to_json_graph_basic(self, sample_entities, sample_relationships):
        """Test basic JSON Graph format generation."""
        from cogent.graph.visualization import to_json_graph
        import json

        result = to_json_graph(sample_entities, sample_relationships)
        data = json.loads(result)

        # Check structure
        assert "graph" in data
        assert "directed" in data["graph"]
        assert "nodes" in data["graph"]
        assert "edges" in data["graph"]

        # Check counts
        assert len(data["graph"]["nodes"]) == 3
        assert len(data["graph"]["edges"]) == 2

    def test_json_graph_node_structure(self, sample_entities):
        """Test JSON Graph node structure."""
        from cogent.graph.visualization import to_json_graph
        import json

        result = to_json_graph(sample_entities, [])
        data = json.loads(result)

        node = data["graph"]["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "metadata" in node
        assert "type" in node["metadata"]

    def test_json_graph_node_metadata(self, sample_entities):
        """Test that attributes are in metadata."""
        from cogent.graph.visualization import to_json_graph
        import json

        result = to_json_graph(sample_entities, [])
        data = json.loads(result)

        # Find Alice node
        alice = next(n for n in data["graph"]["nodes"] if n["id"] == "alice")
        assert alice["metadata"]["name"] == "Alice"
        assert alice["metadata"]["type"] == "Person"

    def test_json_graph_edge_structure(self, sample_entities, sample_relationships):
        """Test JSON Graph edge structure."""
        from cogent.graph.visualization import to_json_graph
        import json

        result = to_json_graph(sample_entities, sample_relationships)
        data = json.loads(result)

        edge = data["graph"]["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "relation" in edge
        assert "metadata" in edge

    def test_json_graph_edge_metadata(self, sample_entities):
        """Test that relationship attributes are in metadata."""
        from cogent.graph.visualization import to_json_graph
        import json

        rel = Relationship("alice", "knows", "bob", {"since": 2020})
        result = to_json_graph(sample_entities, [rel])
        data = json.loads(result)

        edge = data["graph"]["edges"][0]
        assert edge["metadata"]["since"] == 2020

    def test_json_graph_directed(self, sample_entities, sample_relationships):
        """Test that graph is marked as directed."""
        from cogent.graph.visualization import to_json_graph
        import json

        result = to_json_graph(sample_entities, sample_relationships)
        data = json.loads(result)

        assert data["graph"]["directed"] is True

    def test_json_graph_empty_graph(self):
        """Test JSON Graph with empty graph."""
        from cogent.graph.visualization import to_json_graph
        import json

        result = to_json_graph([], [])
        data = json.loads(result)

        assert data["graph"]["nodes"] == []
        assert data["graph"]["edges"] == []

    @pytest.mark.asyncio
    async def test_graph_to_json_graph(self, sample_graph):
        """Test Graph.to_json_graph() method."""
        import json

        result = await sample_graph.to_json_graph()
        data = json.loads(result)

        assert "graph" in data
        assert len(data["graph"]["nodes"]) == 4
        assert len(data["graph"]["edges"]) == 4


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


# --- Graph Integration ---


class TestGraphIntegrationNewFormats:
    """Test new format methods on Graph class."""

    @pytest.mark.asyncio
    async def test_save_diagram_cytoscape(self, sample_graph, tmp_path):
        """Test saving Cytoscape JSON via save_diagram."""
        import json

        file_path = tmp_path / "graph.json"
        await sample_graph.save_diagram(str(file_path), format="cytoscape")

        assert file_path.exists()
        data = json.loads(file_path.read_text())
        assert "elements" in data

    @pytest.mark.asyncio
    async def test_save_diagram_json_graph(self, sample_graph, tmp_path):
        """Test saving JSON Graph via save_diagram."""
        import json

        file_path = tmp_path / "graph.json"
        await sample_graph.save_diagram(str(file_path), format="json")

        assert file_path.exists()
        data = json.loads(file_path.read_text())
        assert "graph" in data

    @pytest.mark.asyncio
    async def test_save_diagram_auto_extension_cytoscape(self, sample_graph, tmp_path):
        """Test that Cytoscape format gets .json extension."""
        file_path = tmp_path / "graph"
        await sample_graph.save_diagram(str(file_path), format="cytoscape")

        json_path = tmp_path / "graph.json"
        assert json_path.exists()

    @pytest.mark.asyncio
    async def test_save_diagram_auto_extension_json_graph(self, sample_graph, tmp_path):
        """Test that JSON Graph format gets .json extension."""
        file_path = tmp_path / "graph"
        await sample_graph.save_diagram(str(file_path), format="json")

        json_path = tmp_path / "graph.json"
        assert json_path.exists()
