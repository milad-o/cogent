"""Tests for query pattern matching system."""

import pytest
from cogent.graph import Graph, Entity, Relationship
from cogent.graph.query import (
    parse_pattern,
    execute_pattern,
    QueryPattern,
    QueryResult,
    match,
)


@pytest.fixture
async def populated_graph():
    """Create a graph with sample data for query testing."""
    graph = Graph()

    # Add entities
    await graph.add_entity("alice", "Person", name="Alice", age=30, city="NYC")
    await graph.add_entity("bob", "Person", name="Bob", age=25, city="SF")
    await graph.add_entity("charlie", "Person", name="Charlie", age=35, city="NYC")
    await graph.add_entity("acme", "Company", name="Acme Corp", industry="Tech")
    await graph.add_entity("beta", "Company", name="Beta Inc", industry="Finance")
    await graph.add_entity("project_x", "Project", name="Project X", status="active")

    # Add relationships
    await graph.add_relationship("alice", "knows", "bob", since=2020)
    await graph.add_relationship("bob", "knows", "charlie", since=2019)
    await graph.add_relationship("alice", "works_at", "acme")
    await graph.add_relationship("bob", "works_at", "acme")
    await graph.add_relationship("charlie", "works_at", "beta")
    await graph.add_relationship("acme", "manages", "project_x")

    return graph


# --- Pattern Parsing Tests ---


class TestPatternParsing:
    """Test parse_pattern() for different pattern types."""

    def test_parse_entity_filter_pattern(self):
        """Test parsing entity filter pattern."""
        pattern = {"type": "Person", "age": 30}
        parsed = parse_pattern(pattern)

        assert parsed.pattern_type == "entity_filter"
        assert parsed.source_filter == {"type": "Person", "age": 30}

    def test_parse_single_hop_pattern(self):
        """Test parsing single-hop relationship pattern."""
        pattern = {
            "source": {"type": "Person"},
            "relation": "knows",
            "target": {"type": "Person"},
        }
        parsed = parse_pattern(pattern)

        assert parsed.pattern_type == "single_hop"
        assert parsed.source_filter == {"type": "Person"}
        assert parsed.relation_filter == "knows"
        assert parsed.target_filter == {"type": "Person"}

    def test_parse_single_hop_partial_pattern(self):
        """Test parsing single-hop with only some fields."""
        pattern = {"source": {"id": "alice"}, "relation": "knows"}
        parsed = parse_pattern(pattern)

        assert parsed.pattern_type == "single_hop"
        assert parsed.source_filter == {"id": "alice"}
        assert parsed.relation_filter == "knows"
        assert parsed.target_filter is None

    def test_parse_multi_hop_pattern(self):
        """Test parsing multi-hop path pattern."""
        pattern = {
            "path": [
                {"source": {"type": "Person"}, "relation": "works_at"},
                {"relation": "manages", "target": {"id": "project_x"}},
            ]
        }
        parsed = parse_pattern(pattern)

        assert parsed.pattern_type == "multi_hop"
        assert parsed.path_hops is not None
        assert len(parsed.path_hops) == 2

    def test_parse_multi_hop_empty_path_raises_error(self):
        """Test that empty path raises ValueError."""
        pattern = {"path": []}

        with pytest.raises(ValueError, match="non-empty list"):
            parse_pattern(pattern)


# --- Entity Filter Tests ---


class TestEntityFiltering:
    """Test entity filtering queries."""

    @pytest.mark.asyncio
    async def test_filter_by_type(self, populated_graph):
        """Test filtering entities by type."""
        result = await match(populated_graph, {"type": "Person"})

        assert len(result.entities) == 3
        assert all(e.entity_type == "Person" for e in result.entities)
        assert {e.id for e in result.entities} == {"alice", "bob", "charlie"}

    @pytest.mark.asyncio
    async def test_filter_by_type_and_attribute(self, populated_graph):
        """Test filtering by type and attribute."""
        result = await match(populated_graph, {"type": "Person", "city": "NYC"})

        assert len(result.entities) == 2
        assert {e.id for e in result.entities} == {"alice", "charlie"}

    @pytest.mark.asyncio
    async def test_filter_by_specific_id(self, populated_graph):
        """Test filtering by specific ID."""
        result = await match(populated_graph, {"id": "alice"})

        assert len(result.entities) == 1
        assert result.entities[0].id == "alice"

    @pytest.mark.asyncio
    async def test_filter_nonexistent_entity(self, populated_graph):
        """Test filtering for nonexistent entity returns empty result."""
        result = await match(populated_graph, {"id": "nonexistent"})

        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_filter_by_multiple_attributes(self, populated_graph):
        """Test filtering by multiple attributes."""
        result = await match(
            populated_graph, {"type": "Person", "age": 30, "city": "NYC"}
        )

        assert len(result.entities) == 1
        assert result.entities[0].id == "alice"


# --- Single-Hop Relationship Tests ---


class TestSingleHopMatching:
    """Test single-hop relationship pattern matching."""

    @pytest.mark.asyncio
    async def test_match_by_relation_type(self, populated_graph):
        """Test matching relationships by type."""
        result = await match(
            populated_graph,
            {"relation": "knows"},
        )

        assert len(result.relationships) == 2
        assert all(r.relation == "knows" for r in result.relationships)

    @pytest.mark.asyncio
    async def test_match_by_source_and_relation(self, populated_graph):
        """Test matching by source entity and relation."""
        result = await match(
            populated_graph,
            {"source": {"id": "alice"}, "relation": "knows"},
        )

        assert len(result.relationships) == 1
        assert result.relationships[0].source_id == "alice"
        assert result.relationships[0].target_id == "bob"

    @pytest.mark.asyncio
    async def test_match_by_source_type_and_relation(self, populated_graph):
        """Test matching by source type and relation."""
        result = await match(
            populated_graph,
            {"source": {"type": "Person"}, "relation": "works_at"},
        )

        assert len(result.relationships) == 3
        assert all(r.relation == "works_at" for r in result.relationships)

    @pytest.mark.asyncio
    async def test_match_by_target(self, populated_graph):
        """Test matching by target entity."""
        result = await match(
            populated_graph,
            {"target": {"id": "acme"}, "relation": "works_at"},
        )

        assert len(result.relationships) == 2
        assert all(r.target_id == "acme" for r in result.relationships)

    @pytest.mark.asyncio
    async def test_match_source_relation_target(self, populated_graph):
        """Test matching with source, relation, and target filters."""
        result = await match(
            populated_graph,
            {
                "source": {"id": "alice"},
                "relation": "works_at",
                "target": {"type": "Company"},
            },
        )

        assert len(result.relationships) == 1
        assert result.relationships[0].source_id == "alice"
        assert result.relationships[0].target_id == "acme"

    @pytest.mark.asyncio
    async def test_match_returns_involved_entities(self, populated_graph):
        """Test that single-hop match returns both source and target entities."""
        result = await match(
            populated_graph,
            {"source": {"id": "alice"}, "relation": "knows"},
        )

        assert len(result.entities) == 2  # alice and bob
        assert {e.id for e in result.entities} == {"alice", "bob"}


# --- Multi-Hop Path Tests ---


class TestMultiHopMatching:
    """Test multi-hop path pattern matching."""

    @pytest.mark.asyncio
    async def test_two_hop_path(self, populated_graph):
        """Test matching a 2-hop path."""
        result = await match(
            populated_graph,
            {
                "path": [
                    {"source": {"id": "alice"}, "relation": "knows"},
                    {"relation": "knows"},
                ]
            },
        )

        # alice -> bob -> charlie
        assert len(result.relationships) == 2
        assert result.relationships[0].source_id == "alice"
        assert result.relationships[1].source_id == "bob"

    @pytest.mark.asyncio
    async def test_two_hop_with_type_filters(self, populated_graph):
        """Test 2-hop path with type filtering."""
        result = await match(
            populated_graph,
            {
                "path": [
                    {"source": {"type": "Person"}, "relation": "works_at"},
                    {"relation": "manages"},
                ]
            },
        )

        # Person -> Company -> Project
        assert len(result.relationships) >= 1
        # Should find: alice->acme->project_x and bob->acme->project_x
        sources = {r.source_id for r in result.relationships}
        assert "acme" in sources  # Company in the middle

    @pytest.mark.asyncio
    async def test_multi_hop_with_target_constraint(self, populated_graph):
        """Test multi-hop with final target constraint."""
        result = await match(
            populated_graph,
            {
                "path": [
                    {"source": {"type": "Person"}, "relation": "works_at"},
                    {"relation": "manages", "target": {"id": "project_x"}},
                ]
            },
        )

        # Should find paths ending at project_x
        assert any(r.target_id == "project_x" for r in result.relationships)

    @pytest.mark.asyncio
    async def test_multi_hop_no_path_returns_empty(self, populated_graph):
        """Test multi-hop returns empty when no path exists."""
        result = await match(
            populated_graph,
            {
                "path": [
                    {"source": {"id": "project_x"}, "relation": "knows"},
                    {"relation": "works_at"},
                ]
            },
        )

        # No such path exists
        assert len(result.relationships) == 0


# --- Wildcard Tests ---


class TestWildcards:
    """Test wildcard support in patterns."""

    @pytest.mark.asyncio
    async def test_wildcard_in_target_id(self, populated_graph):
        """Test wildcard for target ID (any target)."""
        result = await match(
            populated_graph,
            {"source": {"id": "alice"}, "relation": "knows", "target": {"id": "?"}},
        )

        # Should match alice->bob
        assert len(result.relationships) >= 1

    @pytest.mark.asyncio
    async def test_wildcard_in_type(self, populated_graph):
        """Test wildcard for entity type (any type)."""
        result = await match(
            populated_graph,
            {"source": {"id": "alice"}, "target": {"type": "?"}},
        )

        # Should match all relationships from alice
        assert len(result.relationships) == 2  # knows bob, works_at acme


# --- Graph.match() Integration Tests ---


class TestGraphMatchMethod:
    """Test Graph.match() method integration."""

    @pytest.mark.asyncio
    async def test_graph_match_entity_filter(self, populated_graph):
        """Test graph.match() for entity filtering."""
        result = await populated_graph.match({"type": "Company"})

        assert len(result.entities) == 2
        assert all(e.entity_type == "Company" for e in result.entities)

    @pytest.mark.asyncio
    async def test_graph_match_single_hop(self, populated_graph):
        """Test graph.match() for single-hop relationship."""
        result = await populated_graph.match(
            {"source": {"type": "Person"}, "relation": "works_at"}
        )

        assert len(result.relationships) == 3

    @pytest.mark.asyncio
    async def test_graph_match_returns_query_result(self, populated_graph):
        """Test that graph.match() returns QueryResult with metadata."""
        result = await populated_graph.match({"type": "Person"})

        assert isinstance(result, QueryResult)
        assert "pattern_type" in result.metadata
        assert result.metadata["pattern_type"] == "entity_filter"
        assert "matched_count" in result.metadata


# --- Edge Cases and Error Handling ---


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_results(self):
        """Test querying empty graph returns empty results."""
        graph = Graph()
        result = await match(graph, {"type": "Person"})

        assert len(result.entities) == 0
        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_no_matching_entities(self, populated_graph):
        """Test pattern with no matches returns empty result."""
        result = await match(populated_graph, {"type": "Robot"})

        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_no_matching_relationships(self, populated_graph):
        """Test relationship pattern with no matches."""
        result = await match(
            populated_graph,
            {"source": {"id": "alice"}, "relation": "dislikes"},
        )

        assert len(result.relationships) == 0
