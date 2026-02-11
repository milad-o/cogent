"""Tests for the base Graph class."""

import pytest

from cogent.graph import Graph, Entity, Relationship
from cogent.graph.engines import NativeEngine, NetworkXEngine
from cogent.graph.storage import MemoryStorage, FileStorage


class TestGraphInitialization:
    """Test Graph initialization and defaults."""

    def test_default_initialization(self):
        """Test that Graph() uses smart defaults."""
        graph = Graph()

        # Should have engine and storage
        assert graph.engine is not None
        assert graph.storage is not None

        # Storage should be MemoryStorage
        assert isinstance(graph.storage, MemoryStorage)

        # Engine should be NetworkX if available, else Native
        assert isinstance(graph.engine, (NetworkXEngine, NativeEngine))

    def test_explicit_engine_and_storage(self):
        """Test explicit engine and storage selection."""
        engine = NativeEngine()
        storage = MemoryStorage()

        graph = Graph(engine=engine, storage=storage)

        assert graph.engine is engine
        assert graph.storage is storage

    def test_mix_and_match(self):
        """Test that engine and storage are independent."""
        # NetworkX engine with Memory storage
        graph1 = Graph(engine=NetworkXEngine(), storage=MemoryStorage())
        assert isinstance(graph1.engine, NetworkXEngine)
        assert isinstance(graph1.storage, MemoryStorage)

        # Native engine with Memory storage
        graph2 = Graph(engine=NativeEngine(), storage=MemoryStorage())
        assert isinstance(graph2.engine, NativeEngine)
        assert isinstance(graph2.storage, MemoryStorage)


class TestEntityOperations:
    """Test entity CRUD operations."""

    @pytest.fixture
    def graph(self):
        """Create a test graph."""
        return Graph()

    @pytest.mark.asyncio
    async def test_add_entity(self, graph):
        """Test adding a single entity."""
        entity = await graph.add_entity("alice", "Person", name="Alice", age=30)

        assert entity.id == "alice"
        assert entity.entity_type == "Person"
        assert entity.attributes["name"] == "Alice"
        assert entity.attributes["age"] == 30

        # Verify it's in storage
        stored = await graph.storage.get_entity("alice")
        assert stored is not None
        assert stored.id == "alice"

        # Verify it's in engine
        assert await graph.engine.has_node("alice")

    @pytest.mark.asyncio
    async def test_add_duplicate_entity_raises_error(self, graph):
        """Test that adding duplicate entity raises ValueError."""
        await graph.add_entity("alice", "Person", name="Alice")

        with pytest.raises(ValueError, match="already exists"):
            await graph.add_entity("alice", "Person", name="Alice2")

    @pytest.mark.asyncio
    async def test_add_entities_bulk(self, graph):
        """Test bulk adding multiple entities."""
        entities = [
            Entity(id="alice", entity_type="Person", attributes={"name": "Alice"}),
            Entity(id="bob", entity_type="Person", attributes={"name": "Bob"}),
            Entity(id="charlie", entity_type="Person", attributes={"name": "Charlie"}),
        ]

        created = await graph.add_entities(entities)

        assert len(created) == 3

        # Verify all in storage
        for ent in entities:
            stored = await graph.storage.get_entity(ent.id)
            assert stored is not None

        # Verify all in engine
        for ent in entities:
            assert await graph.engine.has_node(ent.id)

    @pytest.mark.asyncio
    async def test_get_entity(self, graph):
        """Test retrieving an entity."""
        await graph.add_entity("alice", "Person", name="Alice")

        entity = await graph.get_entity("alice")

        assert entity is not None
        assert entity.id == "alice"
        assert entity.entity_type == "Person"

    @pytest.mark.asyncio
    async def test_get_nonexistent_entity(self, graph):
        """Test that getting non-existent entity returns None."""
        entity = await graph.get_entity("nonexistent")

        assert entity is None

    @pytest.mark.asyncio
    async def test_remove_entity(self, graph):
        """Test removing an entity."""
        await graph.add_entity("alice", "Person", name="Alice")

        removed = await graph.remove_entity("alice")

        assert removed is True

        # Verify removed from storage
        assert await graph.storage.get_entity("alice") is None

        # Verify removed from engine
        assert not await graph.engine.has_node("alice")

    @pytest.mark.asyncio
    async def test_remove_entity_cascades_relationships(self, graph):
        """Test that removing entity removes its relationships."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_relationship("alice", "knows", "bob")

        # Remove alice
        await graph.remove_entity("alice")

        # Verify relationship removed from storage
        rels = await graph.storage.get_relationships(source_id="alice")
        assert len(rels) == 0

        # Verify edge removed from engine
        edges = await graph.engine.get_edges(source_id="alice")
        assert len(edges) == 0

    @pytest.mark.asyncio
    async def test_get_all_entities(self, graph):
        """Test getting all entities."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_entity("charlie", "Person", name="Charlie")

        entities = await graph.get_all_entities()

        assert len(entities) == 3
        ids = {e.id for e in entities}
        assert ids == {"alice", "bob", "charlie"}


class TestRelationshipOperations:
    """Test relationship CRUD operations."""

    @pytest.fixture
    def graph(self):
        """Create a test graph."""
        return Graph()

    @pytest.mark.asyncio
    async def test_add_relationship(self, graph):
        """Test adding a relationship."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")

        rel = await graph.add_relationship("alice", "knows", "bob", since=2020)

        assert rel.source_id == "alice"
        assert rel.relation == "knows"
        assert rel.target_id == "bob"
        assert rel.attributes["since"] == 2020

        # Verify in storage
        rels = await graph.storage.get_relationships(source_id="alice")
        assert len(rels) == 1

        # Verify in engine
        edges = await graph.engine.get_edges(source_id="alice")
        assert len(edges) == 1

    @pytest.mark.asyncio
    async def test_add_relationship_missing_source(self, graph):
        """Test that adding relationship with missing source raises error."""
        await graph.add_entity("bob", "Person", name="Bob")

        with pytest.raises(ValueError, match="does not exist"):
            await graph.add_relationship("alice", "knows", "bob")

    @pytest.mark.asyncio
    async def test_add_relationship_missing_target(self, graph):
        """Test that adding relationship with missing target raises error."""
        await graph.add_entity("alice", "Person", name="Alice")

        with pytest.raises(ValueError, match="does not exist"):
            await graph.add_relationship("alice", "knows", "bob")

    @pytest.mark.asyncio
    async def test_add_relationships_bulk(self, graph):
        """Test bulk adding relationships."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_entity("charlie", "Person", name="Charlie")

        rels = [
            Relationship(source_id="alice", relation="knows", target_id="bob"),
            Relationship(source_id="bob", relation="knows", target_id="charlie"),
        ]

        created = await graph.add_relationships(rels)

        assert len(created) == 2

        # Verify in storage
        alice_rels = await graph.storage.get_relationships(source_id="alice")
        assert len(alice_rels) == 1

        # Verify in engine
        alice_edges = await graph.engine.get_edges(source_id="alice")
        assert len(alice_edges) == 1

    @pytest.mark.asyncio
    async def test_get_relationships_by_source(self, graph):
        """Test filtering relationships by source."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_entity("charlie", "Person", name="Charlie")

        await graph.add_relationship("alice", "knows", "bob")
        await graph.add_relationship("alice", "likes", "charlie")
        await graph.add_relationship("bob", "knows", "charlie")

        rels = await graph.get_relationships(source_id="alice")

        assert len(rels) == 2
        assert all(r.source_id == "alice" for r in rels)

    @pytest.mark.asyncio
    async def test_get_relationships_by_relation(self, graph):
        """Test filtering relationships by relation type."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_entity("charlie", "Person", name="Charlie")

        await graph.add_relationship("alice", "knows", "bob")
        await graph.add_relationship("alice", "likes", "charlie")
        await graph.add_relationship("bob", "knows", "charlie")

        rels = await graph.get_relationships(relation="knows")

        assert len(rels) == 2
        assert all(r.relation == "knows" for r in rels)

    @pytest.mark.asyncio
    async def test_get_relationships_combined_filters(self, graph):
        """Test filtering relationships by multiple criteria."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_entity("charlie", "Person", name="Charlie")

        await graph.add_relationship("alice", "knows", "bob")
        await graph.add_relationship("alice", "knows", "charlie")
        await graph.add_relationship("alice", "likes", "charlie")

        rels = await graph.get_relationships(source_id="alice", relation="knows")

        assert len(rels) == 2
        assert all(r.source_id == "alice" and r.relation == "knows" for r in rels)


class TestQueryAndTraversal:
    """Test query and graph traversal methods."""

    @pytest.fixture
    def graph(self):
        """Create a test graph."""
        return Graph()

    @pytest.mark.asyncio
    async def test_find_path(self, graph):
        """Test finding path between entities."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_entity("charlie", "Person", name="Charlie")

        await graph.add_relationship("alice", "knows", "bob")
        await graph.add_relationship("bob", "knows", "charlie")

        path = await graph.find_path("alice", "charlie")

        assert path == ["alice", "bob", "charlie"]

    @pytest.mark.asyncio
    async def test_find_path_no_path(self, graph):
        """Test finding path when no connection exists."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")

        path = await graph.find_path("alice", "bob")

        assert path is None

    @pytest.mark.asyncio
    async def test_get_neighbors(self, graph):
        """Test getting neighboring entities."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_entity("charlie", "Person", name="Charlie")

        await graph.add_relationship("alice", "knows", "bob")
        await graph.add_relationship("alice", "likes", "charlie")

        neighbors = await graph.get_neighbors("alice", direction="outgoing")

        assert len(neighbors) == 2
        assert set(neighbors) == {"bob", "charlie"}

    @pytest.mark.asyncio
    async def test_find_entities_by_type(self, graph):
        """Test finding entities by type."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_entity("acme", "Company", name="Acme Corp")

        people = await graph.find_entities(entity_type="Person")

        assert len(people) == 2
        assert all(e.entity_type == "Person" for e in people)

    @pytest.mark.asyncio
    async def test_find_entities_by_attributes(self, graph):
        """Test finding entities by attributes."""
        await graph.add_entity("alice", "Person", name="Alice", age=30)
        await graph.add_entity("bob", "Person", name="Bob", age=25)
        await graph.add_entity("charlie", "Person", name="Charlie", age=30)

        entities = await graph.find_entities(age=30)

        assert len(entities) == 2
        assert all(e.attributes["age"] == 30 for e in entities)

    @pytest.mark.asyncio
    async def test_find_entities_combined_filters(self, graph):
        """Test finding entities with type and attribute filters."""
        await graph.add_entity("alice", "Person", name="Alice", age=30)
        await graph.add_entity("bob", "Person", name="Bob", age=25)
        await graph.add_entity("acme", "Company", name="Acme Corp", age=30)

        entities = await graph.find_entities(entity_type="Person", age=30)

        assert len(entities) == 1
        assert entities[0].id == "alice"


class TestStatisticsAndUtilities:
    """Test graph statistics and utility methods."""

    @pytest.fixture
    def graph(self):
        """Create a test graph."""
        return Graph()

    @pytest.mark.asyncio
    async def test_stats(self, graph):
        """Test getting graph statistics."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_relationship("alice", "knows", "bob")

        stats = await graph.stats()

        assert stats["entity_count"] == 2
        assert stats["relationship_count"] == 1

    @pytest.mark.asyncio
    async def test_node_count(self, graph):
        """Test getting node count from engine."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")

        count = await graph.node_count()

        assert count == 2

    @pytest.mark.asyncio
    async def test_edge_count(self, graph):
        """Test getting edge count from engine."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_relationship("alice", "knows", "bob")

        count = await graph.edge_count()

        assert count == 1

    @pytest.mark.asyncio
    async def test_clear(self, graph):
        """Test clearing all data."""
        await graph.add_entity("alice", "Person", name="Alice")
        await graph.add_entity("bob", "Person", name="Bob")
        await graph.add_relationship("alice", "knows", "bob")

        await graph.clear()

        # Verify storage cleared
        storage_stats = await graph.storage.stats()
        assert storage_stats["entity_count"] == 0
        assert storage_stats["relationship_count"] == 0

        # Verify engine cleared
        assert await graph.node_count() == 0
        assert await graph.edge_count() == 0
