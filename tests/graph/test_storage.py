"""Tests for the Storage protocol.

This module tests that the async protocol is correctly defined and that
example implementations conform to the protocol interface with bulk operations.
"""

import pytest
from cogent.graph.storage import Storage
from cogent.graph.models import Entity, Relationship


class MockBackend:
    """Mock async implementation of Storage for testing protocol conformance."""

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []

    async def add_entity(
        self,
        id: str,
        entity_type: str,
        **attributes: object,
    ) -> Entity:
        """Mock implementation of add_entity."""
        if id in self._entities:
            raise ValueError(f"Entity {id} already exists")

        entity = Entity(id=id, entity_type=entity_type, attributes=attributes)
        self._entities[id] = entity
        return entity

    async def add_entities(self, entities: list[Entity]) -> list[Entity]:
        """Mock implementation of add_entities."""
        for entity in entities:
            if entity.id in self._entities:
                raise ValueError(f"Entity {entity.id} already exists")
            self._entities[entity.id] = entity
        return entities

    async def get_entity(self, id: str) -> Entity | None:
        """Mock implementation of get_entity."""
        return self._entities.get(id)

    async def remove_entity(self, id: str) -> bool:
        """Mock implementation of remove_entity."""
        if id in self._entities:
            del self._entities[id]
            self._relationships = [
                rel
                for rel in self._relationships
                if rel.source_id != id and rel.target_id != id
            ]
            return True
        return False

    async def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        **attributes: object,
    ) -> Relationship:
        """Mock implementation of add_relationship."""
        if source_id not in self._entities:
            raise ValueError(f"Source entity {source_id} not found")
        if target_id not in self._entities:
            raise ValueError(f"Target entity {target_id} not found")

        rel = Relationship(
            source_id=source_id,
            relation=relation,
            target_id=target_id,
            attributes=attributes,
        )
        self._relationships.append(rel)
        return rel

    async def add_relationships(self, relationships: list[Relationship]) -> list[Relationship]:
        """Mock implementation of add_relationships."""
        for rel in relationships:
            if rel.source_id not in self._entities:
                raise ValueError(f"Source entity {rel.source_id} not found")
            if rel.target_id not in self._entities:
                raise ValueError(f"Target entity {rel.target_id} not found")
            self._relationships.append(rel)
        return relationships

    async def get_relationships(
        self,
        entity_id: str | None = None,
        relation: str | None = None,
    ) -> list[Relationship]:
        """Mock implementation of get_relationships."""
        results = self._relationships

        if entity_id is not None:
            results = [
                rel
                for rel in results
                if rel.source_id == entity_id or rel.target_id == entity_id
            ]

        if relation is not None:
            results = [rel for rel in results if rel.relation == relation]

        return results

    async def query(self, pattern: str) -> list[dict[str, object]]:
        """Mock implementation of query."""
        return []

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[str] | None:
        """Mock implementation of find_path."""
        return None

    async def get_all_entities(self) -> list[Entity]:
        """Mock implementation of get_all_entities."""
        return list(self._entities.values())

    async def stats(self) -> dict[str, int]:
        """Mock implementation of stats."""
        types = set(entity.entity_type for entity in self._entities.values())
        return {
            "entities": len(self._entities),
            "relationships": len(self._relationships),
            "types": len(types),
        }

    async def clear(self) -> None:
        """Mock implementation of clear."""
        self._entities.clear()
        self._relationships.clear()


class TestStorageProtocol:
    """Test suite for Storage protocol definition."""

    @pytest.mark.asyncio
    async def test_mock_backend_conforms_to_protocol(self):
        """Test that MockBackend conforms to Storage protocol."""
        backend: Storage = MockBackend()
        assert isinstance(backend, MockBackend)

    @pytest.mark.asyncio
    async def test_protocol_add_entity(self):
        """Test protocol conformance for add_entity method."""
        backend: Storage = MockBackend()
        entity = await backend.add_entity("test:1", "TestType", name="Test")
        assert entity.id == "test:1"
        assert entity.entity_type == "TestType"
        assert entity.attributes["name"] == "Test"

    @pytest.mark.asyncio
    async def test_protocol_add_entities_bulk(self):
        """Test bulk entity creation."""
        backend: Storage = MockBackend()
        entities = [
            Entity("entity:1", "Type1", {"name": "Entity 1"}),
            Entity("entity:2", "Type2", {"name": "Entity 2"}),
            Entity("entity:3", "Type1", {"name": "Entity 3"}),
        ]
        created = await backend.add_entities(entities)
        assert len(created) == 3
        all_entities = await backend.get_all_entities()
        assert len(all_entities) == 3

    @pytest.mark.asyncio
    async def test_protocol_get_entity(self):
        """Test protocol conformance for get_entity method."""
        backend: Storage = MockBackend()
        await backend.add_entity("test:1", "TestType")
        entity = await backend.get_entity("test:1")
        assert entity is not None
        assert entity.id == "test:1"

    @pytest.mark.asyncio
    async def test_protocol_get_nonexistent_entity(self):
        """Test protocol conformance for getting nonexistent entity."""
        backend: Storage = MockBackend()
        entity = await backend.get_entity("nonexistent")
        assert entity is None

    @pytest.mark.asyncio
    async def test_protocol_remove_entity(self):
        """Test protocol conformance for remove_entity method."""
        backend: Storage = MockBackend()
        await backend.add_entity("test:1", "TestType")
        removed = await backend.remove_entity("test:1")
        assert removed is True
        assert await backend.get_entity("test:1") is None

    @pytest.mark.asyncio
    async def test_protocol_remove_nonexistent_entity(self):
        """Test protocol conformance for removing nonexistent entity."""
        backend: Storage = MockBackend()
        removed = await backend.remove_entity("nonexistent")
        assert removed is False

    @pytest.mark.asyncio
    async def test_protocol_add_relationship(self):
        """Test protocol conformance for add_relationship method."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        await backend.add_entity("entity:2", "Type2")
        rel = await backend.add_relationship("entity:1", "RELATES_TO", "entity:2", weight=0.9)
        assert rel.source_id == "entity:1"
        assert rel.relation == "RELATES_TO"
        assert rel.target_id == "entity:2"
        assert rel.attributes["weight"] == 0.9

    @pytest.mark.asyncio
    async def test_protocol_add_relationships_bulk(self):
        """Test bulk relationship creation."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        await backend.add_entity("entity:2", "Type2")
        await backend.add_entity("entity:3", "Type3")
        relationships = [
            Relationship("entity:1", "KNOWS", "entity:2"),
            Relationship("entity:2", "KNOWS", "entity:3"),
            Relationship("entity:1", "LIKES", "entity:3"),
        ]
        created = await backend.add_relationships(relationships)
        assert len(created) == 3
        all_rels = await backend.get_relationships()
        assert len(all_rels) == 3

    @pytest.mark.asyncio
    async def test_protocol_add_relationship_missing_source(self):
        """Test that adding relationship with missing source raises error."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:2", "Type2")
        with pytest.raises(ValueError, match="Source entity .* not found"):
            await backend.add_relationship("nonexistent", "REL", "entity:2")

    @pytest.mark.asyncio
    async def test_protocol_add_relationship_missing_target(self):
        """Test that adding relationship with missing target raises error."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        with pytest.raises(ValueError, match="Target entity .* not found"):
            await backend.add_relationship("entity:1", "REL", "nonexistent")

    @pytest.mark.asyncio
    async def test_protocol_get_relationships_by_entity(self):
        """Test getting relationships filtered by entity."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        await backend.add_entity("entity:2", "Type2")
        await backend.add_entity("entity:3", "Type3")
        await backend.add_relationship("entity:1", "REL_A", "entity:2")
        await backend.add_relationship("entity:2", "REL_B", "entity:3")
        rels = await backend.get_relationships(entity_id="entity:2")
        assert len(rels) == 2

    @pytest.mark.asyncio
    async def test_protocol_get_relationships_by_relation(self):
        """Test getting relationships filtered by relation type."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        await backend.add_entity("entity:2", "Type2")
        await backend.add_entity("entity:3", "Type3")
        await backend.add_relationship("entity:1", "KNOWS", "entity:2")
        await backend.add_relationship("entity:2", "KNOWS", "entity:3")
        await backend.add_relationship("entity:1", "LIKES", "entity:3")
        rels = await backend.get_relationships(relation="KNOWS")
        assert len(rels) == 2

    @pytest.mark.asyncio
    async def test_protocol_get_relationships_filtered(self):
        """Test getting relationships with both filters."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        await backend.add_entity("entity:2", "Type2")
        await backend.add_entity("entity:3", "Type3")
        await backend.add_relationship("entity:1", "KNOWS", "entity:2")
        await backend.add_relationship("entity:1", "LIKES", "entity:2")
        await backend.add_relationship("entity:2", "KNOWS", "entity:3")
        rels = await backend.get_relationships(entity_id="entity:1", relation="KNOWS")
        assert len(rels) == 1
        assert rels[0].relation == "KNOWS"

    @pytest.mark.asyncio
    async def test_protocol_get_all_relationships(self):
        """Test getting all relationships without filters."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        await backend.add_entity("entity:2", "Type2")
        await backend.add_relationship("entity:1", "REL_A", "entity:2")
        await backend.add_relationship("entity:2", "REL_B", "entity:1")
        rels = await backend.get_relationships()
        assert len(rels) == 2

    @pytest.mark.asyncio
    async def test_protocol_query(self):
        """Test protocol conformance for query method."""
        backend: Storage = MockBackend()
        results = await backend.query("? -REL-> target")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_protocol_find_path(self):
        """Test protocol conformance for find_path method."""
        backend: Storage = MockBackend()
        path = await backend.find_path("entity:1", "entity:2")
        assert path is None or isinstance(path, list)

    @pytest.mark.asyncio
    async def test_protocol_get_all_entities(self):
        """Test getting all entities."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        await backend.add_entity("entity:2", "Type2")
        entities = await backend.get_all_entities()
        assert len(entities) == 2
        assert all(isinstance(e, Entity) for e in entities)

    @pytest.mark.asyncio
    async def test_protocol_stats(self):
        """Test protocol conformance for stats method."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        await backend.add_entity("entity:2", "Type2")
        await backend.add_entity("entity:3", "Type1")
        await backend.add_relationship("entity:1", "REL", "entity:2")
        stats = await backend.stats()
        assert stats["entities"] == 3
        assert stats["relationships"] == 1
        assert stats["types"] == 2

    @pytest.mark.asyncio
    async def test_protocol_clear(self):
        """Test protocol conformance for clear method."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        await backend.add_entity("entity:2", "Type2")
        await backend.add_relationship("entity:1", "REL", "entity:2")
        await backend.clear()
        assert len(await backend.get_all_entities()) == 0
        assert len(await backend.get_relationships()) == 0

    @pytest.mark.asyncio
    async def test_protocol_remove_entity_cascades_relationships(self):
        """Test that removing entity also removes related relationships."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        await backend.add_entity("entity:2", "Type2")
        await backend.add_entity("entity:3", "Type3")
        await backend.add_relationship("entity:1", "REL_A", "entity:2")
        await backend.add_relationship("entity:2", "REL_B", "entity:3")
        await backend.remove_entity("entity:2")
        rels = await backend.get_relationships()
        assert len(rels) == 0

    @pytest.mark.asyncio
    async def test_protocol_duplicate_entity_raises_error(self):
        """Test that adding duplicate entity raises error."""
        backend: Storage = MockBackend()
        await backend.add_entity("entity:1", "Type1")
        with pytest.raises(ValueError, match="already exists"):
            await backend.add_entity("entity:1", "Type1")
