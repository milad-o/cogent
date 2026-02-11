"""Tests for storage implementations (MemoryStorage, FileStorage, SQLStorage)."""

import pytest
import tempfile
from pathlib import Path

from cogent.graph.models import Entity, Relationship
from cogent.graph.storage import MemoryStorage, FileStorage, SQLStorage


# --- MemoryStorage Tests ---


class TestMemoryStorage:
    """Test suite for MemoryStorage."""

    @pytest.fixture
    async def storage(self):
        """Create a fresh MemoryStorage instance."""
        return MemoryStorage()

    @pytest.mark.asyncio
    async def test_add_entity(self, storage):
        """Test adding a single entity."""
        entity = await storage.add_entity("alice", "Person", name="Alice", age=30)

        assert entity.id == "alice"
        assert entity.entity_type == "Person"
        assert entity.attributes["name"] == "Alice"
        assert entity.attributes["age"] == 30
        assert entity.created_at is not None
        assert entity.updated_at is not None

    @pytest.mark.asyncio
    async def test_add_duplicate_entity_raises_error(self, storage):
        """Test that adding duplicate entity raises ValueError."""
        await storage.add_entity("alice", "Person", name="Alice")

        with pytest.raises(ValueError, match="already exists"):
            await storage.add_entity("alice", "Person", name="Alice2")

    @pytest.mark.asyncio
    async def test_add_entities_bulk(self, storage):
        """Test bulk adding multiple entities."""
        entities_to_add = [
            Entity(id="alice", entity_type="Person", attributes={"name": "Alice"}),
            Entity(id="bob", entity_type="Person", attributes={"name": "Bob"}),
            Entity(id="charlie", entity_type="Person", attributes={"name": "Charlie"}),
        ]

        result = await storage.add_entities(entities_to_add)

        assert len(result) == 3
        assert all(ent.created_at is not None for ent in result)

        # Verify all entities were stored
        alice = await storage.get_entity("alice")
        bob = await storage.get_entity("bob")
        charlie = await storage.get_entity("charlie")

        assert alice is not None
        assert bob is not None
        assert charlie is not None

    @pytest.mark.asyncio
    async def test_get_entity(self, storage):
        """Test retrieving an entity by ID."""
        await storage.add_entity("alice", "Person", name="Alice")

        entity = await storage.get_entity("alice")

        assert entity is not None
        assert entity.id == "alice"
        assert entity.entity_type == "Person"
        assert entity.attributes["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_get_nonexistent_entity_returns_none(self, storage):
        """Test that getting a non-existent entity returns None."""
        entity = await storage.get_entity("nonexistent")

        assert entity is None

    @pytest.mark.asyncio
    async def test_remove_entity(self, storage):
        """Test removing an entity."""
        await storage.add_entity("alice", "Person", name="Alice")

        result = await storage.remove_entity("alice")

        assert result is True

        # Verify entity was removed
        entity = await storage.get_entity("alice")
        assert entity is None

    @pytest.mark.asyncio
    async def test_remove_entity_cascades_relationships(self, storage):
        """Test that removing an entity removes its relationships."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_relationship("alice", "knows", "bob")

        # Remove alice
        await storage.remove_entity("alice")

        # Verify relationship was removed
        rels = await storage.get_relationships(source_id="alice")
        assert len(rels) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_entity_returns_false(self, storage):
        """Test that removing a non-existent entity returns False."""
        result = await storage.remove_entity("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_add_relationship(self, storage):
        """Test adding a relationship between entities."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")

        rel = await storage.add_relationship("alice", "knows", "bob", since=2020)

        assert rel.source_id == "alice"
        assert rel.relation == "knows"
        assert rel.target_id == "bob"
        assert rel.attributes["since"] == 2020

    @pytest.mark.asyncio
    async def test_add_relationship_without_source_raises_error(self, storage):
        """Test that adding relationship without source entity raises ValueError."""
        await storage.add_entity("bob", "Person", name="Bob")

        with pytest.raises(ValueError, match="does not exist"):
            await storage.add_relationship("alice", "knows", "bob")

    @pytest.mark.asyncio
    async def test_add_relationship_without_target_raises_error(self, storage):
        """Test that adding relationship without target entity raises ValueError."""
        await storage.add_entity("alice", "Person", name="Alice")

        with pytest.raises(ValueError, match="does not exist"):
            await storage.add_relationship("alice", "knows", "bob")

    @pytest.mark.asyncio
    async def test_add_relationships_bulk(self, storage):
        """Test bulk adding multiple relationships."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_entity("charlie", "Person", name="Charlie")

        rels_to_add = [
            Relationship(source_id="alice", relation="knows", target_id="bob"),
            Relationship(source_id="bob", relation="knows", target_id="charlie"),
        ]

        result = await storage.add_relationships(rels_to_add)

        assert len(result) == 2

        # Verify relationships were stored
        alice_rels = await storage.get_relationships(source_id="alice")
        assert len(alice_rels) == 1

    @pytest.mark.asyncio
    async def test_get_relationships_by_source(self, storage):
        """Test filtering relationships by source_id."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_entity("charlie", "Person", name="Charlie")

        await storage.add_relationship("alice", "knows", "bob")
        await storage.add_relationship("alice", "likes", "charlie")
        await storage.add_relationship("bob", "knows", "charlie")

        rels = await storage.get_relationships(source_id="alice")

        assert len(rels) == 2
        assert all(rel.source_id == "alice" for rel in rels)

    @pytest.mark.asyncio
    async def test_get_relationships_by_relation(self, storage):
        """Test filtering relationships by relation."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_entity("charlie", "Person", name="Charlie")

        await storage.add_relationship("alice", "knows", "bob")
        await storage.add_relationship("alice", "likes", "charlie")
        await storage.add_relationship("bob", "knows", "charlie")

        rels = await storage.get_relationships(relation="knows")

        assert len(rels) == 2
        assert all(rel.relation == "knows" for rel in rels)

    @pytest.mark.asyncio
    async def test_get_relationships_by_target(self, storage):
        """Test filtering relationships by target_id."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_entity("charlie", "Person", name="Charlie")

        await storage.add_relationship("alice", "knows", "bob")
        await storage.add_relationship("alice", "likes", "charlie")
        await storage.add_relationship("bob", "knows", "charlie")

        rels = await storage.get_relationships(target_id="charlie")

        assert len(rels) == 2
        assert all(rel.target_id == "charlie" for rel in rels)

    @pytest.mark.asyncio
    async def test_get_relationships_combined_filters(self, storage):
        """Test filtering relationships by multiple criteria."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_entity("charlie", "Person", name="Charlie")

        await storage.add_relationship("alice", "knows", "bob")
        await storage.add_relationship("alice", "knows", "charlie")
        await storage.add_relationship("alice", "likes", "charlie")

        rels = await storage.get_relationships(source_id="alice", relation="knows")

        assert len(rels) == 2
        assert all(rel.source_id == "alice" and rel.relation == "knows" for rel in rels)

    @pytest.mark.asyncio
    async def test_get_all_entities(self, storage):
        """Test retrieving all entities."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_entity("charlie", "Person", name="Charlie")

        entities = await storage.get_all_entities()

        assert len(entities) == 3
        assert {ent.id for ent in entities} == {"alice", "bob", "charlie"}

    @pytest.mark.asyncio
    async def test_stats(self, storage):
        """Test getting storage statistics."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_relationship("alice", "knows", "bob")

        stats = await storage.stats()

        assert stats["entity_count"] == 2
        assert stats["relationship_count"] == 1

    @pytest.mark.asyncio
    async def test_clear(self, storage):
        """Test clearing all data."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_relationship("alice", "knows", "bob")

        await storage.clear()

        stats = await storage.stats()
        assert stats["entity_count"] == 0
        assert stats["relationship_count"] == 0


# --- FileStorage Tests ---


class TestFileStorage:
    """Test suite for FileStorage."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for storage."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
            path = f.name
        yield path
        # Cleanup
        Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_json_format_persistence(self, temp_file):
        """Test JSON format storage and persistence."""
        # Create storage and add data
        storage1 = FileStorage(temp_file, format="json")
        await storage1.add_entity("alice", "Person", name="Alice")
        await storage1.add_entity("bob", "Person", name="Bob")
        await storage1.add_relationship("alice", "knows", "bob")

        # Create new storage instance from same file
        storage2 = FileStorage(temp_file, format="json")

        # Verify data persisted
        alice = await storage2.get_entity("alice")
        assert alice is not None
        assert alice.attributes["name"] == "Alice"

        rels = await storage2.get_relationships(source_id="alice")
        assert len(rels) == 1

    @pytest.mark.asyncio
    async def test_pickle_format_persistence(self, temp_file):
        """Test pickle format storage and persistence."""
        pickle_file = temp_file.replace(".json", ".pkl")

        # Create storage and add data
        storage1 = FileStorage(pickle_file, format="pickle")
        await storage1.add_entity("alice", "Person", name="Alice", age=30)
        await storage1.add_relationship("alice", "knows", "alice")

        # Create new storage instance from same file
        storage2 = FileStorage(pickle_file, format="pickle")

        # Verify data persisted
        alice = await storage2.get_entity("alice")
        assert alice is not None
        assert alice.attributes["age"] == 30

        # Cleanup
        Path(pickle_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_auto_save_enabled(self, temp_file):
        """Test that auto_save immediately persists changes."""
        storage = FileStorage(temp_file, format="json", auto_save=True)
        await storage.add_entity("alice", "Person", name="Alice")

        # Read file directly
        with open(temp_file) as f:
            import json
            data = json.load(f)

        assert any(ent["id"] == "alice" for ent in data["entities"])

    @pytest.mark.asyncio
    async def test_auto_save_disabled_requires_manual_save(self, temp_file):
        """Test that auto_save=False requires manual save."""
        storage = FileStorage(temp_file, format="json", auto_save=False)
        await storage.add_entity("alice", "Person", name="Alice")

        # File should be empty (or contain only structure)
        storage2 = FileStorage(temp_file, format="json", auto_save=False)
        alice = await storage2.get_entity("alice")
        assert alice is None  # Not saved yet

        # Manual save
        await storage.save()

        # Now it should be there
        storage3 = FileStorage(temp_file, format="json", auto_save=False)
        alice = await storage3.get_entity("alice")
        assert alice is not None

    @pytest.mark.asyncio
    async def test_load_from_nonexistent_file(self, temp_file):
        """Test that loading from non-existent file creates empty storage."""
        Path(temp_file).unlink(missing_ok=True)  # Ensure file doesn't exist

        storage = FileStorage(temp_file, format="json")

        stats = await storage.stats()
        assert stats["entity_count"] == 0
        assert stats["relationship_count"] == 0

    @pytest.mark.asyncio
    async def test_all_storage_protocol_methods(self, temp_file):
        """Test that FileStorage implements all Storage protocol methods."""
        storage = FileStorage(temp_file, format="json")

        # Test add_entity
        await storage.add_entity("alice", "Person", name="Alice")

        # Test get_entity
        alice = await storage.get_entity("alice")
        assert alice is not None

        # Test add_relationship
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_relationship("alice", "knows", "bob")

        # Test get_relationships
        rels = await storage.get_relationships(source_id="alice")
        assert len(rels) == 1

        # Test get_all_entities
        entities = await storage.get_all_entities()
        assert len(entities) == 2

        # Test stats
        stats = await storage.stats()
        assert stats["entity_count"] == 2
        assert stats["relationship_count"] == 1

        # Test remove_entity
        result = await storage.remove_entity("bob")
        assert result is True

        # Test clear
        await storage.clear()
        stats = await storage.stats()
        assert stats["entity_count"] == 0


# --- SQLStorage Tests ---


class TestSQLStorage:
    """Test suite for SQLStorage."""

    @pytest.fixture
    async def storage(self):
        """Create an in-memory SQLite storage for testing."""
        storage = SQLStorage("sqlite+aiosqlite:///:memory:")
        await storage.initialize()
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self):
        """Test that initialize creates database tables."""
        storage = SQLStorage("sqlite+aiosqlite:///:memory:")
        await storage.initialize()

        # Should not raise error
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.close()

    @pytest.mark.asyncio
    async def test_add_and_get_entity(self, storage):
        """Test adding and retrieving an entity."""
        entity = await storage.add_entity("alice", "Person", name="Alice", age=30)

        assert entity.id == "alice"
        assert entity.entity_type == "Person"
        assert entity.attributes["name"] == "Alice"

        # Retrieve it
        retrieved = await storage.get_entity("alice")
        assert retrieved is not None
        assert retrieved.id == "alice"
        assert retrieved.entity_type == "Person"
        assert retrieved.entity_type == "Person"

    @pytest.mark.asyncio
    async def test_cascade_delete_relationships(self, storage):
        """Test that deleting an entity cascades to relationships."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_relationship("alice", "knows", "bob")

        # Delete alice
        await storage.remove_entity("alice")

        # Relationship should be gone
        rels = await storage.get_relationships(source_id="alice")
        assert len(rels) == 0

    @pytest.mark.asyncio
    async def test_bulk_operations(self, storage):
        """Test bulk add operations."""
        entities = [
            Entity(id="alice", entity_type="Person", attributes={"name": "Alice"}),
            Entity(id="bob", entity_type="Person", attributes={"name": "Bob"}),
            Entity(id="charlie", entity_type="Person", attributes={"name": "Charlie"}),
        ]

        result = await storage.add_entities(entities)
        assert len(result) == 3

        # Bulk relationships
        rels = [
            Relationship(source_id="alice", relation="knows", target_id="bob"),
            Relationship(source_id="bob", relation="knows", target_id="charlie"),
        ]

        result = await storage.add_relationships(rels)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_stats_and_clear(self, storage):
        """Test stats and clear operations."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_relationship("alice", "knows", "bob")

        stats = await storage.stats()
        assert stats["entity_count"] == 2
        assert stats["relationship_count"] == 1

        await storage.clear()

        stats = await storage.stats()
        assert stats["entity_count"] == 0
        assert stats["relationship_count"] == 0

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, storage):
        """Test that errors during operations rollback transactions."""
        await storage.add_entity("alice", "Person", name="Alice")

        # Try to add duplicate - should raise error
        with pytest.raises(ValueError):
            await storage.add_entity("alice", "Person", name="Alice2")

        # Original entity should still be there with original data
        alice = await storage.get_entity("alice")
        assert alice.attributes["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_relationship_filtering(self, storage):
        """Test filtering relationships by various criteria."""
        await storage.add_entity("alice", "Person", name="Alice")
        await storage.add_entity("bob", "Person", name="Bob")
        await storage.add_entity("charlie", "Person", name="Charlie")

        await storage.add_relationship("alice", "knows", "bob")
        await storage.add_relationship("alice", "likes", "charlie")
        await storage.add_relationship("bob", "knows", "charlie")

        # Filter by source
        rels = await storage.get_relationships(source_id="alice")
        assert len(rels) == 2

        # Filter by relation
        rels = await storage.get_relationships(relation="knows")
        assert len(rels) == 2

        # Filter by target
        rels = await storage.get_relationships(target_id="charlie")
        assert len(rels) == 2

        # Combined filters
        rels = await storage.get_relationships(source_id="alice", relation="knows")
        assert len(rels) == 1
        assert rels[0].target_id == "bob"
