"""Tests for the Knowledge Graph models module."""

import pytest
from datetime import datetime, timezone
from cogent.graph.models import Entity, Relationship


class TestEntity:
    """Test suite for the Entity class."""

    def test_create_basic_entity(self):
        """Test creating a basic entity with required fields."""
        entity = Entity(id="person:alice", entity_type="Person")

        assert entity.id == "person:alice"
        assert entity.entity_type == "Person"
        assert entity.attributes == {}
        assert entity.source is None
        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.updated_at, datetime)

    def test_create_entity_with_attributes(self):
        """Test creating an entity with attributes."""
        entity = Entity(
            id="person:bob",
            entity_type="Person",
            attributes={"name": "Bob Smith", "age": 35, "active": True},
        )

        assert entity.attributes["name"] == "Bob Smith"
        assert entity.attributes["age"] == 35
        assert entity.attributes["active"] is True

    def test_create_entity_with_source(self):
        """Test creating an entity with source attribution."""
        entity = Entity(
            id="doc:123",
            entity_type="Document",
            source="wikipedia",
        )

        assert entity.source == "wikipedia"

    def test_entity_id_validation_empty(self):
        """Test that empty entity ID raises ValueError."""
        with pytest.raises(ValueError, match="Entity ID cannot be empty"):
            Entity(id="", entity_type="Person")

    def test_entity_id_validation_whitespace(self):
        """Test that whitespace-only entity ID raises ValueError."""
        with pytest.raises(ValueError, match="Entity ID cannot be empty"):
            Entity(id="   ", entity_type="Person")

    def test_entity_type_validation_empty(self):
        """Test that empty entity type raises ValueError."""
        with pytest.raises(ValueError, match="Entity type cannot be empty"):
            Entity(id="entity:1", entity_type="")

    def test_entity_type_validation_whitespace(self):
        """Test that whitespace-only entity type raises ValueError."""
        with pytest.raises(ValueError, match="Entity type cannot be empty"):
            Entity(id="entity:1", entity_type="  ")

    def test_entity_normalizes_whitespace(self):
        """Test that entity normalizes whitespace in fields."""
        entity = Entity(
            id="  person:charlie  ",
            entity_type="  Person  ",
            source="  tool  ",
        )

        assert entity.id == "person:charlie"
        assert entity.entity_type == "Person"
        assert entity.source == "tool"

    def test_update_attributes(self):
        """Test updating entity attributes."""
        entity = Entity(id="entity:1", entity_type="Thing")
        initial_updated_at = entity.updated_at

        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.01)

        entity.update_attributes(status="active", score=100)

        assert entity.attributes["status"] == "active"
        assert entity.attributes["score"] == 100
        assert entity.updated_at > initial_updated_at

    def test_get_attribute(self):
        """Test getting attribute values."""
        entity = Entity(
            id="entity:1",
            entity_type="Thing",
            attributes={"key": "value"},
        )

        assert entity.get_attribute("key") == "value"
        assert entity.get_attribute("missing") is None
        assert entity.get_attribute("missing", "default") == "default"

    def test_entity_equality(self):
        """Test entity equality based on ID."""
        entity1 = Entity(id="entity:1", entity_type="TypeA")
        entity2 = Entity(id="entity:1", entity_type="TypeB")
        entity3 = Entity(id="entity:2", entity_type="TypeA")

        assert entity1 == entity2  # Same ID
        assert entity1 != entity3  # Different ID

    def test_entity_hash(self):
        """Test entity hashing for use in sets/dicts."""
        entity1 = Entity(id="entity:1", entity_type="Type")
        entity2 = Entity(id="entity:1", entity_type="Type")
        entity3 = Entity(id="entity:2", entity_type="Type")

        # Same ID = same hash
        assert hash(entity1) == hash(entity2)

        # Can be used in sets
        entity_set = {entity1, entity2, entity3}
        assert len(entity_set) == 2  # entity1 and entity2 are duplicates

    def test_entity_mutable_attributes(self):
        """Test that entity attributes are mutable."""
        entity = Entity(id="entity:1", entity_type="Thing")

        entity.attributes["key"] = "value"
        assert entity.attributes["key"] == "value"

        entity.attributes["key"] = "new_value"
        assert entity.attributes["key"] == "new_value"


class TestRelationship:
    """Test suite for the Relationship class."""

    def test_create_basic_relationship(self):
        """Test creating a basic relationship with required fields."""
        rel = Relationship(
            source_id="person:alice",
            relation="KNOWS",
            target_id="person:bob",
        )

        assert rel.source_id == "person:alice"
        assert rel.relation == "KNOWS"
        assert rel.target_id == "person:bob"
        assert rel.attributes == {}
        assert rel.source is None
        assert isinstance(rel.created_at, datetime)

    def test_create_relationship_with_attributes(self):
        """Test creating a relationship with attributes."""
        rel = Relationship(
            source_id="person:alice",
            relation="WORKS_AT",
            target_id="company:acme",
            attributes={"since": 2020, "role": "Engineer"},
        )

        assert rel.attributes["since"] == 2020
        assert rel.attributes["role"] == "Engineer"

    def test_create_relationship_with_source(self):
        """Test creating a relationship with source attribution."""
        rel = Relationship(
            source_id="entity:1",
            relation="RELATES_TO",
            target_id="entity:2",
            source="document:123",
        )

        assert rel.source == "document:123"

    def test_relationship_source_validation_empty(self):
        """Test that empty source ID raises ValueError."""
        with pytest.raises(ValueError, match="Source ID cannot be empty"):
            Relationship(source_id="", relation="REL", target_id="entity:2")

    def test_relationship_source_validation_whitespace(self):
        """Test that whitespace-only source ID raises ValueError."""
        with pytest.raises(ValueError, match="Source ID cannot be empty"):
            Relationship(source_id="  ", relation="REL", target_id="entity:2")

    def test_relationship_relation_validation_empty(self):
        """Test that empty relation type raises ValueError."""
        with pytest.raises(ValueError, match="Relation type cannot be empty"):
            Relationship(source_id="entity:1", relation="", target_id="entity:2")

    def test_relationship_relation_validation_whitespace(self):
        """Test that whitespace-only relation type raises ValueError."""
        with pytest.raises(ValueError, match="Relation type cannot be empty"):
            Relationship(source_id="entity:1", relation="  ", target_id="entity:2")

    def test_relationship_target_validation_empty(self):
        """Test that empty target ID raises ValueError."""
        with pytest.raises(ValueError, match="Target ID cannot be empty"):
            Relationship(source_id="entity:1", relation="REL", target_id="")

    def test_relationship_target_validation_whitespace(self):
        """Test that whitespace-only target ID raises ValueError."""
        with pytest.raises(ValueError, match="Target ID cannot be empty"):
            Relationship(source_id="entity:1", relation="REL", target_id="  ")

    def test_relationship_normalizes_whitespace(self):
        """Test that relationship normalizes whitespace in fields."""
        rel = Relationship(
            source_id="  entity:1  ",
            relation="  RELATES_TO  ",
            target_id="  entity:2  ",
            source="  doc  ",
        )

        assert rel.source_id == "entity:1"
        assert rel.relation == "RELATES_TO"
        assert rel.target_id == "entity:2"
        assert rel.source == "doc"

    def test_update_attributes(self):
        """Test updating relationship attributes."""
        rel = Relationship(
            source_id="entity:1",
            relation="REL",
            target_id="entity:2",
        )

        rel.update_attributes(weight=0.95, verified=True)

        assert rel.attributes["weight"] == 0.95
        assert rel.attributes["verified"] is True

    def test_get_attribute(self):
        """Test getting attribute values."""
        rel = Relationship(
            source_id="entity:1",
            relation="REL",
            target_id="entity:2",
            attributes={"key": "value"},
        )

        assert rel.get_attribute("key") == "value"
        assert rel.get_attribute("missing") is None
        assert rel.get_attribute("missing", "default") == "default"

    def test_relationship_reverse(self):
        """Test reversing a relationship."""
        rel = Relationship(
            source_id="person:alice",
            relation="KNOWS",
            target_id="person:bob",
            attributes={"since": 2020},
        )

        reversed_rel = rel.reverse()

        assert reversed_rel.source_id == "person:bob"
        assert reversed_rel.target_id == "person:alice"
        assert reversed_rel.relation == "KNOWS"
        assert reversed_rel.attributes["since"] == 2020
        assert reversed_rel.created_at == rel.created_at

        # Original should be unchanged
        assert rel.source_id == "person:alice"
        assert rel.target_id == "person:bob"

    def test_relationship_reverse_independent_attributes(self):
        """Test that reversed relationship has independent attributes."""
        rel = Relationship(
            source_id="entity:1",
            relation="REL",
            target_id="entity:2",
            attributes={"key": "value"},
        )

        reversed_rel = rel.reverse()

        # Modify reversed relationship's attributes
        reversed_rel.attributes["key"] = "new_value"

        # Original should be unchanged
        assert rel.attributes["key"] == "value"

    def test_relationship_equality(self):
        """Test relationship equality based on source, relation, and target."""
        rel1 = Relationship("entity:1", "RELATES_TO", "entity:2")
        rel2 = Relationship("entity:1", "RELATES_TO", "entity:2")
        rel3 = Relationship("entity:1", "DIFFERENT", "entity:2")
        rel4 = Relationship("entity:2", "RELATES_TO", "entity:1")

        assert rel1 == rel2  # Same source, relation, target
        assert rel1 != rel3  # Different relation
        assert rel1 != rel4  # Different source/target

    def test_relationship_hash(self):
        """Test relationship hashing for use in sets/dicts."""
        rel1 = Relationship("entity:1", "REL", "entity:2")
        rel2 = Relationship("entity:1", "REL", "entity:2")
        rel3 = Relationship("entity:2", "REL", "entity:3")

        # Same components = same hash
        assert hash(rel1) == hash(rel2)

        # Can be used in sets
        rel_set = {rel1, rel2, rel3}
        assert len(rel_set) == 2  # rel1 and rel2 are duplicates

    def test_relationship_mutable_attributes(self):
        """Test that relationship attributes are mutable."""
        rel = Relationship("entity:1", "REL", "entity:2")

        rel.attributes["key"] = "value"
        assert rel.attributes["key"] == "value"

        rel.attributes["key"] = "new_value"
        assert rel.attributes["key"] == "new_value"
