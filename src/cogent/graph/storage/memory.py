"""In-memory storage implementation for graphs.

This module provides MemoryStorage - a fast, transient storage backend
that keeps all graph data in memory using native Python data structures.
"""

from cogent.graph.models import Entity, Relationship


class MemoryStorage:
    """In-memory storage backend with zero persistence.

    Fast, transient storage using dicts and lists. All data is lost when
    the process exits. Ideal for testing, prototyping, or temporary graphs.

    Storage:
        - Entities: dict[entity_id, Entity]
        - Relationships: list[Relationship]

    Example:
        >>> storage = MemoryStorage()
        >>> entity = await storage.add_entity("alice", "Person", name="Alice")
        >>> relationships = await storage.get_relationships(source_id="alice")
    """

    def __init__(self) -> None:
        """Initialize empty in-memory storage."""
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []

    async def add_entity(
        self,
        id: str,
        entity_type: str,
        **attributes: object,
    ) -> Entity:
        """Add a new entity to storage."""
        if id in self._entities:
            raise ValueError(f"Entity with ID '{id}' already exists")

        entity = Entity(id=id, entity_type=entity_type, attributes=attributes)
        self._entities[id] = entity
        return entity

    async def add_entities(self, entities: list[Entity]) -> list[Entity]:
        """Bulk add multiple entities."""
        for entity in entities:
            if entity.id in self._entities:
                raise ValueError(f"Entity with ID '{entity.id}' already exists")
            self._entities[entity.id] = entity
        return entities

    async def get_entity(self, id: str) -> Entity | None:
        """Retrieve an entity by ID."""
        return self._entities.get(id)

    async def remove_entity(self, id: str) -> bool:
        """Remove an entity and all its relationships."""
        if id not in self._entities:
            return False

        # Remove entity
        del self._entities[id]

        # Remove all relationships involving this entity
        self._relationships = [
            rel
            for rel in self._relationships
            if rel.source_id != id and rel.target_id != id
        ]

        return True

    async def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        **attributes: object,
    ) -> Relationship:
        """Add a new relationship between entities."""
        # Verify entities exist
        if source_id not in self._entities:
            raise ValueError(f"Source entity '{source_id}' does not exist")
        if target_id not in self._entities:
            raise ValueError(f"Target entity '{target_id}' does not exist")

        relationship = Relationship(
            source_id=source_id,
            relation=relation,
            target_id=target_id,
            attributes=attributes,
        )
        self._relationships.append(relationship)
        return relationship

    async def add_relationships(
        self, relationships: list[Relationship]
    ) -> list[Relationship]:
        """Bulk add multiple relationships."""
        # Verify all entities exist
        for rel in relationships:
            if rel.source_id not in self._entities:
                raise ValueError(f"Source entity '{rel.source_id}' does not exist")
            if rel.target_id not in self._entities:
                raise ValueError(f"Target entity '{rel.target_id}' does not exist")

        self._relationships.extend(relationships)
        return relationships

    async def get_relationships(
        self,
        source_id: str | None = None,
        relation: str | None = None,
        target_id: str | None = None,
    ) -> list[Relationship]:
        """Query relationships by source, relation, and/or target."""
        results = []

        for rel in self._relationships:
            # Apply filters
            if source_id is not None and rel.source_id != source_id:
                continue
            if relation is not None and rel.relation != relation:
                continue
            if target_id is not None and rel.target_id != target_id:
                continue

            results.append(rel)

        return results

    async def query(self, pattern: str) -> list[dict[str, object]]:
        """Execute a pattern-based query.

        Simple pattern matching for demo purposes. Real implementations
        would parse and execute structured queries.
        """
        # Placeholder implementation
        return []

    async def get_all_entities(self) -> list[Entity]:
        """Retrieve all entities."""
        return list(self._entities.values())

    async def stats(self) -> dict[str, int]:
        """Get storage statistics."""
        return {
            "entity_count": len(self._entities),
            "relationship_count": len(self._relationships),
        }

    async def clear(self) -> None:
        """Remove all entities and relationships."""
        self._entities.clear()
        self._relationships.clear()
