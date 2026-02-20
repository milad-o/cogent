"""Storage backend protocol for Knowledge Graph.

This module defines the contract that all storage backends must implement.
Backends can use different storage mechanisms (in-memory, SQLite, Neo4j, etc.)
while providing a consistent async-first interface with bulk operations.
"""

from typing import Protocol

from cogent.graph.models import Entity, Relationship


class Storage(Protocol):
    """Protocol defining the async interface for KG storage backends.

    All storage backends must implement these async methods to be compatible
    with the KnowledgeGraph class. This enables swapping storage implementations
    while maintaining a consistent async API.

    All methods are async-first to support modern async/await patterns and
    enable efficient I/O operations with databases and external services.

    Example:
        >>> class MyBackend:
        ...     async def add_entity(self, id: str, entity_type: str, **attributes) -> Entity:
        ...         # Implementation here
        ...         pass
        ...
        >>> backend: Storage = MyBackend()  # Type-safe
    """

    async def add_entity(
        self,
        id: str,
        entity_type: str,
        **attributes: object,
    ) -> Entity:
        """Add a new entity to the knowledge graph.

        Args:
            id: Unique identifier for the entity.
            entity_type: Classification or category of the entity.
            **attributes: Key-value pairs for entity properties.

        Returns:
            The created Entity object.

        Raises:
            ValueError: If entity with this ID already exists.

        Example:
            >>> entity = await backend.add_entity(
            ...     "person:alice",
            ...     "Person",
            ...     name="Alice",
            ...     age=30
            ... )
        """
        ...

    async def add_entities(self, entities: list[Entity]) -> list[Entity]:
        """Add multiple entities in bulk.

        Args:
            entities: List of Entity objects to add.

        Returns:
            List of created Entity objects.

        Raises:
            ValueError: If any entity ID already exists.

        Example:
            >>> entities = [
            ...     Entity("person:alice", "Person", {"name": "Alice"}),
            ...     Entity("person:bob", "Person", {"name": "Bob"}),
            ... ]
            >>> created = await backend.add_entities(entities)
        """
        ...

    async def get_entity(self, id: str) -> Entity | None:
        """Retrieve an entity by its ID.

        Args:
            id: The entity ID to look up.

        Returns:
            The Entity if found, None otherwise.

        Example:
            >>> entity = await backend.get_entity("person:alice")
            >>> if entity:
            ...     print(entity.attributes["name"])
        """
        ...

    async def remove_entity(self, id: str) -> bool:
        """Remove an entity and all its relationships.

        Args:
            id: The entity ID to remove.

        Returns:
            True if entity was removed, False if not found.

        Example:
            >>> removed = await backend.remove_entity("person:alice")
        """
        ...

    async def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        **attributes: object,
    ) -> Relationship:
        """Add a relationship between two entities.

        Args:
            source_id: ID of the source entity.
            relation: Type/label of the relationship.
            target_id: ID of the target entity.
            **attributes: Key-value pairs for relationship properties.

        Returns:
            The created Relationship object.

        Raises:
            ValueError: If source or target entity doesn't exist.

        Example:
            >>> rel = await backend.add_relationship(
            ...     "person:alice",
            ...     "KNOWS",
            ...     "person:bob",
            ...     since=2020
            ... )
        """
        ...

    async def add_relationships(
        self, relationships: list[Relationship]
    ) -> list[Relationship]:
        """Add multiple relationships in bulk.

        Args:
            relationships: List of Relationship objects to add.

        Returns:
            List of created Relationship objects.

        Raises:
            ValueError: If any source or target entity doesn't exist.

        Example:
            >>> rels = [
            ...     Relationship("person:alice", "KNOWS", "person:bob"),
            ...     Relationship("person:bob", "KNOWS", "person:charlie"),
            ... ]
            >>> created = await backend.add_relationships(rels)
        """
        ...

    async def get_relationships(
        self,
        entity_id: str | None = None,
        relation: str | None = None,
    ) -> list[Relationship]:
        """Get relationships filtered by entity and/or relation type.

        Args:
            entity_id: Filter by source or target entity ID. If None, return all.
            relation: Filter by relationship type. If None, include all types.

        Returns:
            List of matching relationships.

        Example:
            >>> # Get all relationships for an entity
            >>> rels = await backend.get_relationships("person:alice")
            >>> # Get all KNOWS relationships
            >>> knows = await backend.get_relationships(relation="KNOWS")
            >>> # Get specific entity's KNOWS relationships
            >>> alice_knows = await backend.get_relationships("person:alice", "KNOWS")
        """
        ...

    async def query(self, pattern: str) -> list[dict[str, object]]:
        """Execute a pattern-based query on the knowledge graph.

        Pattern syntax:
        - `? -RELATION-> target_id`: Find all sources
        - `source_id -RELATION-> ?`: Find all targets
        - `source_id -?-> target_id`: Find all relations between entities
        - `? -REL1-> ? -REL2-> target`: Multi-hop queries

        Args:
            pattern: Query pattern string.

        Returns:
            List of result dictionaries containing matched entities/relationships.

        Example:
            >>> results = await backend.query("? -KNOWS-> person:bob")
            >>> # Returns [{"source": Entity(...), "relation": "KNOWS", "target": Entity(...)}]
        """
        ...

    async def get_all_entities(self) -> list[Entity]:
        """Retrieve all entities in the knowledge graph.

        Returns:
            List of all Entity objects.

        Example:
            >>> entities = await backend.get_all_entities()
            >>> print(f"Total entities: {len(entities)}")
        """
        ...

    async def stats(self) -> dict[str, int]:
        """Get statistics about the knowledge graph.

        Returns:
            Dictionary containing:
            - "entities": Total number of entities
            - "relationships": Total number of relationships
            - "types": Number of unique entity types

        Example:
            >>> stats = await backend.stats()
            >>> print(f"Entities: {stats['entities']}, Relationships: {stats['relationships']}")
        """
        ...

    async def clear(self) -> None:
        """Remove all entities and relationships from the knowledge graph.

        This operation cannot be undone.

        Example:
            >>> await backend.clear()
            >>> assert len(await backend.get_all_entities()) == 0
        """
        ...

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[str] | None:
        """Find shortest path between two entities.

        Args:
            source_id: Starting entity ID.
            target_id: Destination entity ID.
            max_depth: Maximum path length to search.

        Returns:
            List of entity IDs representing the path, or None if no path exists.

        Example:
            >>> path = await backend.find_path("person:alice", "person:charlie")
            >>> # Returns ["person:alice", "person:bob", "person:charlie"]
        """
        ...
