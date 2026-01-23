"""Knowledge Graph backend protocol.

Defines the interface all storage backends must implement.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agenticflow.capabilities.knowledge_graph.models import Entity, Relationship


@runtime_checkable
class GraphBackend(Protocol):
    """Protocol for graph storage backends."""

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Entity:
        """Add or update an entity."""
        ...

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        ...

    def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Relationship:
        """Add a relationship between entities."""
        ...

    def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        direction: str = "outgoing",
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        ...

    def query(self, pattern: str) -> list[dict[str, Any]]:
        """Query the graph with a pattern."""
        ...

    def find_path(
        self, source_id: str, target_id: str, max_depth: int = 3
    ) -> list[list[str]] | None:
        """Find paths between two entities."""
        ...

    def get_all_entities(
        self,
        entity_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Entity]:
        """Get all entities, optionally filtered by type with pagination."""
        ...

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and its relationships."""
        ...

    def stats(self) -> dict[str, int]:
        """Get graph statistics."""
        ...

    def clear(self) -> None:
        """Clear all entities and relationships."""
        ...
