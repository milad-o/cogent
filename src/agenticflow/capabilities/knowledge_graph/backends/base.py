"""
Knowledge Graph backend abstract base class.

Defines the interface all storage backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agenticflow.capabilities.knowledge_graph.models import Entity, Relationship


class GraphBackend(ABC):
    """Abstract base class for graph storage backends."""

    @abstractmethod
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Entity:
        """Add or update an entity."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        pass

    @abstractmethod
    def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Relationship:
        """Add a relationship between entities."""
        pass

    @abstractmethod
    def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        direction: str = "outgoing",
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        pass

    @abstractmethod
    def query(self, pattern: str) -> list[dict[str, Any]]:
        """Query the graph with a pattern."""
        pass

    @abstractmethod
    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> list[list[str]] | None:
        """Find paths between two entities."""
        pass

    @abstractmethod
    def get_all_entities(
        self,
        entity_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Entity]:
        """Get all entities, optionally filtered by type with pagination."""
        pass

    @abstractmethod
    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and its relationships."""
        pass

    @abstractmethod
    def stats(self) -> dict[str, int]:
        """Get graph statistics."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entities and relationships."""
        pass

    # Optional batch methods for performance (default implementations)
    def add_entities_batch(
        self,
        entities: list[tuple[str, str, dict[str, Any] | None]],
    ) -> int:
        """Bulk insert entities. Override for better performance."""
        for eid, etype, attrs in entities:
            self.add_entity(eid, etype, attrs)
        return len(entities)

    def add_relationships_batch(
        self,
        relationships: list[tuple[str, str, str]],
    ) -> int:
        """Bulk insert relationships. Override for better performance."""
        for src, rel, tgt in relationships:
            self.add_relationship(src, rel, tgt)
        return len(relationships)

    def save(self, path: str | Path | None = None) -> None:
        """Save graph to persistent storage (optional)."""
        pass

    def load(self, path: str | Path | None = None) -> None:
        """Load graph from persistent storage (optional)."""
        pass

    def close(self) -> None:
        """Close any open connections (optional)."""
        pass
