"""File-based storage implementation for graphs.

This module provides FileStorage - persistent storage using JSON or pickle
serialization to save graph data to disk.
"""

import json
import pickle
from pathlib import Path
from typing import Literal

from cogent.graph.models import Entity, Relationship


class FileStorage:
    """File-based persistent storage with JSON or pickle serialization.

    Saves graph data to disk in JSON (human-readable) or pickle (faster)
    format. Auto-loads on initialization and auto-saves on modifications.

    Args:
        path: File path for storage.
        format: Serialization format ("json" or "pickle").
        auto_save: Automatically save after modifications (default: True).

    Example:
        >>> storage = FileStorage("graph.json", format="json")
        >>> await storage.add_entity("alice", "Person", name="Alice")
        >>> # Data automatically saved to graph.json
    """

    def __init__(
        self,
        path: str,
        format: Literal["json", "pickle"] = "json",
        auto_save: bool = True,
    ) -> None:
        """Initialize file storage."""
        self.path = Path(path)
        self.format = format
        self.auto_save = auto_save

        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []

        # Load existing data if file exists
        if self.path.exists():
            self._load_sync()

    def _load_sync(self) -> None:
        """Load data from file (synchronous)."""
        try:
            if self.format == "json":
                with self.path.open("r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:  # Empty file
                        return
                    data = json.loads(content)
                    # Reconstruct entities
                    self._entities = {
                        ent["id"]: Entity(
                            id=ent["id"],
                            entity_type=ent["type"],
                            attributes=ent.get("attributes", {}),
                            created_at=ent.get("created_at"),
                            updated_at=ent.get("updated_at"),
                        )
                        for ent in data.get("entities", [])
                    }
                    # Reconstruct relationships
                    self._relationships = [
                        Relationship(
                            source_id=rel["source_id"],
                            relation=rel["relation"],
                            target_id=rel["target_id"],
                            attributes=rel.get("attributes", {}),
                        )
                        for rel in data.get("relationships", [])
                    ]
            else:  # pickle
                with self.path.open("rb") as f:
                    data = pickle.load(f)
                    self._entities = data.get("entities", {})
                    self._relationships = data.get("relationships", [])
        except Exception as e:
            raise IOError(f"Failed to load from {self.path}: {e}") from e

    def _save_sync(self) -> None:
        """Save data to file (synchronous)."""
        try:
            # Ensure parent directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)

            if self.format == "json":
                data = {
                    "entities": [
                        {
                            "id": ent.id,
                            "type": ent.entity_type,
                            "attributes": ent.attributes,
                            "created_at": ent.created_at.isoformat() if ent.created_at else None,
                            "updated_at": ent.updated_at.isoformat() if ent.updated_at else None,
                        }
                        for ent in self._entities.values()
                    ],
                    "relationships": [
                        {
                            "source_id": rel.source_id,
                            "relation": rel.relation,
                            "target_id": rel.target_id,
                            "attributes": rel.attributes,
                        }
                        for rel in self._relationships
                    ],
                }
                with self.path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:  # pickle
                data = {
                    "entities": self._entities,
                    "relationships": self._relationships,
                }
                with self.path.open("wb") as f:
                    pickle.dump(data, f)
        except Exception as e:
            raise IOError(f"Failed to save to {self.path}: {e}") from e

    async def save(self) -> None:
        """Manually save data to file."""
        self._save_sync()

    async def load(self) -> None:
        """Manually load data from file."""
        self._load_sync()

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

        if self.auto_save:
            self._save_sync()

        return entity

    async def add_entities(self, entities: list[Entity]) -> list[Entity]:
        """Bulk add multiple entities."""
        for entity in entities:
            if entity.id in self._entities:
                raise ValueError(f"Entity with ID '{entity.id}' already exists")
            self._entities[entity.id] = entity

        if self.auto_save:
            self._save_sync()

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

        if self.auto_save:
            self._save_sync()

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

        if self.auto_save:
            self._save_sync()

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

        if self.auto_save:
            self._save_sync()

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
        """Execute a pattern-based query."""
        # Placeholder implementation
        return []

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int | None = None,
    ) -> list[str] | None:
        """Find shortest path between two entities using BFS."""
        if source_id not in self._entities or target_id not in self._entities:
            return None

        if source_id == target_id:
            return [source_id]

        # BFS
        from collections import deque

        queue: deque[tuple[str, list[str]]] = deque([(source_id, [source_id])])
        visited: set[str] = {source_id}

        while queue:
            current_id, path = queue.popleft()

            # Check max depth
            if max_depth is not None and len(path) - 1 >= max_depth:
                continue

            # Get outgoing relationships
            for rel in self._relationships:
                if rel.source_id == current_id:
                    next_id = rel.target_id

                    if next_id == target_id:
                        return path + [next_id]

                    if next_id not in visited:
                        visited.add(next_id)
                        queue.append((next_id, path + [next_id]))

        return None

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

        if self.auto_save:
            self._save_sync()
