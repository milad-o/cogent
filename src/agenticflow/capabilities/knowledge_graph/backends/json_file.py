"""JSON file-backed graph storage with auto-save."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticflow.capabilities.knowledge_graph.models import Entity, Relationship
from agenticflow.capabilities.knowledge_graph.backends.base import GraphBackend
from agenticflow.capabilities.knowledge_graph.backends.memory import InMemoryGraph


class JSONFileGraph(GraphBackend):
    """JSON file-backed graph with auto-save."""

    def __init__(self, path: str | Path, auto_save: bool = True) -> None:
        self._path = Path(path)
        self._auto_save = auto_save
        self._memory = InMemoryGraph()
        self.load()

    def _maybe_save(self) -> None:
        """Auto-save if enabled."""
        if self._auto_save:
            self.save()

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Entity:
        result = self._memory.add_entity(entity_id, entity_type, attributes, source)
        self._maybe_save()
        return result

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._memory.get_entity(entity_id)

    def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Relationship:
        result = self._memory.add_relationship(
            source_id, relation, target_id, attributes, source
        )
        self._maybe_save()
        return result

    def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        direction: str = "outgoing",
    ) -> list[Relationship]:
        return self._memory.get_relationships(entity_id, relation, direction)

    def query(self, pattern: str) -> list[dict[str, Any]]:
        return self._memory.query(pattern)

    def find_path(
        self, source_id: str, target_id: str, max_depth: int = 3
    ) -> list[list[str]] | None:
        return self._memory.find_path(source_id, target_id, max_depth)

    def get_all_entities(
        self,
        entity_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Entity]:
        return self._memory.get_all_entities(entity_type, limit=limit, offset=offset)

    def remove_entity(self, entity_id: str) -> bool:
        result = self._memory.remove_entity(entity_id)
        if result:
            self._maybe_save()
        return result

    def stats(self) -> dict[str, int]:
        return self._memory.stats()

    def clear(self) -> None:
        self._memory.clear()
        self._maybe_save()

    def save(self, path: str | Path | None = None) -> None:
        """Save to JSON file."""
        self._memory.save(path or self._path)

    def load(self, path: str | Path | None = None) -> None:
        """Load from JSON file."""
        self._memory.load(path or self._path)
