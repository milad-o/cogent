"""In-memory graph backend powered by the async Graph API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cogent.capabilities.knowledge_graph.backends.base import (
    GraphBackend,
    _SyncGraphBackend,
)
from cogent.capabilities.knowledge_graph.models import Entity, Relationship
from cogent.graph import Graph
from cogent.graph.storage import MemoryStorage


class InMemoryGraph(_SyncGraphBackend, GraphBackend):
    """Synchronous facade over Graph+MemoryStorage with optional persistence."""

    def __init__(self, path: str | Path | None = None, auto_save: bool = False) -> None:
        self._path = Path(path) if path else None
        self._auto_save = auto_save and self._path is not None
        super().__init__(Graph(storage=MemoryStorage()))

        # Load existing snapshot if provided
        if self._path and self._path.exists():
            self.load(self._path)

    def _maybe_save(self) -> None:
        if self._auto_save and self._path:
            self.save(self._path)

    # Override to inject auto-save hooks
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Entity:
        entity = super().add_entity(entity_id, entity_type, attributes, source)
        self._maybe_save()
        return entity

    def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Relationship:
        # Auto-create entities for convenience
        if not self.get_entity(source_id):
            super().add_entity(source_id, "Unknown")
        if not self.get_entity(target_id):
            super().add_entity(target_id, "Unknown")

        rel = super().add_relationship(source_id, relation, target_id, attributes, source)
        self._maybe_save()
        return rel

    def clear(self) -> None:
        super().clear()
        self._maybe_save()

    # Persistence helpers for legacy behaviour
    def save(self, path: str | Path | None = None) -> None:
        if path is None and not self._path:
            raise ValueError("Path required for InMemoryGraph.save()")

        target = Path(path) if path else self._path
        if target is None:
            raise ValueError("Path required for InMemoryGraph.save()")

        entities = self.get_all_entities()
        relationships = self.get_relationships(None, direction="both")

        data = {
            "entities": [
                {
                    "id": e.id,
                    "type": e.type,
                    "attributes": e.attributes,
                    "source": getattr(e, "source", None),
                }
                for e in entities
            ],
            "relationships": [
                {
                    "source": r.source_id,
                    "relation": r.relation,
                    "target": r.target_id,
                    "attributes": r.attributes,
                }
                for r in relationships
            ],
        }

        target.write_text(json.dumps(data, indent=2, default=str))

    def load(self, path: str | Path | None = None) -> None:
        if path is None and not self._path:
            raise ValueError("Path required for InMemoryGraph.load()")

        target = Path(path) if path else self._path
        if target is None or not target.exists():
            return

        data = json.loads(target.read_text())
        self.clear()

        for ent in data.get("entities", []):
            super().add_entity(ent["id"], ent.get("type", "Entity"), ent.get("attributes", {}), ent.get("source"))

        for rel in data.get("relationships", []):
            super().add_relationship(
                rel["source"], rel["relation"], rel["target"], rel.get("attributes", {}), rel.get("source")
            )

        self._maybe_save()
