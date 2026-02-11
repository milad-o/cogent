"""Shared utilities for knowledge graph backends."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable
from typing import Any, Protocol, runtime_checkable

from cogent.graph import Graph
from cogent.graph.models import Entity, Relationship


@runtime_checkable
class GraphBackend(Protocol):
    """Protocol for graph storage backends (synchronous facade)."""

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Entity:
        ...

    def get_entity(self, entity_id: str) -> Entity | None:
        ...

    def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Relationship:
        ...

    def get_relationships(
        self,
        entity_id: str | None,
        relation: str | None = None,
        direction: str = "outgoing",
    ) -> list[Relationship]:
        ...

    def query(self, pattern: str) -> list[dict[str, Any]]:
        ...

    def find_path(
        self, source_id: str, target_id: str, max_depth: int = 3
    ) -> list[list[str]] | None:
        ...

    def get_all_entities(
        self,
        entity_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Entity]:
        ...

    def remove_entity(self, entity_id: str) -> bool:
        ...

    def stats(self) -> dict[str, int]:
        ...

    def clear(self) -> None:
        ...

    def close(self) -> None:
        ...


class _GraphRunner:
    """Run async Graph operations on a dedicated event loop thread."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="kg-graph-loop", daemon=True
        )
        self._thread.start()

    def run(self, coro: Awaitable[Any]) -> Any:
        """Synchronously execute a coroutine on the background loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def shutdown(self) -> None:
        """Stop the background loop and wait for thread exit."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=1)


class _SyncGraphBackend(GraphBackend):
    """Synchronous facade built on top of the async Graph API."""

    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._runner = _GraphRunner(graph)

    # --- Entity operations ---
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Entity:
        attrs = attributes or {}
        if source:
            attrs = {**attrs, "source": source}
        # Upsert semantics to mirror legacy behaviour
        existing = self._runner.run(self._graph.get_entity(entity_id))
        if existing:
            if entity_type and entity_type != existing.entity_type:
                existing.entity_type = entity_type
            existing.update_attributes(**attrs)
            # Keep engine in sync when attributes change
            self._runner.run(self._graph.engine.add_node(entity_id, **existing.attributes))
            return existing

        return self._runner.run(self._graph.add_entity(entity_id, entity_type, **attrs))

    def add_entities_batch(
        self, entities: list[tuple[str, str, dict[str, Any] | None]]
    ) -> int:
        count = 0
        for eid, etype, attrs in entities:
            self.add_entity(eid, etype, attrs)
            count += 1
        return count

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._runner.run(self._graph.get_entity(entity_id))

    def remove_entity(self, entity_id: str) -> bool:
        return bool(self._runner.run(self._graph.remove_entity(entity_id)))

    def get_all_entities(
        self, entity_type: str | None = None, limit: int | None = None, offset: int = 0
    ) -> list[Entity]:
        entities = self._runner.run(self._graph.get_all_entities())
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        if offset:
            entities = entities[offset:]
        if limit is not None:
            entities = entities[:limit]
        return entities

    # --- Relationship operations ---
    def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Relationship:
        attrs = attributes or {}
        if source:
            attrs = {**attrs, "source": source}
        existing = self._runner.run(
            self._graph.get_relationships(
                source_id=source_id, relation=relation, target_id=target_id
            )
        )
        if existing:
            return existing[0]
        return self._runner.run(
            self._graph.add_relationship(source_id, relation, target_id, **attrs)
        )

    def add_relationships_batch(
        self, relationships: list[tuple[str, str, str]]
    ) -> int:
        count = 0
        for src, rel, dst in relationships:
            self.add_relationship(src, rel, dst)
            count += 1
        return count

    def get_relationships(
        self,
        entity_id: str | None,
        relation: str | None = None,
        direction: str = "outgoing",
    ) -> list[Relationship]:
        relationships: list[Relationship] = []

        if direction in ("outgoing", "both") and entity_id is not None:
            relationships.extend(
                self._runner.run(
                    self._graph.get_relationships(
                        source_id=entity_id, relation=relation, target_id=None
                    )
                )
            )

        if direction in ("incoming", "both") and entity_id is not None:
            relationships.extend(
                self._runner.run(
                    self._graph.get_relationships(
                        source_id=None, relation=relation, target_id=entity_id
                    )
                )
            )

        if entity_id is None:
            relationships.extend(self._runner.run(self._graph.get_relationships()))

        return relationships

    def find_path(
        self, source_id: str, target_id: str, max_depth: int = 3
    ) -> list[list[str]] | None:
        path = self._runner.run(self._graph.find_path(source_id, target_id, max_depth))
        if path is None:
            return None
        if path and isinstance(path[0], list):
            return path  # type: ignore[return-value]
        return [path]  # type: ignore[list-item]

    # --- Query and utility operations ---
    def query(self, pattern: str) -> list[dict[str, Any]]:
        pattern = pattern.strip()

        # Pattern: source -relation-> target with wildcards
        if " -" in pattern and "-> " in pattern:
            source_part, rest = pattern.split(" -", 1)
            relation_part, target_part = rest.split("->", 1)

            source = None if source_part.strip() == "?" else source_part.strip()
            relation = None if relation_part.strip() == "?" else relation_part.strip()
            target = None if target_part.strip() == "?" else target_part.strip()

            results: list[dict[str, Any]] = []

            # Outgoing lookup
            if source is not None:
                rels = self._runner.run(
                    self._graph.get_relationships(
                        source_id=source, relation=relation, target_id=target
                    )
                )
                for rel in rels:
                    target_entity = self._runner.run(
                        self._graph.get_entity(rel.target_id)
                    )
                    results.append(
                        {
                            "source": rel.source_id,
                            "relation": rel.relation,
                            "target": rel.target_id,
                            "target_type": target_entity.entity_type
                            if target_entity
                            else None,
                        }
                    )
                return results

            # Incoming lookup
            if target is not None:
                rels = self._runner.run(
                    self._graph.get_relationships(
                        source_id=source, relation=relation, target_id=target
                    )
                )
                for rel in rels:
                    source_entity = self._runner.run(
                        self._graph.get_entity(rel.source_id)
                    )
                    results.append(
                        {
                            "source": rel.source_id,
                            "source_type": source_entity.entity_type
                            if source_entity
                            else None,
                            "relation": rel.relation,
                            "target": rel.target_id,
                        }
                    )
                return results

        # Fallback: entity lookup with relationships
        entity = self._runner.run(self._graph.get_entity(pattern))
        if entity:
            return [
                {
                    "entity": {
                        "id": entity.id,
                        "type": entity.entity_type,
                        "attributes": entity.attributes,
                    },
                    "outgoing": [
                        {
                            "source_id": rel.source_id,
                            "relation": rel.relation,
                            "target_id": rel.target_id,
                            "attributes": rel.attributes,
                        }
                        for rel in self.get_relationships(entity.id, direction="outgoing")
                    ],
                    "incoming": [
                        {
                            "source_id": rel.source_id,
                            "relation": rel.relation,
                            "target_id": rel.target_id,
                            "attributes": rel.attributes,
                        }
                        for rel in self.get_relationships(entity.id, direction="incoming")
                    ],
                }
            ]

        return []

    def stats(self) -> dict[str, int]:
        stats = self._runner.run(self._graph.stats())
        return {
            "entities": stats.get("entity_count", 0),
            "relationships": stats.get("relationship_count", 0),
        }

    def clear(self) -> None:
        self._runner.run(self._graph.clear())

    def close(self) -> None:
        self._runner.shutdown()
