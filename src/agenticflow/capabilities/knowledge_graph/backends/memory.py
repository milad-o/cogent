"""In-memory graph backend using networkx (if available) or simple dict-based storage."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agenticflow.capabilities.knowledge_graph.backends.base import GraphBackend
from agenticflow.capabilities.knowledge_graph.models import Entity, Relationship


class InMemoryGraph(GraphBackend):
    """Simple in-memory graph storage using networkx (if available)."""

    def __init__(self) -> None:
        try:
            import networkx as nx

            self._nx = nx
            self.graph = nx.DiGraph()
        except ImportError:
            # Fallback to simple dict-based storage
            self._nx = None
            self._entities: dict[str, Entity] = {}
            self._relationships: list[Relationship] = []

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Entity:
        """Add or update an entity."""
        now = datetime.now(UTC)

        if self._nx:
            existing = self.graph.nodes.get(entity_id, {})
            # Parse created_at if it's a string (from previous to_dict())
            existing_created = existing.get("created_at", now)
            if isinstance(existing_created, str):
                existing_created = datetime.fromisoformat(existing_created)

            entity = Entity(
                id=entity_id,
                type=entity_type,
                attributes={**existing.get("attributes", {}), **(attributes or {})},
                created_at=existing_created,
                updated_at=now,
                source=source or existing.get("source"),
            )
            self.graph.add_node(entity_id, **entity.to_dict())
        else:
            existing = self._entities.get(entity_id)
            entity = Entity(
                id=entity_id,
                type=entity_type,
                attributes={**(existing.attributes if existing else {}), **(attributes or {})},
                created_at=existing.created_at if existing else now,
                updated_at=now,
                source=source or (existing.source if existing else None),
            )
            self._entities[entity_id] = entity

        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        if self._nx:
            if entity_id in self.graph.nodes:
                data = self.graph.nodes[entity_id]
                return Entity(
                    id=entity_id,
                    type=data.get("type", "unknown"),
                    attributes=data.get("attributes", {}),
                    created_at=(
                        datetime.fromisoformat(data["created_at"])
                        if "created_at" in data
                        else datetime.now(UTC)
                    ),
                    updated_at=(
                        datetime.fromisoformat(data["updated_at"])
                        if "updated_at" in data
                        else datetime.now(UTC)
                    ),
                    source=data.get("source"),
                )
            return None
        else:
            return self._entities.get(entity_id)

    def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Relationship:
        """Add a relationship between entities."""
        rel = Relationship(
            source_id=source_id,
            relation=relation,
            target_id=target_id,
            attributes=attributes or {},
            source=source,
        )

        if self._nx:
            # Don't duplicate 'relation' - it's already in to_dict()
            edge_data = rel.to_dict()
            self.graph.add_edge(source_id, target_id, **edge_data)
        else:
            # Check if relationship already exists
            existing = [
                r
                for r in self._relationships
                if r.source_id == source_id
                and r.relation == relation
                and r.target_id == target_id
            ]
            if not existing:
                self._relationships.append(rel)

        return rel

    def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        direction: str = "outgoing",
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        results: list[Relationship] = []

        if self._nx:
            if direction in ("outgoing", "both"):
                for _, target, data in self.graph.out_edges(entity_id, data=True):
                    if relation is None or data.get("relation") == relation:
                        results.append(
                            Relationship(
                                source_id=entity_id,
                                relation=data.get("relation", "related_to"),
                                target_id=target,
                                attributes=data.get("attributes", {}),
                            )
                        )
            if direction in ("incoming", "both"):
                for source, _, data in self.graph.in_edges(entity_id, data=True):
                    if relation is None or data.get("relation") == relation:
                        results.append(
                            Relationship(
                                source_id=source,
                                relation=data.get("relation", "related_to"),
                                target_id=entity_id,
                                attributes=data.get("attributes", {}),
                            )
                        )
        else:
            for rel in self._relationships:
                if direction in ("outgoing", "both") and rel.source_id == entity_id:
                    if relation is None or rel.relation == relation:
                        results.append(rel)
                if direction in ("incoming", "both") and rel.target_id == entity_id:
                    if relation is None or rel.relation == relation:
                        results.append(rel)

        return results

    def query(self, pattern: str) -> list[dict[str, Any]]:
        """
        Query the graph with a simple pattern.

        Patterns:
        - "entity_id" - Get entity and its relationships
        - "entity_id -relation-> ?" - Find targets of relation
        - "? -relation-> entity_id" - Find sources of relation
        - "entity_id -?-> ?" - Find all outgoing relationships
        """
        results: list[dict[str, Any]] = []
        pattern = pattern.strip()

        # Pattern: entity_id -relation-> ?
        if " -" in pattern and "-> ?" in pattern:
            parts = pattern.split(" -")
            source = parts[0].strip()
            relation = parts[1].replace("-> ?", "").strip()
            relation = None if relation == "?" else relation

            for rel in self.get_relationships(source, relation, "outgoing"):
                target = self.get_entity(rel.target_id)
                results.append(
                    {
                        "source": source,
                        "relation": rel.relation,
                        "target": rel.target_id,
                        "target_type": target.type if target else "unknown",
                        "target_attributes": target.attributes if target else {},
                    }
                )

        # Pattern: ? -relation-> entity_id
        elif "? -" in pattern and "-> " in pattern:
            parts = pattern.split("-> ")
            target = parts[1].strip()
            relation = parts[0].replace("? -", "").strip()
            relation = None if relation == "?" else relation

            for rel in self.get_relationships(target, relation, "incoming"):
                source_entity = self.get_entity(rel.source_id)
                results.append(
                    {
                        "source": rel.source_id,
                        "source_type": source_entity.type if source_entity else "unknown",
                        "relation": rel.relation,
                        "target": target,
                    }
                )

        # Pattern: just entity_id - return entity + relationships
        else:
            entity = self.get_entity(pattern)
            if entity:
                results.append(
                    {
                        "entity": entity.to_dict(),
                        "outgoing": [
                            r.to_dict()
                            for r in self.get_relationships(pattern, direction="outgoing")
                        ],
                        "incoming": [
                            r.to_dict()
                            for r in self.get_relationships(pattern, direction="incoming")
                        ],
                    }
                )

        return results

    def find_path(
        self, source_id: str, target_id: str, max_depth: int = 3
    ) -> list[list[str]] | None:
        """Find paths between two entities."""
        if self._nx:
            try:
                paths = list(
                    self._nx.all_simple_paths(
                        self.graph, source_id, target_id, cutoff=max_depth
                    )
                )
                return paths if paths else None
            except self._nx.NetworkXNoPath:
                return None
        else:
            # Simple BFS for non-networkx
            visited: set[str] = set()
            queue: list[list[str]] = [[source_id]]

            while queue:
                path = queue.pop(0)
                node = path[-1]

                if node == target_id:
                    return [path]

                if node in visited or len(path) > max_depth:
                    continue

                visited.add(node)

                for rel in self.get_relationships(node, direction="outgoing"):
                    new_path = path + [rel.target_id]
                    queue.append(new_path)

            return None

    def get_all_entities(
        self,
        entity_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Entity]:
        """Get all entities, optionally filtered by type with pagination."""
        if self._nx:
            entities = []
            for node_id, data in self.graph.nodes(data=True):
                if entity_type is None or data.get("type") == entity_type:
                    entities.append(
                        Entity(
                            id=node_id,
                            type=data.get("type", "unknown"),
                            attributes=data.get("attributes", {}),
                        )
                    )
            # Apply pagination
            if limit is not None:
                return entities[offset : offset + limit]
            return entities[offset:] if offset else entities
        else:
            if entity_type is None:
                entities = list(self._entities.values())
            else:
                entities = [e for e in self._entities.values() if e.type == entity_type]
            # Apply pagination
            if limit is not None:
                return entities[offset : offset + limit]
            return entities[offset:] if offset else entities

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and its relationships."""
        if self._nx:
            if entity_id in self.graph.nodes:
                self.graph.remove_node(entity_id)
                return True
            return False
        else:
            if entity_id in self._entities:
                del self._entities[entity_id]
                self._relationships = [
                    r
                    for r in self._relationships
                    if r.source_id != entity_id and r.target_id != entity_id
                ]
                return True
            return False

    def stats(self) -> dict[str, int]:
        """Get graph statistics."""
        if self._nx:
            return {
                "entities": self.graph.number_of_nodes(),
                "relationships": self.graph.number_of_edges(),
            }
        else:
            return {
                "entities": len(self._entities),
                "relationships": len(self._relationships),
            }

    def clear(self) -> None:
        """Clear all entities and relationships."""
        if self._nx:
            self.graph.clear()
        else:
            self._entities.clear()
            self._relationships.clear()

    def save(self, path: str | Path | None = None) -> None:
        """Save graph to JSON file."""
        if path is None:
            raise ValueError("Path required for InMemoryGraph.save()")

        path = Path(path)

        # Serialize entities and relationships
        data: dict[str, list[dict[str, Any]]] = {
            "entities": [],
            "relationships": [],
        }

        for entity in self.get_all_entities():
            data["entities"].append(entity.to_dict())

        if self._nx:
            for source, target, edge_data in self.graph.edges(data=True):
                data["relationships"].append(
                    {
                        "source": source,
                        "target": target,
                        "relation": edge_data.get("relation", "related_to"),
                        "attributes": edge_data.get("attributes", {}),
                        "created_at": edge_data.get(
                            "created_at", datetime.now(UTC).isoformat()
                        ),
                    }
                )
        else:
            for rel in self._relationships:
                data["relationships"].append(rel.to_dict())

        path.write_text(json.dumps(data, indent=2, default=str))

    def load(self, path: str | Path | None = None) -> None:
        """Load graph from JSON file."""
        if path is None:
            raise ValueError("Path required for InMemoryGraph.load()")

        path = Path(path)
        if not path.exists():
            return  # Nothing to load

        data = json.loads(path.read_text())

        # Clear existing data
        self.clear()

        # Load entities
        for e in data.get("entities", []):
            self.add_entity(
                e["id"],
                e["type"],
                e.get("attributes", {}),
                e.get("source"),
            )

        # Load relationships
        for r in data.get("relationships", []):
            self.add_relationship(
                r["source"],
                r["relation"],
                r["target"],
                r.get("attributes", {}),
            )
