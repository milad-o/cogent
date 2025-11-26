"""
KnowledgeGraph capability - persistent entity/relationship memory.

Adds tools for storing and querying entities and their relationships,
enabling multi-hop reasoning and grounded responses.

Supports multiple backends:
- "memory": In-memory storage (default, uses networkx if available)
- "sqlite": SQLite database for persistence and large graphs
- "json": JSON file for simple persistence

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import KnowledgeGraph
    
    # In-memory (default)
    kg = KnowledgeGraph()
    
    # SQLite for persistence
    kg = KnowledgeGraph(backend="sqlite", path="knowledge.db")
    
    # JSON file for simple persistence
    kg = KnowledgeGraph(backend="json", path="knowledge.json")
    
    agent = Agent(
        name="Assistant",
        model=model,
        capabilities=[kg],
    )
    
    # Agent can now remember facts and query relationships
    await agent.run("Remember that John works at Acme as a senior engineer")
    await agent.run("Who works at Acme?")  # → John
    ```
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from agenticflow.capabilities.base import BaseCapability


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    
    id: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None  # Where this fact came from
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
        }


@dataclass
class Relationship:
    """A relationship between two entities."""
    
    source_id: str
    relation: str
    target_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None  # Where this fact came from
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "relation": self.relation,
            "target": self.target_id,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat(),
        }


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
    def get_all_entities(self, entity_type: str | None = None) -> list[Entity]:
        """Get all entities, optionally filtered by type."""
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
    
    def save(self, path: str | Path | None = None) -> None:
        """Save graph to persistent storage (optional)."""
        pass
    
    def load(self, path: str | Path | None = None) -> None:
        """Load graph from persistent storage (optional)."""
        pass
    
    def close(self) -> None:
        """Close any open connections (optional)."""
        pass


class InMemoryGraph(GraphBackend):
    """Simple in-memory graph storage using networkx."""
    
    def __init__(self):
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
        now = datetime.now(timezone.utc)
        
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
                    created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
                    updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
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
            existing = [r for r in self._relationships 
                       if r.source_id == source_id and r.relation == relation and r.target_id == target_id]
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
        results = []
        
        if self._nx:
            if direction in ("outgoing", "both"):
                for _, target, data in self.graph.out_edges(entity_id, data=True):
                    if relation is None or data.get("relation") == relation:
                        results.append(Relationship(
                            source_id=entity_id,
                            relation=data.get("relation", "related_to"),
                            target_id=target,
                            attributes=data.get("attributes", {}),
                        ))
            if direction in ("incoming", "both"):
                for source, _, data in self.graph.in_edges(entity_id, data=True):
                    if relation is None or data.get("relation") == relation:
                        results.append(Relationship(
                            source_id=source,
                            relation=data.get("relation", "related_to"),
                            target_id=entity_id,
                            attributes=data.get("attributes", {}),
                        ))
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
        results = []
        pattern = pattern.strip()
        
        # Pattern: entity_id -relation-> ?
        if " -" in pattern and "-> ?" in pattern:
            parts = pattern.split(" -")
            source = parts[0].strip()
            relation = parts[1].replace("-> ?", "").strip()
            relation = None if relation == "?" else relation
            
            for rel in self.get_relationships(source, relation, "outgoing"):
                target = self.get_entity(rel.target_id)
                results.append({
                    "source": source,
                    "relation": rel.relation,
                    "target": rel.target_id,
                    "target_type": target.type if target else "unknown",
                    "target_attributes": target.attributes if target else {},
                })
        
        # Pattern: ? -relation-> entity_id
        elif "? -" in pattern and "-> " in pattern:
            parts = pattern.split("-> ")
            target = parts[1].strip()
            relation = parts[0].replace("? -", "").strip()
            relation = None if relation == "?" else relation
            
            for rel in self.get_relationships(target, relation, "incoming"):
                source_entity = self.get_entity(rel.source_id)
                results.append({
                    "source": rel.source_id,
                    "source_type": source_entity.type if source_entity else "unknown",
                    "relation": rel.relation,
                    "target": target,
                })
        
        # Pattern: just entity_id - return entity + relationships
        else:
            entity = self.get_entity(pattern)
            if entity:
                results.append({
                    "entity": entity.to_dict(),
                    "outgoing": [r.to_dict() for r in self.get_relationships(pattern, direction="outgoing")],
                    "incoming": [r.to_dict() for r in self.get_relationships(pattern, direction="incoming")],
                })
        
        return results
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> list[list[str]] | None:
        """Find paths between two entities."""
        if self._nx:
            try:
                paths = list(self._nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_depth))
                return paths if paths else None
            except self._nx.NetworkXNoPath:
                return None
        else:
            # Simple BFS for non-networkx
            visited = set()
            queue = [[source_id]]
            
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
    
    def get_all_entities(self, entity_type: str | None = None) -> list[Entity]:
        """Get all entities, optionally filtered by type."""
        if self._nx:
            entities = []
            for node_id, data in self.graph.nodes(data=True):
                if entity_type is None or data.get("type") == entity_type:
                    entities.append(Entity(
                        id=node_id,
                        type=data.get("type", "unknown"),
                        attributes=data.get("attributes", {}),
                    ))
            return entities
        else:
            if entity_type is None:
                return list(self._entities.values())
            return [e for e in self._entities.values() if e.type == entity_type]
    
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
                    r for r in self._relationships
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
        data = {
            "entities": [],
            "relationships": [],
        }
        
        for entity in self.get_all_entities():
            data["entities"].append(entity.to_dict())
        
        if self._nx:
            for source, target, edge_data in self.graph.edges(data=True):
                data["relationships"].append({
                    "source": source,
                    "target": target,
                    "relation": edge_data.get("relation", "related_to"),
                    "attributes": edge_data.get("attributes", {}),
                    "created_at": edge_data.get("created_at", datetime.now(timezone.utc).isoformat()),
                })
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


class SQLiteGraph(GraphBackend):
    """SQLite-backed graph storage for persistence and large graphs."""
    
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._local = threading.local()
        self._init_db()
    
    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                attributes TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source TEXT
            );
            
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_id TEXT NOT NULL,
                attributes TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                source TEXT,
                UNIQUE(source_id, relation, target_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
            CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
            CREATE INDEX IF NOT EXISTS idx_rel_relation ON relationships(relation);
        """)
        conn.commit()
    
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Entity:
        """Add or update an entity."""
        now = datetime.now(timezone.utc).isoformat()
        attrs_json = json.dumps(attributes or {})
        
        conn = self._conn
        
        # Check if exists
        row = conn.execute(
            "SELECT attributes, created_at FROM entities WHERE id = ?",
            (entity_id,)
        ).fetchone()
        
        if row:
            # Merge attributes
            existing_attrs = json.loads(row["attributes"])
            merged_attrs = {**existing_attrs, **(attributes or {})}
            conn.execute(
                "UPDATE entities SET type = ?, attributes = ?, updated_at = ?, source = COALESCE(?, source) WHERE id = ?",
                (entity_type, json.dumps(merged_attrs), now, source, entity_id)
            )
            created_at = row["created_at"]
        else:
            conn.execute(
                "INSERT INTO entities (id, type, attributes, created_at, updated_at, source) VALUES (?, ?, ?, ?, ?, ?)",
                (entity_id, entity_type, attrs_json, now, now, source)
            )
            created_at = now
        
        conn.commit()
        
        return Entity(
            id=entity_id,
            type=entity_type,
            attributes=attributes or {},
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(now),
            source=source,
        )
    
    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id = ?",
            (entity_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return Entity(
            id=row["id"],
            type=row["type"],
            attributes=json.loads(row["attributes"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            source=row["source"],
        )
    
    def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Relationship:
        """Add a relationship between entities."""
        now = datetime.now(timezone.utc)
        attrs_json = json.dumps(attributes or {})
        
        conn = self._conn
        try:
            conn.execute(
                "INSERT OR REPLACE INTO relationships (source_id, relation, target_id, attributes, created_at, source) VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, relation, target_id, attrs_json, now.isoformat(), source)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # Relationship already exists
        
        return Relationship(
            source_id=source_id,
            relation=relation,
            target_id=target_id,
            attributes=attributes or {},
            created_at=now,
            source=source,
        )
    
    def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        direction: str = "outgoing",
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        results = []
        conn = self._conn
        
        if direction in ("outgoing", "both"):
            if relation:
                rows = conn.execute(
                    "SELECT * FROM relationships WHERE source_id = ? AND relation = ?",
                    (entity_id, relation)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM relationships WHERE source_id = ?",
                    (entity_id,)
                ).fetchall()
            
            for row in rows:
                results.append(Relationship(
                    source_id=row["source_id"],
                    relation=row["relation"],
                    target_id=row["target_id"],
                    attributes=json.loads(row["attributes"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                ))
        
        if direction in ("incoming", "both"):
            if relation:
                rows = conn.execute(
                    "SELECT * FROM relationships WHERE target_id = ? AND relation = ?",
                    (entity_id, relation)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM relationships WHERE target_id = ?",
                    (entity_id,)
                ).fetchall()
            
            for row in rows:
                results.append(Relationship(
                    source_id=row["source_id"],
                    relation=row["relation"],
                    target_id=row["target_id"],
                    attributes=json.loads(row["attributes"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                ))
        
        return results
    
    def query(self, pattern: str) -> list[dict[str, Any]]:
        """Query the graph with a pattern."""
        results = []
        pattern = pattern.strip()
        
        # Pattern: entity_id -relation-> ?
        if " -" in pattern and "-> ?" in pattern:
            parts = pattern.split(" -")
            source = parts[0].strip()
            relation = parts[1].replace("-> ?", "").strip()
            relation = None if relation == "?" else relation
            
            for rel in self.get_relationships(source, relation, "outgoing"):
                target = self.get_entity(rel.target_id)
                results.append({
                    "source": source,
                    "relation": rel.relation,
                    "target": rel.target_id,
                    "target_type": target.type if target else "unknown",
                    "target_attributes": target.attributes if target else {},
                })
        
        # Pattern: ? -relation-> entity_id
        elif "? -" in pattern and "-> " in pattern:
            parts = pattern.split("-> ")
            target = parts[1].strip()
            relation = parts[0].replace("? -", "").strip()
            relation = None if relation == "?" else relation
            
            for rel in self.get_relationships(target, relation, "incoming"):
                source_entity = self.get_entity(rel.source_id)
                results.append({
                    "source": rel.source_id,
                    "source_type": source_entity.type if source_entity else "unknown",
                    "relation": rel.relation,
                    "target": target,
                })
        
        # Pattern: just entity_id - return entity + relationships
        else:
            entity = self.get_entity(pattern)
            if entity:
                results.append({
                    "entity": entity.to_dict(),
                    "outgoing": [r.to_dict() for r in self.get_relationships(pattern, direction="outgoing")],
                    "incoming": [r.to_dict() for r in self.get_relationships(pattern, direction="incoming")],
                })
        
        return results
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> list[list[str]] | None:
        """Find paths between two entities using BFS."""
        visited = set()
        queue = [[source_id]]
        
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
    
    def get_all_entities(self, entity_type: str | None = None, limit: int | None = None, offset: int = 0) -> list[Entity]:
        """Get all entities, optionally filtered by type with pagination."""
        conn = self._conn
        
        if entity_type:
            query = "SELECT * FROM entities WHERE type = ?"
            params: tuple = (entity_type,)
        else:
            query = "SELECT * FROM entities"
            params = ()
        
        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        rows = conn.execute(query, params).fetchall()
        
        return [
            Entity(
                id=row["id"],
                type=row["type"],
                attributes=json.loads(row["attributes"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                source=row["source"],
            )
            for row in rows
        ]
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and its relationships."""
        conn = self._conn
        
        cursor = conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        if cursor.rowcount == 0:
            return False
        
        conn.execute("DELETE FROM relationships WHERE source_id = ? OR target_id = ?", (entity_id, entity_id))
        conn.commit()
        return True
    
    def stats(self) -> dict[str, int]:
        """Get graph statistics."""
        conn = self._conn
        entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relationships = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        return {
            "entities": entities,
            "relationships": relationships,
        }
    
    def clear(self) -> None:
        """Clear all entities and relationships."""
        conn = self._conn
        conn.execute("DELETE FROM entities")
        conn.execute("DELETE FROM relationships")
        conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


class JSONFileGraph(GraphBackend):
    """JSON file-backed graph with auto-save."""
    
    def __init__(self, path: str | Path, auto_save: bool = True):
        self._path = Path(path)
        self._auto_save = auto_save
        self._memory = InMemoryGraph()
        self.load()
    
    def _maybe_save(self) -> None:
        """Auto-save if enabled."""
        if self._auto_save:
            self.save()
    
    def add_entity(self, entity_id: str, entity_type: str, attributes: dict[str, Any] | None = None, source: str | None = None) -> Entity:
        result = self._memory.add_entity(entity_id, entity_type, attributes, source)
        self._maybe_save()
        return result
    
    def get_entity(self, entity_id: str) -> Entity | None:
        return self._memory.get_entity(entity_id)
    
    def add_relationship(self, source_id: str, relation: str, target_id: str, attributes: dict[str, Any] | None = None, source: str | None = None) -> Relationship:
        result = self._memory.add_relationship(source_id, relation, target_id, attributes, source)
        self._maybe_save()
        return result
    
    def get_relationships(self, entity_id: str, relation: str | None = None, direction: str = "outgoing") -> list[Relationship]:
        return self._memory.get_relationships(entity_id, relation, direction)
    
    def query(self, pattern: str) -> list[dict[str, Any]]:
        return self._memory.query(pattern)
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> list[list[str]] | None:
        return self._memory.find_path(source_id, target_id, max_depth)
    
    def get_all_entities(self, entity_type: str | None = None) -> list[Entity]:
        return self._memory.get_all_entities(entity_type)
    
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


class KnowledgeGraph(BaseCapability):
    """
    Knowledge graph capability for persistent entity/relationship memory.
    
    Supports multiple backends:
    - "memory": In-memory graph (default, uses networkx if available)
    - "sqlite": SQLite database for persistence and large graphs
    - "json": JSON file for simple persistence with auto-save
    
    Provides tools for:
    - remember: Store entities and facts
    - recall: Retrieve entity information
    - connect: Create relationships between entities
    - query: Query the graph with patterns
    - forget: Remove entities
    
    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.capabilities import KnowledgeGraph
        
        # In-memory (default)
        kg = KnowledgeGraph()
        
        # SQLite for persistence and large graphs
        kg = KnowledgeGraph(backend="sqlite", path="knowledge.db")
        
        # JSON file with auto-save
        kg = KnowledgeGraph(backend="json", path="knowledge.json")
        
        agent = Agent(
            name="Assistant",
            model=model,
            capabilities=[kg],
        )
        
        # Agent can now use knowledge tools
        await agent.run("Remember that Alice is a software engineer at Acme Corp")
        await agent.run("What do you know about Alice?")
        
        # Direct access to graph
        kg.graph.add_entity("Bob", "Person", {"role": "Manager"})
        kg.graph.add_relationship("Alice", "reports_to", "Bob")
        
        # Persistence (for memory backend, explicit save needed)
        kg.save("backup.json")  # Save to file
        kg.load("backup.json")  # Load from file
        ```
    """
    
    def __init__(
        self,
        backend: str = "memory",
        path: str | Path | None = None,
        name: str | None = None,
        auto_save: bool = True,
    ):
        """
        Initialize KnowledgeGraph capability.
        
        Args:
            backend: Storage backend:
                - "memory": In-memory graph (uses networkx if available)
                - "sqlite": SQLite database for persistence
                - "json": JSON file with auto-save
            path: Path to storage file (required for sqlite/json backends)
            name: Optional custom name for this capability instance
            auto_save: For json backend, save after each modification (default: True)
        """
        self._backend = backend
        self._path = Path(path) if path else None
        self._name = name
        self._tools_cache: dict[str, BaseTool] | None = None
        
        if backend == "memory":
            self.graph: GraphBackend = InMemoryGraph()
        elif backend == "sqlite":
            if not path:
                raise ValueError("Path required for sqlite backend")
            self.graph = SQLiteGraph(path)
        elif backend == "json":
            if not path:
                raise ValueError("Path required for json backend")
            self.graph = JSONFileGraph(path, auto_save=auto_save)
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: 'memory', 'sqlite', 'json'")
    
    @property
    def name(self) -> str:
        return self._name or "knowledge_graph"
    
    @property
    def description(self) -> str:
        return "Persistent knowledge graph for storing and querying entities and relationships"
    
    @property
    def tools(self) -> list[BaseTool]:
        # Cache tools to avoid recreating them
        if self._tools_cache is None:
            self._tools_cache = {
                "remember": self._remember_tool(),
                "recall": self._recall_tool(),
                "connect": self._connect_tool(),
                "query": self._query_tool(),
                "forget": self._forget_tool(),
                "list": self._list_entities_tool(),
            }
        return list(self._tools_cache.values())
    
    # === Convenience methods for direct use ===
    
    def remember(self, entity: str, entity_type: str, facts: dict[str, Any] | None = None) -> str:
        """Remember an entity with attributes."""
        import json
        facts_str = json.dumps(facts) if facts else "{}"
        return self._tools_cache["remember"].invoke({
            "entity": entity,
            "entity_type": entity_type,
            "facts": facts_str,
        })
    
    def recall(self, entity: str) -> str:
        """Recall information about an entity."""
        if self._tools_cache is None:
            _ = self.tools  # Initialize cache
        return self._tools_cache["recall"].invoke({"entity": entity})
    
    def connect(self, source: str, relation: str, target: str) -> str:
        """Create a relationship between entities."""
        if self._tools_cache is None:
            _ = self.tools
        return self._tools_cache["connect"].invoke({
            "source": source,
            "relation": relation,
            "target": target,
        })
    
    def query(self, pattern: str) -> str:
        """Query the knowledge graph with a pattern."""
        if self._tools_cache is None:
            _ = self.tools
        return self._tools_cache["query"].invoke({"pattern": pattern})
    
    def forget(self, entity: str) -> str:
        """Remove an entity from the graph."""
        if self._tools_cache is None:
            _ = self.tools
        return self._tools_cache["forget"].invoke({"entity": entity})
    
    def list_entities(self, entity_type: str = "") -> str:
        """List all known entities."""
        if self._tools_cache is None:
            _ = self.tools
        return self._tools_cache["list"].invoke({"entity_type": entity_type})
    
    def _remember_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def remember(
            entity: str,
            entity_type: str,
            facts: str,
        ) -> str:
            """
            Remember an entity and its attributes.
            
            Args:
                entity: Name/ID of the entity (e.g., "Alice", "Acme Corp")
                entity_type: Type of entity (e.g., "Person", "Company", "Project")
                facts: JSON string of facts/attributes (e.g., '{"role": "engineer", "team": "backend"}')
            
            Returns:
                Confirmation message
            """
            try:
                attributes = json.loads(facts) if facts else {}
            except json.JSONDecodeError:
                # Try to parse as simple key-value
                attributes = {"info": facts}
            
            graph.add_entity(entity, entity_type, attributes)
            return f"Remembered: {entity} ({entity_type}) with {len(attributes)} attributes"
        
        return remember
    
    def _recall_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def recall(entity: str) -> str:
            """
            Recall information about an entity.
            
            Args:
                entity: Name/ID of the entity to recall
            
            Returns:
                Entity information and relationships, or "not found"
            """
            e = graph.get_entity(entity)
            if not e:
                return f"No information found about '{entity}'"
            
            # Get relationships
            outgoing = graph.get_relationships(entity, direction="outgoing")
            incoming = graph.get_relationships(entity, direction="incoming")
            
            info = [f"**{entity}** ({e.type})"]
            
            if e.attributes:
                info.append("Attributes:")
                for k, v in e.attributes.items():
                    info.append(f"  - {k}: {v}")
            
            if outgoing:
                info.append("Relationships:")
                for rel in outgoing:
                    info.append(f"  - {rel.relation} → {rel.target_id}")
            
            if incoming:
                info.append("Referenced by:")
                for rel in incoming:
                    info.append(f"  - {rel.source_id} {rel.relation} → this")
            
            return "\n".join(info)
        
        return recall
    
    def _connect_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def connect(
            source: str,
            relation: str,
            target: str,
        ) -> str:
            """
            Create a relationship between two entities.
            
            Args:
                source: Source entity (e.g., "Alice")
                relation: Relationship type (e.g., "works_at", "manages", "knows")
                target: Target entity (e.g., "Acme Corp")
            
            Returns:
                Confirmation message
            """
            # Auto-create entities if they don't exist
            if not graph.get_entity(source):
                graph.add_entity(source, "Unknown")
            if not graph.get_entity(target):
                graph.add_entity(target, "Unknown")
            
            graph.add_relationship(source, relation, target)
            return f"Connected: {source} --[{relation}]--> {target}"
        
        return connect
    
    def _query_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def query_knowledge(pattern: str) -> str:
            """
            Query the knowledge graph for relationships.
            
            The pattern uses arrow syntax: SOURCE -RELATION-> TARGET
            Use '?' as a wildcard to find unknowns.
            
            Args:
                pattern: Query pattern. IMPORTANT: Use '?' for what you want to find:
                    - "? -works_on-> Project X" - Who works on Project X?
                    - "Alice -works_at-> ?" - Where does Alice work?
                    - "? -reports_to-> Bob" - Who reports to Bob?
                    - "Alice -?-> ?" - All relationships FROM Alice
                    - "Alice" - Get all info about Alice (simple lookup)
            
            Returns:
                Query results as JSON
            """
            results = graph.query(pattern)
            
            if not results:
                return f"No results for pattern: {pattern}"
            
            return json.dumps(results, indent=2, default=str)
        
        return query_knowledge
    
    def _forget_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def forget(entity: str) -> str:
            """
            Remove an entity and all its relationships from memory.
            
            Args:
                entity: Name/ID of the entity to forget
            
            Returns:
                Confirmation message
            """
            if graph.remove_entity(entity):
                return f"Forgot: {entity} (and all its relationships)"
            else:
                return f"Entity '{entity}' not found in memory"
        
        return forget
    
    def _list_entities_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def list_knowledge(entity_type: str = "") -> str:
            """
            List all known entities.
            
            Args:
                entity_type: Optional filter by type (e.g., "Person", "Company")
            
            Returns:
                List of entities
            """
            entities = graph.get_all_entities(entity_type if entity_type else None)
            
            if not entities:
                filter_msg = f" of type '{entity_type}'" if entity_type else ""
                return f"No entities{filter_msg} in knowledge graph"
            
            lines = [f"Knowledge graph ({len(entities)} entities):"]
            for e in entities:
                attrs = ", ".join(f"{k}={v}" for k, v in list(e.attributes.items())[:3])
                lines.append(f"  - {e.id} ({e.type}){': ' + attrs if attrs else ''}")
            
            return "\n".join(lines)
        
        return list_knowledge

    # =========================================================================
    # Convenience Methods - Direct access without tools
    # =========================================================================

    def add_entity(
        self,
        entity_id: str,
        entity_type: str = "Entity",
        attributes: dict[str, Any] | None = None,
    ) -> Entity:
        """
        Add an entity to the knowledge graph.

        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type/category of the entity
            attributes: Optional attributes dictionary

        Returns:
            The created Entity object
        """
        return self.graph.add_entity(entity_id, entity_type, attributes)

    def add_relationship(
        self,
        source: str,
        relation: str,
        target: str,
        attributes: dict[str, Any] | None = None,
    ) -> Relationship:
        """
        Add a relationship between two entities.

        Args:
            source: Source entity ID
            relation: Relationship type
            target: Target entity ID
            attributes: Optional relationship attributes

        Returns:
            The created Relationship object
        """
        return self.graph.add_relationship(source, relation, target, attributes)

    def get_entity(self, entity_id: str) -> Entity | None:
        """
        Get an entity by ID.

        Args:
            entity_id: The entity identifier

        Returns:
            Entity object or None if not found
        """
        return self.graph.get_entity(entity_id)

    def get_entities(self, entity_type: str | None = None) -> list[Entity]:
        """
        Get all entities, optionally filtered by type.

        Args:
            entity_type: Optional type filter

        Returns:
            List of Entity objects
        """
        return self.graph.get_all_entities(entity_type)

    def get_relationships(
        self,
        entity_id: str,
        relation_type: str | None = None,
        direction: str = "both",
    ) -> list[Relationship]:
        """
        Get relationships for an entity.

        Args:
            entity_id: The entity to get relationships for
            relation_type: Optional filter by relationship type
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of Relationship objects
        """
        return self.graph.get_relationships(entity_id, relation_type, direction)

    def query_graph(self, pattern: str) -> list[dict[str, Any]]:
        """
        Query the knowledge graph with a pattern.

        Args:
            pattern: Query pattern (e.g., "Alice -works_at-> ?")

        Returns:
            List of matching results
        """
        return self.graph.query(pattern)

    def find_path(
        self,
        source: str,
        target: str,
        max_depth: int = 5,
    ) -> list[str] | None:
        """
        Find a path between two entities.

        Args:
            source: Starting entity ID
            target: Target entity ID
            max_depth: Maximum path length to search

        Returns:
            List of entity IDs forming the path, or None if no path exists
        """
        return self.graph.find_path(source, target, max_depth)

    def remove_entity(self, entity_id: str) -> bool:
        """
        Remove an entity and all its relationships.

        Args:
            entity_id: Entity to remove

        Returns:
            True if removed, False if not found
        """
        return self.graph.remove_entity(entity_id)

    def get_tool(self, name: str) -> BaseTool | None:
        """
        Get a specific tool by name.

        Args:
            name: Tool name (remember, recall, connect, query, forget, list)

        Returns:
            The tool or None if not found
        """
        if self._tools_cache is None:
            _ = self.tools  # Initialize cache
        return self._tools_cache.get(name)

    def stats(self) -> dict[str, int]:
        """
        Get graph statistics.

        Returns:
            Dictionary with entity_count, relationship_count, type_counts
        """
        return self.graph.stats()

    def clear(self) -> None:
        """Clear all entities and relationships from the graph."""
        self.graph.clear()

    def save(self, path: str | Path | None = None) -> None:
        """
        Save graph to file.
        
        For memory backend: saves to specified path (required).
        For sqlite backend: no-op (data is already persisted).
        For json backend: saves to original path or specified path.
        
        Args:
            path: File path (required for memory backend)
        """
        if self._backend == "sqlite":
            return  # SQLite auto-persists
        
        save_path = path or self._path
        if not save_path and self._backend == "memory":
            raise ValueError("Path required for save() on memory backend")
        
        self.graph.save(save_path)
    
    def load(self, path: str | Path | None = None) -> None:
        """
        Load graph from file.
        
        For memory backend: loads from specified path (required).
        For sqlite backend: no-op (data is already loaded).
        For json backend: reloads from file.
        
        Args:
            path: File path (required for memory backend)
        """
        if self._backend == "sqlite":
            return  # SQLite auto-loads
        
        load_path = path or self._path
        if not load_path and self._backend == "memory":
            raise ValueError("Path required for load() on memory backend")
        
        self.graph.load(load_path)
    
    def close(self) -> None:
        """Close any open connections (for sqlite backend)."""
        self.graph.close()
    
    def __enter__(self) -> "KnowledgeGraph":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes connections."""
        self.close()

    def to_dict(self) -> dict[str, Any]:
        """Convert capability info to dictionary."""
        base = super().to_dict()
        base["backend"] = self._backend
        base["path"] = str(self._path) if self._path else None
        base["stats"] = self.graph.stats()
        return base
