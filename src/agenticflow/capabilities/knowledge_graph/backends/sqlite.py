"""SQLite-backed graph storage for persistence and large graphs."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agenticflow.capabilities.knowledge_graph.models import Entity, Relationship
from agenticflow.capabilities.knowledge_graph.backends.base import GraphBackend


class SQLiteGraph(GraphBackend):
    """SQLite-backed graph storage for persistence and large graphs."""

    def __init__(self, path: str | Path) -> None:
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
            "SELECT attributes, created_at FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()

        if row:
            # Merge attributes
            existing_attrs = json.loads(row["attributes"])
            merged_attrs = {**existing_attrs, **(attributes or {})}
            conn.execute(
                "UPDATE entities SET type = ?, attributes = ?, updated_at = ?, source = COALESCE(?, source) WHERE id = ?",
                (entity_type, json.dumps(merged_attrs), now, source, entity_id),
            )
            created_at = row["created_at"]
        else:
            conn.execute(
                "INSERT INTO entities (id, type, attributes, created_at, updated_at, source) VALUES (?, ?, ?, ?, ?, ?)",
                (entity_id, entity_type, attrs_json, now, now, source),
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

    def add_entities_batch(
        self,
        entities: list[tuple[str, str, dict[str, Any] | None]],
    ) -> int:
        """
        Bulk insert entities for high performance.

        Args:
            entities: List of (entity_id, entity_type, attributes) tuples

        Returns:
            Number of entities inserted
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn

        # Use executemany for batch insert
        data = [
            (eid, etype, json.dumps(attrs or {}), now, now, None)
            for eid, etype, attrs in entities
        ]

        conn.executemany(
            "INSERT OR REPLACE INTO entities (id, type, attributes, created_at, updated_at, source) VALUES (?, ?, ?, ?, ?, ?)",
            data,
        )
        conn.commit()

        return len(entities)

    def add_relationships_batch(
        self,
        relationships: list[tuple[str, str, str]],
    ) -> int:
        """
        Bulk insert relationships for high performance.

        Args:
            relationships: List of (source_id, relation, target_id) tuples

        Returns:
            Number of relationships inserted
        """
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn

        data = [(src, rel, tgt, "{}", now, None) for src, rel, tgt in relationships]

        conn.executemany(
            "INSERT OR IGNORE INTO relationships (source_id, relation, target_id, attributes, created_at, source) VALUES (?, ?, ?, ?, ?, ?)",
            data,
        )
        conn.commit()

        return len(relationships)

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
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
                (source_id, relation, target_id, attrs_json, now.isoformat(), source),
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
        results: list[Relationship] = []
        conn = self._conn

        if direction in ("outgoing", "both"):
            if relation:
                rows = conn.execute(
                    "SELECT * FROM relationships WHERE source_id = ? AND relation = ?",
                    (entity_id, relation),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM relationships WHERE source_id = ?", (entity_id,)
                ).fetchall()

            for row in rows:
                results.append(
                    Relationship(
                        source_id=row["source_id"],
                        relation=row["relation"],
                        target_id=row["target_id"],
                        attributes=json.loads(row["attributes"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )

        if direction in ("incoming", "both"):
            if relation:
                rows = conn.execute(
                    "SELECT * FROM relationships WHERE target_id = ? AND relation = ?",
                    (entity_id, relation),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM relationships WHERE target_id = ?", (entity_id,)
                ).fetchall()

            for row in rows:
                results.append(
                    Relationship(
                        source_id=row["source_id"],
                        relation=row["relation"],
                        target_id=row["target_id"],
                        attributes=json.loads(row["attributes"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )

        return results

    def query(self, pattern: str) -> list[dict[str, Any]]:
        """Query the graph with a pattern."""
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
        """Find paths between two entities using BFS."""
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

        conn.execute(
            "DELETE FROM relationships WHERE source_id = ? OR target_id = ?",
            (entity_id, entity_id),
        )
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
