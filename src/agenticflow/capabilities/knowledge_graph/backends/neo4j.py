"""Neo4j-backed graph storage for production knowledge graphs.

Neo4j is the industry-standard graph database, offering:
- Native graph storage and traversal
- Cypher query language for complex queries
- ACID transactions
- Horizontal scaling with clusters
- Built-in graph algorithms

Requirements:
    uv add neo4j

Example:
    ```python
    from agenticflow.capabilities import KnowledgeGraph
    from agenticflow.capabilities.knowledge_graph.backends import Neo4jGraph

    # Connect to Neo4j
    backend = Neo4jGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
    )

    # Use with KnowledgeGraph capability
    kg = KnowledgeGraph(backend=backend)

    # Or use the convenience class method
    kg = KnowledgeGraph.neo4j(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
    )
    ```
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agenticflow.capabilities.knowledge_graph.backends.base import GraphBackend
from agenticflow.capabilities.knowledge_graph.models import Entity, Relationship


class Neo4jGraph(GraphBackend):
    """
    Neo4j-backed graph storage for production knowledge graphs.

    Neo4j provides native graph storage with the Cypher query language,
    making it ideal for complex relationship queries, graph algorithms,
    and large-scale knowledge graphs.

    Args:
        uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
        user: Neo4j username
        password: Neo4j password
        database: Database name (default: "neo4j")

    Example:
        ```python
        from agenticflow.capabilities.knowledge_graph.backends import Neo4jGraph

        # Connect to local Neo4j
        graph = Neo4jGraph(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="your-password",
        )

        # Add entities
        graph.add_entity("Alice", "Person", {"role": "Engineer"})
        graph.add_entity("Acme", "Company", {"industry": "Tech"})

        # Add relationships
        graph.add_relationship("Alice", "WORKS_AT", "Acme")

        # Query using Cypher
        results = graph.cypher(
            "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN p.id, c.id"
        )

        graph.close()
        ```
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver = None
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "neo4j package required for Neo4j backend. "
                "Install with: uv add neo4j"
            )

        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        # Verify connection
        self._driver.verify_connectivity()

    def _session(self):
        """Get a new session."""
        return self._driver.session(database=self._database)

    def cypher(self, query: str, **params) -> list[dict[str, Any]]:
        """
        Execute a raw Cypher query.

        This gives you full access to Neo4j's Cypher query language
        for complex graph operations.

        Args:
            query: Cypher query string
            **params: Query parameters

        Returns:
            List of result records as dictionaries

        Example:
            ```python
            # Find all people who work at a company
            results = graph.cypher(
                "MATCH (p:Person)-[:WORKS_AT]->(c:Company {id: $company}) "
                "RETURN p.id as name, p.role as role",
                company="Acme"
            )
            ```
        """
        with self._session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Entity:
        """Add or update an entity (node) in Neo4j."""
        now = datetime.now(UTC)
        attrs = attributes or {}

        # Neo4j node labels can't have spaces, normalize type
        label = entity_type.replace(" ", "_")

        with self._session() as session:
            # MERGE to create or update
            query = f"""
                MERGE (n:{label} {{id: $id}})
                ON CREATE SET
                    n.created_at = $created_at,
                    n.updated_at = $updated_at,
                    n.source = $source,
                    n += $attrs
                ON MATCH SET
                    n.updated_at = $updated_at,
                    n.source = COALESCE($source, n.source),
                    n += $attrs
                RETURN n
            """
            session.run(
                query,
                id=entity_id,
                created_at=now.isoformat(),
                updated_at=now.isoformat(),
                source=source,
                attrs=attrs,
            )

        return Entity(
            id=entity_id,
            type=entity_type,
            attributes=attrs,
            created_at=now,
            updated_at=now,
            source=source,
        )

    def add_entities_batch(
        self,
        entities: list[tuple[str, str, dict[str, Any] | None]],
    ) -> int:
        """Bulk insert entities using UNWIND for performance."""
        now = datetime.now(UTC).isoformat()

        # Group by type for efficient batch inserts
        by_type: dict[str, list[dict]] = {}
        for eid, etype, attrs in entities:
            label = etype.replace(" ", "_")
            if label not in by_type:
                by_type[label] = []
            by_type[label].append({
                "id": eid,
                "attrs": attrs or {},
            })

        count = 0
        with self._session() as session:
            for label, items in by_type.items():
                query = f"""
                    UNWIND $items AS item
                    MERGE (n:{label} {{id: item.id}})
                    ON CREATE SET
                        n.created_at = $now,
                        n.updated_at = $now,
                        n += item.attrs
                    ON MATCH SET
                        n.updated_at = $now,
                        n += item.attrs
                """
                session.run(query, items=items, now=now)
                count += len(items)

        return count

    def add_relationships_batch(
        self,
        relationships: list[tuple[str, str, str]],
    ) -> int:
        """Bulk insert relationships using UNWIND."""
        now = datetime.now(UTC).isoformat()

        # Group by relation type
        by_type: dict[str, list[dict]] = {}
        for src, rel, tgt in relationships:
            rel_type = rel.upper().replace(" ", "_")
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append({"src": src, "tgt": tgt})

        count = 0
        with self._session() as session:
            for rel_type, items in by_type.items():
                query = f"""
                    UNWIND $items AS item
                    MATCH (a {{id: item.src}})
                    MATCH (b {{id: item.tgt}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    ON CREATE SET r.created_at = $now
                """
                session.run(query, items=items, now=now)
                count += len(items)

        return count

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        with self._session() as session:
            result = session.run(
                "MATCH (n {id: $id}) RETURN n, labels(n) as labels",
                id=entity_id,
            )
            record = result.single()
            if not record:
                return None

            node = record["n"]
            labels = record["labels"]
            props = dict(node)

            # Extract metadata
            entity_id = props.pop("id", entity_id)
            created_at = props.pop("created_at", None)
            updated_at = props.pop("updated_at", None)
            source = props.pop("source", None)

            return Entity(
                id=entity_id,
                type=labels[0] if labels else "Entity",
                attributes=props,
                created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
                updated_at=datetime.fromisoformat(updated_at) if updated_at else datetime.now(UTC),
                source=source,
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
        now = datetime.now(UTC)
        rel_type = relation.upper().replace(" ", "_")
        attrs = attributes or {}

        with self._session() as session:
            query = f"""
                MATCH (a {{id: $source_id}})
                MATCH (b {{id: $target_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                ON CREATE SET
                    r.created_at = $created_at,
                    r.source = $source
                SET r += $attrs
                RETURN r
            """
            session.run(
                query,
                source_id=source_id,
                target_id=target_id,
                created_at=now.isoformat(),
                source=source,
                attrs=attrs,
            )

        return Relationship(
            source_id=source_id,
            relation=relation,
            target_id=target_id,
            attributes=attrs,
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

        with self._session() as session:
            if direction in ("outgoing", "both"):
                if relation:
                    rel_type = relation.upper().replace(" ", "_")
                    query = f"""
                        MATCH (a {{id: $id}})-[r:{rel_type}]->(b)
                        RETURN type(r) as rel, b.id as target, properties(r) as props
                    """
                else:
                    query = """
                        MATCH (a {id: $id})-[r]->(b)
                        RETURN type(r) as rel, b.id as target, properties(r) as props
                    """
                for record in session.run(query, id=entity_id):
                    props = dict(record["props"])
                    created_at = props.pop("created_at", None)
                    source = props.pop("source", None)
                    results.append(Relationship(
                        source_id=entity_id,
                        relation=record["rel"].lower().replace("_", " "),
                        target_id=record["target"],
                        attributes=props,
                        created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
                        source=source,
                    ))

            if direction in ("incoming", "both"):
                if relation:
                    rel_type = relation.upper().replace(" ", "_")
                    query = f"""
                        MATCH (a)-[r:{rel_type}]->(b {{id: $id}})
                        RETURN a.id as source, type(r) as rel, properties(r) as props
                    """
                else:
                    query = """
                        MATCH (a)-[r]->(b {id: $id})
                        RETURN a.id as source, type(r) as rel, properties(r) as props
                    """
                for record in session.run(query, id=entity_id):
                    props = dict(record["props"])
                    created_at = props.pop("created_at", None)
                    source = props.pop("source", None)
                    results.append(Relationship(
                        source_id=record["source"],
                        relation=record["rel"].lower().replace("_", " "),
                        target_id=entity_id,
                        attributes=props,
                        created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
                        source=source,
                    ))

        return results

    def query(self, pattern: str) -> list[dict[str, Any]]:
        """
        Query the graph with a pattern.

        Supports simple patterns like:
        - "Alice" - Get entity info
        - "Alice -works_at-> ?" - Find targets
        - "? -works_at-> Acme" - Find sources

        For complex queries, use the cypher() method directly.
        """
        results: list[dict[str, Any]] = []
        pattern = pattern.strip()

        # Pattern: entity_id -relation-> ?
        if " -" in pattern and "-> ?" in pattern:
            parts = pattern.split(" -")
            source = parts[0].strip()
            relation = parts[1].replace("-> ?", "").strip()

            for rel in self.get_relationships(source, relation if relation != "?" else None, "outgoing"):
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

            for rel in self.get_relationships(target, relation if relation != "?" else None, "incoming"):
                source_entity = self.get_entity(rel.source_id)
                results.append({
                    "source": rel.source_id,
                    "source_type": source_entity.type if source_entity else "unknown",
                    "relation": rel.relation,
                    "target": target,
                })

        # Simple entity lookup
        else:
            entity = self.get_entity(pattern)
            if entity:
                results.append({
                    "entity": entity.to_dict(),
                    "outgoing": [r.to_dict() for r in self.get_relationships(pattern, direction="outgoing")],
                    "incoming": [r.to_dict() for r in self.get_relationships(pattern, direction="incoming")],
                })

        return results

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[list[str]] | None:
        """Find shortest path between two entities using Neo4j's built-in algorithm."""
        with self._session() as session:
            result = session.run(
                f"""
                MATCH path = shortestPath(
                    (a {{id: $source}})-[*1..{max_depth}]-(b {{id: $target}})
                )
                RETURN [n in nodes(path) | n.id] as path
                """,
                source=source_id,
                target=target_id,
            )
            record = result.single()
            if record:
                return [record["path"]]
            return None

    def get_all_entities(
        self,
        entity_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Entity]:
        """Get all entities, optionally filtered by type."""
        with self._session() as session:
            if entity_type:
                label = entity_type.replace(" ", "_")
                query = f"MATCH (n:{label}) RETURN n, labels(n) as labels"
            else:
                query = "MATCH (n) RETURN n, labels(n) as labels"

            if limit is not None:
                query += f" SKIP {offset} LIMIT {limit}"

            results = []
            for record in session.run(query):
                node = record["n"]
                labels = record["labels"]
                props = dict(node)

                entity_id = props.pop("id", "unknown")
                created_at = props.pop("created_at", None)
                updated_at = props.pop("updated_at", None)
                source = props.pop("source", None)

                results.append(Entity(
                    id=entity_id,
                    type=labels[0] if labels else "Entity",
                    attributes=props,
                    created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(UTC),
                    updated_at=datetime.fromisoformat(updated_at) if updated_at else datetime.now(UTC),
                    source=source,
                ))

            return results

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its relationships."""
        with self._session() as session:
            result = session.run(
                "MATCH (n {id: $id}) DETACH DELETE n RETURN count(n) as deleted",
                id=entity_id,
            )
            record = result.single()
            return record["deleted"] > 0 if record else False

    def stats(self) -> dict[str, int]:
        """Get graph statistics."""
        with self._session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            return {
                "entities": node_count,
                "relationships": rel_count,
            }

    def clear(self) -> None:
        """Clear all nodes and relationships."""
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
