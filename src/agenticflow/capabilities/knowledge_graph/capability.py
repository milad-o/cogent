"""KnowledgeGraph capability for persistent entity/relationship memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agenticflow.capabilities.base import BaseCapability
from agenticflow.capabilities.knowledge_graph.backends import (
    GraphBackend,
    InMemoryGraph,
    JSONFileGraph,
    SQLiteGraph,
)
from agenticflow.capabilities.knowledge_graph.models import Entity, Relationship
from agenticflow.tools.base import BaseTool, tool


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
    ) -> None:
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
        elif isinstance(backend, GraphBackend):
            # Allow passing a custom backend instance directly
            self.graph = backend
            self._backend = "custom"
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Supported: 'memory', 'sqlite', 'json', or GraphBackend instance"
            )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        name: str | None = None,
    ) -> KnowledgeGraph:
        """
        Load a KnowledgeGraph from an existing file.

        Automatically detects the backend based on file extension:
        - .db, .sqlite, .sqlite3 → SQLite backend
        - .json → JSON backend (with auto-save)

        This is the simplest way to load an existing knowledge graph.

        Args:
            path: Path to existing knowledge store file
            name: Optional custom name for this capability instance

        Returns:
            KnowledgeGraph instance loaded from the file

        Example:
            ```python
            # Load existing SQLite database
            kg = KnowledgeGraph.from_file("knowledge.db")

            # Load existing JSON file
            kg = KnowledgeGraph.from_file("knowledge.json")

            # Use with agent
            agent = Agent(
                name="Assistant",
                capabilities=[KnowledgeGraph.from_file("company.db")],
            )
            ```
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in (".db", ".sqlite", ".sqlite3"):
            return cls(backend="sqlite", path=path, name=name)
        elif suffix == ".json":
            return cls(backend="json", path=path, name=name, auto_save=True)
        else:
            raise ValueError(
                f"Unknown file extension: {suffix}. "
                "Use .db/.sqlite for SQLite or .json for JSON. "
                "Or use KnowledgeGraph(backend=..., path=...) directly."
            )

    @classmethod
    def neo4j(
        cls,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "",
        database: str = "neo4j",
        name: str | None = None,
    ) -> KnowledgeGraph:
        """
        Create a KnowledgeGraph with Neo4j backend.

        Neo4j is a production-grade graph database with:
        - Native graph storage and Cypher query language
        - ACID transactions and horizontal scaling
        - Built-in graph algorithms

        Requires: uv add neo4j

        Args:
            uri: Neo4j connection URI (default: "bolt://localhost:7687")
            user: Neo4j username (default: "neo4j")
            password: Neo4j password
            database: Database name (default: "neo4j")
            name: Optional custom name for this capability instance

        Returns:
            KnowledgeGraph with Neo4j backend

        Example:
            ```python
            # Connect to local Neo4j
            kg = KnowledgeGraph.neo4j(password="your-password")

            # Connect to remote Neo4j Aura
            kg = KnowledgeGraph.neo4j(
                uri="neo4j+s://xxxx.databases.neo4j.io",
                user="neo4j",
                password="your-password",
            )

            # Use with agent
            agent = Agent(
                name="Assistant",
                capabilities=[kg],
            )
            ```
        """
        from agenticflow.capabilities.knowledge_graph.backends.neo4j import Neo4jGraph

        backend = Neo4jGraph(
            uri=uri,
            user=user,
            password=password,
            database=database,
        )
        instance = cls(backend=backend, name=name)  # type: ignore
        instance._backend = "neo4j"
        return instance

    @property
    def name(self) -> str:
        return self._name or "knowledge_graph"

    @property
    def description(self) -> str:
        return (
            "Persistent knowledge graph for storing and querying entities and relationships"
        )

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

    def remember(
        self, entity: str, entity_type: str, facts: dict[str, Any] | None = None
    ) -> str:
        """Remember an entity with attributes."""
        facts_str = json.dumps(facts) if facts else "{}"
        if self._tools_cache is None:
            _ = self.tools  # Initialize cache
        return self._tools_cache["remember"].invoke(
            {
                "entity": entity,
                "entity_type": entity_type,
                "facts": facts_str,
            }
        )

    def recall(self, entity: str) -> str:
        """Recall information about an entity."""
        if self._tools_cache is None:
            _ = self.tools  # Initialize cache
        return self._tools_cache["recall"].invoke({"entity": entity})

    def connect(self, source: str, relation: str, target: str) -> str:
        """Create a relationship between entities."""
        if self._tools_cache is None:
            _ = self.tools
        return self._tools_cache["connect"].invoke(
            {
                "source": source,
                "relation": relation,
                "target": target,
            }
        )

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
                attrs = ", ".join(
                    f"{k}={v}" for k, v in list(e.attributes.items())[:3]
                )
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

    def add_entities_batch(
        self,
        entities: list[tuple[str, str, dict[str, Any] | None]],
    ) -> int:
        """
        Bulk insert entities for high performance.

        This is much faster than calling add_entity() in a loop for large datasets.

        Args:
            entities: List of (entity_id, entity_type, attributes) tuples

        Returns:
            Number of entities inserted

        Example:
            ```python
            kg.add_entities_batch([
                ("Alice", "Person", {"role": "engineer"}),
                ("Bob", "Person", {"role": "manager"}),
                ("Acme", "Company", {"industry": "tech"}),
            ])
            ```
        """
        return self.graph.add_entities_batch(entities)

    def add_relationships_batch(
        self,
        relationships: list[tuple[str, str, str]],
    ) -> int:
        """
        Bulk insert relationships for high performance.

        This is much faster than calling add_relationship() in a loop.

        Args:
            relationships: List of (source_id, relation, target_id) tuples

        Returns:
            Number of relationships inserted

        Example:
            ```python
            kg.add_relationships_batch([
                ("Alice", "works_at", "Acme"),
                ("Bob", "manages", "Alice"),
                ("Alice", "knows", "Bob"),
            ])
            ```
        """
        return self.graph.add_relationships_batch(relationships)

    def get_entity(self, entity_id: str) -> Entity | None:
        """
        Get an entity by ID.

        Args:
            entity_id: The entity identifier

        Returns:
            Entity object or None if not found
        """
        return self.graph.get_entity(entity_id)

    def get_entities(
        self,
        entity_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Entity]:
        """
        Get all entities, optionally filtered by type with pagination.

        Args:
            entity_type: Optional type filter
            limit: Maximum number of entities to return (for pagination)
            offset: Number of entities to skip (for pagination)

        Returns:
            List of Entity objects
        """
        return self.graph.get_all_entities(entity_type, limit=limit, offset=offset)

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

    def __enter__(self) -> KnowledgeGraph:
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
