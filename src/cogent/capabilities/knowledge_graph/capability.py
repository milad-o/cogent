"""KnowledgeGraph capability for persistent entity/relationship memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cogent.capabilities.base import BaseCapability
from cogent.capabilities.knowledge_graph.backends import (
    GraphBackend,
    InMemoryGraph,
    JSONFileGraph,
    SQLiteGraph,
)
from cogent.capabilities.knowledge_graph.models import Entity, Relationship
from cogent.tools.base import BaseTool, tool


class KnowledgeGraph(BaseCapability):
    """
    Knowledge graph capability for persistent entity/relationship memory.

    Supports multiple backends:
    - "memory": In-memory graph (default, uses networkx if available)
    - "sqlite": SQLite database for persistence and large graphs
    - "json": JSON file for simple persistence with auto-save
    - Custom GraphBackend instance for your own implementation

    Provides tools for:
    - kg_remember: Store entities and facts
    - kg_recall: Retrieve entity information
    - kg_connect: Create relationships between entities
    - kg_query: Query the graph with patterns
    - kg_forget: Remove entities
    - kg_list: List known entities

    Example:
        ```python
        from cogent import Agent
        from cogent.capabilities import KnowledgeGraph
        from cogent.capabilities.knowledge_graph.backends import GraphBackend

        # In-memory (default)
        kg = KnowledgeGraph()

        # In-memory with auto-save to file
        kg = KnowledgeGraph(backend="memory", path="memory.json", auto_save=True)

        # SQLite for persistence and large graphs
        kg = KnowledgeGraph(backend="sqlite", path="knowledge.db")

        # JSON file with auto-save
        kg = KnowledgeGraph(backend="json", path="knowledge.json")

        # Custom backend instance
        custom_backend = MyCustomBackend()  # Your GraphBackend implementation
        kg = KnowledgeGraph(backend=custom_backend)

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
        backend: str | GraphBackend = "memory",
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
                - Custom GraphBackend instance for your own implementation
            path: Path to storage file (optional for memory, required for sqlite/json backends)
            name: Optional custom name for this capability instance
            auto_save: Auto-save after each modification (default: True)
                - For memory: saves to path if provided
                - For json: saves to path
                - For sqlite/neo4j: always saves (ignored)
        """
        self._backend = backend
        self._path = Path(path) if path else None
        self._name = name
        self._tools_cache: dict[str, BaseTool] | None = None

        if backend == "memory":
            self.graph: GraphBackend = InMemoryGraph(path=path, auto_save=auto_save)
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

    def set_backend(
        self,
        backend: str | GraphBackend,
        path: str | Path | None = None,
        auto_save: bool = True,
        migrate: bool = True,
    ) -> None:
        """
        Change the backend of this KnowledgeGraph instance.

        Optionally migrates existing data from the current backend to the new one.

        Args:
            backend: New storage backend ("memory", "sqlite", "json", or GraphBackend instance)
            path: Path for sqlite/json backends
            auto_save: For json backend, save after each modification
            migrate: If True, copy all entities and relationships to new backend

        Example:
            ```python
            # Start with in-memory
            kg = KnowledgeGraph()
            kg.remember("Alice", "Person", {"role": "Engineer"})

            # Switch to SQLite with migration
            kg.set_backend("sqlite", path="knowledge.db", migrate=True)

            # Data is now persisted in SQLite
            # Switch to JSON
            kg.set_backend("json", path="knowledge.json", migrate=True)
            ```
        """
        # Create new backend
        previous_graph = self.graph
        new_graph: GraphBackend

        if backend == "memory":
            new_graph = InMemoryGraph(path=path, auto_save=auto_save)
        elif backend == "sqlite":
            if not path:
                raise ValueError("Path required for sqlite backend")
            new_graph = SQLiteGraph(path)
        elif backend == "json":
            if not path:
                raise ValueError("Path required for json backend")
            new_graph = JSONFileGraph(path, auto_save=auto_save)
        elif isinstance(backend, GraphBackend):
            new_graph = backend
            backend = "custom"
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Supported: 'memory', 'sqlite', 'json', or GraphBackend instance"
            )

        # Migrate data if requested
        if migrate:
            # Get all entities and relationships from current backend
            entities = self.graph.get_all_entities()

            # Copy entities
            for entity in entities:
                new_graph.add_entity(
                    entity.id,
                    entity.type,
                    entity.attributes,
                    entity.source,
                )

            # Copy relationships
            for entity in entities:
                relationships = self.graph.get_relationships(
                    entity.id, direction="outgoing"
                )
                for rel in relationships:
                    new_graph.add_relationship(
                        rel.source_id,
                        rel.relation,
                        rel.target_id,
                        rel.attributes,
                        rel.source,
                    )

        # Switch to new backend
        self.graph = new_graph
        self._backend = backend
        self._path = Path(path) if path else None

        # Close the previous backend to release resources
        try:
            previous_graph.close()
        except Exception:
            pass

        # Clear tools cache to regenerate with new backend
        self._tools_cache = None

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
        from cogent.capabilities.knowledge_graph.backends.neo4j import Neo4jGraph

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
        return "Persistent knowledge graph for storing and querying entities and relationships"

    @property
    def tools(self) -> list[BaseTool]:
        # Cache tools to avoid recreating them
        if self._tools_cache is None:
            self._tools_cache = {
                "kg_remember": self._remember_tool(),
                "kg_recall": self._recall_tool(),
                "kg_connect": self._connect_tool(),
                "kg_query": self._query_tool(),
                "kg_forget": self._forget_tool(),
                "kg_list": self._list_entities_tool(),
            }
        return list(self._tools_cache.values())

    # === Convenience methods for direct use ===

    def remember(
        self, entity: str, entity_type: str, facts: dict[str, Any] | None = None
    ) -> str:
        """Remember an entity with attributes."""
        if self._tools_cache is None:
            _ = self.tools  # Initialize cache
        return self._tools_cache["kg_remember"].invoke(
            {
                "entity": entity,
                "entity_type": entity_type,
                "attributes": facts,
            }
        )

    def recall(self, entity: str) -> str:
        """Recall information about an entity."""
        if self._tools_cache is None:
            _ = self.tools  # Initialize cache
        return self._tools_cache["kg_recall"].invoke({"entity": entity})

    def connect(self, source: str, relation: str, target: str) -> str:
        """Create a relationship between entities."""
        if self._tools_cache is None:
            _ = self.tools
        return self._tools_cache["kg_connect"].invoke(
            {
                "source": source,
                "relation": relation,
                "target": target,
            }
        )

    def query(self, pattern: str) -> str:
        """Query the knowledge graph with a pattern.

        Args:
            pattern: Pattern string like "Alice -works_at-> ?" or "? -reports_to-> Bob"
                     Use ? for wildcards to find unknowns.
        """
        if self._tools_cache is None:
            _ = self.tools

        # Parse pattern: "source -relation-> target"
        import re

        match = re.match(r"(.+?)\s*-(.+?)->\s*(.+)", pattern)
        if not match:
            return f"Invalid pattern format: {pattern}. Expected: 'source -relation-> target'"

        source_str, relation_str, target_str = match.groups()

        # Convert ? to None for wildcards
        source = None if source_str.strip() == "?" else source_str.strip()
        relation = None if relation_str.strip() == "?" else relation_str.strip()
        target = None if target_str.strip() == "?" else target_str.strip()

        return self._tools_cache["kg_query"].invoke(
            {"source": source, "relation": relation, "target": target}
        )

    def forget(self, entity: str) -> str:
        """Remove an entity from the graph."""
        if self._tools_cache is None:
            _ = self.tools
        return self._tools_cache["kg_forget"].invoke({"entity": entity})

    def list_entities(self, entity_type: str = "") -> str:
        """List all known entities."""
        if self._tools_cache is None:
            _ = self.tools
        return self._tools_cache["kg_list"].invoke({"entity_type": entity_type})

    def _remember_tool(self) -> BaseTool:
        graph = self.graph

        @tool(name="kg_remember")
        def kg_remember(
            entity: str,
            entity_type: str,
            attributes: dict[str, Any] | str | None = None,
        ) -> str:
            """
            Remember an entity and its attributes.

            Args:
                entity: Name/ID of the entity (e.g., "Alice", "Acme Corp")
                entity_type: Type of entity (e.g., "Person", "Company", "Project")
                attributes: Dictionary of facts/attributes or JSON string
                    Examples: {"role": "engineer", "team": "backend"}
                             '{"role": "engineer", "team": "backend"}'

            Returns:
                Confirmation message

            Examples:
                kg_remember(entity="Alice", entity_type="Person", attributes={"role": "CEO", "age": 35})
                kg_remember(entity="TechCorp", entity_type="Company", attributes={"founded": 2015})
            """
            # Handle both dict and JSON string for compatibility
            if isinstance(attributes, str):
                try:
                    attrs = json.loads(attributes) if attributes else {}
                except json.JSONDecodeError:
                    attrs = {"info": attributes}
            elif isinstance(attributes, dict):
                attrs = attributes
            else:
                attrs = {}

            graph.add_entity(entity, entity_type, attrs)
            return f"Remembered: {entity} ({entity_type}) with {len(attrs)} attributes"

        return kg_remember

    def _recall_tool(self) -> BaseTool:
        graph = self.graph

        @tool(name="kg_recall")
        def kg_recall(entity: str) -> str:
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

        return kg_recall

    def _connect_tool(self) -> BaseTool:
        graph = self.graph

        @tool(name="kg_connect")
        def kg_connect(
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

        return kg_connect

    def _query_tool(self) -> BaseTool:
        graph = self.graph

        @tool(name="kg_query")
        def kg_query(
            source: str | None = None,
            relation: str | None = None,
            target: str | None = None,
        ) -> str:
            """
            Query the knowledge graph for relationships matching a pattern.

            Use None for wildcards to find unknowns.

            Args:
                source: Source entity name (use None to find all sources)
                relation: Relationship type (use None to find all relations)
                target: Target entity name (use None to find all targets)

            Examples:
                                - kg_query(source=None, relation="works_at", target="TechCorp")
                  → Who works at TechCorp?
                                - kg_query(source="Alice", relation="works_at", target=None)
                  → Where does Alice work?
                                - kg_query(source=None, relation="reports_to", target="Bob")
                  → Who reports to Bob?
                                - kg_query(source="Alice", relation=None, target=None)
                  → All relationships from Alice

            Returns:
                Query results as JSON
            """
            # Build pattern string for internal query method
            s = source or "?"
            r = relation or "?"
            t = target or "?"
            pattern = f"{s} -{r}-> {t}"

            results = graph.query(pattern)

            if not results:
                return f"No results for pattern: {pattern}"

            return json.dumps(results, indent=2, default=str)

        return kg_query

    def _forget_tool(self) -> BaseTool:
        graph = self.graph

        @tool(name="kg_forget")
        def kg_forget(entity: str) -> str:
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

        return kg_forget

    def _list_entities_tool(self) -> BaseTool:
        graph = self.graph

        @tool(name="kg_list")
        def kg_list(entity_type: str = "") -> str:
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

        return kg_list

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
            name: Tool name (kg_remember, kg_recall, kg_connect, kg_query, kg_forget, kg_list)

        Returns:
            The tool or None if not found
        """
        if self._tools_cache is None:
            _ = self.tools  # Initialize cache
        tool = self._tools_cache.get(name)
        if tool is not None:
            return tool

        for candidate in self._tools_cache.values():
            if candidate.name == name:
                return candidate

        return None

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

    # === Visualization ===

    def _collect_visualization_data(
        self,
        *,
        max_entities: int | None = None,
    ) -> tuple[list[Entity], list[Relationship]]:
        entities = self.graph.get_all_entities()
        if max_entities and len(entities) > max_entities:
            entities = entities[:max_entities]

        relationships: list[Relationship] = []
        seen_rels: set[tuple[str, str, str]] = set()
        for entity in entities:
            rels = self.graph.get_relationships(entity.id, direction="outgoing")
            for rel in rels:
                rel_key = (rel.source_id, rel.relation, rel.target_id)
                if rel_key not in seen_rels:
                    relationships.append(rel)
                    seen_rels.add(rel_key)

        return entities, relationships

    # =========================================================================
    # Convenience Visualization APIs (Low, Medium, High Level)
    # =========================================================================

    def mermaid(
        self,
        *,
        direction: str = "LR",
        group_by_type: bool = False,
        scheme: str = "default",
        title: str | None = None,
        max_entities: int | None = None,
    ) -> str:
        """Get Mermaid diagram code for the knowledge graph."""
        from cogent.graph.visualization import to_mermaid

        entities, relationships = self._collect_visualization_data(
            max_entities=max_entities
        )
        return to_mermaid(
            entities,
            relationships,
            direction=direction,
            group_by_type=group_by_type,
            scheme=scheme,
            title=title,
        )

    def render(
        self,
        format: str = "mermaid",
        *,
        direction: str = "LR",
        group_by_type: bool = False,
        scheme: str = "default",
        title: str | None = None,
        max_entities: int | None = None,
    ) -> str:
        """Render the graph to Mermaid format."""
        from cogent.graph.visualization import (
            to_mermaid,
        )

        entities, relationships = self._collect_visualization_data(
            max_entities=max_entities
        )

        normalized = format.lower()
        if normalized in ("mermaid", "mmd"):
            return to_mermaid(
                entities,
                relationships,
                direction=direction,
                group_by_type=group_by_type,
                scheme=scheme,
                title=title,
            )

        raise ValueError(f"Unsupported format: {format}. Valid: mermaid")

    def display(self, **kwargs) -> None:
        """Print Mermaid diagram code for inline display."""
        print(self.mermaid(**kwargs))

    def interactive(
        self,
        output_path: str | Path = "knowledge_graph.html",
        *,
        height: str = "600px",
        width: str = "100%",
        physics_config: dict[str, Any] | None = None,
        entity_color: str | None = None,
        relationship_color: str = "#2B7CE9",
        notebook: bool = False,
        max_entities: int | None = None,
        color_by_type: bool = True,
        show_type_in_label: bool = True,
        scheme: str = "default",
    ) -> Path:
        """
        Generate interactive HTML visualization using PyVis and NetworkX.

        Creates a force-directed graph with drag, zoom, and hover capabilities.
        Nodes are automatically colored by entity type for better visual organization.

        Args:
            output_path: Where to save the HTML file
            height: Canvas height (e.g., "600px", "100vh")
            width: Canvas width (e.g., "100%", "800px")
            physics_config: Custom physics configuration for layout
            entity_color: Override color for all entities (hex). If None and color_by_type=True,
                colors are assigned by type from the style scheme.
            relationship_color: Color for relationship edges (hex)
            notebook: True if running in Jupyter notebook
            max_entities: Limit number of entities to visualize
            color_by_type: If True, color nodes by their entity type (recommended)
            show_type_in_label: If True, include entity type in node labels
            scheme: Style scheme for coloring ("default" or "minimal")

        Returns:
            Path to the generated HTML file

        Example:
            ```python
            # Basic usage with type-based coloring
            kg.interactive("my_graph.html")

            # Jupyter notebook
            kg.interactive(notebook=True)

            # Disable type coloring
            kg.interactive(color_by_type=False, entity_color="#7BE382")

            # Custom physics
            kg.interactive(
                physics_config={
                    "barnesHut": {
                        "gravitationalConstant": -3000,
                        "centralGravity": 0.5,
                        "springLength": 200
                    }
                }
            )
            ```
        """
        from cogent.graph.visualization import to_pyvis

        # Collect graph data
        entities, relationships = self._collect_visualization_data(
            max_entities=max_entities
        )

        # Create PyVis network using renderer
        net = to_pyvis(
            entities,
            relationships,
            height=height,
            width=width,
            physics_config=physics_config,
            entity_color=entity_color,
            relationship_color=relationship_color,
            notebook=notebook,
            directed=True,
            scheme=scheme,
            color_by_type=color_by_type,
            show_type_in_label=show_type_in_label,
        )

        # Save and return path
        output = Path(output_path)
        net.save_graph(str(output))

        return output

    # plot() method removed - use mermaid(), interactive(), or visualize() instead
    # For static diagrams: mermaid() -> render to PNG/SVG/PDF
    # For web-based: interactive() (PyVis) or visualize() (gravis 2D/3D)

    def visualize(
        self,
        *,
        mode: str = "2d",
        renderer: str = "vis",
        output_path: str | Path | None = None,
        max_entities: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create interactive web visualization using gravis.

        Generates interactive 2D or 3D visualizations powered by d3.js, vis.js,
        or three.js. Supports pan, zoom, drag nodes, and rich metadata display.

        Args:
            mode: Visualization mode - "2d" or "3d"
            renderer: Rendering engine:
                - "vis": vis.js force-directed (default, best for exploration)
                - "d3": d3.js-based (good for export to SVG)
                - "three": three.js 3D visualization
            output_path: Optional path to save HTML file
            max_entities: Limit number of entities to visualize
            **kwargs: Additional arguments passed to to_gravis():
                - node_size_data: Attribute to map to node size
                - node_label_data: Attribute for labels
                - edge_curvature: Edge curvature (0.0-1.0)
                - zoom_factor: Initial zoom level
                - show_node_label: Show/hide node labels (bool)
                - show_edge_label: Show/hide edge labels (bool)
                - layout_algorithm: Force layout algorithm (vis only)
                - graph_height: Height in pixels (int, default 450)

        Returns:
            gravis Figure object with methods:
                - .display() - Open in browser
                - .export_html(path) - Save as HTML
                - .export_svg(path) - Save as SVG (2D only)
                - .export_png(path) - Save as PNG (requires Selenium)

        Example:
            ```python
            # Basic 2D interactive visualization
            fig = kg.visualize()
            fig.display()  # Opens in browser

            # 3D visualization
            fig = kg.visualize(mode="3d")
            fig.export_html("graph_3d.html")

            # Advanced 2D with styling
            fig = kg.visualize(
                renderer="d3",
                edge_curvature=0.3,
                zoom_factor=0.8,
                show_edge_label=True
            )
            fig.export_svg("graph.svg")

            # Auto-save to file
            fig = kg.visualize(
                output_path="knowledge_graph.html",
                max_entities=100
            )
            ```
        """
        from cogent.graph.visualization import to_gravis

        # Collect graph data
        entities, relationships = self._collect_visualization_data(
            max_entities=max_entities
        )

        # Create gravis visualization
        fig = to_gravis(
            entities,
            relationships,
            mode=mode,
            renderer=renderer,
            **kwargs,
        )

        # Save if path provided
        if output_path:
            path = Path(output_path)
            fig.export_html(str(path))

        return fig

    def visualize_3d(
        self,
        *,
        output_path: str | Path | None = None,
        max_entities: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create 3D interactive visualization using gravis + three.js.

        Convenient shortcut for 3D graph exploration with physics simulation.

        Args:
            output_path: Optional path to save HTML file
            max_entities: Limit number of entities to visualize
            **kwargs: Additional arguments passed to to_gravis()

        Returns:
            gravis Figure object (call .display() or .export_html())

        Example:
            ```python
            # Open 3D view in browser
            fig = kg.visualize_3d()
            fig.display()

            # Save 3D visualization
            fig = kg.visualize_3d(output_path="graph_3d.html")

            # Custom zoom and labels
            fig = kg.visualize_3d(
                zoom_factor=1.5,
                show_edge_label=True,
            )
            ```
        """
        return self.visualize(
            mode="3d",
            renderer="three",
            output_path=output_path,
            max_entities=max_entities,
            **kwargs,
        )

