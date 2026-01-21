"""KnowledgeGraph capability for persistent entity/relationship memory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenticflow.capabilities.base import BaseCapability
from agenticflow.capabilities.knowledge_graph.backends import (
    GraphBackend,
    InMemoryGraph,
    JSONFileGraph,
    SQLiteGraph,
)
from agenticflow.capabilities.knowledge_graph.models import Entity, Relationship
from agenticflow.tools.base import BaseTool, tool

if TYPE_CHECKING:
    from agenticflow.graph import GraphView


class KnowledgeGraph(BaseCapability):
    """
    Knowledge graph capability for persistent entity/relationship memory.

    Supports multiple backends:
    - "memory": In-memory graph (default, uses networkx if available)
    - "sqlite": SQLite database for persistence and large graphs
    - "json": JSON file for simple persistence with auto-save
    - Custom GraphBackend instance for your own implementation

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
        from agenticflow.capabilities.knowledge_graph.backends import GraphBackend

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
                relationships = self.graph.get_relationships(entity.id, direction="outgoing")
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
        if self._tools_cache is None:
            _ = self.tools  # Initialize cache
        return self._tools_cache["remember"].invoke(
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
        """Query the knowledge graph with a pattern.
        
        Args:
            pattern: Pattern string like "Alice -works_at-> ?" or "? -reports_to-> Bob"
                     Use ? for wildcards to find unknowns.
        """
        if self._tools_cache is None:
            _ = self.tools
        
        # Parse pattern: "source -relation-> target"
        import re
        match = re.match(r'(.+?)\s*-(.+?)->\s*(.+)', pattern)
        if not match:
            return f"Invalid pattern format: {pattern}. Expected: 'source -relation-> target'"
        
        source_str, relation_str, target_str = match.groups()
        
        # Convert ? to None for wildcards
        source = None if source_str.strip() == "?" else source_str.strip()
        relation = None if relation_str.strip() == "?" else relation_str.strip()
        target = None if target_str.strip() == "?" else target_str.strip()
        
        return self._tools_cache["query"].invoke({
            "source": source,
            "relation": relation,
            "target": target
        })

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
                remember(entity="Alice", entity_type="Person", attributes={"role": "CEO", "age": 35})
                remember(entity="TechCorp", entity_type="Company", attributes={"founded": 2015})
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
        def query_knowledge(
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
                - query_knowledge(source=None, relation="works_at", target="TechCorp")
                  → Who works at TechCorp?
                - query_knowledge(source="Alice", relation="works_at", target=None)
                  → Where does Alice work?
                - query_knowledge(source=None, relation="reports_to", target="Bob")
                  → Who reports to Bob?
                - query_knowledge(source="Alice", relation=None, target=None)
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

    # === Visualization ===

    def visualize(
        self,
        *,
        layout: str = "hierarchical",
        direction: str = "LR",
        show_attributes: bool = False,
        max_entities: int | None = None,
        group_by_type: bool = True,
    ) -> GraphView:
        """
        Get a graph visualization of the knowledge graph.

        Returns a GraphView that provides a unified interface for
        rendering to Mermaid, Graphviz, ASCII, or other formats.

        Args:
            layout: Layout algorithm - "hierarchical", "circular", "force" (default: "hierarchical")
            direction: Graph direction - "TB" (top-bottom), "LR" (left-right), "BT", "RL" (default: "LR")
            show_attributes: Include entity attributes in labels (default: False)
            max_entities: Maximum number of entities to show (default: all)
            group_by_type: Group entities by type in subgraphs (default: True)

        Returns:
            GraphView instance for rendering.

        Example:
            ```python
            # Build knowledge graph
            kg = KnowledgeGraph()
            kg.remember("Alice", "Person", {"role": "Engineer"})
            kg.remember("Bob", "Person", {"role": "Manager"})
            kg.connect("Alice", "reports_to", "Bob")

            # Get visualization
            view = kg.visualize()

            # Render in different formats
            print(view.mermaid())    # Mermaid diagram
            print(view.ascii())      # Terminal-friendly
            print(view.dot())        # Graphviz DOT

            # Save to file
            view.save("knowledge.png")

            # Get shareable URL
            print(view.url())
            ```
        """
        from agenticflow.graph import GraphView
        from agenticflow.graph.config import GraphConfig, GraphDirection
        from agenticflow.graph.primitives import (
            ClassDef,
            Edge,
            EdgeType,
            Graph,
            Node,
            NodeShape,
            NodeStyle,
            Subgraph,
        )

        g = Graph()

        # Get entities and relationships
        stats = self.graph.stats()
        entities = self.graph.get_all_entities()
        
        # Collect all relationships by iterating through entities
        relationships: list[Relationship] = []
        seen_rels: set[tuple[str, str, str]] = set()
        for entity in entities:
            rels = self.graph.get_relationships(entity.id, direction="outgoing")
            for rel in rels:
                rel_key = (rel.source_id, rel.relation, rel.target_id)
                if rel_key not in seen_rels:
                    relationships.append(rel)
                    seen_rels.add(rel_key)

        # Limit entities if specified
        if max_entities and len(entities) > max_entities:
            entities = entities[:max_entities]

        # Group entities by type
        entity_types: dict[str, list[Entity]] = {}
        for entity in entities:
            entity_type = entity.type
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity)

        # Create nodes for each entity (with optional subgraphs)
        if group_by_type and len(entity_types) > 1:
            # Create subgraphs for each entity type
            for entity_type, type_entities in sorted(entity_types.items()):
                subgraph_id = f"type_{entity_type.replace(' ', '_')}"
                subgraph_nodes = []
                
                for entity in type_entities:
                    node_id = entity.id.replace(" ", "_").replace(".", "_")
                    subgraph_nodes.append(node_id)
                    
                    # Build label
                    if show_attributes and entity.attributes:
                        attr_str = "\\n".join(
                            f"{k}: {str(v)[:20]}" for k, v in list(entity.attributes.items())[:3]
                        )
                        label = f"{entity.id}\\n<small>{attr_str}</small>"
                    else:
                        label = entity.id

                    # Determine shape and class based on type
                    shape = NodeShape.ROUNDED
                    if entity.type.lower() in ("person", "user", "agent"):
                        css_class = "person"
                    elif entity.type.lower() in ("organization", "company", "org"):
                        css_class = "org"
                    elif entity.type.lower() in ("location", "place"):
                        css_class = "location"
                    elif entity.type.lower() in ("event", "action"):
                        css_class = "event"
                    else:
                        css_class = "entity"

                    g.add_node(
                        Node(
                            id=node_id,
                            label=label,
                            shape=shape,
                            css_class=css_class,
                        )
                    )
                
                # Add subgraph
                g.add_subgraph(
                    Subgraph(
                        id=subgraph_id,
                        label=entity_type,
                        node_ids=subgraph_nodes,
                    )
                )
        else:
            # Create nodes without subgraphs
            for entity in entities:
                node_id = entity.id.replace(" ", "_").replace(".", "_")

                # Build label
                if show_attributes and entity.attributes:
                    attr_str = "\\n".join(
                        f"{k}: {str(v)[:20]}" for k, v in list(entity.attributes.items())[:3]
                    )
                    label = f"{entity.id}\\n<small><i>{entity.type}</i></small>\\n<small>{attr_str}</small>"
                else:
                    label = f"{entity.id}\\n<small><i>{entity.type}</i></small>"

                # Determine shape and class based on type
                shape = NodeShape.ROUNDED
                if entity.type.lower() in ("person", "user", "agent"):
                    css_class = "person"
                elif entity.type.lower() in ("organization", "company", "org"):
                    css_class = "org"
                elif entity.type.lower() in ("location", "place"):
                    css_class = "location"
                elif entity.type.lower() in ("event", "action"):
                    css_class = "event"
                else:
                    css_class = "entity"

                g.add_node(
                    Node(
                        id=node_id,
                        label=label,
                        shape=shape,
                        css_class=css_class,
                    )
                )

        # Create edges for relationships
        for rel in relationships:
            source_id = rel.source_id.replace(" ", "_").replace(".", "_")
            target_id = rel.target_id.replace(" ", "_").replace(".", "_")

            # Only add edge if both nodes exist
            if source_id in g.nodes and target_id in g.nodes:
                edge_label = rel.relation

                # Determine edge type based on relation
                if "parent" in rel.relation.lower() or "child" in rel.relation.lower():
                    edge_type = EdgeType.ARROW
                elif "knows" in rel.relation.lower() or "friend" in rel.relation.lower():
                    edge_type = EdgeType.BIDIRECTIONAL
                else:
                    edge_type = EdgeType.ARROW

                g.add_edge(
                    Edge(
                        source=source_id,
                        target=target_id,
                        label=edge_label,
                        edge_type=edge_type,
                    )
                )

        # Add class definitions for styling
        g.add_class_def(
            ClassDef(
                name="person",
                style=NodeStyle(fill="#60a5fa", stroke="#3b82f6", color="#fff"),
            )
        )
        g.add_class_def(
            ClassDef(
                name="org",
                style=NodeStyle(fill="#7eb36a", stroke="#4a7a3d", color="#fff"),
            )
        )
        g.add_class_def(
            ClassDef(
                name="location",
                style=NodeStyle(fill="#f59e0b", stroke="#d97706", color="#fff"),
            )
        )
        g.add_class_def(
            ClassDef(
                name="event",
                style=NodeStyle(fill="#9b59b6", stroke="#7b3a96", color="#fff"),
            )
        )
        g.add_class_def(
            ClassDef(
                name="entity",
                style=NodeStyle(fill="#94a3b8", stroke="#64748b", color="#fff"),
            )
        )

        # Create config without title in frontmatter (to avoid redundancy in inline rendering)
        # Title is added separately in HTML rendering
        title = f"Knowledge Graph ({stats['entities']} entities, {stats['relationships']} relationships)"
        
        # Map direction string to GraphDirection enum
        direction_map = {
            "TB": GraphDirection.TOP_DOWN,
            "LR": GraphDirection.LEFT_RIGHT,
            "BT": GraphDirection.BOTTOM_UP,
            "RL": GraphDirection.RIGHT_LEFT,
        }
        graph_direction = direction_map.get(direction.upper(), GraphDirection.TOP_DOWN)
        
        config = GraphConfig(
            title=title,
            direction=graph_direction,
        )

        return GraphView(g, config)

    # =========================================================================
    # Convenience Visualization APIs (Low, Medium, High Level)
    # =========================================================================

    def mermaid(self, **kwargs) -> str:
        """Low-level API: Get Mermaid diagram code.
        
        This is the simplest API - just returns the raw Mermaid code string.
        Use this when you want to copy/paste the diagram or save it to a file.
        
        Args:
            **kwargs: Visualization options (direction, show_attributes, etc.)
            
        Returns:
            Mermaid diagram code as a string.
            
        Example:
            ```python
            kg = KnowledgeGraph()
            kg.remember("Alice", "Person")
            kg.remember("Bob", "Person")
            kg.connect("Alice", "knows", "Bob")
            
            # Get mermaid code
            code = kg.mermaid()
            print(code)
            
            # Save to file
            with open("graph.mmd", "w") as f:
                f.write(kg.mermaid())
            ```
        """
        view = self.visualize(**kwargs)
        return view.mermaid()
    
    def render(self, format: str = "auto", **kwargs) -> str | bytes:
        """Medium-level API: Render to various formats.
        
        Choose from multiple output formats. Good for when you need
        flexibility but don't want to deal with GraphView directly.
        
        Args:
            format: Output format - "auto", "mermaid", "ascii", "html", "png", "svg"
            **kwargs: Visualization options (direction, show_attributes, etc.)
            
        Returns:
            Rendered content (string or bytes for images).
            
        Example:
            ```python
            kg = KnowledgeGraph()
            # ... add entities ...
            
            # Get different formats
            mermaid_code = kg.render("mermaid")
            ascii_art = kg.render("ascii")
            html = kg.render("html")
            png_bytes = kg.render("png")
            
            # Save PNG
            with open("graph.png", "wb") as f:
                f.write(kg.render("png"))
            ```
        """
        view = self.visualize(**kwargs)
        return view.render(format)
    
    def display(self, **kwargs) -> None:
        """High-level API: Display graph in Jupyter notebook.
        
        This is the easiest way to visualize in Jupyter - just call display()
        and the graph will render inline with proper styling.
        
        Args:
            **kwargs: Visualization options (direction, show_attributes, etc.)
            
        Example:
            ```python
            # In Jupyter notebook:
            kg = KnowledgeGraph()
            kg.remember("Alice", "Person", {"role": "Engineer"})
            kg.remember("TechCorp", "Company")
            kg.connect("Alice", "works_at", "TechCorp")
            
            # Display inline
            kg.display()
            
            # Or with options
            kg.display(direction="TB", show_attributes=True)
            
            # Alternative: just call the object in Jupyter
            view = kg.visualize()
            view  # Automatically renders
            ```
        """
        view = self.visualize(**kwargs)
        view.display()
