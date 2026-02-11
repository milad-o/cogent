"""Base Graph class - composable graph with engine + storage.

This module provides the core Graph class that combines an Engine (for graph
operations and algorithms) with Storage (for data persistence), creating a
fully composable and flexible graph system.
"""

from typing import Any

from cogent.graph.engines.base import Engine
from cogent.graph.models import Entity, Relationship
from cogent.graph.storage.base import Storage


class Graph:
    """Composable graph combining engine (how) and storage (where).

    The Graph class provides a high-level interface for working with knowledge
    graphs by coordinating between an Engine (graph algorithms and operations)
    and Storage (data persistence).

    Args:
        engine: Graph engine for operations. If None, auto-selects NetworkXEngine
            if available, otherwise NativeEngine.
        storage: Storage backend for persistence. If None, uses MemoryStorage.

    Example:
        >>> # Default: NetworkX engine + memory storage
        >>> graph = Graph()
        >>> await graph.add_entity("alice", "Person", name="Alice")

        >>> # Explicit engine and storage
        >>> from cogent.graph.engines import NativeEngine
        >>> from cogent.graph.storage import FileStorage
        >>> graph = Graph(
        ...     engine=NativeEngine(),
        ...     storage=FileStorage("graph.json", format="json")
        ... )

        >>> # SQL persistence with NetworkX algorithms
        >>> from cogent.graph.storage import SQLStorage
        >>> graph = Graph(
        ...     storage=SQLStorage("postgresql://localhost/mydb")
        ... )
    """

    def __init__(
        self,
        engine: Engine | None = None,
        storage: Storage | None = None,
    ) -> None:
        """Initialize graph with engine and storage."""
        # Auto-select engine if not provided
        if engine is None:
            engine = self._get_default_engine()

        # Auto-select storage if not provided
        if storage is None:
            from cogent.graph.storage import MemoryStorage
            storage = MemoryStorage()

        self.engine = engine
        self.storage = storage

    def _get_default_engine(self) -> Engine:
        """Get default engine (NetworkX if available, else Native)."""
        try:
            from cogent.graph.engines import NetworkXEngine
            return NetworkXEngine()
        except ImportError:
            from cogent.graph.engines import NativeEngine
            return NativeEngine()

    # --- Entity Operations ---

    async def add_entity(
        self,
        id: str,
        entity_type: str,
        **attributes: Any,
    ) -> Entity:
        """Add a new entity to the graph.

        Args:
            id: Unique identifier for the entity.
            entity_type: Type/category of the entity.
            **attributes: Additional key-value attributes.

        Returns:
            The created Entity.

        Raises:
            ValueError: If entity with this ID already exists.

        Example:
            >>> entity = await graph.add_entity("alice", "Person", name="Alice", age=30)
        """
        # Add to storage first (validates uniqueness)
        entity = await self.storage.add_entity(id, entity_type, **attributes)

        # Add node to engine
        await self.engine.add_node(id, **attributes)

        return entity

    async def add_entities(self, entities: list[Entity]) -> list[Entity]:
        """Bulk add multiple entities.

        Args:
            entities: List of entities to add.

        Returns:
            List of created entities.

        Raises:
            ValueError: If any entity ID already exists.

        Example:
            >>> entities = [
            ...     Entity(id="alice", entity_type="Person", attributes={"name": "Alice"}),
            ...     Entity(id="bob", entity_type="Person", attributes={"name": "Bob"}),
            ... ]
            >>> created = await graph.add_entities(entities)
        """
        # Add to storage (validates uniqueness)
        created_entities = await self.storage.add_entities(entities)

        # Add nodes to engine
        for entity in created_entities:
            await self.engine.add_node(entity.id, **entity.attributes)

        return created_entities

    async def get_entity(self, id: str) -> Entity | None:
        """Get an entity by ID.

        Args:
            id: Entity identifier.

        Returns:
            The Entity if found, None otherwise.

        Example:
            >>> entity = await graph.get_entity("alice")
        """
        return await self.storage.get_entity(id)

    async def remove_entity(self, id: str) -> bool:
        """Remove an entity and all its relationships.

        Args:
            id: Entity identifier.

        Returns:
            True if entity was removed, False if not found.

        Example:
            >>> removed = await graph.remove_entity("alice")
        """
        # Remove from storage (cascades to relationships)
        removed = await self.storage.remove_entity(id)

        # Remove from engine
        if removed:
            await self.engine.remove_node(id)

        return removed

    async def get_all_entities(self) -> list[Entity]:
        """Get all entities in the graph.

        Returns:
            List of all entities.

        Example:
            >>> entities = await graph.get_all_entities()
        """
        return await self.storage.get_all_entities()

    # --- Relationship Operations ---

    async def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        **attributes: Any,
    ) -> Relationship:
        """Add a relationship between two entities.

        Args:
            source_id: Source entity ID.
            relation: Relationship type/label.
            target_id: Target entity ID.
            **attributes: Additional relationship attributes.

        Returns:
            The created Relationship.

        Raises:
            ValueError: If source or target entity doesn't exist.

        Example:
            >>> rel = await graph.add_relationship("alice", "knows", "bob", since=2020)
        """
        # Add to storage (validates entity existence)
        relationship = await self.storage.add_relationship(
            source_id, relation, target_id, **attributes
        )

        # Add edge to engine
        await self.engine.add_edge(source_id, target_id, relation=relation, **attributes)

        return relationship

    async def add_relationships(
        self, relationships: list[Relationship]
    ) -> list[Relationship]:
        """Bulk add multiple relationships.

        Args:
            relationships: List of relationships to add.

        Returns:
            List of created relationships.

        Raises:
            ValueError: If any source or target entity doesn't exist.

        Example:
            >>> rels = [
            ...     Relationship(source_id="alice", relation="knows", target_id="bob"),
            ...     Relationship(source_id="bob", relation="knows", target_id="charlie"),
            ... ]
            >>> created = await graph.add_relationships(rels)
        """
        # Add to storage (validates entity existence)
        created_rels = await self.storage.add_relationships(relationships)

        # Add edges to engine
        for rel in created_rels:
            await self.engine.add_edge(
                rel.source_id, rel.target_id, relation=rel.relation, **rel.attributes
            )

        return created_rels

    async def get_relationships(
        self,
        source_id: str | None = None,
        relation: str | None = None,
        target_id: str | None = None,
    ) -> list[Relationship]:
        """Query relationships by source, relation, and/or target.

        Args:
            source_id: Filter by source entity ID.
            relation: Filter by relationship type.
            target_id: Filter by target entity ID.

        Returns:
            List of matching relationships.

        Example:
            >>> # All relationships from alice
            >>> rels = await graph.get_relationships(source_id="alice")

            >>> # All "knows" relationships
            >>> rels = await graph.get_relationships(relation="knows")

            >>> # Specific relationship
            >>> rels = await graph.get_relationships(
            ...     source_id="alice", relation="knows", target_id="bob"
            ... )
        """
        return await self.storage.get_relationships(source_id, relation, target_id)

    # --- Query and Traversal Methods ---

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int | None = None,
    ) -> list[str] | None:
        """Find shortest path between two entities.

        Args:
            source_id: Starting entity ID.
            target_id: Target entity ID.
            max_depth: Maximum path length (optional).

        Returns:
            List of entity IDs forming the path, or None if no path exists.

        Example:
            >>> path = await graph.find_path("alice", "charlie")
            >>> print(path)  # ["alice", "bob", "charlie"]
        """
        # Use engine for pathfinding (faster algorithms)
        return await self.engine.find_path(source_id, target_id, max_depth)

    async def get_neighbors(
        self,
        entity_id: str,
        direction: str = "outgoing",
    ) -> list[str]:
        """Get neighboring entities.

        Args:
            entity_id: Entity ID to get neighbors for.
            direction: "outgoing", "incoming", or "both".

        Returns:
            List of neighbor entity IDs.

        Example:
            >>> neighbors = await graph.get_neighbors("alice", direction="outgoing")
        """
        return await self.engine.get_neighbors(entity_id, direction)

    async def find_entities(
        self,
        entity_type: str | None = None,
        **attributes: Any,
    ) -> list[Entity]:
        """Find entities matching type and/or attributes.

        Args:
            entity_type: Filter by entity type.
            **attributes: Filter by attribute key-value pairs.

        Returns:
            List of matching entities.

        Example:
            >>> # Find all Person entities
            >>> people = await graph.find_entities(entity_type="Person")

            >>> # Find entities with specific attribute
            >>> adults = await graph.find_entities(entity_type="Person", age=30)
        """
        all_entities = await self.storage.get_all_entities()

        # Filter by type
        if entity_type is not None:
            all_entities = [e for e in all_entities if e.entity_type == entity_type]

        # Filter by attributes
        if attributes:
            filtered = []
            for entity in all_entities:
                if all(
                    entity.attributes.get(key) == value
                    for key, value in attributes.items()
                ):
                    filtered.append(entity)
            return filtered

        return all_entities

    async def match(self, pattern: dict[str, Any]) -> Any:
        """Execute a pattern-based query against the graph.

        This method enables complex graph queries using dict-based patterns
        instead of string query languages. Supports single-hop, multi-hop,
        and entity filtering patterns.

        Args:
            pattern: Dict-based query pattern.

        Returns:
            QueryResult with matched entities, relationships, and metadata.

        Examples:
            >>> # Find all Person entities
            >>> result = await graph.match({"type": "Person"})
            >>> print(result.entities)

            >>> # Find specific relationship
            >>> result = await graph.match({
            ...     "source": {"id": "alice"},
            ...     "relation": "knows",
            ...     "target": {"type": "Person"}
            ... })
            >>> print(result.relationships)

            >>> # Multi-hop path query
            >>> result = await graph.match({
            ...     "path": [
            ...         {"source": {"type": "Person"}, "relation": "works_at"},
            ...         {"relation": "manages", "target": {"id": "project_x"}}
            ...     ]
            ... })
            >>> print(f"Found {len(result.entities)} entities in path")
        """
        from cogent.graph.query import match as query_match
        return await query_match(self, pattern)

    # --- Graph Statistics and Utilities ---

    async def stats(self) -> dict[str, int]:
        """Get graph statistics.

        Returns:
            Dictionary with entity_count and relationship_count.

        Example:
            >>> stats = await graph.stats()
            >>> print(f"Entities: {stats['entity_count']}")
        """
        return await self.storage.stats()

    async def clear(self) -> None:
        """Remove all entities and relationships.

        Example:
            >>> await graph.clear()
        """
        await self.storage.clear()
        await self.engine.clear()

    async def node_count(self) -> int:
        """Get number of nodes in the graph.

        Returns:
            Number of nodes.

        Example:
            >>> count = await graph.node_count()
        """
        return await self.engine.node_count()

    async def edge_count(self) -> int:
        """Get number of edges in the graph.

        Returns:
            Number of edges.

        Example:
            >>> count = await graph.edge_count()
        """
        return await self.engine.edge_count()

    # --- Visualization Methods ---

    async def to_mermaid(
        self,
        direction: str = "LR",
        group_by_type: bool = False,
        scheme: str = "default",
        title: str | None = None,
    ) -> str:
        """Generate Mermaid diagram for the graph.

        Args:
            direction: Flow direction ("LR" for left-to-right, "TB" for top-to-bottom).
            group_by_type: If True, group entities by type in subgraphs.
            scheme: Style scheme name ("default", "minimal").
            title: Optional diagram title.

        Returns:
            Mermaid diagram code as string.

        Example:
            >>> diagram = await graph.to_mermaid(direction="TB", group_by_type=True)
            >>> print(diagram)
        """
        from cogent.graph.visualization import to_mermaid

        entities = await self.get_all_entities()
        relationships = await self.get_relationships()

        return to_mermaid(
            entities,
            relationships,
            direction=direction,
            group_by_type=group_by_type,
            scheme=scheme,
            title=title,
        )

    async def save_diagram(
        self,
        file_path: str,
        format: str = "mermaid",
        **options: Any,
    ) -> None:
        """Save graph diagram to a file.

        Args:
            file_path: Path to save the file.
            format: Output format ("mermaid").
            **options: Format-specific options (direction, group_by_type, scheme, title).

        Raises:
            ValueError: If format is not recognized.

        Example:
            >>> await graph.save_diagram("graph.mmd", format="mermaid", direction="TB")
        """
        from cogent.graph.visualization import save_diagram

        # Generate content based on format
        if format == "mermaid":
            content = await self.to_mermaid(
                direction=options.get("direction", "LR"),
                group_by_type=options.get("group_by_type", False),
                scheme=options.get("scheme", "default"),
                title=options.get("title"),
            )
        else:
            raise ValueError(
                f"Unknown format: {format}. Use: mermaid"
            )

        save_diagram(content, file_path, format=format)

    async def render_to_image(
        self,
        output_path: str,
        format: str = "png",
        diagram_format: str = "mermaid",
        width: int = 1920,
        height: int = 1080,
        **diagram_options: Any,
    ) -> None:
        """Render graph to image file (PNG, SVG, PDF).

        Requires Playwright to be installed:
            uv add playwright
            uv run playwright install chromium

        Args:
            output_path: Path to save the image.
            format: Image format ("png", "svg", "pdf").
            diagram_format: Diagram format to render ("mermaid" only for now).
            width: Viewport width in pixels.
            height: Viewport height in pixels.
            **diagram_options: Options for diagram generation (direction, group_by_type, etc.).

        Raises:
            ImportError: If Playwright is not installed.
            ValueError: If diagram format is not supported.

        Example:
            >>> await graph.render_to_image("graph.png", direction="TB", group_by_type=True)
            >>> await graph.render_to_image("graph.svg", format="svg")
            >>> await graph.render_to_image("graph.pdf", format="pdf")
        """
        if diagram_format != "mermaid":
            raise ValueError("Only 'mermaid' diagram format is supported for image rendering")

        from cogent.graph.visualization import render_mermaid_to_image

        # Generate Mermaid diagram
        diagram = await self.to_mermaid(
            direction=diagram_options.get("direction", "LR"),
            group_by_type=diagram_options.get("group_by_type", False),
            scheme=diagram_options.get("scheme", "default"),
            title=diagram_options.get("title"),
        )

        # Render to image
        await render_mermaid_to_image(
            diagram, output_path, format=format, width=width, height=height
        )
