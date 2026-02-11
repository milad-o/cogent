"""Base Engine protocol for graph operations.

This module defines the Engine protocol that all graph engines must implement.
Engines handle graph operations, algorithms, and queries, while remaining
stateless - all data persistence is handled by Storage backends.
"""

from typing import Protocol


class Engine(Protocol):
    """Protocol defining the interface for graph execution engines.

    Engines provide graph operations and algorithms without managing
    persistence. They work with node IDs and attributes as simple dicts,
    keeping the implementation flexible and stateless.

    All methods use async to support both sync (NativeEngine) and
    async (NetworkXEngine with async operations) implementations.

    Node representation: Simple dict with arbitrary attributes
    Edge representation: (source_id, target_id, attributes_dict)

    Example:
        >>> class MyEngine:
        ...     async def add_node(self, node_id: str, **attributes) -> None:
        ...         pass  # Implementation
        ...
        >>> engine: Engine = MyEngine()  # Type-safe
    """

    # --- Node Operations ---

    async def add_node(self, node_id: str, **attributes: object) -> None:
        """Add a node to the graph.

        Args:
            node_id: Unique identifier for the node.
            **attributes: Arbitrary key-value pairs for node properties.

        Example:
            >>> await engine.add_node("person:alice", name="Alice", age=30)
        """
        ...

    async def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges.

        Args:
            node_id: ID of the node to remove.

        Returns:
            True if node was removed, False if it didn't exist.

        Example:
            >>> removed = await engine.remove_node("person:alice")
        """
        ...

    async def get_node(self, node_id: str) -> dict[str, object] | None:
        """Get node attributes.

        Args:
            node_id: ID of the node to retrieve.

        Returns:
            Dict of node attributes if found, None otherwise.

        Example:
            >>> node = await engine.get_node("person:alice")
            >>> if node:
            ...     print(node["name"])
        """
        ...

    async def has_node(self, node_id: str) -> bool:
        """Check if node exists.

        Args:
            node_id: ID of the node to check.

        Returns:
            True if node exists, False otherwise.

        Example:
            >>> if await engine.has_node("person:alice"):
            ...     print("Alice exists")
        """
        ...

    async def get_neighbors(
        self,
        node_id: str,
        direction: str = "outgoing",
    ) -> list[str]:
        """Get neighboring node IDs.

        Args:
            node_id: ID of the node.
            direction: "outgoing" (successors), "incoming" (predecessors),
                or "both" (all neighbors).

        Returns:
            List of neighbor node IDs.

        Example:
            >>> neighbors = await engine.get_neighbors("person:alice", "outgoing")
        """
        ...

    async def node_count(self) -> int:
        """Get total number of nodes.

        Returns:
            Number of nodes in the graph.

        Example:
            >>> count = await engine.node_count()
        """
        ...

    # --- Edge Operations ---

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        **attributes: object,
    ) -> None:
        """Add an edge between two nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            **attributes: Arbitrary key-value pairs for edge properties.

        Example:
            >>> await engine.add_edge(
            ...     "person:alice",
            ...     "person:bob",
            ...     relation="knows",
            ...     since=2020
            ... )
        """
        ...

    async def remove_edge(self, source_id: str, target_id: str) -> bool:
        """Remove an edge between two nodes.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.

        Returns:
            True if edge was removed, False if it didn't exist.

        Example:
            >>> removed = await engine.remove_edge("person:alice", "person:bob")
        """
        ...

    async def get_edges(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
    ) -> list[tuple[str, str, dict[str, object]]]:
        """Get edges matching criteria.

        Args:
            source_id: Filter by source node (optional).
            target_id: Filter by target node (optional).

        Returns:
            List of (source_id, target_id, attributes) tuples.

        Example:
            >>> # All edges
            >>> edges = await engine.get_edges()
            >>> # Edges from alice
            >>> edges = await engine.get_edges(source_id="person:alice")
            >>> # Edges to bob
            >>> edges = await engine.get_edges(target_id="person:bob")
        """
        ...

    async def edge_count(self) -> int:
        """Get total number of edges.

        Returns:
            Number of edges in the graph.

        Example:
            >>> count = await engine.edge_count()
        """
        ...

    # --- Graph Operations ---

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int | None = None,
    ) -> list[str] | None:
        """Find shortest path between two nodes.

        Args:
            source_id: Starting node ID.
            target_id: Destination node ID.
            max_depth: Maximum path length (optional).

        Returns:
            List of node IDs forming path (including source and target),
            or None if no path exists.

        Example:
            >>> path = await engine.find_path("person:alice", "person:charlie")
            >>> if path:
            ...     print(" -> ".join(path))
        """
        ...

    async def connected_components(self) -> list[set[str]]:
        """Find all weakly connected components.

        Returns:
            List of sets, each containing node IDs in a component.

        Example:
            >>> components = await engine.connected_components()
            >>> print(f"Graph has {len(components)} components")
        """
        ...

    async def get_subgraph(self, node_ids: set[str]) -> "Engine":
        """Extract subgraph containing specified nodes.

        Args:
            node_ids: Set of node IDs to include.

        Returns:
            New Engine instance containing only specified nodes and
            edges between them.

        Example:
            >>> subgraph = await engine.get_subgraph({"person:alice", "person:bob"})
        """
        ...

    async def clear(self) -> None:
        """Remove all nodes and edges from the graph.

        Example:
            >>> await engine.clear()
        """
        ...
