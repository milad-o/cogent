"""NetworkX-based graph engine implementation.

This module provides NetworkXEngine - a wrapper around NetworkX's DiGraph
that provides access to 100+ graph algorithms while maintaining the
Engine protocol interface.
"""

import networkx as nx


class NetworkXEngine:
    """NetworkX-based graph engine with rich algorithm support.

    Wraps NetworkX's DiGraph to provide the Engine protocol interface
    while enabling access to NetworkX's extensive algorithm library
    (shortest paths, centrality, community detection, etc.).

    Example:
        >>> if NETWORKX_AVAILABLE:
        ...     engine = NetworkXEngine()
        ...     await engine.add_node("alice", name="Alice")
        ...     # Access NetworkX algorithms
        ...     centrality = nx.betweenness_centrality(engine.graph)
    """

    def __init__(self) -> None:
        """Initialize NetworkX DiGraph."""
        self.graph = nx.DiGraph()

    # --- Node Operations ---

    async def add_node(self, node_id: str, **attributes: object) -> None:
        """Add a node to the graph."""
        self.graph.add_node(node_id, **attributes)

    async def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        if node_id not in self.graph:
            return False

        self.graph.remove_node(node_id)
        return True

    async def get_node(self, node_id: str) -> dict[str, object] | None:
        """Get node attributes."""
        if node_id not in self.graph:
            return None

        return dict(self.graph.nodes[node_id])

    async def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return self.graph.has_node(node_id)

    async def get_neighbors(
        self,
        node_id: str,
        direction: str = "outgoing",
    ) -> list[str]:
        """Get neighboring node IDs."""
        if node_id not in self.graph:
            return []

        if direction == "outgoing":
            return list(self.graph.successors(node_id))

        if direction == "incoming":
            return list(self.graph.predecessors(node_id))

        if direction == "both":
            successors = set(self.graph.successors(node_id))
            predecessors = set(self.graph.predecessors(node_id))
            return list(successors | predecessors)

        raise ValueError(f"Invalid direction: {direction}")

    async def node_count(self) -> int:
        """Get total number of nodes."""
        return self.graph.number_of_nodes()

    # --- Edge Operations ---

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        **attributes: object,
    ) -> None:
        """Add an edge between two nodes."""
        self.graph.add_edge(source_id, target_id, **attributes)

    async def remove_edge(self, source_id: str, target_id: str) -> bool:
        """Remove an edge between two nodes."""
        if not self.graph.has_edge(source_id, target_id):
            return False

        self.graph.remove_edge(source_id, target_id)
        return True

    async def get_edges(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
    ) -> list[tuple[str, str, dict[str, object]]]:
        """Get edges matching criteria."""
        results = []

        for src, tgt, attrs in self.graph.edges(data=True):
            # Apply filters
            if source_id is not None and src != source_id:
                continue
            if target_id is not None and tgt != target_id:
                continue

            results.append((src, tgt, dict(attrs)))

        return results

    async def edge_count(self) -> int:
        """Get total number of edges."""
        return self.graph.number_of_edges()

    # --- Graph Operations ---

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int | None = None,
    ) -> list[str] | None:
        """Find shortest path using NetworkX."""
        try:
            if max_depth is None:
                path = nx.shortest_path(self.graph, source_id, target_id)
            else:
                # Use single_source_shortest_path with cutoff for max_depth
                paths = nx.single_source_shortest_path(
                    self.graph,
                    source_id,
                    cutoff=max_depth,
                )
                # Check if target is reachable within max_depth
                if target_id not in paths:
                    return None
                path = paths[target_id]
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    async def connected_components(self) -> list[set[str]]:
        """Find all weakly connected components."""
        return [set(component) for component in nx.weakly_connected_components(self.graph)]

    async def get_subgraph(self, node_ids: set[str]) -> "NetworkXEngine":
        """Extract subgraph containing specified nodes."""
        subgraph_engine = NetworkXEngine()
        subgraph_engine.graph = self.graph.subgraph(node_ids).copy()
        return subgraph_engine

    async def clear(self) -> None:
        """Remove all nodes and edges."""
        self.graph.clear()

    # --- NetworkX-Specific Methods ---

    def betweenness_centrality(self) -> dict[str, float]:
        """Calculate betweenness centrality for all nodes.

        Returns:
            Dict mapping node IDs to centrality scores.

        Example:
            >>> centrality = engine.betweenness_centrality()
            >>> most_central = max(centrality, key=centrality.get)
        """
        return nx.betweenness_centrality(self.graph)

    def pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """Calculate PageRank for all nodes.

        Args:
            alpha: Damping parameter (default 0.85).

        Returns:
            Dict mapping node IDs to PageRank scores.

        Example:
            >>> scores = engine.pagerank()
        """
        return nx.pagerank(self.graph, alpha=alpha)

    def degree_centrality(self) -> dict[str, float]:
        """Calculate degree centrality for all nodes.

        Returns:
            Dict mapping node IDs to degree centrality scores.
        """
        return nx.degree_centrality(self.graph)

    def clustering_coefficient(self) -> dict[str, float]:
        """Calculate clustering coefficient for all nodes.

        Returns:
            Dict mapping node IDs to clustering coefficients.
        """
        # Convert to undirected for clustering
        undirected = self.graph.to_undirected()
        return nx.clustering(undirected)

    def all_simple_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int | None = None,
    ) -> list[list[str]]:
        """Find all simple paths between two nodes.

        Args:
            source_id: Starting node ID.
            target_id: Destination node ID.
            max_length: Maximum path length (optional).

        Returns:
            List of paths, each path is a list of node IDs.

        Example:
            >>> paths = engine.all_simple_paths("alice", "charlie", max_length=5)
        """
        try:
            paths = nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_length,
            )
            return list(paths)
        except nx.NodeNotFound:
            return []

    def is_directed_acyclic(self) -> bool:
        """Check if graph is a directed acyclic graph (DAG).

        Returns:
            True if graph has no cycles, False otherwise.

        Example:
            >>> if engine.is_directed_acyclic():
            ...     print("Graph is a DAG")
        """
        return nx.is_directed_acyclic_graph(self.graph)

    def topological_sort(self) -> list[str]:
        """Topologically sort nodes (if graph is a DAG).

        Returns:
            List of node IDs in topological order.

        Raises:
            NetworkXError: If graph has cycles.

        Example:
            >>> if engine.is_directed_acyclic():
            ...     order = engine.topological_sort()
        """
        return list(nx.topological_sort(self.graph))
