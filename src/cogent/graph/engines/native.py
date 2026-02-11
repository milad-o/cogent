"""Pure Python graph engine implementation.

This module provides NativeEngine - a lightweight, dependency-free graph
engine using native Python data structures (dicts and lists).
"""

from collections import deque


class NativeEngine:
    """Pure Python graph engine with zero dependencies.

    Uses native Python data structures for graph storage and implements
    graph algorithms from scratch. Suitable as a lightweight default or
    fallback when NetworkX is unavailable.

    Storage:
        - Nodes: dict[node_id, attributes]
        - Adjacency: dict[node_id, list[neighbor_ids]]
        - Edges: dict[(source, target), attributes]

    Example:
        >>> engine = NativeEngine()
        >>> await engine.add_node("alice", name="Alice")
        >>> await engine.add_edge("alice", "bob", relation="knows")
        >>> path = await engine.find_path("alice", "bob")
    """

    def __init__(self) -> None:
        """Initialize empty graph."""
        self._nodes: dict[str, dict[str, object]] = {}
        self._adjacency: dict[str, list[str]] = {}
        self._edges: dict[tuple[str, str], dict[str, object]] = {}

    # --- Node Operations ---

    async def add_node(self, node_id: str, **attributes: object) -> None:
        """Add a node to the graph."""
        self._nodes[node_id] = dict(attributes)
        if node_id not in self._adjacency:
            self._adjacency[node_id] = []

    async def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            return False

        # Remove node
        del self._nodes[node_id]
        del self._adjacency[node_id]

        # Remove all edges involving this node
        edges_to_remove = [
            (src, tgt)
            for src, tgt in self._edges
            if src == node_id or tgt == node_id
        ]
        for edge in edges_to_remove:
            del self._edges[edge]

        # Remove from adjacency lists
        for neighbors in self._adjacency.values():
            if node_id in neighbors:
                neighbors.remove(node_id)

        return True

    async def get_node(self, node_id: str) -> dict[str, object] | None:
        """Get node attributes."""
        return self._nodes.get(node_id)

    async def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self._nodes

    async def get_neighbors(
        self,
        node_id: str,
        direction: str = "outgoing",
    ) -> list[str]:
        """Get neighboring node IDs."""
        if node_id not in self._nodes:
            return []

        if direction == "outgoing":
            return list(self._adjacency.get(node_id, []))

        if direction == "incoming":
            return [
                src
                for src, neighbors in self._adjacency.items()
                if node_id in neighbors
            ]

        if direction == "both":
            outgoing = set(self._adjacency.get(node_id, []))
            incoming = {
                src
                for src, neighbors in self._adjacency.items()
                if node_id in neighbors
            }
            return list(outgoing | incoming)

        raise ValueError(f"Invalid direction: {direction}")

    async def node_count(self) -> int:
        """Get total number of nodes."""
        return len(self._nodes)

    # --- Edge Operations ---

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        **attributes: object,
    ) -> None:
        """Add an edge between two nodes."""
        # Ensure nodes exist
        if source_id not in self._nodes:
            await self.add_node(source_id)
        if target_id not in self._nodes:
            await self.add_node(target_id)

        # Add edge
        self._edges[(source_id, target_id)] = dict(attributes)

        # Update adjacency list
        if target_id not in self._adjacency[source_id]:
            self._adjacency[source_id].append(target_id)

    async def remove_edge(self, source_id: str, target_id: str) -> bool:
        """Remove an edge between two nodes."""
        edge_key = (source_id, target_id)
        if edge_key not in self._edges:
            return False

        del self._edges[edge_key]
        if source_id in self._adjacency and target_id in self._adjacency[source_id]:
            self._adjacency[source_id].remove(target_id)

        return True

    async def get_edges(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
    ) -> list[tuple[str, str, dict[str, object]]]:
        """Get edges matching criteria."""
        results = []

        for (src, tgt), attrs in self._edges.items():
            # Apply filters
            if source_id is not None and src != source_id:
                continue
            if target_id is not None and tgt != target_id:
                continue

            results.append((src, tgt, attrs.copy()))

        return results

    async def edge_count(self) -> int:
        """Get total number of edges."""
        return len(self._edges)

    # --- Graph Operations ---

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int | None = None,
    ) -> list[str] | None:
        """Find shortest path using BFS.

        Implements breadth-first search from scratch to find the
        shortest path between two nodes.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        if source_id == target_id:
            return [source_id]

        # BFS with path tracking
        queue: deque[tuple[str, list[str]]] = deque([(source_id, [source_id])])
        visited: set[str] = {source_id}

        while queue:
            current, path = queue.popleft()

            # Check max depth
            if max_depth is not None and len(path) > max_depth:
                continue

            # Explore neighbors
            for neighbor in self._adjacency.get(current, []):
                if neighbor == target_id:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    async def connected_components(self) -> list[set[str]]:
        """Find all weakly connected components using DFS.

        Treats the graph as undirected for component detection.
        """
        visited: set[str] = set()
        components: list[set[str]] = []

        def dfs(node_id: str, component: set[str]) -> None:
            """Depth-first search to find component."""
            visited.add(node_id)
            component.add(node_id)

            # Explore outgoing edges
            for neighbor in self._adjacency.get(node_id, []):
                if neighbor not in visited:
                    dfs(neighbor, component)

            # Explore incoming edges (treat as undirected)
            for src, neighbors in self._adjacency.items():
                if node_id in neighbors and src not in visited:
                    dfs(src, component)

        for node_id in self._nodes:
            if node_id not in visited:
                component: set[str] = set()
                dfs(node_id, component)
                components.append(component)

        return components

    async def get_subgraph(self, node_ids: set[str]) -> "NativeEngine":
        """Extract subgraph containing specified nodes."""
        subgraph = NativeEngine()

        # Add nodes
        for node_id in node_ids:
            if node_id in self._nodes:
                await subgraph.add_node(node_id, **self._nodes[node_id])

        # Add edges between included nodes
        for (src, tgt), attrs in self._edges.items():
            if src in node_ids and tgt in node_ids:
                await subgraph.add_edge(src, tgt, **attrs)

        return subgraph

    async def clear(self) -> None:
        """Remove all nodes and edges."""
        self._nodes.clear()
        self._adjacency.clear()
        self._edges.clear()
