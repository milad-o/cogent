"""
Low-level graph primitives - Nodes, Edges, and Graphs.

These are backend-agnostic data structures that represent graph elements.
Backends consume these primitives to generate output in their specific format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeShape(str, Enum):
    """Node shape types."""

    RECTANGLE = "rectangle"
    ROUNDED = "rounded"
    CIRCLE = "circle"
    DIAMOND = "diamond"
    HEXAGON = "hexagon"
    PARALLELOGRAM = "parallelogram"
    CYLINDER = "cylinder"
    STADIUM = "stadium"  # Pill shape
    SUBROUTINE = "subroutine"  # Double border


@dataclass(slots=True, kw_only=True)
class NodeStyle:
    """Visual styling for nodes.

    Attributes:
        fill: Background fill color (hex or named).
        stroke: Border stroke color.
        stroke_width: Border width in pixels.
        color: Text color.
        font_size: Font size in pixels.
        font_weight: Font weight (normal, bold).
        opacity: Node opacity (0.0 to 1.0).
        dashed: Whether border is dashed.
    """

    fill: str | None = None
    stroke: str | None = None
    stroke_width: int | None = None
    color: str | None = None
    font_size: int | None = None
    font_weight: str | None = None
    opacity: float | None = None
    dashed: bool = False


@dataclass(slots=True, kw_only=True)
class EdgeStyle:
    """Visual styling for edges.

    Attributes:
        stroke: Line color.
        stroke_width: Line width in pixels.
        stroke_dasharray: Dash pattern (e.g., "5,5").
        opacity: Line opacity.
        animated: Whether edge should animate.
    """

    stroke: str | None = None
    stroke_width: int | None = None
    stroke_dasharray: str | None = None
    opacity: float | None = None
    animated: bool = False


class EdgeType(str, Enum):
    """Edge arrow/line types."""

    ARROW = "arrow"  # -->
    OPEN = "open"  # ---
    DOTTED = "dotted"  # -.->
    THICK = "thick"  # ==>
    BIDIRECTIONAL = "bidirectional"  # <-->


@dataclass(slots=True, kw_only=True)
class Node:
    """A graph node.

    Attributes:
        id: Unique node identifier (must be valid identifier).
        label: Display label (supports HTML in some backends).
        shape: Node shape.
        style: Visual styling.
        css_class: CSS class name for styling.
        metadata: Additional node metadata.
        tooltip: Hover tooltip text.
        url: Click URL (supported by some backends).
    """

    id: str
    label: str | None = None
    shape: NodeShape = NodeShape.ROUNDED
    style: NodeStyle | None = None
    css_class: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tooltip: str | None = None
    url: str | None = None

    def __post_init__(self) -> None:
        """Validate node ID."""
        if not self.id:
            raise ValueError("Node ID cannot be empty")
        # Use label as ID if not provided
        if self.label is None:
            self.label = self.id

    @property
    def display_label(self) -> str:
        """Get display label, falling back to ID."""
        return self.label or self.id


@dataclass(slots=True, kw_only=True)
class Edge:
    """A graph edge connecting two nodes.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        label: Edge label text.
        edge_type: Type of edge (arrow, dotted, etc.).
        style: Visual styling.
        metadata: Additional edge metadata.
    """

    source: str
    target: str
    label: str | None = None
    edge_type: EdgeType = EdgeType.ARROW
    style: EdgeStyle | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate edge."""
        if not self.source or not self.target:
            raise ValueError("Edge source and target cannot be empty")


@dataclass(slots=True, kw_only=True)
class Subgraph:
    """A subgraph (group of nodes).

    Attributes:
        id: Unique subgraph identifier.
        label: Display label for the subgraph.
        node_ids: IDs of nodes contained in this subgraph.
        direction: Internal layout direction (overrides parent).
        style: Visual styling for the subgraph container.
    """

    id: str
    label: str | None = None
    node_ids: list[str] = field(default_factory=list)
    direction: str | None = None
    style: NodeStyle | None = None


@dataclass(slots=True, kw_only=True)
class ClassDef:
    """A CSS class definition for nodes.

    Attributes:
        name: Class name (used with :::className syntax in Mermaid).
        style: Style to apply to nodes with this class.
    """

    name: str
    style: NodeStyle


@dataclass(slots=True)
class Graph:
    """A complete graph structure.

    This is the core data structure that backends render.
    Supports nodes, edges, subgraphs, and class definitions.

    Example:
        >>> g = Graph()
        >>> g.add_node(Node(id="a", label="Agent A"))
        >>> g.add_node(Node(id="b", label="Agent B"))
        >>> g.add_edge(Edge(source="a", target="b"))
        >>> g.render()  # Uses default Mermaid backend
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    subgraphs: dict[str, Subgraph] = field(default_factory=dict)
    class_defs: dict[str, ClassDef] = field(default_factory=dict)

    def add_node(self, node: Node) -> "Graph":
        """Add a node to the graph.

        Args:
            node: Node to add.

        Returns:
            Self for chaining.
        """
        self.nodes[node.id] = node
        return self

    def add_edge(self, edge: Edge) -> "Graph":
        """Add an edge to the graph.

        Args:
            edge: Edge to add.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If source or target node doesn't exist.
        """
        if edge.source not in self.nodes:
            raise ValueError(f"Source node '{edge.source}' not in graph")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node '{edge.target}' not in graph")
        self.edges.append(edge)
        return self

    def add_subgraph(self, subgraph: Subgraph) -> "Graph":
        """Add a subgraph to the graph.

        Args:
            subgraph: Subgraph to add.

        Returns:
            Self for chaining.
        """
        self.subgraphs[subgraph.id] = subgraph
        return self

    def add_class_def(self, class_def: ClassDef) -> "Graph":
        """Add a class definition.

        Args:
            class_def: Class definition to add.

        Returns:
            Self for chaining.
        """
        self.class_defs[class_def.name] = class_def
        return self

    def node(
        self,
        id: str,
        label: str | None = None,
        *,
        shape: NodeShape = NodeShape.ROUNDED,
        css_class: str | None = None,
        **metadata: Any,
    ) -> "Graph":
        """Fluent method to add a node.

        Args:
            id: Node ID.
            label: Display label.
            shape: Node shape.
            css_class: CSS class for styling.
            **metadata: Additional metadata.

        Returns:
            Self for chaining.
        """
        self.add_node(
            Node(
                id=id,
                label=label,
                shape=shape,
                css_class=css_class,
                metadata=metadata,
            )
        )
        return self

    def edge(
        self,
        source: str,
        target: str,
        label: str | None = None,
        *,
        edge_type: EdgeType = EdgeType.ARROW,
    ) -> "Graph":
        """Fluent method to add an edge.

        Args:
            source: Source node ID.
            target: Target node ID.
            label: Edge label.
            edge_type: Type of edge.

        Returns:
            Self for chaining.
        """
        self.add_edge(
            Edge(source=source, target=target, label=label, edge_type=edge_type)
        )
        return self

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID.

        Args:
            node_id: Node ID to look up.

        Returns:
            Node if found, None otherwise.
        """
        return self.nodes.get(node_id)

    def get_edges_from(self, node_id: str) -> list[Edge]:
        """Get all edges originating from a node.

        Args:
            node_id: Source node ID.

        Returns:
            List of edges from this node.
        """
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> list[Edge]:
        """Get all edges targeting a node.

        Args:
            node_id: Target node ID.

        Returns:
            List of edges to this node.
        """
        return [e for e in self.edges if e.target == node_id]

    def validate(self) -> list[str]:
        """Validate graph structure.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        # Check edge references
        for edge in self.edges:
            if edge.source not in self.nodes:
                errors.append(f"Edge references missing source node: {edge.source}")
            if edge.target not in self.nodes:
                errors.append(f"Edge references missing target node: {edge.target}")

        # Check subgraph references
        for sg in self.subgraphs.values():
            for node_id in sg.node_ids:
                if node_id not in self.nodes:
                    errors.append(
                        f"Subgraph '{sg.id}' references missing node: {node_id}"
                    )

        return errors

    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if node exists."""
        return node_id in self.nodes
