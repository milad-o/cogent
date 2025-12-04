"""
Graph rendering backends.

Backends transform Graph primitives into specific output formats:
- MermaidBackend: Mermaid diagram syntax (default)
- GraphvizBackend: DOT/Graphviz format
- ASCIIBackend: Terminal-friendly text diagrams
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from agenticflow.graph.config import GraphConfig, GraphDirection, GraphTheme
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

if TYPE_CHECKING:
    pass


class Backend(ABC):
    """Abstract base class for graph rendering backends.

    Backends transform a Graph into a specific output format.
    Each backend implements its own rendering logic.
    """

    @abstractmethod
    def render(self, graph: Graph, config: GraphConfig | None = None) -> str:
        """Render graph to string output.

        Args:
            graph: Graph to render.
            config: Optional rendering configuration.

        Returns:
            Rendered string in backend-specific format.
        """
        ...

    @abstractmethod
    def get_format_name(self) -> str:
        """Get the name of this format.

        Returns:
            Format name (e.g., "mermaid", "graphviz", "ascii").
        """
        ...

    def render_to_file(
        self,
        graph: Graph,
        path: str,
        config: GraphConfig | None = None,
    ) -> None:
        """Render graph and save to file.

        Args:
            graph: Graph to render.
            path: Output file path.
            config: Optional rendering configuration.
        """
        content = self.render(graph, config)
        with open(path, "w") as f:
            f.write(content)


class MermaidBackend(Backend):
    """Mermaid diagram backend (default).

    Renders graphs as Mermaid flowchart syntax.
    Supports rendering to URL, PNG, SVG, and HTML.
    """

    MERMAID_INK_URL = "https://mermaid.ink"

    # Shape mappings to Mermaid syntax
    SHAPE_MAP: dict[NodeShape, tuple[str, str]] = {
        NodeShape.RECTANGLE: ("[", "]"),
        NodeShape.ROUNDED: ("(", ")"),
        NodeShape.CIRCLE: ("((", "))"),
        NodeShape.DIAMOND: ("{", "}"),
        NodeShape.HEXAGON: ("{{", "}}"),
        NodeShape.PARALLELOGRAM: ("[/", "/]"),
        NodeShape.CYLINDER: ("[(", ")]"),
        NodeShape.STADIUM: ("([", "])"),
        NodeShape.SUBROUTINE: ("[[", "]]"),
    }

    # Edge type mappings
    EDGE_MAP: dict[EdgeType, str] = {
        EdgeType.ARROW: "-->",
        EdgeType.OPEN: "---",
        EdgeType.DOTTED: "-.->",
        EdgeType.THICK: "==>",
        EdgeType.BIDIRECTIONAL: "<-->",
    }

    def get_format_name(self) -> str:
        """Get format name."""
        return "mermaid"

    def render(self, graph: Graph, config: GraphConfig | None = None) -> str:
        """Render graph as Mermaid code.

        Args:
            graph: Graph to render.
            config: Optional configuration.

        Returns:
            Mermaid flowchart code.
        """
        cfg = config or GraphConfig()
        lines: list[str] = []

        # Add frontmatter if needed
        if cfg.title or cfg.theme != GraphTheme.DEFAULT:
            lines.append(self._render_frontmatter(cfg))

        # Start flowchart
        lines.append(f"flowchart {cfg.direction.value}")

        # Collect nodes in subgraphs
        nodes_in_subgraphs: set[str] = set()
        for sg in graph.subgraphs.values():
            nodes_in_subgraphs.update(sg.node_ids)

        # Render subgraphs first
        for sg in graph.subgraphs.values():
            lines.extend(self._render_subgraph(sg, graph))
            lines.append("")

        # Render nodes not in subgraphs
        for node_id, node in graph.nodes.items():
            if node_id not in nodes_in_subgraphs:
                lines.append(f"    {self._render_node(node)}")

        lines.append("")

        # Render edges
        for edge in graph.edges:
            lines.append(f"    {self._render_edge(edge)}")

        # Render class definitions
        if graph.class_defs:
            lines.append("")
            for class_def in graph.class_defs.values():
                lines.append(f"    {self._render_class_def(class_def)}")

        return "\n".join(lines)

    def _render_frontmatter(self, cfg: GraphConfig) -> str:
        """Render YAML frontmatter."""
        lines = ["---"]

        if cfg.title:
            # Quote if contains special chars
            if any(c in cfg.title for c in ":{}[]|>&*!?#"):
                lines.append(f'title: "{cfg.title}"')
            else:
                lines.append(f"title: {cfg.title}")

        lines.append("config:")
        lines.append(f"  theme: {cfg.theme.value}")
        lines.append("  flowchart:")
        lines.append(f"    curve: {cfg.curve}")
        lines.append(f"    nodeSpacing: {cfg.node_spacing}")
        lines.append(f"    rankSpacing: {cfg.rank_spacing}")
        lines.append(f"    padding: {cfg.padding}")
        lines.append("---")

        return "\n".join(lines)

    def _render_node(self, node: Node) -> str:
        """Render a single node."""
        # Get shape delimiters
        left, right = self.SHAPE_MAP.get(node.shape, ("[", "]"))

        # Escape label
        label = self._escape_label(node.display_label)

        # Build node string
        node_str = f'{node.id}{left}"{label}"{right}'

        # Add class if specified
        if node.css_class:
            node_str += f":::{node.css_class}"

        return node_str

    def _render_edge(self, edge: Edge) -> str:
        """Render a single edge."""
        arrow = self.EDGE_MAP.get(edge.edge_type, "-->")

        if edge.label:
            escaped_label = self._escape_label(edge.label)
            return f"{edge.source} {arrow}|{escaped_label}| {edge.target}"
        else:
            return f"{edge.source} {arrow} {edge.target}"

    def _render_subgraph(self, sg: Subgraph, graph: Graph) -> list[str]:
        """Render a subgraph."""
        lines = []

        label = sg.label or sg.id
        lines.append(f'    subgraph {sg.id}["{label}"]')

        if sg.direction:
            lines.append(f"        direction {sg.direction}")

        for node_id in sg.node_ids:
            node = graph.nodes.get(node_id)
            if node:
                lines.append(f"        {self._render_node(node)}")

        lines.append("    end")

        return lines

    def _render_class_def(self, class_def: ClassDef) -> str:
        """Render a class definition."""
        style_parts = []
        style = class_def.style

        if style.fill:
            style_parts.append(f"fill:{style.fill}")
        if style.stroke:
            style_parts.append(f"stroke:{style.stroke}")
        if style.color:
            style_parts.append(f"color:{style.color}")
        if style.stroke_width:
            style_parts.append(f"stroke-width:{style.stroke_width}px")
        if style.dashed:
            style_parts.append("stroke-dasharray:5 5")

        return f"classDef {class_def.name} {','.join(style_parts)}"

    def _escape_label(self, text: str) -> str:
        """Escape text for Mermaid labels."""
        text = text.replace('"', "&quot;")
        text = text.replace("(", "&#40;")
        text = text.replace(")", "&#41;")
        text = text.replace("[", "&#91;")
        text = text.replace("]", "&#93;")
        text = text.replace("{", "&#123;")
        text = text.replace("}", "&#125;")
        return text

    # --- Rendering methods ---

    def to_url(self, graph: Graph, config: GraphConfig | None = None) -> str:
        """Get mermaid.ink SVG URL.

        Args:
            graph: Graph to render.
            config: Optional configuration.

        Returns:
            URL to SVG rendering.
        """
        code = self.render(graph, config)
        encoded = base64.urlsafe_b64encode(code.encode()).decode()
        return f"{self.MERMAID_INK_URL}/svg/{encoded}"

    def to_png_url(self, graph: Graph, config: GraphConfig | None = None) -> str:
        """Get mermaid.ink PNG URL.

        Args:
            graph: Graph to render.
            config: Optional configuration.

        Returns:
            URL to PNG rendering.
        """
        code = self.render(graph, config)
        encoded = base64.urlsafe_b64encode(code.encode()).decode()
        return f"{self.MERMAID_INK_URL}/img/{encoded}"

    def draw_png(self, graph: Graph, config: GraphConfig | None = None) -> bytes:
        """Render as PNG bytes.

        Args:
            graph: Graph to render.
            config: Optional configuration.

        Returns:
            PNG image bytes.

        Raises:
            ImportError: If httpx not installed.
            RuntimeError: If rendering fails.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx required for PNG rendering. Install with: uv add httpx"
            )

        url = self.to_png_url(graph, config)

        try:
            response = httpx.get(url, timeout=30, follow_redirects=True)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to render diagram: {e}")

    def to_html(self, graph: Graph, config: GraphConfig | None = None) -> str:
        """Generate HTML with embedded Mermaid.

        Args:
            graph: Graph to render.
            config: Optional configuration.

        Returns:
            HTML string.
        """
        code = self.render(graph, config)
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        return f"""<div class="mermaid">
{escaped}
</div>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({{ startOnLoad: true }});
</script>"""


class GraphvizBackend(Backend):
    """Graphviz DOT format backend.

    Renders graphs as DOT syntax for Graphviz.
    Requires graphviz to be installed for image rendering.
    """

    # Shape mappings to Graphviz
    SHAPE_MAP: dict[NodeShape, str] = {
        NodeShape.RECTANGLE: "box",
        NodeShape.ROUNDED: "box",  # with style=rounded
        NodeShape.CIRCLE: "circle",
        NodeShape.DIAMOND: "diamond",
        NodeShape.HEXAGON: "hexagon",
        NodeShape.PARALLELOGRAM: "parallelogram",
        NodeShape.CYLINDER: "cylinder",
        NodeShape.STADIUM: "stadium",
        NodeShape.SUBROUTINE: "box",  # with double border
    }

    # Direction mappings
    DIRECTION_MAP: dict[GraphDirection, str] = {
        GraphDirection.TOP_DOWN: "TB",
        GraphDirection.BOTTOM_UP: "BT",
        GraphDirection.LEFT_RIGHT: "LR",
        GraphDirection.RIGHT_LEFT: "RL",
    }

    def get_format_name(self) -> str:
        """Get format name."""
        return "graphviz"

    def render(self, graph: Graph, config: GraphConfig | None = None) -> str:
        """Render graph as DOT code.

        Args:
            graph: Graph to render.
            config: Optional configuration.

        Returns:
            DOT/Graphviz code.
        """
        cfg = config or GraphConfig()
        lines: list[str] = []

        # Start digraph
        lines.append("digraph G {")

        # Graph attributes
        direction = self.DIRECTION_MAP.get(cfg.direction, "TB")
        lines.append(f'    rankdir="{direction}";')
        lines.append(f'    nodesep={cfg.node_spacing / 72:.2f};')  # Convert to inches
        lines.append(f'    ranksep={cfg.rank_spacing / 72:.2f};')

        if cfg.title:
            lines.append(f'    label="{cfg.title}";')
            lines.append("    labelloc=t;")

        # Default node style
        lines.append('    node [fontname="Arial", fontsize=12];')
        lines.append("")

        # Render nodes
        for node in graph.nodes.values():
            lines.append(f"    {self._render_node(node)}")

        lines.append("")

        # Render subgraphs
        for sg in graph.subgraphs.values():
            lines.extend(self._render_subgraph(sg, graph))

        # Render edges
        for edge in graph.edges:
            lines.append(f"    {self._render_edge(edge)}")

        lines.append("}")

        return "\n".join(lines)

    def _render_node(self, node: Node) -> str:
        """Render a single node."""
        attrs = []

        # Label
        attrs.append(f'label="{node.display_label}"')

        # Shape
        shape = self.SHAPE_MAP.get(node.shape, "box")
        attrs.append(f"shape={shape}")

        # Rounded for rounded rectangle
        if node.shape == NodeShape.ROUNDED:
            attrs.append("style=rounded")

        # Style attributes
        if node.style:
            if node.style.fill:
                attrs.append(f'fillcolor="{node.style.fill}"')
                attrs.append("style=filled")
            if node.style.stroke:
                attrs.append(f'color="{node.style.stroke}"')
            if node.style.color:
                attrs.append(f'fontcolor="{node.style.color}"')

        return f'{node.id} [{", ".join(attrs)}];'

    def _render_edge(self, edge: Edge) -> str:
        """Render a single edge."""
        attrs = []

        if edge.label:
            attrs.append(f'label="{edge.label}"')

        # Edge type styling
        if edge.edge_type == EdgeType.DOTTED:
            attrs.append("style=dotted")
        elif edge.edge_type == EdgeType.THICK:
            attrs.append("penwidth=2.0")
        elif edge.edge_type == EdgeType.OPEN:
            attrs.append("arrowhead=none")
        elif edge.edge_type == EdgeType.BIDIRECTIONAL:
            attrs.append("dir=both")

        attrs_str = f" [{', '.join(attrs)}]" if attrs else ""
        return f"{edge.source} -> {edge.target}{attrs_str};"

    def _render_subgraph(self, sg: Subgraph, graph: Graph) -> list[str]:
        """Render a subgraph (cluster)."""
        lines = []

        lines.append(f"    subgraph cluster_{sg.id} {{")
        if sg.label:
            lines.append(f'        label="{sg.label}";')

        for node_id in sg.node_ids:
            lines.append(f"        {node_id};")

        lines.append("    }")

        return lines

    def draw_png(self, graph: Graph, config: GraphConfig | None = None) -> bytes:
        """Render as PNG using graphviz.

        Args:
            graph: Graph to render.
            config: Optional configuration.

        Returns:
            PNG image bytes.

        Raises:
            ImportError: If graphviz not installed.
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "graphviz required. Install with: uv add graphviz"
            )

        dot_code = self.render(graph, config)
        gv = graphviz.Source(dot_code)
        return gv.pipe(format="png")

    def draw_svg(self, graph: Graph, config: GraphConfig | None = None) -> str:
        """Render as SVG string using graphviz.

        Args:
            graph: Graph to render.
            config: Optional configuration.

        Returns:
            SVG string.
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "graphviz required. Install with: uv add graphviz"
            )

        dot_code = self.render(graph, config)
        gv = graphviz.Source(dot_code)
        return gv.pipe(format="svg").decode()


class ASCIIBackend(Backend):
    """ASCII art backend for terminal display.

    Renders simple text-based diagrams suitable for terminals
    and environments without graphics support.
    """

    def get_format_name(self) -> str:
        """Get format name."""
        return "ascii"

    def render(self, graph: Graph, config: GraphConfig | None = None) -> str:
        """Render graph as ASCII art.

        Args:
            graph: Graph to render.
            config: Optional configuration.

        Returns:
            ASCII diagram string.
        """
        cfg = config or GraphConfig()
        lines: list[str] = []

        # Title
        if cfg.title:
            lines.append(f"┌{'─' * (len(cfg.title) + 2)}┐")
            lines.append(f"│ {cfg.title} │")
            lines.append(f"└{'─' * (len(cfg.title) + 2)}┘")
            lines.append("")

        # Determine layout direction
        horizontal = cfg.direction in (
            GraphDirection.LEFT_RIGHT,
            GraphDirection.RIGHT_LEFT,
        )

        if horizontal:
            lines.extend(self._render_horizontal(graph))
        else:
            lines.extend(self._render_vertical(graph))

        return "\n".join(lines)

    def _render_vertical(self, graph: Graph) -> list[str]:
        """Render nodes vertically with connections."""
        lines: list[str] = []

        # Get node order from edges (topological-ish)
        ordered = self._get_ordered_nodes(graph)

        # Track which nodes connect to which
        connections: dict[str, list[str]] = {}
        for edge in graph.edges:
            connections.setdefault(edge.source, []).append(edge.target)

        for i, node_id in enumerate(ordered):
            node = graph.nodes[node_id]
            label = node.display_label

            # Draw node box
            box_width = len(label) + 4
            lines.append(f"┌{'─' * box_width}┐")
            lines.append(f"│  {label}  │")
            lines.append(f"└{'─' * box_width}┘")

            # Draw connection to next if exists
            if node_id in connections:
                targets = connections[node_id]
                if targets:
                    lines.append("        │")
                    lines.append("        ▼")

        return lines

    def _render_horizontal(self, graph: Graph) -> list[str]:
        """Render nodes horizontally with connections."""
        ordered = self._get_ordered_nodes(graph)

        # Build horizontal layout
        node_boxes: list[list[str]] = []

        for node_id in ordered:
            node = graph.nodes[node_id]
            label = node.display_label
            width = len(label) + 4

            box = [
                f"┌{'─' * width}┐",
                f"│  {label}  │",
                f"└{'─' * width}┘",
            ]
            node_boxes.append(box)

        # Combine horizontally with arrows
        if not node_boxes:
            return []

        max_height = max(len(box) for box in node_boxes)
        result_lines: list[str] = ["" for _ in range(max_height)]

        for i, box in enumerate(node_boxes):
            # Pad shorter boxes
            while len(box) < max_height:
                width = len(box[0]) if box else 0
                box.insert(0, " " * width)

            for j, line in enumerate(box):
                result_lines[j] += line

            # Add arrow between nodes
            if i < len(node_boxes) - 1:
                for j in range(max_height):
                    if j == max_height // 2:
                        result_lines[j] += " ──▶ "
                    else:
                        result_lines[j] += "     "

        return result_lines

    def _get_ordered_nodes(self, graph: Graph) -> list[str]:
        """Get nodes in rough topological order."""
        # Simple ordering: nodes with no incoming edges first
        incoming: dict[str, int] = {n: 0 for n in graph.nodes}
        for edge in graph.edges:
            if edge.target in incoming:
                incoming[edge.target] += 1

        # Sort by incoming edge count
        ordered = sorted(graph.nodes.keys(), key=lambda n: incoming.get(n, 0))
        return ordered


# Default backend instance
_default_backend: Backend = MermaidBackend()


def get_default_backend() -> Backend:
    """Get the default rendering backend.

    Returns:
        Default Backend instance (MermaidBackend).
    """
    return _default_backend


def set_default_backend(backend: Backend) -> None:
    """Set the default rendering backend.

    Args:
        backend: Backend to use as default.
    """
    global _default_backend
    _default_backend = backend
