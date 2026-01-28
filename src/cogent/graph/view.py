"""GraphView - Unified visualization interface for graphable entities.

Provides a consistent API for rendering agents and topologies
to various formats without exposing internal graph primitives.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from cogent.graph.backends import (
    ASCIIBackend,
    GraphvizBackend,
    MermaidBackend,
)
from cogent.graph.config import GraphConfig
from cogent.graph.primitives import (
    ClassDef,
    Edge,
    EdgeType,
    Graph,
    Node,
    NodeShape,
    NodeStyle,
)

if TYPE_CHECKING:
    from cogent.agent.base import Agent


# Standard role styles (4-role system)
ROLE_STYLES: dict[str, str] = {
    "supervisor": "super",
    "worker": "work",
    "autonomous": "auto",
    "reviewer": "review",
}


def _make_node_id(name: str) -> str:
    """Create a valid node ID from a name."""
    node_id = "".join(c if c.isalnum() else "_" for c in name)
    if node_id and node_id[0].isdigit():
        node_id = "n_" + node_id
    return node_id or "node"


def _make_agent_label(name: str, role: str, tools: list[str] | None = None) -> str:
    """Create agent label with name, role, and optional tools."""
    parts = [name, f"<small><i>{role}</i></small>"]
    if tools:
        tools_str = ", ".join(tools[:3])
        if len(tools) > 3:
            tools_str += "..."
        parts.append(f"<small>{tools_str}</small>")
    return "<br/>".join(parts)


def _get_standard_class_defs() -> dict[str, ClassDef]:
    """Get standard class definitions for the 4-role system."""
    return {
        "super": ClassDef(
            name="super",
            style=NodeStyle(fill="#4a90d9", stroke="#2d5986", color="#fff"),
        ),
        "work": ClassDef(
            name="work",
            style=NodeStyle(fill="#7eb36a", stroke="#4a7a3d", color="#fff"),
        ),
        "auto": ClassDef(
            name="auto",
            style=NodeStyle(fill="#9b59b6", stroke="#7b3a96", color="#fff"),
        ),
        "review": ClassDef(
            name="review",
            style=NodeStyle(fill="#f56c6c", stroke="#c45656", color="#fff"),
        ),
        "tool": ClassDef(
            name="tool",
            style=NodeStyle(fill="#f5f5f5", stroke="#999", color="#333"),
        ),
        "config": ClassDef(
            name="config",
            style=NodeStyle(
                fill="#fff3e0", stroke="#ff9800", color="#333", dashed=True
            ),
        ),
        "start": ClassDef(
            name="start",
            style=NodeStyle(fill="#4ade80", stroke="#22c55e", color="#fff"),
        ),
        "end": ClassDef(
            name="end",
            style=NodeStyle(fill="#f87171", stroke="#ef4444", color="#fff"),
        ),
        "step": ClassDef(
            name="step",
            style=NodeStyle(fill="#60a5fa", stroke="#3b82f6", color="#fff"),
        ),
    }


class GraphView:
    """Unified graph visualization interface.

    Returned by `.to_graph()` on Agent and Topology objects.
    Provides consistent rendering API across all entity types.

    Example:
        >>> graph = agent.to_graph()
        >>> graph.mermaid()      # Get Mermaid code
        >>> graph.ascii()        # Terminal-friendly
        >>> graph.url()          # mermaid.ink URL
        >>> graph.save("out.png")  # Save to file
    """

    def __init__(
        self,
        graph: Graph,
        config: GraphConfig | None = None,
    ) -> None:
        """Initialize with internal graph representation.

        Args:
            graph: Internal Graph object (not exposed to users).
            config: Rendering configuration.
        """
        self._graph = graph
        self._config = config or GraphConfig()

    # ─────────────────────────────────────────────────────────────────────
    # Factory Methods (for internal use by Agent, Topology)
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_agent(
        cls,
        agent: Agent,
        *,
        show_tools: bool = True,
        show_config: bool = False,
    ) -> GraphView:
        """Create a graph from an agent.

        Args:
            agent: Agent to visualize.
            show_tools: Whether to show tools in label.
            show_config: Whether to show configuration node.

        Returns:
            GraphView instance.
        """
        g = Graph()

        # Get agent info
        agent_id = _make_node_id(agent.name)
        role = agent.role.value
        role_class = ROLE_STYLES.get(role, "work")

        # Build label
        tools = agent.config.tools if show_tools else None
        label = _make_agent_label(agent.name, role, tools)

        # Add agent node
        g.add_node(
            Node(
                id=agent_id,
                label=label,
                shape=NodeShape.ROUNDED,
                css_class=role_class,
            )
        )

        # Add config node if enabled
        if show_config and agent.config.model is not None:
            model_obj = agent.config.model
            model_info = (
                getattr(model_obj, "model_name", None)
                or getattr(model_obj, "model", "model")
                or "model"
            )
            config_id = f"cfg_{agent_id}"
            g.add_node(
                Node(
                    id=config_id,
                    label=str(model_info),
                    shape=NodeShape.DIAMOND,
                    css_class="config",
                )
            )
            g.add_edge(
                Edge(source=agent_id, target=config_id, edge_type=EdgeType.DOTTED)
            )

        # Add standard class definitions
        for class_def in _get_standard_class_defs().values():
            g.add_class_def(class_def)

        config = GraphConfig(title=f"Agent: {agent.name}")
        return cls(g, config)

    # ─────────────────────────────────────────────────────────────────────
    # Rendering Methods
    # ─────────────────────────────────────────────────────────────────────

    def mermaid(self) -> str:
        """Render as Mermaid diagram code.

        Returns:
            Mermaid diagram code string.
        """
        backend = MermaidBackend()
        return backend.render(self._graph, self._config)

    def ascii(self) -> str:
        """Render as ASCII art diagram (terminal-friendly).

        Returns:
            ASCII diagram string.
        """
        backend = ASCIIBackend()
        return backend.render(self._graph, self._config)

    def dot(self) -> str:
        """Render as Graphviz DOT format.

        Returns:
            DOT format string.
        """
        backend = GraphvizBackend()
        return backend.render(self._graph, self._config)

    def url(self) -> str:
        """Get mermaid.ink URL for online viewing.

        Returns:
            Shareable URL to rendered diagram.
        """
        backend = MermaidBackend()
        return backend.to_url(self._graph, self._config)

    def html(self) -> str:
        """Generate HTML with embedded diagram.

        Returns:
            HTML string with embedded Mermaid diagram.
        """
        backend = MermaidBackend()
        return backend.to_html(self._graph, self._config)

    def png(self) -> bytes:
        """Render as PNG image bytes.

        Returns:
            PNG image data.
        """
        backend = MermaidBackend()
        return backend.draw_png(self._graph, self._config)

    def svg(self) -> bytes:
        """Render as SVG image bytes.

        Returns:
            SVG image data.
        """
        backend = MermaidBackend()
        return backend.draw_svg(self._graph, self._config)

    def save(self, path: str | Path, backend: str | None = None) -> None:
        """Save diagram to file (format from extension).

        Args:
            path: Output file path (.png, .svg, .mmd, .dot, etc).
            backend: Optional backend override ("mermaid", "graphviz", "ascii").

        Example:
            >>> view.save("diagram.png")  # PNG image
            >>> view.save("diagram.svg")  # SVG image
            >>> view.save("diagram.mmd")  # Mermaid code
            >>> view.save("diagram.dot")  # Graphviz DOT
        """
        path = Path(path) if isinstance(path, str) else path
        suffix = path.suffix.lower()

        # Auto-detect backend from extension if not specified
        if backend is None:
            if suffix in (".mmd", ".mermaid"):
                backend = "mermaid"
            elif suffix in (".dot", ".gv"):
                backend = "graphviz"
            elif suffix in (".txt", ".ascii"):
                backend = "ascii"
            elif suffix in (".png", ".svg", ".pdf"):
                backend = "mermaid"  # Use Mermaid for images
            else:
                backend = "mermaid"  # Default

        # Render based on format
        if suffix == ".png":
            data = self.png()
            path.write_bytes(data)
        elif suffix == ".svg":
            data = self.svg()
            path.write_bytes(data)
        elif suffix in (".mmd", ".mermaid"):
            content = self.mermaid()
            path.write_text(content, encoding="utf-8")
        elif suffix in (".dot", ".gv"):
            content = self.dot()
            path.write_text(content, encoding="utf-8")
        elif suffix in (".txt", ".ascii"):
            content = self.ascii()
            path.write_text(content, encoding="utf-8")
        elif suffix == ".html":
            content = self.html()
            path.write_text(content, encoding="utf-8")
        else:
            # Default to mermaid text
            content = self.mermaid()
            path.write_text(content, encoding="utf-8")

    def display(self) -> None:
        """Display the graph in Jupyter notebook.

        Uses _repr_markdown_() for native Mermaid rendering in VS Code.

        Example:
            >>> view = kg.visualize()
            >>> view.display()  # Shows inline in Jupyter
        """
        try:
            from IPython.display import display as ipy_display

            ipy_display(self)
        except ImportError:
            print("IPython not available. Use .html() or .mermaid() instead.")

    def render(self, format: str = "auto") -> str | bytes:
        """High-level rendering method with auto-detection.

        Args:
            format: Output format - "auto", "mermaid", "ascii", "dot", "html", "png", "svg"
                   "auto" will use HTML in Jupyter, mermaid otherwise.

        Returns:
            Rendered content as string or bytes (for images).

        Example:
            >>> view.render()           # Auto-detect context
            >>> view.render("mermaid")  # Get mermaid code
            >>> view.render("png")      # Get PNG bytes
        """
        if format == "auto":
            # Auto-detect: HTML in Jupyter, mermaid otherwise
            try:
                get_ipython()  # type: ignore
                return self.html()
            except NameError:
                return self.mermaid()
        elif format == "mermaid":
            return self.mermaid()
        elif format == "ascii":
            return self.ascii()
        elif format == "dot":
            return self.dot()
        elif format == "html":
            return self.html()
        elif format == "png":
            return self.png()
        elif format == "svg":
            return self.svg()
        else:
            raise ValueError(
                f"Unknown format: {format}. "
                "Use: 'auto', 'mermaid', 'ascii', 'dot', 'html', 'png', 'svg'"
            )

    def _repr_markdown_(self) -> str:
        """IPython/Jupyter Markdown representation.

        This enables native Mermaid rendering in VS Code notebooks.
        Just evaluate the GraphView object and it will render as a diagram.
        """
        mermaid_code = self.mermaid()
        title = self._config.title if self._config and self._config.title else ""

        if title:
            return f"### {title}\n\n```mermaid\n{mermaid_code}\n```"
        else:
            return f"```mermaid\n{mermaid_code}\n```"

    def _repr_html_(self) -> str:
        """IPython/Jupyter HTML representation."""
        return self.html()

    @staticmethod
    def _get_step_label(step: Any) -> str:
        """Get label for a pipeline step."""
        if hasattr(step, "name"):
            return step.name
        if hasattr(step, "__name__"):
            return step.__name__
        if callable(step):
            return getattr(step, "__name__", "step")
        return str(step)[:20]
