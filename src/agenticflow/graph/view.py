"""
GraphView - Unified visualization interface for all graphable entities.

Provides a consistent API for rendering agents, topologies, and flows
to various formats without exposing internal graph primitives.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from agenticflow.graph.backends import (
    ASCIIBackend,
    GraphvizBackend,
    MermaidBackend,
)

if TYPE_CHECKING:
    from agenticflow.agent.base import Agent
    from agenticflow.flow import Flow
    from agenticflow.topologies.core import BaseTopology


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
            style=NodeStyle(fill="#fff3e0", stroke="#ff9800", color="#333", dashed=True),
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

    Returned by `.to_graph()` on Agent, Topology, and Flow objects.
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
    # Factory Methods (for internal use by Agent, Topology, Flow)
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_agent(
        cls,
        agent: "Agent",
        *,
        show_tools: bool = True,
        show_config: bool = False,
    ) -> "GraphView":
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

    @classmethod
    def from_topology(
        cls,
        topology: "BaseTopology",
        *,
        show_tools: bool = True,
    ) -> "GraphView":
        """Create a graph from a topology.

        Args:
            topology: Topology to visualize.
            show_tools: Whether to show tools in labels.

        Returns:
            GraphView instance.
        """
        g = Graph()

        # Determine topology type and build accordingly
        topo_type = cls._get_topology_type(topology)

        if topo_type == "supervisor":
            cls._build_supervisor_graph(g, topology, show_tools)
        elif topo_type == "pipeline":
            cls._build_pipeline_graph(g, topology, show_tools)
        elif topo_type == "mesh":
            cls._build_mesh_graph(g, topology, show_tools)
        elif topo_type == "hierarchical":
            cls._build_hierarchical_graph(g, topology, show_tools)
        elif topo_type == "custom":
            cls._build_custom_graph(g, topology, show_tools)
        else:
            cls._build_generic_graph(g, topology, show_tools)

        # Add standard class definitions
        for class_def in _get_standard_class_defs().values():
            g.add_class_def(class_def)

        config = GraphConfig(title=topology.config.name)
        return cls(g, config)

    @classmethod
    def from_flow(
        cls,
        flow: "Flow",
    ) -> "GraphView":
        """Create a graph from a Flow.

        Args:
            flow: Flow to visualize.

        Returns:
            GraphView instance.
        """
        g = Graph()

        # Add start node
        g.add_node(
            Node(id="start", label="Start", shape=NodeShape.CIRCLE, css_class="start")
        )

        # Process flow steps
        steps = getattr(flow, "_steps", [])
        prev_node = "start"

        for i, step in enumerate(steps):
            step_id = f"step_{i}"
            step_label = cls._get_step_label(step)

            g.add_node(
                Node(
                    id=step_id,
                    label=step_label,
                    shape=NodeShape.ROUNDED,
                    css_class="step",
                )
            )

            g.add_edge(Edge(source=prev_node, target=step_id))
            prev_node = step_id

        # Add end node
        g.add_node(
            Node(id="end", label="End", shape=NodeShape.CIRCLE, css_class="end")
        )
        if prev_node:
            g.add_edge(Edge(source=prev_node, target="end"))

        # Add standard class definitions
        for class_def in _get_standard_class_defs().values():
            g.add_class_def(class_def)

        flow_name = getattr(flow, "name", "Flow")
        config = GraphConfig(title=flow_name)
        return cls(g, config)

    # ─────────────────────────────────────────────────────────────────────
    # Public Rendering API
    # ─────────────────────────────────────────────────────────────────────

    def mermaid(self) -> str:
        """Render as Mermaid diagram code.

        Returns:
            Mermaid flowchart syntax string.
        """
        backend = MermaidBackend()
        return backend.render(self._graph, self._config)

    def ascii(self) -> str:
        """Render as ASCII art for terminal display.

        Returns:
            ASCII diagram string.
        """
        backend = ASCIIBackend()
        return backend.render(self._graph, self._config)

    def dot(self) -> str:
        """Render as Graphviz DOT format.

        Returns:
            DOT syntax string.
        """
        backend = GraphvizBackend()
        return backend.render(self._graph, self._config)

    def url(self) -> str:
        """Get mermaid.ink URL for the diagram.

        Returns:
            URL to rendered SVG image.
        """
        backend = MermaidBackend()
        return backend.to_url(self._graph, self._config)

    def png_url(self) -> str:
        """Get mermaid.ink PNG URL.

        Returns:
            URL to rendered PNG image.
        """
        backend = MermaidBackend()
        return backend.to_png_url(self._graph, self._config)

    def png(self) -> bytes:
        """Render as PNG image bytes.

        Requires internet connection (uses mermaid.ink API).

        Returns:
            PNG image bytes.
        """
        backend = MermaidBackend()
        return backend.draw_png(self._graph, self._config)

    def html(self) -> str:
        """Generate embeddable HTML with the diagram.

        Returns:
            HTML string with Mermaid JS.
        """
        backend = MermaidBackend()
        return backend.to_html(self._graph, self._config)

    def save(self, path: str | Path) -> None:
        """Save diagram to file.

        Format is determined by file extension:
        - .png: PNG image (via mermaid.ink)
        - .svg: SVG via mermaid.ink URL reference
        - .md, .mermaid, .mmd: Mermaid code
        - .dot, .gv: Graphviz DOT format
        - .html: Embeddable HTML
        - .txt: ASCII art

        Args:
            path: Output file path.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".png":
            path.write_bytes(self.png())
        elif suffix in (".svg",):
            # For SVG, embed the URL in an SVG wrapper
            url = self.url()
            svg_content = f'<svg xmlns="http://www.w3.org/2000/svg"><image href="{url}"/></svg>'
            path.write_text(svg_content)
        elif suffix in (".md", ".mermaid", ".mmd"):
            # Wrap in markdown code block
            content = f"```mermaid\n{self.mermaid()}\n```"
            path.write_text(content)
        elif suffix in (".dot", ".gv"):
            path.write_text(self.dot())
        elif suffix == ".html":
            path.write_text(self.html())
        elif suffix == ".txt":
            path.write_text(self.ascii())
        else:
            # Default to Mermaid
            path.write_text(self.mermaid())

    # ─────────────────────────────────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────────────────────────────────

    def with_title(self, title: str) -> "GraphView":
        """Set diagram title.

        Args:
            title: New title.

        Returns:
            New GraphView with updated config.
        """
        new_config = self._config.with_title(title)
        return GraphView(self._graph, new_config)

    def with_theme(self, theme: GraphTheme | str) -> "GraphView":
        """Set diagram theme.

        Args:
            theme: Theme name or GraphTheme enum.

        Returns:
            New GraphView with updated config.
        """
        if isinstance(theme, str):
            theme = GraphTheme(theme)
        new_config = self._config.with_theme(theme)
        return GraphView(self._graph, new_config)

    def with_direction(self, direction: GraphDirection | str) -> "GraphView":
        """Set diagram layout direction.

        Args:
            direction: Direction (TD, LR, BT, RL) or GraphDirection enum.

        Returns:
            New GraphView with updated config.
        """
        if isinstance(direction, str):
            direction = GraphDirection(direction)
        new_config = self._config.with_direction(direction)
        return GraphView(self._graph, new_config)

    # ─────────────────────────────────────────────────────────────────────
    # Jupyter/IPython Integration
    # ─────────────────────────────────────────────────────────────────────

    def _repr_html_(self) -> str:
        """IPython/Jupyter HTML representation."""
        return self.html()

    def __repr__(self) -> str:
        """String representation."""
        node_count = len(self._graph.nodes)
        edge_count = len(self._graph.edges)
        title = self._config.title or "Untitled"
        return f"GraphView({title!r}, nodes={node_count}, edges={edge_count})"

    # ─────────────────────────────────────────────────────────────────────
    # Internal Topology Builders
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_topology_type(topology: "BaseTopology") -> str:
        """Determine topology type from class name."""
        class_name = type(topology).__name__.lower()
        for t in ("custom", "supervisor", "pipeline", "mesh", "hierarchical"):
            if t in class_name:
                return t
        return "generic"

    @classmethod
    def _build_supervisor_graph(
        cls,
        g: Graph,
        topology: "BaseTopology",
        show_tools: bool,
    ) -> None:
        """Build graph for supervisor topology."""
        supervisor_name = getattr(topology, "supervisor_name", None)
        worker_names = getattr(topology, "worker_names", [])

        if not supervisor_name:
            agents_dict = topology.get_agents_dict()
            supervisor_name = next(iter(agents_dict), "supervisor")
            worker_names = [n for n in agents_dict if n != supervisor_name]

        # Add supervisor node
        cls._add_agent_node(g, topology, supervisor_name, show_tools)
        sup_id = _make_node_id(supervisor_name)

        # Add workers in subgraph if multiple
        if len(worker_names) > 1:
            worker_ids = []
            for name in worker_names:
                cls._add_agent_node(g, topology, name, show_tools)
                worker_ids.append(_make_node_id(name))

            g.add_subgraph(
                Subgraph(id="workers", label=" ", node_ids=worker_ids, direction="LR")
            )

            for worker_id in worker_ids:
                g.add_edge(Edge(source=sup_id, target=worker_id))
        else:
            for name in worker_names:
                cls._add_agent_node(g, topology, name, show_tools)
                g.add_edge(Edge(source=sup_id, target=_make_node_id(name)))

    @classmethod
    def _build_pipeline_graph(
        cls,
        g: Graph,
        topology: "BaseTopology",
        show_tools: bool,
    ) -> None:
        """Build graph for pipeline topology."""
        raw_stages = getattr(topology, "stages", None)
        if raw_stages:
            stages = [s.name if hasattr(s, "name") else str(s) for s in raw_stages]
        else:
            stages = list(topology.get_agents_dict().keys())

        for name in stages:
            cls._add_agent_node(g, topology, name, show_tools)

        for i in range(len(stages) - 1):
            g.add_edge(
                Edge(source=_make_node_id(stages[i]), target=_make_node_id(stages[i + 1]))
            )

    @classmethod
    def _build_mesh_graph(
        cls,
        g: Graph,
        topology: "BaseTopology",
        show_tools: bool,
    ) -> None:
        """Build graph for mesh topology."""
        agent_names = list(topology.get_agents_dict().keys())

        for name in agent_names:
            cls._add_agent_node(g, topology, name, show_tools)

        for i in range(len(agent_names)):
            for j in range(i + 1, len(agent_names)):
                g.add_edge(
                    Edge(
                        source=_make_node_id(agent_names[i]),
                        target=_make_node_id(agent_names[j]),
                        edge_type=EdgeType.BIDIRECTIONAL,
                    )
                )

    @classmethod
    def _build_hierarchical_graph(
        cls,
        g: Graph,
        topology: "BaseTopology",
        show_tools: bool,
    ) -> None:
        """Build graph for hierarchical topology."""
        hierarchy = getattr(topology, "hierarchy", {})
        root = getattr(topology, "root", None)

        if not hierarchy or not root:
            cls._build_generic_graph(g, topology, show_tools)
            return

        added: set[str] = set()

        def add_node_recursive(name: str) -> None:
            if name in added:
                return
            added.add(name)
            cls._add_agent_node(g, topology, name, show_tools)

            children = hierarchy.get(name, [])
            parent_id = _make_node_id(name)
            for child in children:
                add_node_recursive(child)
                g.add_edge(Edge(source=parent_id, target=_make_node_id(child)))

        add_node_recursive(root)

    @classmethod
    def _build_custom_graph(
        cls,
        g: Graph,
        topology: "BaseTopology",
        show_tools: bool,
    ) -> None:
        """Build graph for custom topology using explicit edges."""
        edges = getattr(topology, "edges", [])

        for name in topology.get_agents_dict():
            cls._add_agent_node(g, topology, name, show_tools)

        for edge in edges:
            edge_type = EdgeType.BIDIRECTIONAL if edge.bidirectional else EdgeType.ARROW
            label = edge.label or edge.condition

            g.add_edge(
                Edge(
                    source=_make_node_id(edge.source),
                    target=_make_node_id(edge.target),
                    label=label,
                    edge_type=edge_type,
                )
            )

    @classmethod
    def _build_generic_graph(
        cls,
        g: Graph,
        topology: "BaseTopology",
        show_tools: bool,
    ) -> None:
        """Build graph for unknown topology types."""
        agent_names = list(topology.get_agents_dict().keys())

        for name in agent_names:
            cls._add_agent_node(g, topology, name, show_tools)

        if hasattr(topology, "policy") and hasattr(topology.policy, "get_edges_for_diagram"):
            policy_edges = topology.policy.get_edges_for_diagram(agent_names)
            for source, target, label in policy_edges:
                g.add_edge(
                    Edge(
                        source=_make_node_id(source),
                        target=_make_node_id(target),
                        label=label or None,
                    )
                )
        elif len(agent_names) > 1:
            for i in range(len(agent_names) - 1):
                g.add_edge(
                    Edge(
                        source=_make_node_id(agent_names[i]),
                        target=_make_node_id(agent_names[i + 1]),
                    )
                )

    @classmethod
    def _add_agent_node(
        cls,
        g: Graph,
        topology: "BaseTopology",
        name: str,
        show_tools: bool,
    ) -> None:
        """Add an agent node to the graph."""
        agent = topology.get_agents_dict().get(name)
        role = agent.role.value if agent else "worker"
        role_class = ROLE_STYLES.get(role, "work")
        node_id = _make_node_id(name)

        tools = None
        if show_tools and agent and agent.config.tools:
            tools = agent.config.tools

        label = _make_agent_label(name, role, tools)

        g.add_node(
            Node(id=node_id, label=label, shape=NodeShape.ROUNDED, css_class=role_class)
        )

    @staticmethod
    def _get_step_label(step: Any) -> str:
        """Get label for a flow step."""
        if hasattr(step, "name"):
            return step.name
        if hasattr(step, "__name__"):
            return step.__name__
        if callable(step):
            return getattr(step, "__name__", "step")
        return str(step)[:20]
