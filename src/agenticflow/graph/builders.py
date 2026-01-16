"""
Mid-level graph builders for agents, flows, and topologies.

These builders provide convenient APIs to create Graph objects from
AgenticFlow components without manual node/edge construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agenticflow.graph.backends import Backend, MermaidBackend, get_default_backend
from agenticflow.graph.config import GraphConfig
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
    from agenticflow.agent.base import Agent
    from agenticflow.flow import Flow


# Standard role styles (4-role system)
ROLE_STYLES: dict[str, NodeStyle] = {
    "supervisor": NodeStyle(fill="#4a90d9", stroke="#2d5986", color="#fff"),
    "worker": NodeStyle(fill="#7eb36a", stroke="#4a7a3d", color="#fff"),
    "autonomous": NodeStyle(fill="#9b59b6", stroke="#7b3a96", color="#fff"),
    "reviewer": NodeStyle(fill="#f56c6c", stroke="#c45656", color="#fff"),
}

ROLE_CLASSES: dict[str, str] = {
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
    }


class AgentGraph:
    """Build graphs from individual agents.

    Creates a visual representation of an agent with its tools,
    role, and configuration.

    Example:
        >>> from agenticflow.graph import AgentGraph
        >>> graph = AgentGraph.from_agent(agent)
        >>> print(graph.render())  # Mermaid code
        >>> graph.to_png("agent.png")
    """

    def __init__(
        self,
        graph: Graph,
        config: GraphConfig | None = None,
    ) -> None:
        """Initialize with a pre-built graph.

        Args:
            graph: The underlying Graph object.
            config: Rendering configuration.
        """
        self._graph = graph
        self._config = config or GraphConfig()

    @property
    def graph(self) -> Graph:
        """Get the underlying Graph object."""
        return self._graph

    @classmethod
    def from_agent(
        cls,
        agent: Agent,
        *,
        show_tools: bool = True,
        show_config: bool = False,
        config: GraphConfig | None = None,
    ) -> AgentGraph:
        """Create a graph from an agent.

        Args:
            agent: Agent to visualize.
            show_tools: Whether to show tools in label.
            show_config: Whether to show configuration node.
            config: Optional graph configuration.

        Returns:
            AgentGraph instance.
        """
        g = Graph()
        cfg = config or GraphConfig()

        # Get agent info
        agent_id = _make_node_id(agent.name)
        role = agent.role.value
        role_class = ROLE_CLASSES.get(role, "work")

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

        # Set title if not provided
        if not cfg.title:
            cfg = cfg.with_title(f"Agent: {agent.name}")

        return cls(g, cfg)

    def render(self, backend: Backend | None = None) -> str:
        """Render graph using specified backend.

        Args:
            backend: Rendering backend (default: Mermaid).

        Returns:
            Rendered graph string.
        """
        b = backend or get_default_backend()
        return b.render(self._graph, self._config)

    def to_url(self) -> str:
        """Get mermaid.ink URL for this graph.

        Returns:
            URL to rendered diagram.
        """
        backend = MermaidBackend()
        return backend.to_url(self._graph, self._config)

    def to_png(self, path: str | None = None, backend: Backend | None = None) -> bytes:
        """Render as PNG.

        Args:
            path: Optional file path to save.
            backend: Rendering backend.

        Returns:
            PNG bytes.
        """
        b = backend or MermaidBackend()
        png_data = b.draw_png(self._graph, self._config)

        if path:
            with open(path, "wb") as f:
                f.write(png_data)

        return png_data

    def to_html(self) -> str:
        """Generate HTML with embedded diagram.

        Returns:
            HTML string.
        """
        backend = MermaidBackend()
        return backend.to_html(self._graph, self._config)

    def _repr_html_(self) -> str:
        """IPython/Jupyter HTML representation."""
        return self.to_html()


class FlowGraph:
    """Build graphs from flows.

    Creates visual representations of Flow execution paths,
    showing conditions, transitions, and agent invocations.

    Example:
        >>> from agenticflow.graph import FlowGraph
        >>> graph = FlowGraph.from_flow(flow)
        >>> graph.render()
    """

    def __init__(
        self,
        graph: Graph,
        config: GraphConfig | None = None,
    ) -> None:
        """Initialize with a pre-built graph.

        Args:
            graph: The underlying Graph object.
            config: Rendering configuration.
        """
        self._graph = graph
        self._config = config or GraphConfig()

    @property
    def graph(self) -> Graph:
        """Get the underlying Graph object."""
        return self._graph

    @classmethod
    def from_flow(
        cls,
        flow: Flow,
        *,
        config: GraphConfig | None = None,
    ) -> FlowGraph:
        """Create a graph from a Flow.

        Args:
            flow: Flow to visualize.
            config: Optional graph configuration.

        Returns:
            FlowGraph instance.
        """
        g = Graph()
        cfg = config or GraphConfig()

        # Add start node
        g.add_node(
            Node(
                id="start",
                label="Start",
                shape=NodeShape.CIRCLE,
                css_class="start",
            )
        )

        # Track added nodes
        added_nodes: set[str] = {"start"}

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
            added_nodes.add(step_id)

            # Connect to previous
            g.add_edge(Edge(source=prev_node, target=step_id))
            prev_node = step_id

        # Add end node
        g.add_node(
            Node(
                id="end",
                label="End",
                shape=NodeShape.CIRCLE,
                css_class="end",
            )
        )

        if prev_node:
            g.add_edge(Edge(source=prev_node, target="end"))

        # Add class definitions
        g.add_class_def(
            ClassDef(
                name="start",
                style=NodeStyle(fill="#4ade80", stroke="#22c55e", color="#fff"),
            )
        )
        g.add_class_def(
            ClassDef(
                name="end",
                style=NodeStyle(fill="#f87171", stroke="#ef4444", color="#fff"),
            )
        )
        g.add_class_def(
            ClassDef(
                name="step",
                style=NodeStyle(fill="#60a5fa", stroke="#3b82f6", color="#fff"),
            )
        )

        if not cfg.title:
            flow_name = getattr(flow, "name", "Flow")
            cfg = cfg.with_title(flow_name)

        return cls(g, cfg)

    @staticmethod
    def _get_step_label(step: Any) -> str:
        """Get label for a flow step."""
        # Try to extract meaningful label from step
        if hasattr(step, "name"):
            return step.name
        if hasattr(step, "__name__"):
            return step.__name__
        if callable(step):
            return getattr(step, "__name__", "step")
        return str(step)[:20]

    def render(self, backend: Backend | None = None) -> str:
        """Render graph using specified backend.

        Args:
            backend: Rendering backend (default: Mermaid).

        Returns:
            Rendered graph string.
        """
        b = backend or get_default_backend()
        return b.render(self._graph, self._config)

    def to_url(self) -> str:
        """Get mermaid.ink URL for this graph.

        Returns:
            URL to rendered diagram.
        """
        backend = MermaidBackend()
        return backend.to_url(self._graph, self._config)

    def to_png(self, path: str | None = None, backend: Backend | None = None) -> bytes:
        """Render as PNG.

        Args:
            path: Optional file path to save.
            backend: Rendering backend.

        Returns:
            PNG bytes.
        """
        b = backend or MermaidBackend()
        png_data = b.draw_png(self._graph, self._config)

        if path:
            with open(path, "wb") as f:
                f.write(png_data)

        return png_data

    def to_html(self) -> str:
        """Generate HTML with embedded diagram.

        Returns:
            HTML string.
        """
        backend = MermaidBackend()
        return backend.to_html(self._graph, self._config)

    def _repr_html_(self) -> str:
        """IPython/Jupyter HTML representation."""
        return self.to_html()
