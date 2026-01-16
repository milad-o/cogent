"""
GraphView - Unified visualization interface for all graphable entities.

Provides a consistent API for rendering agents, topologies, and flows
to various formats without exposing internal graph primitives.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from agenticflow.graph.backends import (
    ASCIIBackend,
    GraphvizBackend,
    MermaidBackend,
)
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
    from agenticflow.agent.base import Agent
    from agenticflow.flow import Flow


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
