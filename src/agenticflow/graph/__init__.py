"""
AgenticFlow Graph API - Multi-level visualization for agents and flows.

Three API levels:
- **Low-level**: Raw graph primitives (Node, Edge, Graph)
- **Mid-level**: Diagram builders (AgentGraph, FlowGraph, TopologyGraph)
- **High-level**: Convenience methods on agents/flows (.visualize(), .to_graph())

Multiple backends:
- **Mermaid** (default): Renders to mermaid.ink, PNG, SVG, HTML
- **Graphviz**: DOT format, high-quality PNG/PDF (requires graphviz)
- **ASCII**: Terminal-friendly text diagrams

Example:
    ```python
    # Mid-level API (recommended)
    from agenticflow.graph import AgentGraph, TopologyGraph

    graph = AgentGraph.from_agent(agent)
    print(graph.render())  # Mermaid code
    graph.to_png("agent.png")

    # With different backend
    from agenticflow.graph import GraphvizBackend
    graph.render(backend=GraphvizBackend())

    # Low-level API
    from agenticflow.graph import Graph, Node, Edge

    g = Graph()
    g.node("agent1", "Researcher", css_class="work")
    g.node("agent2", "Writer", css_class="work")
    g.edge("agent1", "agent2", "handoff")
    print(MermaidBackend().render(g))
    ```
"""

# Low-level primitives
from agenticflow.graph.primitives import (
    ClassDef,
    Edge,
    EdgeStyle,
    EdgeType,
    Graph,
    Node,
    NodeShape,
    NodeStyle,
    Subgraph,
)

# Configuration
from agenticflow.graph.config import (
    GraphConfig,
    GraphDirection,
    GraphTheme,
)

# Backends
from agenticflow.graph.backends import (
    ASCIIBackend,
    Backend,
    GraphvizBackend,
    MermaidBackend,
    get_default_backend,
    set_default_backend,
)

# Mid-level builders
from agenticflow.graph.builders import (
    AgentGraph,
    FlowGraph,
    TopologyGraph,
)

__all__ = [
    # Primitives (low-level)
    "Node",
    "Edge",
    "EdgeType",
    "Graph",
    "Subgraph",
    "NodeStyle",
    "NodeShape",
    "EdgeStyle",
    "ClassDef",
    # Config
    "GraphConfig",
    "GraphTheme",
    "GraphDirection",
    # Backends
    "Backend",
    "MermaidBackend",
    "GraphvizBackend",
    "ASCIIBackend",
    "get_default_backend",
    "set_default_backend",
    # Builders (mid-level)
    "AgentGraph",
    "FlowGraph",
    "TopologyGraph",
]
