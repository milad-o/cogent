"""
Example: Graph API - Multi-level visualization.

Demonstrates the three API levels:
1. Low-level: Raw Graph, Node, Edge primitives
2. Mid-level: AgentGraph, TopologyGraph builders
3. High-level: agent.to_graph(), topology.visualize()

Also shows multiple backends:
- Mermaid (default): Web-based rendering
- Graphviz: High-quality DOT format
- ASCII: Terminal-friendly text diagrams
"""

import asyncio

from agenticflow import Agent, tool
from agenticflow.topologies import Supervisor, Pipeline, AgentConfig

# Graph API imports
from agenticflow.graph import (
    # Low-level primitives
    Graph,
    Node,
    Edge,
    EdgeType,
    NodeShape,
    Subgraph,
    ClassDef,
    NodeStyle,
    # Configuration
    GraphConfig,
    GraphTheme,
    GraphDirection,
    # Backends
    MermaidBackend,
    GraphvizBackend,
    ASCIIBackend,
    # Mid-level builders
    AgentGraph,
    TopologyGraph,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tools for agents
# ─────────────────────────────────────────────────────────────────────────────


@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"


@tool
def write(content: str) -> str:
    """Write content to a document."""
    return f"Written: {content[:50]}..."


@tool
def review(content: str) -> str:
    """Review and provide feedback on content."""
    return f"Review: {content[:50]} - Looks good!"


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Low-Level API - Raw Graph Primitives
# ─────────────────────────────────────────────────────────────────────────────


def demo_low_level_api():
    """Demonstrate low-level graph primitives."""
    print("\n" + "=" * 60)
    print("Level 1: Low-Level API - Raw Graph Primitives")
    print("=" * 60)

    # Create a graph from scratch
    g = Graph()

    # Add nodes with various shapes and styles
    g.add_node(
        Node(
            id="start",
            label="Start",
            shape=NodeShape.CIRCLE,
            css_class="start",
        )
    )

    g.add_node(
        Node(
            id="research",
            label="Research Agent",
            shape=NodeShape.ROUNDED,
            css_class="work",
        )
    )

    g.add_node(
        Node(
            id="write",
            label="Writer Agent",
            shape=NodeShape.ROUNDED,
            css_class="work",
        )
    )

    g.add_node(
        Node(
            id="end",
            label="End",
            shape=NodeShape.CIRCLE,
            css_class="end",
        )
    )

    # Add edges
    g.add_edge(Edge(source="start", target="research"))
    g.add_edge(Edge(source="research", target="write", label="handoff"))
    g.add_edge(Edge(source="write", target="end"))

    # Add class definitions for styling
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
            name="work",
            style=NodeStyle(fill="#7eb36a", stroke="#4a7a3d", color="#fff"),
        )
    )

    # Render with Mermaid backend
    backend = MermaidBackend()
    config = GraphConfig(title="Simple Workflow", direction=GraphDirection.LEFT_RIGHT)
    mermaid_code = backend.render(g, config)

    print("\nMermaid output:")
    print(mermaid_code)

    # Also render with ASCII backend
    ascii_backend = ASCIIBackend()
    ascii_output = ascii_backend.render(g, config)

    print("\nASCII output:")
    print(ascii_output)

    return g


def demo_fluent_api():
    """Demonstrate fluent API for building graphs."""
    print("\n" + "=" * 60)
    print("Level 1: Fluent API - Chained Method Calls")
    print("=" * 60)

    # Fluent API - chain methods
    g = (
        Graph()
        .node("a", "Step A", css_class="work")
        .node("b", "Step B", css_class="work")
        .node("c", "Step C", css_class="work")
        .edge("a", "b", "process")
        .edge("b", "c", "validate")
    )

    # Add bidirectional edge
    g.edge("a", "c", "feedback", edge_type=EdgeType.BIDIRECTIONAL)

    backend = MermaidBackend()
    print("\nFluent API output:")
    print(backend.render(g))

    return g


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: Mid-Level API - Graph Builders
# ─────────────────────────────────────────────────────────────────────────────


def demo_mid_level_api():
    """Demonstrate mid-level AgentGraph and TopologyGraph builders."""
    print("\n" + "=" * 60)
    print("Level 2: Mid-Level API - Graph Builders")
    print("=" * 60)

    # Create agents (model is optional for visualization)
    researcher = Agent(
        name="Researcher",
        tools=[search],
        instructions="You research topics thoroughly.",
    )

    writer = Agent(
        name="Writer",
        tools=[write],
        instructions="You write clear, engaging content.",
    )

    # AgentGraph - visualize a single agent
    print("\n--- AgentGraph (single agent) ---")
    agent_graph = AgentGraph.from_agent(researcher, show_tools=True)
    print(agent_graph.render())

    # TopologyGraph - visualize a supervisor topology
    print("\n--- TopologyGraph (supervisor topology) ---")
    supervisor = Agent(
        name="Manager",
        instructions="You coordinate the team.",
    )

    topology = Supervisor(
        coordinator=AgentConfig(agent=supervisor, role="coordinator"),
        workers=[
            AgentConfig(agent=researcher, role="research"),
            AgentConfig(agent=writer, role="writing"),
        ],
    )

    topo_graph = TopologyGraph.from_topology(topology, show_tools=True)
    print(topo_graph.render())

    # TopologyGraph - visualize a pipeline
    print("\n--- TopologyGraph (pipeline topology) ---")
    editor = Agent(
        name="Editor",
        tools=[review],
        instructions="You edit and polish content.",
    )

    pipeline = Pipeline(
        stages=[
            AgentConfig(agent=researcher, role="gather info"),
            AgentConfig(agent=writer, role="draft"),
            AgentConfig(agent=editor, role="polish"),
        ]
    )

    pipeline_graph = TopologyGraph.from_topology(pipeline)
    print(pipeline_graph.render())

    return agent_graph, topo_graph


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: High-Level API - Methods on Objects
# ─────────────────────────────────────────────────────────────────────────────


def demo_high_level_api():
    """Demonstrate high-level API with to_graph() and visualize() methods."""
    print("\n" + "=" * 60)
    print("Level 3: High-Level API - Object Methods")
    print("=" * 60)

    # Create an agent (no model needed for visualization)
    agent = Agent(
        name="Assistant",
        tools=[search, write],
        instructions="A helpful assistant.",
    )

    # High-level API on Agent
    print("\n--- agent.to_graph() ---")
    graph = agent.to_graph()  # Returns AgentGraph
    print(graph.render())

    print("\n--- agent.visualize('ascii') ---")
    print(agent.visualize("ascii"))

    # Create a topology
    researcher = Agent(name="Researcher", tools=[search])
    writer = Agent(name="Writer", tools=[write])

    topology = Supervisor(
        coordinator=AgentConfig(agent=agent, role="manager"),
        workers=[
            AgentConfig(agent=researcher, role="research"),
            AgentConfig(agent=writer, role="writing"),
        ],
    )

    # High-level API on Topology
    print("\n--- topology.to_graph() ---")
    topo_graph = topology.to_graph()
    print(topo_graph.render())

    print("\n--- topology.visualize('ascii') ---")
    print(topology.visualize("ascii"))

    return graph, topo_graph


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Backends Demo
# ─────────────────────────────────────────────────────────────────────────────


def demo_backends():
    """Demonstrate different rendering backends."""
    print("\n" + "=" * 60)
    print("Multiple Backends")
    print("=" * 60)

    # Create a simple graph
    g = (
        Graph()
        .node("a", "Agent A")
        .node("b", "Agent B")
        .node("c", "Agent C")
        .edge("a", "b")
        .edge("b", "c")
        .edge("a", "c", "direct")
    )

    config = GraphConfig(title="Multi-Backend Demo")

    # Mermaid Backend (default)
    print("\n--- Mermaid Backend ---")
    mermaid = MermaidBackend()
    print(mermaid.render(g, config))

    # Graphviz Backend
    print("\n--- Graphviz Backend (DOT format) ---")
    graphviz = GraphvizBackend()
    print(graphviz.render(g, config))

    # ASCII Backend
    print("\n--- ASCII Backend ---")
    ascii_backend = ASCIIBackend()
    print(ascii_backend.render(g, config))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Run all demos."""
    print("AgenticFlow Graph API Demo")
    print("=" * 60)

    # Level 1: Low-level
    demo_low_level_api()
    demo_fluent_api()

    # Level 2: Mid-level
    demo_mid_level_api()

    # Level 3: High-level
    demo_high_level_api()

    # Backends
    demo_backends()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
