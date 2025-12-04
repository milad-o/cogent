"""
Example: Graph API - Unified Visualization.

Demonstrates the clean, unified Graph API:
- Single `graph()` method on agents and topologies
- Returns `GraphView` with all rendering options
- Three backends: Mermaid (default), Graphviz, ASCII

Key API:
    view = agent.graph()         # or topology.graph()
    
    view.mermaid() -> str        # Mermaid diagram code
    view.ascii() -> str          # Terminal-friendly text
    view.dot() -> str            # Graphviz DOT format
    view.url() -> str            # mermaid.ink URL
    view.png() -> bytes          # PNG image data
    view.html() -> str           # HTML with embedded diagram
    view.save("file.png")        # Auto-detect format
"""

from agenticflow import Agent, tool
from agenticflow.topologies import Supervisor, Pipeline, AgentConfig


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
# Agent Graph Visualization
# ─────────────────────────────────────────────────────────────────────────────


def demo_agent_graph():
    """Demonstrate agent visualization."""
    print("\n" + "=" * 60)
    print("Agent Graph Visualization")
    print("=" * 60)

    # Create an agent
    agent = Agent(
        name="Assistant",
        tools=[search, write],
        instructions="A helpful assistant.",
    )

    # Get graph view
    view = agent.graph()

    # Mermaid (default)
    print("\n--- Mermaid ---")
    print(view.mermaid())

    # Get shareable URL
    print("\n--- Mermaid.ink URL ---")
    print(view.url())

    # ASCII for terminal
    print("\n--- ASCII (terminal-friendly) ---")
    print(view.ascii())

    # Graphviz DOT format
    print("\n--- Graphviz (DOT format) ---")
    print(view.dot())

    # Hide tools for simpler diagram
    print("\n--- Without tools ---")
    simple_view = agent.graph(show_tools=False)
    print(simple_view.mermaid())

    return view


# ─────────────────────────────────────────────────────────────────────────────
# Topology Graph Visualization
# ─────────────────────────────────────────────────────────────────────────────


def demo_topology_graph():
    """Demonstrate topology visualization."""
    print("\n" + "=" * 60)
    print("Topology Graph Visualization")
    print("=" * 60)

    # Create agents
    manager = Agent(
        name="Manager",
        instructions="You coordinate the team.",
    )

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

    editor = Agent(
        name="Editor",
        tools=[review],
        instructions="You edit and polish content.",
    )

    # Supervisor topology
    print("\n--- Supervisor Topology ---")
    supervisor = Supervisor(
        coordinator=AgentConfig(agent=manager, role="coordinator"),
        workers=[
            AgentConfig(agent=researcher, role="research"),
            AgentConfig(agent=writer, role="writing"),
        ],
    )

    view = supervisor.graph()
    print(view.mermaid())

    # Pipeline topology
    print("\n--- Pipeline Topology ---")
    pipeline = Pipeline(
        stages=[
            AgentConfig(agent=researcher, role="gather info"),
            AgentConfig(agent=writer, role="draft"),
            AgentConfig(agent=editor, role="polish"),
        ]
    )

    pipeline_view = pipeline.graph()
    print(pipeline_view.mermaid())

    # ASCII for terminal viewing
    print("\n--- Pipeline (ASCII) ---")
    print(pipeline.graph().ascii())

    return view


# ─────────────────────────────────────────────────────────────────────────────
# Backend Comparison
# ─────────────────────────────────────────────────────────────────────────────


def demo_backends():
    """Compare different backends."""
    print("\n" + "=" * 60)
    print("Backend Comparison")
    print("=" * 60)

    # Simple agent for comparison
    agent = Agent(
        name="Agent",
        tools=[search, write],
        instructions="Test agent.",
    )

    view = agent.graph()

    print("\n--- Mermaid ---")
    print(view.mermaid())

    print("\n--- Graphviz ---")
    print(view.dot())

    print("\n--- ASCII ---")
    print(view.ascii())


# ─────────────────────────────────────────────────────────────────────────────
# Saving Diagrams
# ─────────────────────────────────────────────────────────────────────────────


def demo_saving():
    """Demonstrate saving diagrams to files."""
    print("\n" + "=" * 60)
    print("Saving Diagrams")
    print("=" * 60)

    agent = Agent(
        name="Assistant",
        tools=[search],
        instructions="A helpful assistant.",
    )

    view = agent.graph()

    # Save options (commented to avoid creating files)
    print("\nAvailable save methods:")
    print("  view.save('agent.png')     # PNG via mermaid.ink")
    print("  view.save('agent.svg')     # SVG (Graphviz)")
    print("  view.save('agent.mmd')     # Mermaid code")
    print("  view.save('agent.html')    # HTML with diagram")
    print("  view.save('agent.dot')     # DOT format")
    
    # Get HTML for embedding
    print("\n--- HTML output (for embedding) ---")
    html = view.html()
    print(html[:500] + "...")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Run all demos."""
    print("AgenticFlow Graph API Demo")
    print("=" * 60)

    # Agent visualization
    demo_agent_graph()

    # Topology visualization
    demo_topology_graph()

    # Backend comparison
    demo_backends()

    # Saving
    demo_saving()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
