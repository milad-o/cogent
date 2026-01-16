"""
Example: Graph API - Visualize Agents

Demonstrates the unified Graph API for generating visual diagrams
of agents.

Key API:
    view = agent.graph()         # Get GraphView from any entity
    
    view.mermaid() -> str        # Mermaid diagram code
    view.ascii() -> str          # Terminal-friendly text
    view.dot() -> str            # Graphviz DOT format
    view.url() -> str            # Shareable mermaid.ink URL
    view.png() -> bytes          # PNG image bytes
    view.html() -> str           # Embeddable HTML
    view.save("file.png")        # Save to file (format from extension)

Usage:
    uv run python examples/observability/graph.py
"""

from agenticflow import Agent, tool


# ─────────────────────────────────────────────────────────────────────────────
# Sample Tools
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
    """Review and provide feedback."""
    return f"Review: {content[:50]} - Approved!"


# ─────────────────────────────────────────────────────────────────────────────
# Agent Visualization
# ─────────────────────────────────────────────────────────────────────────────


def demo_agent_graph():
    """Visualize a single agent."""
    print("\n" + "=" * 60)
    print("Agent Graph")
    print("=" * 60)

    agent = Agent(
        name="Assistant",
        tools=[search, write],
        instructions="A helpful assistant.",
    )

    # Get graph view
    view = agent.graph()

    # Mermaid (default, great for docs/markdown)
    print("\n--- Mermaid ---")
    print(view.mermaid())

    # ASCII (terminal-friendly)
    print("\n--- ASCII ---")
    print(view.ascii())

    # Shareable URL
    print("\n--- Shareable URL ---")
    print(view.url())

    # Without tools (simpler view)
    print("\n--- Without tools ---")
    print(agent.graph(show_tools=False).mermaid())


# ─────────────────────────────────────────────────────────────────────────────
# Saving Diagrams
# ─────────────────────────────────────────────────────────────────────────────


def demo_save_options():
    """Show saving options."""
    print("\n" + "=" * 60)
    print("Saving Diagrams")
    print("=" * 60)

    agent = Agent(name="Agent", tools=[search])
    view = agent.graph()

    print("\nSave methods (format from extension):")
    print("  view.save('agent.png')   # PNG image")
    print("  view.save('agent.svg')   # SVG image")
    print("  view.save('agent.mmd')   # Mermaid code")
    print("  view.save('agent.html')  # Embeddable HTML")
    print("  view.save('agent.dot')   # Graphviz DOT")
    print("  view.save('agent.txt')   # ASCII art")

    # HTML preview
    print("\n--- HTML (first 300 chars) ---")
    print(view.html()[:300] + "...")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Run all graph demos."""
    print("AgenticFlow Graph API")
    print("=" * 60)

    demo_agent_graph()
    demo_save_options()

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
