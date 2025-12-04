"""
AgenticFlow Graph API - Unified visualization for agents, topologies, and flows.

Simple API:
    ```python
    # Get a graph from any entity
    view = agent.graph()
    view = topology.graph()
    view = flow.graph()

    # Render in any format
    print(view.mermaid())    # Mermaid code (default)
    print(view.ascii())      # Terminal-friendly text
    print(view.dot())        # Graphviz DOT format
    print(view.url())        # mermaid.ink URL
    print(view.html())       # Embeddable HTML

    # Save to file (format auto-detected)
    view.save("diagram.png")
    view.save("diagram.svg")
    view.save("diagram.mmd")
    ```
"""

from agenticflow.graph.view import GraphView
from agenticflow.graph.config import GraphConfig, GraphTheme, GraphDirection

__all__ = [
    "GraphView",
    "GraphConfig",
    "GraphTheme",
    "GraphDirection",
]
