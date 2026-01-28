"""
Cogent Graph API - Unified visualization for agents and topologies.

Simple API:
    ```python
    # Get a graph from any entity
    view = agent.graph()
    view = topology.graph()

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

from cogent.graph.config import GraphConfig, GraphDirection, GraphTheme
from cogent.graph.view import GraphView

__all__ = [
    "GraphView",
    "GraphConfig",
    "GraphTheme",
    "GraphDirection",
]
