"""Graph visualization module.

Provides rendering capabilities for knowledge graphs in multiple formats:
- Mermaid diagrams (with PNG/SVG/PDF image rendering via Playwright)
- Graphviz DOT files
- GraphML XML
- Cytoscape.js JSON
- JSON Graph Format

Usage:
    >>> from cogent.graph import Graph
    >>> from cogent.graph.visualization import to_mermaid
    >>>
    >>> graph = Graph()
    >>> # ... add entities and relationships ...
    >>>
    >>> # Generate Mermaid diagram
    >>> diagram = await graph.to_mermaid(direction="LR")
    >>>
    >>> # Or use standalone functions
    >>> entities = await graph.get_all_entities()
    >>> relationships = await graph.get_relationships()
    >>> diagram = to_mermaid(entities, relationships)
    >>>
    >>> # Render to image (requires Playwright)
    >>> from cogent.graph.visualization import render_mermaid_to_image
    >>> await render_mermaid_to_image(diagram, "graph.png")
"""

from cogent.graph.visualization.renderer import (
    entity_to_mermaid_node,
    relationship_to_mermaid_edge,
    render_mermaid_to_image,
    save_diagram,
    to_cytoscape_json,
    to_graphml,
    to_graphviz,
    to_json_graph,
    to_mermaid,
)
from cogent.graph.visualization.styles import (
    SCHEMES,
    DefaultScheme,
    EdgeStyle,
    MinimalScheme,
    NodeStyle,
    StyleScheme,
    get_scheme,
)

__all__ = [
    # Renderers
    "to_mermaid",
    "to_graphviz",
    "to_graphml",
    "to_cytoscape_json",
    "to_json_graph",
    "render_mermaid_to_image",
    "save_diagram",
    "entity_to_mermaid_node",
    "relationship_to_mermaid_edge",
    # Styles
    "StyleScheme",
    "NodeStyle",
    "EdgeStyle",
    "DefaultScheme",
    "MinimalScheme",
    "get_scheme",
    "SCHEMES",
]
