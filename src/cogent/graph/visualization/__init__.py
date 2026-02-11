"""Graph visualization module.

Provides rendering capabilities for knowledge graphs in multiple formats:
- Mermaid diagrams (with PNG/SVG/PDF image rendering via Mermaid CLI)
- PyVis interactive HTML (force-directed layout, vis.js-based)
- gravis interactive 2D/3D visualizations (d3.js/vis.js/three.js-based)

Usage:
    >>> from cogent.graph import Graph
    >>> from cogent.graph.visualization import to_mermaid, to_pyvis, to_gravis
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
    >>> # Render to image (requires Mermaid CLI)
    >>> from cogent.graph.visualization import render_mermaid_to_image
    >>> await render_mermaid_to_image(diagram, "graph.png")
    >>>
    >>> # Interactive PyVis visualization (requires networkx, pyvis)
    >>> net = to_pyvis(entities, relationships)
    >>> net.save_graph("graph.html")
    >>>
    >>> # Interactive gravis visualization (requires gravis, networkx)
    >>> fig = to_gravis(entities, relationships, mode="2d", renderer="vis")
    >>> fig.display()  # Opens in browser
    >>> fig.export_html("graph.html")
    >>>
    >>> # 3D visualization
    >>> fig = to_gravis(entities, relationships, mode="3d")
    >>> fig.display()
"""

from cogent.graph.visualization.renderer import (
    entity_to_mermaid_node,
    relationship_to_mermaid_edge,
    render_mermaid_to_image,
    save_diagram,
    to_gravis,
    to_mermaid,
    to_pyvis,
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
    "to_pyvis",
    "to_gravis",
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
