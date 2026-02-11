"""Graph visualization module.

Provides rendering capabilities for knowledge graphs in multiple formats:
- Mermaid diagrams
- Graphviz DOT files
- GraphML XML

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
"""

from cogent.graph.visualization.renderer import (
    entity_to_mermaid_node,
    relationship_to_mermaid_edge,
    save_diagram,
    to_graphml,
    to_graphviz,
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
