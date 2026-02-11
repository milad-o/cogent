"""Visualization styles for knowledge graphs.

This module provides color schemes, shapes, and styling options for rendering
knowledge graphs in different formats (Mermaid, PyVis, gravis, etc.).
"""

from dataclasses import dataclass


@dataclass
class NodeStyle:
    """Style configuration for a graph node.

    Attributes:
        shape: Node shape (e.g., "rectangle", "circle", "rounded").
        color: Fill color (hex code or name).
        border_color: Border color.
        text_color: Text color.
    """

    shape: str = "rectangle"
    color: str = "#f0f0f0"
    border_color: str = "#333333"
    text_color: str = "#000000"


@dataclass
class EdgeStyle:
    """Style configuration for a graph edge.

    Attributes:
        color: Line color.
        width: Line width.
        style: Line style ("solid", "dashed", "dotted").
        arrow: Arrow style ("normal", "thick", "none").
    """

    color: str = "#666666"
    width: int = 2
    style: str = "solid"
    arrow: str = "normal"


class StyleScheme:
    """Base style scheme with default colors and shapes.

    Provides mapping from entity types to visual styles for consistent
    rendering across different visualization formats.
    """

    def __init__(self) -> None:
        """Initialize default style mappings."""
        self.node_styles: dict[str, NodeStyle] = {}
        self.edge_styles: dict[str, EdgeStyle] = {}
        self.default_node_style = NodeStyle()
        self.default_edge_style = EdgeStyle()

    def get_node_style(self, entity_type: str) -> NodeStyle:
        """Get node style for an entity type.

        Args:
            entity_type: Type of the entity.

        Returns:
            NodeStyle for this entity type.
        """
        return self.node_styles.get(entity_type, self.default_node_style)

    def get_edge_style(self, relation: str) -> EdgeStyle:
        """Get edge style for a relationship type.

        Args:
            relation: Relationship type/label.

        Returns:
            EdgeStyle for this relationship.
        """
        return self.edge_styles.get(relation, self.default_edge_style)

    def set_node_style(self, entity_type: str, style: NodeStyle) -> None:
        """Set custom style for an entity type.

        Args:
            entity_type: Type of the entity.
            style: NodeStyle to apply.
        """
        self.node_styles[entity_type] = style

    def set_edge_style(self, relation: str, style: EdgeStyle) -> None:
        """Set custom style for a relationship type.

        Args:
            relation: Relationship type/label.
            style: EdgeStyle to apply.
        """
        self.edge_styles[relation] = style


class DefaultScheme(StyleScheme):
    """Default color scheme with sensible colors for common entity types."""

    def __init__(self) -> None:
        """Initialize with default color mappings."""
        super().__init__()

        # Common entity type styles
        self.node_styles = {
            "Person": NodeStyle(
                shape="rounded",
                color="#90CAF9",
                border_color="#1976D2",
                text_color="#000000",
            ),
            "Company": NodeStyle(
                shape="rectangle",
                color="#A5D6A7",
                border_color="#388E3C",
                text_color="#000000",
            ),
            "Organization": NodeStyle(
                shape="rectangle",
                color="#A5D6A7",
                border_color="#388E3C",
                text_color="#000000",
            ),
            "Project": NodeStyle(
                shape="rectangle",
                color="#FFCC80",
                border_color="#F57C00",
                text_color="#000000",
            ),
            "Location": NodeStyle(
                shape="rounded",
                color="#CE93D8",
                border_color="#7B1FA2",
                text_color="#000000",
            ),
            "Document": NodeStyle(
                shape="rectangle",
                color="#FFF59D",
                border_color="#F9A825",
                text_color="#000000",
            ),
            "Event": NodeStyle(
                shape="rounded",
                color="#EF9A9A",
                border_color="#C62828",
                text_color="#000000",
            ),
        }

        # Common relationship styles
        self.edge_styles = {
            "knows": EdgeStyle(color="#1976D2", width=2, style="solid"),
            "works_at": EdgeStyle(color="#388E3C", width=2, style="solid"),
            "manages": EdgeStyle(color="#F57C00", width=3, style="solid"),
            "located_in": EdgeStyle(color="#7B1FA2", width=2, style="dashed"),
            "part_of": EdgeStyle(color="#666666", width=2, style="dashed"),
            "created": EdgeStyle(color="#F9A825", width=2, style="solid"),
            "attended": EdgeStyle(color="#C62828", width=2, style="dotted"),
        }


class MinimalScheme(StyleScheme):
    """Minimal monochrome scheme for simple diagrams."""

    def __init__(self) -> None:
        """Initialize with minimal styling."""
        super().__init__()

        self.default_node_style = NodeStyle(
            shape="rectangle",
            color="#ffffff",
            border_color="#000000",
            text_color="#000000",
        )

        self.default_edge_style = EdgeStyle(
            color="#000000", width=1, style="solid", arrow="normal"
        )


SCHEMES: dict[str, type[StyleScheme]] = {
    "default": DefaultScheme,
    "minimal": MinimalScheme,
}


def get_scheme(name: str = "default") -> StyleScheme:
    """Get a style scheme by name.

    Args:
        name: Name of the scheme ("default", "minimal").

    Returns:
        StyleScheme instance.

    Raises:
        ValueError: If scheme name is not recognized.

    Example:
        >>> scheme = get_scheme("default")
        >>> style = scheme.get_node_style("Person")
    """
    if name not in SCHEMES:
        raise ValueError(
            f"Unknown scheme: {name}. Available: {list(SCHEMES.keys())}"
        )

    return SCHEMES[name]()
