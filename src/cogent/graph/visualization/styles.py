"""Visualization styles for knowledge graphs.

This module provides color schemes, shapes, and styling options for rendering
knowledge graphs in different formats (Mermaid, PyVis, gravis, etc.).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cogent.graph.models import Entity


# Distinct color palette for automatic type assignment
# Material Design colors - visually distinct and accessible
COLOR_PALETTE = [
    "#90CAF9",  # Blue
    "#FFD54F",  # Amber
    "#A5D6A7",  # Green
    "#CE93D8",  # Purple
    "#FFAB91",  # Deep Orange
    "#80DEEA",  # Cyan
    "#F48FB1",  # Pink
    "#C5E1A5",  # Light Green
    "#FFCC80",  # Orange
    "#B39DDB",  # Deep Purple
    "#81C784",  # Green
    "#FFE082",  # Yellow
    "#E57373",  # Red
    "#64B5F6",  # Light Blue
    "#AED581",  # Lime
    "#BA68C8",  # Purple
    "#FF8A65",  # Deep Orange
    "#4DD0E1",  # Cyan
    "#9575CD",  # Deep Purple
    "#4DB6AC",  # Teal
]


def _darken_color(hex_color: str, factor: float = 0.3) -> str:
    """Darken a hex color for borders.
    
    Args:
        hex_color: Hex color code (e.g., "#90CAF9")
        factor: Darkening factor (0-1, higher = darker)
        
    Returns:
        Darkened hex color
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def assign_colors_to_types(entity_types: list[str]) -> dict[str, str]:
    """Automatically assign distinct colors to entity types.
    
    Args:
        entity_types: List of unique entity types
        
    Returns:
        Dictionary mapping entity type to hex color
        
    Example:
        >>> types = ["Person", "Project", "Company"]
        >>> colors = assign_colors_to_types(types)
        >>> print(colors["Person"])
        #90CAF9
    """
    color_map = {}
    for i, entity_type in enumerate(sorted(entity_types)):
        color_map[entity_type] = COLOR_PALETTE[i % len(COLOR_PALETTE)]
    return color_map


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


class AutoScheme(StyleScheme):
    """Automatic color scheme that assigns distinct colors to unknown types.
    
    This scheme combines predefined colors for common types with automatic
    color assignment for unknown types. Colors are assigned from a palette
    in a deterministic, repeatable way.
    
    Example:
        >>> scheme = AutoScheme()
        >>> # Known types get predefined colors
        >>> style = scheme.get_node_style("Person")  
        >>> # Unknown types get auto-assigned colors
        >>> style = scheme.get_node_style("Spacecraft")  # Auto-assigned from palette
    """

    def __init__(self) -> None:
        """Initialize with common types + automatic assignment."""
        super().__init__()
        
        # Start with common entity type styles (same as DefaultScheme)
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
        
        # Track assigned types for deterministic color assignment
        self._color_index = len(self.node_styles)
    
    def get_node_style(self, entity_type: str) -> NodeStyle:
        """Get node style for an entity type, auto-creating if needed.
        
        Args:
            entity_type: Type of the entity.
            
        Returns:
            NodeStyle for this entity type (existing or newly created).
        """
        if entity_type not in self.node_styles:
            # Auto-assign color from palette
            color = COLOR_PALETTE[self._color_index % len(COLOR_PALETTE)]
            border_color = _darken_color(color)
            
            self.node_styles[entity_type] = NodeStyle(
                shape="circle",  # Default shape for unknown types
                color=color,
                border_color=border_color,
                text_color="#000000",
            )
            self._color_index += 1
        
        return self.node_styles[entity_type]


def create_scheme_from_entities(entities: list["Entity"]) -> AutoScheme:
    """Create an automatic color scheme from a list of entities.
    
    Analyzes entity types and pre-assigns colors to all types found,
    ensuring consistent colors across multiple visualizations.
    
    Args:
        entities: List of Entity objects
        
    Returns:
        AutoScheme with colors pre-assigned to all entity types
        
    Example:
        >>> from cogent.graph import Graph
        >>> graph = Graph()
        >>> await graph.add_entity("sun", "Star", name="Sun")
        >>> await graph.add_entity("earth", "Planet", name="Earth")
        >>> entities = await graph.get_all_entities()
        >>> 
        >>> scheme = create_scheme_from_entities(entities)
        >>> # Use in visualizations
        >>> net = to_pyvis(entities, relationships, scheme=scheme)
    """
    scheme = AutoScheme()
    
    # Pre-assign colors to all types
    entity_types = {e.entity_type for e in entities}
    for entity_type in sorted(entity_types):
        _ = scheme.get_node_style(entity_type)  # Trigger auto-assignment
    
    return scheme


SCHEMES: dict[str, type[StyleScheme]] = {
    "default": DefaultScheme,
    "minimal": MinimalScheme,
    "auto": AutoScheme,
}


def get_scheme(name: str = "default") -> StyleScheme:
    """Get a style scheme by name.

    Args:
        name: Name of the scheme:
            - "default": Predefined colors for common types (Person, Company, etc.)
            - "auto": Automatic color assignment for any entity types
            - "minimal": Monochrome scheme

    Returns:
        StyleScheme instance.

    Raises:
        ValueError: If scheme name is not recognized.

    Example:
        >>> scheme = get_scheme("auto")
        >>> style = scheme.get_node_style("Spacecraft")  # Auto-assigned color
    """
    if name not in SCHEMES:
        raise ValueError(
            f"Unknown scheme: {name}. Available: {list(SCHEMES.keys())}"
        )

    return SCHEMES[name]()
