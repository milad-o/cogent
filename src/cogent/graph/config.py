"""
Graph configuration - themes, directions, and rendering options.
"""

from dataclasses import dataclass, field
from enum import Enum


class GraphTheme(str, Enum):
    """Graph visual themes."""

    DEFAULT = "default"
    DARK = "dark"
    FOREST = "forest"
    NEUTRAL = "neutral"
    BASE = "base"


class GraphDirection(str, Enum):
    """Graph layout direction."""

    TOP_DOWN = "TD"
    BOTTOM_UP = "BT"
    LEFT_RIGHT = "LR"
    RIGHT_LEFT = "RL"


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphConfig:
    """Configuration for graph rendering.

    Attributes:
        title: Optional title for the graph.
        theme: Visual theme to apply.
        direction: Layout direction.
        show_legend: Whether to include a legend.
        node_spacing: Horizontal spacing between nodes.
        rank_spacing: Vertical spacing between ranks.
        curve: Line curve style (basis, linear, step).
        padding: Diagram padding.
        font_family: Font family for labels.
        font_size: Base font size in pixels.
    """

    title: str | None = None
    theme: GraphTheme = GraphTheme.DEFAULT
    direction: GraphDirection = GraphDirection.TOP_DOWN
    show_legend: bool = False
    node_spacing: int = 40
    rank_spacing: int = 40
    curve: str = "basis"
    padding: int = 8
    font_family: str = "Arial"
    font_size: int = 14
    custom_styles: dict[str, str] = field(default_factory=dict)

    def with_title(self, title: str) -> "GraphConfig":
        """Create a new config with the specified title."""
        return GraphConfig(
            title=title,
            theme=self.theme,
            direction=self.direction,
            show_legend=self.show_legend,
            node_spacing=self.node_spacing,
            rank_spacing=self.rank_spacing,
            font_family=self.font_family,
            font_size=self.font_size,
            custom_styles=self.custom_styles,
        )

    def with_theme(self, theme: GraphTheme) -> "GraphConfig":
        """Create a new config with the specified theme."""
        return GraphConfig(
            title=self.title,
            theme=theme,
            direction=self.direction,
            show_legend=self.show_legend,
            node_spacing=self.node_spacing,
            rank_spacing=self.rank_spacing,
            font_family=self.font_family,
            font_size=self.font_size,
            custom_styles=self.custom_styles,
        )

    def with_direction(self, direction: GraphDirection) -> "GraphConfig":
        """Create a new config with the specified direction."""
        return GraphConfig(
            title=self.title,
            theme=self.theme,
            direction=direction,
            show_legend=self.show_legend,
            node_spacing=self.node_spacing,
            rank_spacing=self.rank_spacing,
            font_family=self.font_family,
            font_size=self.font_size,
            custom_styles=self.custom_styles,
        )
