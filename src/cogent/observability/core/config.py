"""
Configuration for Observability v2.

Minimal, composable configuration that avoids the 25+ field explosion of v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    pass


class Level(IntEnum):
    """
    Observability levels.

    Each level includes all events from lower levels plus additional detail.

    Attributes:
        OFF: No output
        RESULT: Only final results
        PROGRESS: Key milestones (agent started/completed)
        DETAILED: Tool calls, timing, intermediate results
        DEBUG: Everything including internal events
        TRACE: Maximum detail + execution graphs
    """

    OFF = 0
    RESULT = 1
    PROGRESS = 2
    DETAILED = 3
    DEBUG = 4
    TRACE = 5

    @classmethod
    def from_string(cls, value: str) -> Level:
        """
        Parse level from string.

        Supports both level names and preset names:
            "off", "result", "progress", "detailed", "debug", "trace"
            "minimal" → RESULT
            "verbose" → PROGRESS
        """
        aliases = {
            "minimal": cls.RESULT,
            "verbose": cls.PROGRESS,
            "normal": cls.PROGRESS,
        }

        name = value.lower().strip()
        if name in aliases:
            return aliases[name]

        try:
            return cls[name.upper()]
        except KeyError:
            valid = [e.name.lower() for e in cls] + list(aliases.keys())
            raise ValueError(
                f"Invalid level '{value}'. Valid levels: {', '.join(valid)}"
            ) from None


@dataclass
class FormatConfig:
    """
    Configuration for event formatting.

    Controls how events are rendered to strings.
    """

    use_colors: bool = True
    """Use ANSI color codes in output."""

    show_timestamps: bool = False
    """Show timestamps in output."""

    show_duration: bool = True
    """Show operation duration."""

    show_trace_ids: bool = False
    """Show correlation IDs."""

    truncate: int = 0
    """Max chars for content (0 = no limit)."""

    indent: str = "  "
    """Indentation string."""


@dataclass
class ObserverConfig:
    """
    Configuration for the Observer.

    Minimal config that composes other configs rather than
    having 25+ fields like v1.
    """

    level: Level = Level.PROGRESS
    """Minimum level for events to be processed."""

    format: FormatConfig = field(default_factory=FormatConfig)
    """Formatting configuration."""

    stream: TextIO | None = None
    """Output stream (None = stderr)."""

    # Filters - glob patterns for event types
    include: list[str] | None = None
    """Only include events matching these patterns (None = all)."""

    exclude: list[str] | None = None
    """Exclude events matching these patterns."""

    @classmethod
    def from_level(cls, level: Level | str) -> ObserverConfig:
        """
        Create config from a level with sensible defaults.

        Args:
            level: Level enum or string name

        Returns:
            Config with appropriate settings for the level
        """
        if isinstance(level, str):
            level = Level.from_string(level)

        # Level-specific defaults
        format_config = FormatConfig()

        if level >= Level.DETAILED:
            format_config.show_timestamps = True

        if level >= Level.DEBUG:
            format_config.show_trace_ids = True
            format_config.truncate = 0  # No truncation at debug+

        return cls(level=level, format=format_config)


# Preset configurations (for backward compatibility with v1)
PRESETS: dict[str, ObserverConfig] = {
    "off": ObserverConfig(level=Level.OFF),
    "minimal": ObserverConfig(level=Level.RESULT),
    "progress": ObserverConfig(level=Level.PROGRESS),
    "normal": ObserverConfig(level=Level.PROGRESS),
    "verbose": ObserverConfig(
        level=Level.PROGRESS,
        format=FormatConfig(show_duration=True, truncate=500),
    ),
    "detailed": ObserverConfig(
        level=Level.DETAILED,
        format=FormatConfig(show_timestamps=True, show_duration=True),
    ),
    "debug": ObserverConfig(
        level=Level.DEBUG,
        format=FormatConfig(
            show_timestamps=True,
            show_duration=True,
            show_trace_ids=True,
            truncate=0,
        ),
    ),
    "trace": ObserverConfig(
        level=Level.TRACE,
        format=FormatConfig(
            show_timestamps=True,
            show_duration=True,
            show_trace_ids=True,
            truncate=0,
        ),
    ),
}


def get_preset(name: str) -> ObserverConfig:
    """
    Get a preset configuration by name.

    Args:
        name: Preset name (off, minimal, progress, verbose, detailed, debug, trace)

    Returns:
        ObserverConfig for the preset

    Raises:
        ValueError: If preset name is not recognized
    """
    name = name.lower().strip()
    if name not in PRESETS:
        valid = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Valid presets: {valid}")
    return PRESETS[name]
