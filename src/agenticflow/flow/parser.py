"""Event pattern parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedPattern:
    """Parsed event pattern with optional source filter.
    
    Attributes:
        event: The event pattern (e.g., "agent.done", "*.done")
        source: Optional source filter (e.g., "researcher", "agent*")
        separator: The separator used, if any ("@")
    
    Examples:
        >>> parse_pattern("agent.done@researcher")
        ParsedPattern(event='agent.done', source='researcher', separator='@')
        
        >>> parse_pattern("*.done")
        ParsedPattern(event='*.done', source=None, separator=None)
    """

    event: str
    source: str | None = None
    separator: str | None = None


def parse_pattern(pattern: str) -> ParsedPattern:
    """Parse event pattern with optional source filter.
    
    Supports the @ separator syntax:
        - "event@source" - @ separator
    
    Both event and source parts support wildcards (*, ?).
    
    Note: The : and -> separators are reserved for future features.
    
    Args:
        pattern: Event pattern string, optionally with source filter
    
    Returns:
        ParsedPattern with event, optional source, and separator
    
    Examples:
        Basic pattern without source:
        >>> parse_pattern("agent.done")
        ParsedPattern(event='agent.done', source=None, separator=None)
        
        Pattern with @ separator:
        >>> parse_pattern("agent.done@researcher")
        ParsedPattern(event='agent.done', source='researcher', separator='@')
        
        Pattern with wildcards:
        >>> parse_pattern("*.done@agent*")
        ParsedPattern(event='*.done', source='agent*', separator='@')
        
        Multiple sources not supported (use list or OR logic):
        >>> # NOT: "event@source1,source2"
        >>> # USE: ["event@source1", "event@source2"]
    """
    # Check for @ separator
    if "@" in pattern:
        parts = pattern.split("@", 1)  # Split only on first occurrence
        event_part = parts[0].strip()
        source_part = parts[1].strip() if len(parts) > 1 else None
        return ParsedPattern(
            event=event_part,
            source=source_part if source_part else None,
            separator="@",
        )

    # No separator found - pattern is just an event
    return ParsedPattern(event=pattern.strip())
