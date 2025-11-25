"""
Utility functions for AgenticFlow.
"""

import uuid
from datetime import datetime, timezone


def generate_id(length: int = 8) -> str:
    """
    Generate a short unique identifier.
    
    Args:
        length: Number of characters (default 8)
        
    Returns:
        A unique ID string
    """
    return str(uuid.uuid4())[:length]


def now_utc() -> datetime:
    """
    Get current UTC timestamp.
    
    Returns:
        Current datetime in UTC timezone
    """
    return datetime.now(timezone.utc)


def format_timestamp(dt: datetime | None) -> str | None:
    """
    Format a datetime for display.
    
    Args:
        dt: Datetime to format (can be None)
        
    Returns:
        Formatted string or None
    """
    if dt is None:
        return None
    return dt.strftime("%H:%M:%S.%f")[:-3]


def format_duration_ms(start: datetime | None, end: datetime | None) -> float | None:
    """
    Calculate duration in milliseconds between two timestamps.
    
    Args:
        start: Start datetime
        end: End datetime
        
    Returns:
        Duration in milliseconds or None if either is None
    """
    if start is None or end is None:
        return None
    return (end - start).total_seconds() * 1000


def truncate_string(s: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length.
    
    Args:
        s: String to truncate
        max_length: Maximum length (default 200)
        suffix: Suffix to add if truncated (default "...")
        
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def safe_json_value(value: object) -> object:
    """
    Convert a value to a JSON-safe representation.
    
    Args:
        value: Any value
        
    Returns:
        JSON-serializable value
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)
