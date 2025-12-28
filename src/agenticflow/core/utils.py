"""
Utility functions for AgenticFlow.
"""

import uuid
from datetime import datetime, timezone


def generate_id(prefix: str | int = "", length: int = 8) -> str:
    """
    Generate a short unique identifier.
    
    Args:
        prefix: Optional prefix string, or length for backwards compatibility
        length: Number of characters (default 8)
        
    Returns:
        A unique ID string, optionally with prefix
    """
    # Handle backwards compatibility where first arg was length
    if isinstance(prefix, int):
        length = prefix
        prefix = ""
    
    uid = str(uuid.uuid4()).replace("-", "")[:length]
    if prefix:
        return f"{prefix}_{uid}"
    return uid


def now_utc() -> datetime:
    """
    Get current UTC timestamp (timezone-aware).
    
    Returns:
        Current datetime in UTC timezone
    """
    return datetime.now(timezone.utc)


def now_local() -> datetime:
    """
    Get current local timestamp (timezone-aware).
    
    Returns:
        Current datetime in local timezone
    """
    return datetime.now(timezone.utc).astimezone()


def to_local(dt: datetime) -> datetime:
    """
    Convert a datetime to local timezone.
    
    Args:
        dt: Datetime to convert (can be naive or timezone-aware)
        
    Returns:
        Datetime in local timezone
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone()


def format_timestamp(dt: datetime | None, local: bool = True) -> str | None:
    """
    Format a datetime for display.
    
    Args:
        dt: Datetime to format (can be None)
        local: If True, convert to local timezone first (default: True)
        
    Returns:
        Formatted string or None
    """
    if dt is None:
        return None
    if local:
        dt = to_local(dt)
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


def model_identifier(model: object) -> str:
    """Return a safe, human-friendly identifier for a model.

    Observability should never log full model reprs because they may contain
    secrets (e.g., API keys). This helper prefers common name fields and falls
    back to the class name.
    """
    if model is None:
        return "unknown"

    for attr in ("model_name", "model", "name", "deployment_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value

    # As a last resort, use the type name (safe) rather than str(model).
    return type(model).__name__
