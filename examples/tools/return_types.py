"""
Tool Return Type Demo

Demonstrates how tool return types are automatically extracted
and included in tool descriptions for LLM visibility.

The @tool decorator:
1. Extracts return type from function annotations (e.g., -> dict[str, int])
2. Parses the Returns section from docstrings
3. Combines both into the tool description

This helps the LLM understand what output to expect from each tool.
"""

import json
from agenticflow import tool


# Example 1: Simple return type
@tool
def search(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: The search query string.
    """
    return f"Found 10 results for: {query}"


# Example 2: Complex return type with docstring description
@tool
def get_weather(city: str) -> dict[str, int]:
    """Get weather data for a city.
    
    Args:
        city: City name to query.
    
    Returns:
        A dictionary with temp (Fahrenheit), humidity (%), and wind_speed (mph).
    """
    return {"temp": 75, "humidity": 45, "wind_speed": 10}


# Example 3: List return type
@tool
def list_files(directory: str) -> list[str]:
    """List all files in a directory.
    
    Args:
        directory: Path to the directory.
    
    Returns:
        List of filenames in the directory.
    """
    return ["file1.txt", "file2.py", "file3.md"]


# Example 4: Optional return type
@tool
def find_user(user_id: str) -> dict | None:
    """Find a user by ID.
    
    Args:
        user_id: The user's unique identifier.
    
    Returns:
        User data dictionary if found, None if not found.
    """
    if user_id == "123":
        return {"id": "123", "name": "Alice"}
    return None


# Example 5: Boolean return with description
@tool
def file_exists(path: str) -> bool:
    """Check if a file exists.
    
    Args:
        path: File path to check.
    
    Returns:
        True if the file exists, False otherwise.
    """
    return True  # Simplified


# Example 6: No return type (void function)
@tool
def log_event(message: str) -> None:
    """Log an event message.
    
    Args:
        message: The message to log.
    """
    print(f"[LOG] {message}")


def main() -> None:
    """Show how return info is included in tool schemas."""
    
    tools = [search, get_weather, list_files, find_user, file_exists, log_event]
    
    print("=" * 70)
    print("Tool Return Type Demo")
    print("=" * 70)
    print()
    
    for t in tools:
        print(f"🔧 {t.name}")
        print(f"   Description: {t.description}")
        print(f"   Return Info: {t.return_info or '(none)'}")
        print()
    
    print("=" * 70)
    print("JSON Schema sent to LLM (example: get_weather)")
    print("=" * 70)
    print()
    print(json.dumps(get_weather.to_dict(), indent=2))
    print()
    
    print("=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("""
1. Return types (-> str, -> dict[str, int]) are automatically extracted
2. Docstring 'Returns:' sections provide human-readable descriptions  
3. Both are combined: "Returns: dict[str, int] - A dictionary with..."
4. This helps the LLM understand what output to expect from tools
5. Functions returning None don't add "Returns:" to avoid noise
""")


if __name__ == "__main__":
    main()
