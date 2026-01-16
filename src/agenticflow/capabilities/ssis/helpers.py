"""Helper functions for SSIS XML parsing."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET


def get_property(element: ET.Element, prop_name: str) -> str | None:
    """Get a DTS property value from an element."""
    for prop in element.iter():
        if prop.tag.endswith("Property") and (
            prop.get("Name") == prop_name
            or prop.get("{www.microsoft.com/SqlServer/Dts}Name") == prop_name
        ):
            return prop.text
    return None


def get_attribute(element: ET.Element, attr_name: str) -> str | None:
    """Get an attribute value, checking with and without namespace."""
    value = element.get(attr_name)
    if not value:
        value = element.get(f"{{www.microsoft.com/SqlServer/Dts}}{attr_name}")
    return value


def extract_tables_from_sql(sql: str) -> list[str]:
    """Extract table names from SQL statement (basic extraction)."""
    tables = []
    sql_upper = sql.upper()

    # Patterns for various SQL statements
    # Using \S+ to match table names (handles schema.table and [bracketed] names)
    patterns = [
        r"FROM\s+(\S+)",
        r"JOIN\s+(\S+)",
        r"INTO\s+(\S+)",
        r"UPDATE\s+(\S+)",
        r"TABLE\s+(\S+)",  # Handles TRUNCATE TABLE, DROP TABLE, etc.
        r"INSERT\s+INTO\s+(\S+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, sql_upper)
        for match in matches:
            # Clean up brackets and schema, get just the table name
            table = match.strip("[]").split(".")[-1].strip("[]")
            # Filter out SQL keywords
            if table and table not in ("SELECT", "FROM", "WHERE", "SET", "VALUES", "("):
                tables.append(table)

    return list(set(tables))


def sanitize_connection_string(conn_str: str) -> str:
    """Remove sensitive info from connection string."""
    # Remove password
    sanitized = re.sub(
        r"(Password|PWD)\s*=\s*[^;]*",
        r"\1=***",
        conn_str,
        flags=re.IGNORECASE,
    )
    return sanitized


def extract_component_from_path(
    path_id: str,
    components: dict[str, dict[str, any]],
) -> str | None:
    """Extract component name from a path reference ID."""
    # Path IDs are like "Package\DataFlow\Component.Outputs[Output 0]"
    for ref_id, comp_info in components.items():
        if ref_id and ref_id in path_id:
            return comp_info["name"]

    # Try to extract from path format
    if "\\" in path_id:
        parts = path_id.split("\\")
        for part in reversed(parts):
            clean_part = part.split(".")[0]
            for ref_id, comp_info in components.items():
                if clean_part in comp_info["name"]:
                    return comp_info["name"]

    return None
