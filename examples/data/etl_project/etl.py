"""
ETL Pipeline - Extract, Transform, Load demo.

A simple ETL process that reads data, transforms it, and outputs results.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Record:
    """A single data record."""
    id: str
    name: str
    value: float
    category: str


class DataExtractor:
    """Extracts data from various sources."""

    def __init__(self, source: str):
        self.source = source
        self._connection = None

    def connect(self) -> bool:
        """Establish connection to data source."""
        print(f"Connecting to {self.source}...")
        self._connection = True
        return True

    def extract(self) -> list[Record]:
        """Extract records from the source."""
        if not self._connection:
            raise RuntimeError("Not connected")
        
        # Simulated data extraction
        return [
            Record("1", "Widget A", 100.0, "electronics"),
            Record("2", "Widget B", 200.0, "electronics"),
            Record("3", "Gadget X", 50.0, "accessories"),
        ]

    def close(self) -> None:
        """Close the connection."""
        self._connection = None


class DataTransformer:
    """Transforms extracted data."""

    def __init__(self, rules: dict[str, Any] | None = None):
        self.rules = rules or {}

    def transform(self, records: list[Record]) -> list[dict]:
        """Apply transformations to records."""
        results = []
        for record in records:
            transformed = self._apply_rules(record)
            results.append(transformed)
        return results

    def _apply_rules(self, record: Record) -> dict:
        """Apply transformation rules to a single record."""
        result = {
            "id": record.id,
            "name": record.name.upper(),
            "value": record.value * self.rules.get("multiplier", 1.0),
            "category": record.category,
            "processed": True,
        }
        return result

    def filter_by_category(self, records: list[Record], category: str) -> list[Record]:
        """Filter records by category."""
        return [r for r in records if r.category == category]


class DataLoader:
    """Loads transformed data to destination."""

    def __init__(self, destination: str):
        self.destination = destination
        self._loaded_count = 0

    def load(self, data: list[dict]) -> int:
        """Load data to destination."""
        print(f"Loading {len(data)} records to {self.destination}...")
        for item in data:
            self._write_record(item)
        return self._loaded_count

    def _write_record(self, record: dict) -> None:
        """Write a single record."""
        # Simulated write
        self._loaded_count += 1

    def get_stats(self) -> dict:
        """Get loading statistics."""
        return {
            "destination": self.destination,
            "records_loaded": self._loaded_count,
        }
