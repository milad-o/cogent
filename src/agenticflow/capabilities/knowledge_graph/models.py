"""
Knowledge Graph data models.

Contains Entity and Relationship dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Entity:
    """An entity in the knowledge graph."""
    
    id: str
    type: str
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None  # Where this fact came from
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
        }


@dataclass
class Relationship:
    """A relationship between two entities."""
    
    source_id: str
    relation: str
    target_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None  # Where this fact came from
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "relation": self.relation,
            "target": self.target_id,
            "attributes": self.attributes,
            "created_at": self.created_at.isoformat(),
        }
