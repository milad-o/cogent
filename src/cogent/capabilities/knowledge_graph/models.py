"""Compatibility aliases for graph data models."""

from cogent.graph.models import Entity as _GraphEntity
from cogent.graph.models import Relationship as _GraphRelationship


# Expose the new graph models while keeping legacy attribute names.
Entity = _GraphEntity
Relationship = _GraphRelationship

# Backward compatibility: old code used .type instead of .entity_type.
Entity.type = property(  # type: ignore[attr-defined]
	lambda self: self.entity_type, lambda self, value: setattr(self, "entity_type", value)
)

__all__ = ["Entity", "Relationship"]
