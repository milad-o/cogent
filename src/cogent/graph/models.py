"""Core data models for the Knowledge Graph.

This module defines the foundational Entity and Relationship models
for representing knowledge graph data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(slots=True, frozen=False)
class Entity:
    """Represents a node/entity in the knowledge graph.

    An entity is a distinct object or concept with a unique identifier,
    type classification, and arbitrary attributes.

    Args:
        id: Unique identifier for the entity.
        entity_type: Classification or category of the entity.
        attributes: Key-value pairs storing entity properties.
        created_at: Timestamp when the entity was created.
        updated_at: Timestamp when the entity was last updated.
        source: Optional source attribution (e.g., document, tool, user).

    Raises:
        ValueError: If id or entity_type is empty/whitespace.

    Example:
        >>> person = Entity(
        ...     id="person:john",
        ...     entity_type="Person",
        ...     attributes={"name": "John Doe", "age": 30}
        ... )
        >>> person.id
        'person:john'
    """

    id: str
    entity_type: str
    attributes: dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None

    def __post_init__(self) -> None:
        """Validate entity data after initialization."""
        if not self.id or not self.id.strip():
            raise ValueError("Entity ID cannot be empty or whitespace")

        if not self.entity_type or not self.entity_type.strip():
            raise ValueError("Entity type cannot be empty or whitespace")

        # Normalize whitespace
        self.id = self.id.strip()
        self.entity_type = self.entity_type.strip()

        if self.source is not None:
            self.source = self.source.strip()

    def update_attributes(self, **kwargs: object) -> None:
        """Update entity attributes and refresh the updated_at timestamp.

        Args:
            **kwargs: Key-value pairs to add or update in attributes.

        Example:
            >>> entity.update_attributes(status="active", score=95)
        """
        self.attributes.update(kwargs)
        self.updated_at = datetime.now(timezone.utc)

    def get_attribute(self, key: str, default: object = None) -> object:
        """Get an attribute value with optional default.

        Args:
            key: Attribute key to retrieve.
            default: Default value if key doesn't exist.

        Returns:
            The attribute value or default.
        """
        return self.attributes.get(key, default)

    def __hash__(self) -> int:
        """Make Entity hashable based on its ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Two entities are equal if they have the same ID."""
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id


@dataclass(slots=True, frozen=False)
class Relationship:
    """Represents an edge/relationship between two entities in the knowledge graph.

    A relationship connects two entities with a labeled relation type
    and optional attributes describing the relationship.

    Args:
        source_id: ID of the source entity.
        relation: Type/label of the relationship.
        target_id: ID of the target entity.
        attributes: Key-value pairs storing relationship properties.
        created_at: Timestamp when the relationship was created.
        source: Optional source attribution (e.g., document, tool, user).

    Raises:
        ValueError: If source_id, relation, or target_id is empty/whitespace.

    Example:
        >>> relationship = Relationship(
        ...     source_id="person:john",
        ...     relation="WORKS_AT",
        ...     target_id="company:acme",
        ...     attributes={"since": 2020, "role": "Engineer"}
        ... )
        >>> relationship.relation
        'WORKS_AT'
    """

    source_id: str
    relation: str
    target_id: str
    attributes: dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None

    def __post_init__(self) -> None:
        """Validate relationship data after initialization."""
        if not self.source_id or not self.source_id.strip():
            raise ValueError("Source ID cannot be empty or whitespace")

        if not self.relation or not self.relation.strip():
            raise ValueError("Relation type cannot be empty or whitespace")

        if not self.target_id or not self.target_id.strip():
            raise ValueError("Target ID cannot be empty or whitespace")

        # Normalize whitespace
        self.source_id = self.source_id.strip()
        self.relation = self.relation.strip()
        self.target_id = self.target_id.strip()

        if self.source is not None:
            self.source = self.source.strip()

    def update_attributes(self, **kwargs: object) -> None:
        """Update relationship attributes.

        Args:
            **kwargs: Key-value pairs to add or update in attributes.

        Example:
            >>> relationship.update_attributes(weight=0.95, verified=True)
        """
        self.attributes.update(kwargs)

    def get_attribute(self, key: str, default: object = None) -> object:
        """Get an attribute value with optional default.

        Args:
            key: Attribute key to retrieve.
            default: Default value if key doesn't exist.

        Returns:
            The attribute value or default.
        """
        return self.attributes.get(key, default)

    def reverse(self) -> "Relationship":
        """Create a reversed copy of this relationship.

        Returns:
            A new Relationship with source and target swapped.

        Example:
            >>> rel = Relationship("A", "RELATES_TO", "B")
            >>> reversed_rel = rel.reverse()
            >>> reversed_rel.source_id
            'B'
            >>> reversed_rel.target_id
            'A'
        """
        return Relationship(
            source_id=self.target_id,
            relation=self.relation,
            target_id=self.source_id,
            attributes=self.attributes.copy(),
            created_at=self.created_at,
            source=self.source,
        )

    def __hash__(self) -> int:
        """Make Relationship hashable based on its components."""
        return hash((self.source_id, self.relation, self.target_id))

    def __eq__(self, other: object) -> bool:
        """Two relationships are equal if they have the same source, relation, and target."""
        if not isinstance(other, Relationship):
            return NotImplemented
        return (
            self.source_id == other.source_id
            and self.relation == other.relation
            and self.target_id == other.target_id
        )
