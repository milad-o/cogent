"""Pattern matching and query execution for knowledge graphs.

This module provides powerful pattern-based querying capabilities for graphs,
allowing complex graph traversals and relationship matching using dict-based
patterns instead of string query languages.
"""

from dataclasses import dataclass
from typing import Any

from cogent.graph.models import Entity, Relationship


@dataclass
class QueryPattern:
    """Parsed query pattern for execution.

    Patterns can be:
    - Single-hop: Match one relationship
    - Multi-hop: Match a path of multiple relationships
    - Entity filter: Match entities by type/attributes
    """

    pattern_type: str  # "single_hop", "multi_hop", "entity_filter"
    source_filter: dict[str, Any] | None = None
    target_filter: dict[str, Any] | None = None
    relation_filter: str | None = None
    path_hops: list[dict[str, Any]] | None = None


@dataclass
class QueryResult:
    """Result from pattern matching.

    Contains matched entities and relationships, along with metadata about
    the match (path length, match confidence, etc.).
    """

    entities: list[Entity]
    relationships: list[Relationship]
    metadata: dict[str, Any]


def parse_pattern(pattern: dict[str, Any]) -> QueryPattern:
    """Parse a dict-based query pattern into a QueryPattern object.

    Args:
        pattern: Dictionary defining the query pattern.

    Returns:
        Parsed QueryPattern ready for execution.

    Raises:
        ValueError: If pattern format is invalid.

    Examples:
        >>> # Single-hop pattern
        >>> pattern = {
        ...     "source": {"type": "Person", "name": "Alice"},
        ...     "relation": "knows",
        ...     "target": {"type": "Person"}
        ... }
        >>> query = parse_pattern(pattern)

        >>> # Multi-hop path
        >>> pattern = {
        ...     "path": [
        ...         {"source": {"type": "Person"}, "relation": "works_at"},
        ...         {"relation": "manages", "target": {"id": "project_x"}}
        ...     ]
        ... }
        >>> query = parse_pattern(pattern)

        >>> # Entity filter
        >>> pattern = {"type": "Person", "age": 30}
        >>> query = parse_pattern(pattern)
    """
    # Multi-hop path pattern
    if "path" in pattern:
        hops = pattern["path"]
        if not isinstance(hops, list) or len(hops) == 0:
            raise ValueError("Path must be a non-empty list of hop patterns")

        return QueryPattern(
            pattern_type="multi_hop",
            path_hops=hops,
        )

    # Single-hop relationship pattern
    if "source" in pattern or "target" in pattern or "relation" in pattern:
        return QueryPattern(
            pattern_type="single_hop",
            source_filter=pattern.get("source"),
            target_filter=pattern.get("target"),
            relation_filter=pattern.get("relation"),
        )

    # Entity filter pattern (no relationship)
    return QueryPattern(
        pattern_type="entity_filter",
        source_filter=pattern,  # The pattern itself is the entity filter
    )


async def execute_pattern(
    graph: Any,  # Graph instance (avoiding circular import)
    pattern: QueryPattern,
) -> QueryResult:
    """Execute a parsed query pattern against a graph.

    This function implements the query execution logic with optimization:
    1. Index lookups first (if available)
    2. Filter early (most restrictive filters first)
    3. Minimize graph traversal

    Args:
        graph: Graph instance to query against.
        pattern: Parsed query pattern.

    Returns:
        QueryResult with matched entities and relationships.

    Example:
        >>> pattern = parse_pattern({"type": "Person", "name": "Alice"})
        >>> result = await execute_pattern(graph, pattern)
        >>> print(result.entities)
    """
    if pattern.pattern_type == "entity_filter":
        return await _execute_entity_filter(graph, pattern)
    elif pattern.pattern_type == "single_hop":
        return await _execute_single_hop(graph, pattern)
    elif pattern.pattern_type == "multi_hop":
        return await _execute_multi_hop(graph, pattern)
    else:
        raise ValueError(f"Unknown pattern type: {pattern.pattern_type}")


async def _execute_entity_filter(
    graph: Any,
    pattern: QueryPattern,
) -> QueryResult:
    """Execute entity filter pattern.

    Finds all entities matching the given type and attributes.
    """
    if pattern.source_filter is None:
        return QueryResult(entities=[], relationships=[], metadata={})

    # Extract type and attributes from filter
    entity_type = pattern.source_filter.get("type")
    id_filter = pattern.source_filter.get("id")

    # If specific ID requested, get single entity
    if id_filter and id_filter != "?":
        entity = await graph.get_entity(id_filter)
        entities = [entity] if entity else []
        return QueryResult(
            entities=entities,
            relationships=[],
            metadata={"pattern_type": "entity_filter", "matched_count": len(entities)},
        )

    # Build attribute filters (exclude special keys)
    attributes = {
        k: v
        for k, v in pattern.source_filter.items()
        if k not in ("type", "id") and v != "?"
    }

    # Use graph's find_entities method
    entities = await graph.find_entities(entity_type=entity_type, **attributes)

    return QueryResult(
        entities=entities,
        relationships=[],
        metadata={"pattern_type": "entity_filter", "matched_count": len(entities)},
    )


async def _execute_single_hop(
    graph: Any,
    pattern: QueryPattern,
) -> QueryResult:
    """Execute single-hop relationship pattern.

    Finds all relationships matching source, relation, and target filters.
    """
    # Step 1: Filter source entities
    source_entities = []
    if pattern.source_filter:
        source_result = await _execute_entity_filter(
            graph, QueryPattern(pattern_type="entity_filter", source_filter=pattern.source_filter)
        )
        source_entities = source_result.entities
    else:
        source_entities = await graph.get_all_entities()

    # Step 2: Filter target entities
    target_entities = []
    if pattern.target_filter:
        target_result = await _execute_entity_filter(
            graph, QueryPattern(pattern_type="entity_filter", source_filter=pattern.target_filter)
        )
        target_entities = target_result.entities
    else:
        target_entities = await graph.get_all_entities()

    # Build target ID set for fast lookup
    target_ids = {e.id for e in target_entities} if target_entities else None

    # Step 3: Find matching relationships
    matched_relationships = []
    matched_entities_dict = {}

    for source in source_entities:
        # Get all relationships from this source
        rels = await graph.get_relationships(
            source_id=source.id,
            relation=pattern.relation_filter,
        )

        # Filter by target if specified
        if target_ids is not None:
            rels = [r for r in rels if r.target_id in target_ids]

        matched_relationships.extend(rels)

        # Track entities involved
        matched_entities_dict[source.id] = source
        for rel in rels:
            target = await graph.get_entity(rel.target_id)
            if target:
                matched_entities_dict[target.id] = target

    return QueryResult(
        entities=list(matched_entities_dict.values()),
        relationships=matched_relationships,
        metadata={
            "pattern_type": "single_hop",
            "matched_count": len(matched_relationships),
        },
    )


async def _execute_multi_hop(
    graph: Any,
    pattern: QueryPattern,
) -> QueryResult:
    """Execute multi-hop path pattern.

    Finds all paths matching the sequence of relationship hops.
    """
    if not pattern.path_hops:
        return QueryResult(entities=[], relationships=[], metadata={})

    # Start with all entities if first hop has no source filter
    first_hop = pattern.path_hops[0]

    if "source" in first_hop:
        # Get starting entities from first hop's source filter
        source_result = await _execute_entity_filter(
            graph,
            QueryPattern(pattern_type="entity_filter", source_filter=first_hop["source"]),
        )
        current_entities = source_result.entities
    else:
        current_entities = await graph.get_all_entities()

    all_matched_relationships = []
    all_matched_entities = {e.id: e for e in current_entities}

    # Execute each hop in sequence
    for _i, hop in enumerate(pattern.path_hops):
        next_entities = []
        hop_relationships = []

        relation_filter = hop.get("relation")
        target_filter = hop.get("target")

        # For each current entity, find outgoing relationships
        for entity in current_entities:
            rels = await graph.get_relationships(
                source_id=entity.id,
                relation=relation_filter,
            )

            # Apply target filter if specified
            if target_filter:
                filtered_rels = []
                for rel in rels:
                    target = await graph.get_entity(rel.target_id)
                    if target and _matches_filter(target, target_filter):
                        filtered_rels.append(rel)
                        next_entities.append(target)
                        all_matched_entities[target.id] = target
                rels = filtered_rels
            else:
                # No target filter - accept all targets
                for rel in rels:
                    target = await graph.get_entity(rel.target_id)
                    if target:
                        next_entities.append(target)
                        all_matched_entities[target.id] = target

            hop_relationships.extend(rels)

        all_matched_relationships.extend(hop_relationships)
        current_entities = next_entities

        # If no entities found at this hop, no complete paths exist
        if not current_entities:
            break

    return QueryResult(
        entities=list(all_matched_entities.values()),
        relationships=all_matched_relationships,
        metadata={
            "pattern_type": "multi_hop",
            "hop_count": len(pattern.path_hops),
            "matched_count": len(all_matched_relationships),
        },
    )


def _matches_filter(entity: Entity, filter_dict: dict[str, Any]) -> bool:
    """Check if an entity matches a filter dictionary.

    Args:
        entity: Entity to check.
        filter_dict: Filter with type, id, and attribute constraints.

    Returns:
        True if entity matches all filter criteria.
    """
    # Check type
    if "type" in filter_dict and filter_dict["type"] != "?":
        if entity.entity_type != filter_dict["type"]:
            return False

    # Check ID
    if "id" in filter_dict and filter_dict["id"] != "?":
        if entity.id != filter_dict["id"]:
            return False

    # Check attributes
    for key, value in filter_dict.items():
        if key in ("type", "id"):
            continue
        if value == "?":  # Wildcard
            continue
        if entity.attributes.get(key) != value:
            return False

    return True


async def match(
    graph: Any,
    pattern: dict[str, Any],
) -> QueryResult:
    """Execute a pattern-based query against the graph.

    This is the main entry point for pattern matching queries. It parses
    the pattern and executes it with optimization.

    Args:
        graph: Graph instance to query.
        pattern: Dict-based query pattern.

    Returns:
        QueryResult with matched entities and relationships.

    Examples:
        >>> # Find all Person entities
        >>> result = await match(graph, {"type": "Person"})

        >>> # Find specific relationship
        >>> result = await match(graph, {
        ...     "source": {"id": "alice"},
        ...     "relation": "knows",
        ...     "target": {"type": "Person"}
        ... })

        >>> # Multi-hop path
        >>> result = await match(graph, {
        ...     "path": [
        ...         {"source": {"type": "Person"}, "relation": "works_at"},
        ...         {"relation": "manages", "target": {"id": "project_x"}}
        ...     ]
        ... })
    """
    parsed = parse_pattern(pattern)
    return await execute_pattern(graph, parsed)
