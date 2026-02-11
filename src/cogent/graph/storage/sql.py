"""SQL-based storage implementation for graphs using SQLAlchemy v2.

This module provides SQLStorage - persistent storage using any SQL database
(PostgreSQL, MySQL, SQLite) with async SQLAlchemy ORM and DRY CRUD patterns.
"""

import json
from datetime import datetime, UTC

from sqlalchemy import JSON, String, Text, ForeignKey, select, delete, func
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from cogent.graph.models import Entity, Relationship


# SQLAlchemy v2 declarative base
class Base(DeclarativeBase):
    pass


# Models
class EntityModel(Base):
    """SQLAlchemy model for entities."""

    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    type: Mapped[str] = mapped_column(String(100), index=True)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class RelationshipModel(Base):
    """SQLAlchemy model for relationships."""

    __tablename__ = "relationships"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    relation: Mapped[str] = mapped_column(String(100), index=True)
    target_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)


class SQLStorage:
    """SQL-based persistent storage with SQLAlchemy v2 ORM.

    Supports any SQL database (PostgreSQL, MySQL, SQLite) with async operations.
    Uses DRY CRUD patterns with generic base methods.

    Args:
        connection_string: SQLAlchemy connection string.
            Examples:
                - "sqlite+aiosqlite:///:memory:" (in-memory SQLite)
                - "sqlite+aiosqlite:///graph.db" (file-based SQLite)
                - "postgresql+asyncpg://user:pass@localhost/db"
                - "mysql+aiomysql://user:pass@localhost/db"

    Example:
        >>> storage = SQLStorage("sqlite+aiosqlite:///:memory:")
        >>> await storage.initialize()
        >>> await storage.add_entity("alice", "Person", name="Alice")
    """

    def __init__(self, connection_string: str) -> None:
        """Initialize SQL storage."""
        self.connection_string = connection_string
        self.engine = create_async_engine(connection_string, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        if self._initialized:
            return

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        self._initialized = True

    async def close(self) -> None:
        """Close database connections."""
        await self.engine.dispose()

    # --- Generic CRUD Base Methods (DRY) ---

    async def _get_by_id[T](
        self, model: type[T], id: str | int, session: AsyncSession
    ) -> T | None:
        """Generic: Get entity by ID."""
        return await session.get(model, id)

    async def _create[T](self, entity: T, session: AsyncSession) -> T:
        """Generic: Create entity."""
        session.add(entity)
        await session.flush()
        await session.refresh(entity)
        return entity

    async def _create_many[T](self, entities: list[T], session: AsyncSession) -> list[T]:
        """Generic: Create multiple entities."""
        session.add_all(entities)
        await session.flush()
        for entity in entities:
            await session.refresh(entity)
        return entities

    async def _delete[T](
        self, model: type[T], id: str | int, session: AsyncSession
    ) -> bool:
        """Generic: Delete entity by ID."""
        entity = await self._get_by_id(model, id, session)
        if not entity:
            return False
        await session.delete(entity)
        return True

    # --- Storage Protocol Implementation ---

    async def add_entity(
        self,
        id: str,
        entity_type: str,
        **attributes: object,
    ) -> Entity:
        """Add a new entity to storage."""
        await self.initialize()

        async with self.async_session() as session:
            async with session.begin():
                # Check if entity exists
                existing = await self._get_by_id(EntityModel, id, session)
                if existing:
                    raise ValueError(f"Entity with ID '{id}' already exists")

                # Create entity model
                entity_model = EntityModel(
                    id=id,
                    type=entity_type,
                    attributes=dict(attributes),
                )

                # Use generic create
                await self._create(entity_model, session)

                # Convert to domain model
                return Entity(
                    id=entity_model.id,
                    entity_type=entity_model.type,
                    attributes=entity_model.attributes,
                    created_at=entity_model.created_at,
                    updated_at=entity_model.updated_at,
                )

    async def add_entities(self, entities: list[Entity]) -> list[Entity]:
        """Bulk add multiple entities."""
        await self.initialize()

        async with self.async_session() as session:
            async with session.begin():
                # Check for existing entities
                for entity in entities:
                    existing = await self._get_by_id(EntityModel, entity.id, session)
                    if existing:
                        raise ValueError(f"Entity with ID '{entity.id}' already exists")

                # Create entity models
                entity_models = [
                    EntityModel(
                        id=ent.id,
                        type=ent.entity_type,
                        attributes=ent.attributes,
                    )
                    for ent in entities
                ]

                # Use generic bulk create
                await self._create_many(entity_models, session)

                # Convert to domain models
                return [
                    Entity(
                        id=model.id,
                        entity_type=model.type,
                        attributes=model.attributes,
                        created_at=model.created_at,
                        updated_at=model.updated_at,
                    )
                    for model in entity_models
                ]

    async def get_entity(self, id: str) -> Entity | None:
        """Retrieve an entity by ID."""
        await self.initialize()

        async with self.async_session() as session:
            entity_model = await self._get_by_id(EntityModel, id, session)

            if not entity_model:
                return None

            return Entity(
                id=entity_model.id,
                entity_type=entity_model.type,
                attributes=entity_model.attributes,
                created_at=entity_model.created_at,
                updated_at=entity_model.updated_at,
            )

    async def remove_entity(self, id: str) -> bool:
        """Remove an entity and all its relationships (cascade)."""
        await self.initialize()

        async with self.async_session() as session:
            async with session.begin():
                # First, manually delete relationships
                await session.execute(
                    delete(RelationshipModel).where(
                        (RelationshipModel.source_id == id) | (RelationshipModel.target_id == id)
                    )
                )
                # Then delete the entity
                return await self._delete(EntityModel, id, session)

    async def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        **attributes: object,
    ) -> Relationship:
        """Add a new relationship between entities."""
        await self.initialize()

        async with self.async_session() as session:
            async with session.begin():
                # Verify entities exist
                source = await self._get_by_id(EntityModel, source_id, session)
                if not source:
                    raise ValueError(f"Source entity '{source_id}' does not exist")

                target = await self._get_by_id(EntityModel, target_id, session)
                if not target:
                    raise ValueError(f"Target entity '{target_id}' does not exist")

                # Create relationship model
                rel_model = RelationshipModel(
                    source_id=source_id,
                    relation=relation,
                    target_id=target_id,
                    attributes=dict(attributes),
                )

                # Use generic create
                await self._create(rel_model, session)

                # Convert to domain model
                return Relationship(
                    source_id=rel_model.source_id,
                    relation=rel_model.relation,
                    target_id=rel_model.target_id,
                    attributes=rel_model.attributes,
                )

    async def add_relationships(
        self, relationships: list[Relationship]
    ) -> list[Relationship]:
        """Bulk add multiple relationships."""
        await self.initialize()

        async with self.async_session() as session:
            async with session.begin():
                # Verify all entities exist
                for rel in relationships:
                    source = await self._get_by_id(EntityModel, rel.source_id, session)
                    if not source:
                        raise ValueError(f"Source entity '{rel.source_id}' does not exist")

                    target = await self._get_by_id(EntityModel, rel.target_id, session)
                    if not target:
                        raise ValueError(f"Target entity '{rel.target_id}' does not exist")

                # Create relationship models
                rel_models = [
                    RelationshipModel(
                        source_id=rel.source_id,
                        relation=rel.relation,
                        target_id=rel.target_id,
                        attributes=rel.attributes,
                    )
                    for rel in relationships
                ]

                # Use generic bulk create
                await self._create_many(rel_models, session)

                # Convert to domain models
                return relationships

    async def get_relationships(
        self,
        source_id: str | None = None,
        relation: str | None = None,
        target_id: str | None = None,
    ) -> list[Relationship]:
        """Query relationships by source, relation, and/or target."""
        await self.initialize()

        async with self.async_session() as session:
            # Build query with filters
            stmt = select(RelationshipModel)

            if source_id is not None:
                stmt = stmt.where(RelationshipModel.source_id == source_id)
            if relation is not None:
                stmt = stmt.where(RelationshipModel.relation == relation)
            if target_id is not None:
                stmt = stmt.where(RelationshipModel.target_id == target_id)

            result = await session.execute(stmt)
            rel_models = result.scalars().all()

            # Convert to domain models
            return [
                Relationship(
                    source_id=model.source_id,
                    relation=model.relation,
                    target_id=model.target_id,
                    attributes=model.attributes,
                )
                for model in rel_models
            ]

    async def query(self, pattern: str) -> list[dict[str, object]]:
        """Execute a pattern-based query."""
        # Placeholder implementation
        return []

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int | None = None,
    ) -> list[str] | None:
        """Find shortest path using recursive CTE (PostgreSQL) or BFS fallback."""
        await self.initialize()

        # Check if entities exist
        async with self.async_session() as session:
            source = await self._get_by_id(EntityModel, source_id, session)
            target = await self._get_by_id(EntityModel, target_id, session)

            if not source or not target:
                return None

            if source_id == target_id:
                return [source_id]

            # Try recursive CTE for PostgreSQL
            if "postgresql" in self.connection_string.lower():
                return await self._find_path_recursive_cte(
                    source_id, target_id, max_depth, session
                )

            # Fallback to BFS for other databases
            return await self._find_path_bfs(source_id, target_id, max_depth, session)

    async def _find_path_recursive_cte(
        self,
        source_id: str,
        target_id: str,
        max_depth: int | None,
        session: AsyncSession,
    ) -> list[str] | None:
        """Find path using PostgreSQL recursive CTE (optimal)."""
        # Recursive CTE implementation
        # For now, fallback to BFS
        return await self._find_path_bfs(source_id, target_id, max_depth, session)

    async def _find_path_bfs(
        self,
        source_id: str,
        target_id: str,
        max_depth: int | None,
        session: AsyncSession,
    ) -> list[str] | None:
        """Find path using BFS (fallback for all DBs)."""
        from collections import deque

        # Fetch all relationships once
        stmt = select(RelationshipModel)
        result = await session.execute(stmt)
        relationships = result.scalars().all()

        # Build adjacency map
        adjacency: dict[str, list[str]] = {}
        for rel in relationships:
            if rel.source_id not in adjacency:
                adjacency[rel.source_id] = []
            adjacency[rel.source_id].append(rel.target_id)

        # BFS
        queue: deque[tuple[str, list[str]]] = deque([(source_id, [source_id])])
        visited: set[str] = {source_id}

        while queue:
            current_id, path = queue.popleft()

            # Check max depth
            if max_depth is not None and len(path) - 1 >= max_depth:
                continue

            # Get neighbors
            for next_id in adjacency.get(current_id, []):
                if next_id == target_id:
                    return path + [next_id]

                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [next_id]))

        return None

    async def get_all_entities(self) -> list[Entity]:
        """Retrieve all entities."""
        await self.initialize()

        async with self.async_session() as session:
            stmt = select(EntityModel)
            result = await session.execute(stmt)
            entity_models = result.scalars().all()

            return [
                Entity(
                    id=model.id,
                    entity_type=model.type,
                    attributes=model.attributes,
                    created_at=model.created_at,
                    updated_at=model.updated_at,
                )
                for model in entity_models
            ]

    async def stats(self) -> dict[str, int]:
        """Get storage statistics."""
        await self.initialize()

        async with self.async_session() as session:
            # Count entities
            entity_count_stmt = select(func.count()).select_from(EntityModel)
            entity_count = await session.scalar(entity_count_stmt) or 0

            # Count relationships
            rel_count_stmt = select(func.count()).select_from(RelationshipModel)
            rel_count = await session.scalar(rel_count_stmt) or 0

            return {
                "entity_count": int(entity_count),
                "relationship_count": int(rel_count),
            }

    async def clear(self) -> None:
        """Remove all entities and relationships."""
        await self.initialize()

        async with self.async_session() as session:
            async with session.begin():
                # Delete all relationships first (foreign keys)
                await session.execute(delete(RelationshipModel))
                # Delete all entities
                await session.execute(delete(EntityModel))
