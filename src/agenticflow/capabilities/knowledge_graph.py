"""
KnowledgeGraph capability - persistent entity/relationship memory.

Adds tools for storing and querying entities and their relationships,
enabling multi-hop reasoning and grounded responses.

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import KnowledgeGraph
    
    agent = Agent(
        name="Assistant",
        model=model,
        capabilities=[KnowledgeGraph()],
    )
    
    # Agent can now remember facts and query relationships
    await agent.run("Remember that John works at Acme as a senior engineer")
    await agent.run("Who works at Acme?")  # → John
    ```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import BaseTool, tool

from agenticflow.capabilities.base import BaseCapability


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


class InMemoryGraph:
    """Simple in-memory graph storage using networkx."""
    
    def __init__(self):
        try:
            import networkx as nx
            self._nx = nx
            self.graph = nx.DiGraph()
        except ImportError:
            # Fallback to simple dict-based storage
            self._nx = None
            self._entities: dict[str, Entity] = {}
            self._relationships: list[Relationship] = []
    
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Entity:
        """Add or update an entity."""
        now = datetime.now(timezone.utc)
        
        if self._nx:
            existing = self.graph.nodes.get(entity_id, {})
            # Parse created_at if it's a string (from previous to_dict())
            existing_created = existing.get("created_at", now)
            if isinstance(existing_created, str):
                existing_created = datetime.fromisoformat(existing_created)
            
            entity = Entity(
                id=entity_id,
                type=entity_type,
                attributes={**existing.get("attributes", {}), **(attributes or {})},
                created_at=existing_created,
                updated_at=now,
                source=source or existing.get("source"),
            )
            self.graph.add_node(entity_id, **entity.to_dict())
        else:
            existing = self._entities.get(entity_id)
            entity = Entity(
                id=entity_id,
                type=entity_type,
                attributes={**(existing.attributes if existing else {}), **(attributes or {})},
                created_at=existing.created_at if existing else now,
                updated_at=now,
                source=source or (existing.source if existing else None),
            )
            self._entities[entity_id] = entity
        
        return entity
    
    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        if self._nx:
            if entity_id in self.graph.nodes:
                data = self.graph.nodes[entity_id]
                return Entity(
                    id=entity_id,
                    type=data.get("type", "unknown"),
                    attributes=data.get("attributes", {}),
                    created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
                    updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(timezone.utc),
                    source=data.get("source"),
                )
            return None
        else:
            return self._entities.get(entity_id)
    
    def add_relationship(
        self,
        source_id: str,
        relation: str,
        target_id: str,
        attributes: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Relationship:
        """Add a relationship between entities."""
        rel = Relationship(
            source_id=source_id,
            relation=relation,
            target_id=target_id,
            attributes=attributes or {},
            source=source,
        )
        
        if self._nx:
            # Don't duplicate 'relation' - it's already in to_dict()
            edge_data = rel.to_dict()
            self.graph.add_edge(source_id, target_id, **edge_data)
        else:
            # Check if relationship already exists
            existing = [r for r in self._relationships 
                       if r.source_id == source_id and r.relation == relation and r.target_id == target_id]
            if not existing:
                self._relationships.append(rel)
        
        return rel
    
    def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        direction: str = "outgoing",
    ) -> list[Relationship]:
        """Get relationships for an entity."""
        results = []
        
        if self._nx:
            if direction in ("outgoing", "both"):
                for _, target, data in self.graph.out_edges(entity_id, data=True):
                    if relation is None or data.get("relation") == relation:
                        results.append(Relationship(
                            source_id=entity_id,
                            relation=data.get("relation", "related_to"),
                            target_id=target,
                            attributes=data.get("attributes", {}),
                        ))
            if direction in ("incoming", "both"):
                for source, _, data in self.graph.in_edges(entity_id, data=True):
                    if relation is None or data.get("relation") == relation:
                        results.append(Relationship(
                            source_id=source,
                            relation=data.get("relation", "related_to"),
                            target_id=entity_id,
                            attributes=data.get("attributes", {}),
                        ))
        else:
            for rel in self._relationships:
                if direction in ("outgoing", "both") and rel.source_id == entity_id:
                    if relation is None or rel.relation == relation:
                        results.append(rel)
                if direction in ("incoming", "both") and rel.target_id == entity_id:
                    if relation is None or rel.relation == relation:
                        results.append(rel)
        
        return results
    
    def query(self, pattern: str) -> list[dict[str, Any]]:
        """
        Query the graph with a simple pattern.
        
        Patterns:
        - "entity_id" - Get entity and its relationships
        - "entity_id -relation-> ?" - Find targets of relation
        - "? -relation-> entity_id" - Find sources of relation
        - "entity_id -?-> ?" - Find all outgoing relationships
        """
        results = []
        pattern = pattern.strip()
        
        # Pattern: entity_id -relation-> ?
        if " -" in pattern and "-> ?" in pattern:
            parts = pattern.split(" -")
            source = parts[0].strip()
            relation = parts[1].replace("-> ?", "").strip()
            relation = None if relation == "?" else relation
            
            for rel in self.get_relationships(source, relation, "outgoing"):
                target = self.get_entity(rel.target_id)
                results.append({
                    "source": source,
                    "relation": rel.relation,
                    "target": rel.target_id,
                    "target_type": target.type if target else "unknown",
                    "target_attributes": target.attributes if target else {},
                })
        
        # Pattern: ? -relation-> entity_id
        elif "? -" in pattern and "-> " in pattern:
            parts = pattern.split("-> ")
            target = parts[1].strip()
            relation = parts[0].replace("? -", "").strip()
            relation = None if relation == "?" else relation
            
            for rel in self.get_relationships(target, relation, "incoming"):
                source_entity = self.get_entity(rel.source_id)
                results.append({
                    "source": rel.source_id,
                    "source_type": source_entity.type if source_entity else "unknown",
                    "relation": rel.relation,
                    "target": target,
                })
        
        # Pattern: just entity_id - return entity + relationships
        else:
            entity = self.get_entity(pattern)
            if entity:
                results.append({
                    "entity": entity.to_dict(),
                    "outgoing": [r.to_dict() for r in self.get_relationships(pattern, direction="outgoing")],
                    "incoming": [r.to_dict() for r in self.get_relationships(pattern, direction="incoming")],
                })
        
        return results
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> list[list[str]] | None:
        """Find paths between two entities."""
        if self._nx:
            try:
                paths = list(self._nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_depth))
                return paths if paths else None
            except self._nx.NetworkXNoPath:
                return None
        else:
            # Simple BFS for non-networkx
            visited = set()
            queue = [[source_id]]
            
            while queue:
                path = queue.pop(0)
                node = path[-1]
                
                if node == target_id:
                    return [path]
                
                if node in visited or len(path) > max_depth:
                    continue
                
                visited.add(node)
                
                for rel in self.get_relationships(node, direction="outgoing"):
                    new_path = path + [rel.target_id]
                    queue.append(new_path)
            
            return None
    
    def get_all_entities(self, entity_type: str | None = None) -> list[Entity]:
        """Get all entities, optionally filtered by type."""
        if self._nx:
            entities = []
            for node_id, data in self.graph.nodes(data=True):
                if entity_type is None or data.get("type") == entity_type:
                    entities.append(Entity(
                        id=node_id,
                        type=data.get("type", "unknown"),
                        attributes=data.get("attributes", {}),
                    ))
            return entities
        else:
            if entity_type is None:
                return list(self._entities.values())
            return [e for e in self._entities.values() if e.type == entity_type]
    
    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and its relationships."""
        if self._nx:
            if entity_id in self.graph.nodes:
                self.graph.remove_node(entity_id)
                return True
            return False
        else:
            if entity_id in self._entities:
                del self._entities[entity_id]
                self._relationships = [
                    r for r in self._relationships
                    if r.source_id != entity_id and r.target_id != entity_id
                ]
                return True
            return False
    
    def stats(self) -> dict[str, int]:
        """Get graph statistics."""
        if self._nx:
            return {
                "entities": self.graph.number_of_nodes(),
                "relationships": self.graph.number_of_edges(),
            }
        else:
            return {
                "entities": len(self._entities),
                "relationships": len(self._relationships),
            }


class KnowledgeGraph(BaseCapability):
    """
    Knowledge graph capability for persistent entity/relationship memory.
    
    Provides tools for:
    - remember: Store entities and facts
    - recall: Retrieve entity information
    - connect: Create relationships between entities
    - query: Query the graph with patterns
    - forget: Remove entities
    
    Example:
        ```python
        from agenticflow import Agent
        from agenticflow.capabilities import KnowledgeGraph
        
        kg = KnowledgeGraph()
        agent = Agent(
            name="Assistant",
            model=model,
            capabilities=[kg],
        )
        
        # Agent can now use knowledge tools
        await agent.run("Remember that Alice is a software engineer at Acme Corp")
        await agent.run("What do you know about Alice?")
        
        # Direct access to graph
        kg.graph.add_entity("Bob", "Person", {"role": "Manager"})
        kg.graph.add_relationship("Alice", "reports_to", "Bob")
        ```
    """
    
    def __init__(self, backend: str = "memory"):
        """
        Initialize KnowledgeGraph capability.
        
        Args:
            backend: Storage backend. Currently supports:
                - "memory": In-memory graph (uses networkx if available)
        """
        if backend == "memory":
            self.graph = InMemoryGraph()
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: 'memory'")
        
        self._backend = backend
    
    @property
    def name(self) -> str:
        return "knowledge_graph"
    
    @property
    def description(self) -> str:
        return "Persistent knowledge graph for storing and querying entities and relationships"
    
    @property
    def tools(self) -> list[BaseTool]:
        return [
            self._remember_tool(),
            self._recall_tool(),
            self._connect_tool(),
            self._query_tool(),
            self._forget_tool(),
            self._list_entities_tool(),
        ]
    
    def _remember_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def remember(
            entity: str,
            entity_type: str,
            facts: str,
        ) -> str:
            """
            Remember an entity and its attributes.
            
            Args:
                entity: Name/ID of the entity (e.g., "Alice", "Acme Corp")
                entity_type: Type of entity (e.g., "Person", "Company", "Project")
                facts: JSON string of facts/attributes (e.g., '{"role": "engineer", "team": "backend"}')
            
            Returns:
                Confirmation message
            """
            try:
                attributes = json.loads(facts) if facts else {}
            except json.JSONDecodeError:
                # Try to parse as simple key-value
                attributes = {"info": facts}
            
            graph.add_entity(entity, entity_type, attributes)
            return f"Remembered: {entity} ({entity_type}) with {len(attributes)} attributes"
        
        return remember
    
    def _recall_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def recall(entity: str) -> str:
            """
            Recall information about an entity.
            
            Args:
                entity: Name/ID of the entity to recall
            
            Returns:
                Entity information and relationships, or "not found"
            """
            e = graph.get_entity(entity)
            if not e:
                return f"No information found about '{entity}'"
            
            # Get relationships
            outgoing = graph.get_relationships(entity, direction="outgoing")
            incoming = graph.get_relationships(entity, direction="incoming")
            
            info = [f"**{entity}** ({e.type})"]
            
            if e.attributes:
                info.append("Attributes:")
                for k, v in e.attributes.items():
                    info.append(f"  - {k}: {v}")
            
            if outgoing:
                info.append("Relationships:")
                for rel in outgoing:
                    info.append(f"  - {rel.relation} → {rel.target_id}")
            
            if incoming:
                info.append("Referenced by:")
                for rel in incoming:
                    info.append(f"  - {rel.source_id} {rel.relation} → this")
            
            return "\n".join(info)
        
        return recall
    
    def _connect_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def connect(
            source: str,
            relation: str,
            target: str,
        ) -> str:
            """
            Create a relationship between two entities.
            
            Args:
                source: Source entity (e.g., "Alice")
                relation: Relationship type (e.g., "works_at", "manages", "knows")
                target: Target entity (e.g., "Acme Corp")
            
            Returns:
                Confirmation message
            """
            # Auto-create entities if they don't exist
            if not graph.get_entity(source):
                graph.add_entity(source, "Unknown")
            if not graph.get_entity(target):
                graph.add_entity(target, "Unknown")
            
            graph.add_relationship(source, relation, target)
            return f"Connected: {source} --[{relation}]--> {target}"
        
        return connect
    
    def _query_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def query_knowledge(pattern: str) -> str:
            """
            Query the knowledge graph.
            
            Args:
                pattern: Query pattern. Examples:
                    - "Alice" - Get all info about Alice
                    - "Alice -works_at-> ?" - Where does Alice work?
                    - "? -works_at-> Acme" - Who works at Acme?
                    - "Alice -?-> ?" - All of Alice's relationships
            
            Returns:
                Query results
            """
            results = graph.query(pattern)
            
            if not results:
                return f"No results for pattern: {pattern}"
            
            return json.dumps(results, indent=2, default=str)
        
        return query_knowledge
    
    def _forget_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def forget(entity: str) -> str:
            """
            Remove an entity and all its relationships from memory.
            
            Args:
                entity: Name/ID of the entity to forget
            
            Returns:
                Confirmation message
            """
            if graph.remove_entity(entity):
                return f"Forgot: {entity} (and all its relationships)"
            else:
                return f"Entity '{entity}' not found in memory"
        
        return forget
    
    def _list_entities_tool(self) -> BaseTool:
        graph = self.graph
        
        @tool
        def list_knowledge(entity_type: str = "") -> str:
            """
            List all known entities.
            
            Args:
                entity_type: Optional filter by type (e.g., "Person", "Company")
            
            Returns:
                List of entities
            """
            entities = graph.get_all_entities(entity_type if entity_type else None)
            
            if not entities:
                filter_msg = f" of type '{entity_type}'" if entity_type else ""
                return f"No entities{filter_msg} in knowledge graph"
            
            lines = [f"Knowledge graph ({len(entities)} entities):"]
            for e in entities:
                attrs = ", ".join(f"{k}={v}" for k, v in list(e.attributes.items())[:3])
                lines.append(f"  - {e.id} ({e.type}){': ' + attrs if attrs else ''}")
            
            return "\n".join(lines)
        
        return list_knowledge
    
    def to_dict(self) -> dict[str, Any]:
        """Convert capability info to dictionary."""
        base = super().to_dict()
        base["backend"] = self._backend
        base["stats"] = self.graph.stats()
        return base
