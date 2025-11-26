"""Tests for capabilities module."""

import pytest
from agenticflow.capabilities import BaseCapability, KnowledgeGraph
from agenticflow.capabilities.knowledge_graph import InMemoryGraph, Entity, Relationship


class TestInMemoryGraph:
    """Tests for InMemoryGraph."""
    
    def test_add_entity(self):
        graph = InMemoryGraph()
        entity = graph.add_entity("Alice", "Person", {"role": "engineer"})
        
        assert entity.id == "Alice"
        assert entity.type == "Person"
        assert entity.attributes["role"] == "engineer"
    
    def test_get_entity(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person", {"role": "engineer"})
        
        entity = graph.get_entity("Alice")
        assert entity is not None
        assert entity.type == "Person"
        
        # Non-existent
        assert graph.get_entity("Bob") is None
    
    def test_update_entity(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person", {"role": "engineer"})
        graph.add_entity("Alice", "Person", {"team": "backend"})
        
        entity = graph.get_entity("Alice")
        assert entity.attributes["role"] == "engineer"
        assert entity.attributes["team"] == "backend"
    
    def test_add_relationship(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person")
        graph.add_entity("Acme", "Company")
        
        rel = graph.add_relationship("Alice", "works_at", "Acme")
        
        assert rel.source_id == "Alice"
        assert rel.relation == "works_at"
        assert rel.target_id == "Acme"
    
    def test_get_relationships_outgoing(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person")
        graph.add_entity("Acme", "Company")
        graph.add_entity("Bob", "Person")
        
        graph.add_relationship("Alice", "works_at", "Acme")
        graph.add_relationship("Alice", "knows", "Bob")
        
        rels = graph.get_relationships("Alice", direction="outgoing")
        assert len(rels) == 2
        
        # Filter by relation
        rels = graph.get_relationships("Alice", relation="works_at", direction="outgoing")
        assert len(rels) == 1
        assert rels[0].target_id == "Acme"
    
    def test_get_relationships_incoming(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person")
        graph.add_entity("Acme", "Company")
        
        graph.add_relationship("Alice", "works_at", "Acme")
        
        rels = graph.get_relationships("Acme", direction="incoming")
        assert len(rels) == 1
        assert rels[0].source_id == "Alice"
    
    def test_query_entity(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person", {"role": "engineer"})
        graph.add_relationship("Alice", "works_at", "Acme")
        
        results = graph.query("Alice")
        assert len(results) == 1
        assert results[0]["entity"]["id"] == "Alice"
    
    def test_query_outgoing_pattern(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person")
        graph.add_entity("Acme", "Company")
        graph.add_relationship("Alice", "works_at", "Acme")
        
        results = graph.query("Alice -works_at-> ?")
        assert len(results) == 1
        assert results[0]["target"] == "Acme"
    
    def test_query_incoming_pattern(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person")
        graph.add_entity("Acme", "Company")
        graph.add_relationship("Alice", "works_at", "Acme")
        
        results = graph.query("? -works_at-> Acme")
        assert len(results) == 1
        assert results[0]["source"] == "Alice"
    
    def test_remove_entity(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person")
        graph.add_entity("Acme", "Company")
        graph.add_relationship("Alice", "works_at", "Acme")
        
        assert graph.remove_entity("Alice")
        assert graph.get_entity("Alice") is None
        
        # Relationships should also be removed
        rels = graph.get_relationships("Acme", direction="incoming")
        assert len(rels) == 0
    
    def test_get_all_entities(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person")
        graph.add_entity("Bob", "Person")
        graph.add_entity("Acme", "Company")
        
        all_entities = graph.get_all_entities()
        assert len(all_entities) == 3
        
        people = graph.get_all_entities("Person")
        assert len(people) == 2
    
    def test_stats(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person")
        graph.add_entity("Acme", "Company")
        graph.add_relationship("Alice", "works_at", "Acme")
        
        stats = graph.stats()
        assert stats["entities"] == 2
        assert stats["relationships"] == 1
    
    def test_find_path(self):
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person")
        graph.add_entity("Bob", "Person")
        graph.add_entity("Charlie", "Person")
        
        graph.add_relationship("Alice", "knows", "Bob")
        graph.add_relationship("Bob", "knows", "Charlie")
        
        paths = graph.find_path("Alice", "Charlie")
        assert paths is not None
        assert len(paths) >= 1
        assert paths[0] == ["Alice", "Bob", "Charlie"]


class TestKnowledgeGraphCapability:
    """Tests for KnowledgeGraph capability."""
    
    def test_name_and_description(self):
        kg = KnowledgeGraph()
        assert kg.name == "knowledge_graph"
        assert "knowledge graph" in kg.description.lower()
    
    def test_tools_provided(self):
        kg = KnowledgeGraph()
        tools = kg.tools
        
        tool_names = [t.name for t in tools]
        assert "remember" in tool_names
        assert "recall" in tool_names
        assert "connect" in tool_names
        assert "query_knowledge" in tool_names
        assert "forget" in tool_names
        assert "list_knowledge" in tool_names
    
    def test_remember_tool(self):
        kg = KnowledgeGraph()
        remember = next(t for t in kg.tools if t.name == "remember")
        
        result = remember.invoke({
            "entity": "Alice",
            "entity_type": "Person",
            "facts": '{"role": "engineer"}'
        })
        
        assert "Remembered" in result
        assert "Alice" in result
        
        # Verify in graph
        entity = kg.graph.get_entity("Alice")
        assert entity is not None
        assert entity.attributes["role"] == "engineer"
    
    def test_recall_tool(self):
        kg = KnowledgeGraph()
        kg.graph.add_entity("Alice", "Person", {"role": "engineer"})
        
        recall = next(t for t in kg.tools if t.name == "recall")
        result = recall.invoke({"entity": "Alice"})
        
        assert "Alice" in result
        assert "Person" in result
        assert "engineer" in result
    
    def test_recall_not_found(self):
        kg = KnowledgeGraph()
        recall = next(t for t in kg.tools if t.name == "recall")
        
        result = recall.invoke({"entity": "Unknown"})
        assert "no information" in result.lower() or "not found" in result.lower()
    
    def test_connect_tool(self):
        kg = KnowledgeGraph()
        connect = next(t for t in kg.tools if t.name == "connect")
        
        result = connect.invoke({
            "source": "Alice",
            "relation": "works_at",
            "target": "Acme"
        })
        
        assert "Connected" in result
        
        # Verify relationship
        rels = kg.graph.get_relationships("Alice", direction="outgoing")
        assert len(rels) == 1
        assert rels[0].target_id == "Acme"
    
    def test_query_tool(self):
        kg = KnowledgeGraph()
        kg.graph.add_entity("Alice", "Person")
        kg.graph.add_entity("Acme", "Company")
        kg.graph.add_relationship("Alice", "works_at", "Acme")
        
        query = next(t for t in kg.tools if t.name == "query_knowledge")
        result = query.invoke({"pattern": "Alice -works_at-> ?"})
        
        assert "Acme" in result
    
    def test_forget_tool(self):
        kg = KnowledgeGraph()
        kg.graph.add_entity("Alice", "Person")
        
        forget = next(t for t in kg.tools if t.name == "forget")
        result = forget.invoke({"entity": "Alice"})
        
        assert "Forgot" in result
        assert kg.graph.get_entity("Alice") is None
    
    def test_list_entities_tool(self):
        kg = KnowledgeGraph()
        kg.graph.add_entity("Alice", "Person")
        kg.graph.add_entity("Bob", "Person")
        kg.graph.add_entity("Acme", "Company")
        
        list_tool = next(t for t in kg.tools if t.name == "list_knowledge")
        
        # All entities
        result = list_tool.invoke({"entity_type": ""})
        assert "3 entities" in result
        
        # Filter by type
        result = list_tool.invoke({"entity_type": "Person"})
        assert "Alice" in result
        assert "Bob" in result
    
    def test_to_dict(self):
        kg = KnowledgeGraph()
        kg.graph.add_entity("Alice", "Person")
        
        info = kg.to_dict()
        assert info["name"] == "knowledge_graph"
        assert info["backend"] == "memory"
        assert info["stats"]["entities"] == 1


class TestAgentWithCapabilities:
    """Tests for Agent with capabilities."""
    
    def test_agent_with_capability(self):
        from agenticflow import Agent
        from unittest.mock import MagicMock
        
        model = MagicMock()
        kg = KnowledgeGraph()
        
        agent = Agent(
            name="TestAgent",
            model=model,
            capabilities=[kg],
        )
        
        # Capability tools should be in agent's tools
        tool_names = [t.name for t in agent.all_tools]
        assert "remember" in tool_names
        assert "recall" in tool_names
    
    def test_agent_capability_list(self):
        from agenticflow import Agent
        from unittest.mock import MagicMock
        
        model = MagicMock()
        kg = KnowledgeGraph()
        
        agent = Agent(
            name="TestAgent",
            model=model,
            capabilities=[kg],
        )
        
        assert len(agent.capabilities) == 1
        assert agent.capabilities[0].name == "knowledge_graph"
    
    def test_get_capability_by_name(self):
        from agenticflow import Agent
        from unittest.mock import MagicMock
        
        model = MagicMock()
        kg = KnowledgeGraph()
        
        agent = Agent(
            name="TestAgent",
            model=model,
            capabilities=[kg],
        )
        
        retrieved = agent.get_capability("knowledge_graph")
        assert retrieved is kg
        
        assert agent.get_capability("nonexistent") is None
    
    def test_invalid_capability_raises(self):
        from agenticflow import Agent
        from unittest.mock import MagicMock
        
        model = MagicMock()
        
        with pytest.raises(TypeError, match="BaseCapability"):
            Agent(
                name="TestAgent",
                model=model,
                capabilities=["not_a_capability"],
            )
