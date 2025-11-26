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
    
    # === Convenience method tests ===
    
    def test_recall_convenience(self):
        kg = KnowledgeGraph()
        kg.graph.add_entity("Alice", "Person", {"role": "engineer"})
        
        result = kg.recall("Alice")
        assert "Alice" in result
        assert "engineer" in result
    
    def test_query_convenience(self):
        kg = KnowledgeGraph()
        kg.graph.add_entity("Alice", "Person")
        kg.graph.add_entity("Acme", "Company")
        kg.graph.add_relationship("Alice", "works_at", "Acme")
        
        result = kg.query("Alice -works_at-> ?")
        assert "Acme" in result
    
    def test_connect_convenience(self):
        kg = KnowledgeGraph()
        result = kg.connect("Alice", "knows", "Bob")
        assert "Connected" in result
        
        rels = kg.graph.get_relationships("Alice", direction="outgoing")
        assert len(rels) == 1
    
    def test_forget_convenience(self):
        kg = KnowledgeGraph()
        kg.graph.add_entity("Alice", "Person")
        
        result = kg.forget("Alice")
        assert "Forgot" in result
        assert kg.graph.get_entity("Alice") is None
    
    def test_list_entities_convenience(self):
        kg = KnowledgeGraph()
        kg.graph.add_entity("Alice", "Person")
        kg.graph.add_entity("Acme", "Company")
        
        result = kg.list_entities()
        assert "2 entities" in result
        
        result = kg.list_entities("Person")
        assert "Alice" in result


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


class TestInMemoryGraphPersistence:
    """Tests for InMemoryGraph save/load functionality."""
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading graph to JSON."""
        graph = InMemoryGraph()
        graph.add_entity("Alice", "Person", {"role": "engineer"})
        graph.add_entity("Acme", "Company", {"industry": "tech"})
        graph.add_relationship("Alice", "works_at", "Acme")
        
        # Save
        save_path = tmp_path / "graph.json"
        graph.save(save_path)
        
        assert save_path.exists()
        
        # Load into new graph
        graph2 = InMemoryGraph()
        graph2.load(save_path)
        
        # Verify
        alice = graph2.get_entity("Alice")
        assert alice is not None
        assert alice.type == "Person"
        assert alice.attributes["role"] == "engineer"
        
        acme = graph2.get_entity("Acme")
        assert acme is not None
        
        rels = graph2.get_relationships("Alice", direction="outgoing")
        assert len(rels) == 1
        assert rels[0].relation == "works_at"
        assert rels[0].target_id == "Acme"
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file does nothing."""
        graph = InMemoryGraph()
        graph.load(tmp_path / "nonexistent.json")
        
        # Graph should still be empty
        assert graph.stats()["entities"] == 0


class TestSQLiteGraph:
    """Tests for SQLiteGraph backend."""
    
    def test_create_and_query(self, tmp_path):
        from agenticflow.capabilities.knowledge_graph import SQLiteGraph
        
        db_path = tmp_path / "test.db"
        graph = SQLiteGraph(db_path)
        
        # Add data
        graph.add_entity("Alice", "Person", {"role": "engineer"})
        graph.add_entity("Acme", "Company")
        graph.add_relationship("Alice", "works_at", "Acme")
        
        # Query
        alice = graph.get_entity("Alice")
        assert alice is not None
        assert alice.type == "Person"
        assert alice.attributes["role"] == "engineer"
        
        rels = graph.get_relationships("Alice", direction="outgoing")
        assert len(rels) == 1
        assert rels[0].target_id == "Acme"
        
        graph.close()
    
    def test_persistence_across_connections(self, tmp_path):
        """Test that data persists across connections."""
        from agenticflow.capabilities.knowledge_graph import SQLiteGraph
        
        db_path = tmp_path / "persist.db"
        
        # First connection
        graph1 = SQLiteGraph(db_path)
        graph1.add_entity("Alice", "Person", {"role": "engineer"})
        graph1.close()
        
        # Second connection
        graph2 = SQLiteGraph(db_path)
        alice = graph2.get_entity("Alice")
        assert alice is not None
        assert alice.attributes["role"] == "engineer"
        graph2.close()
    
    def test_query_patterns(self, tmp_path):
        """Test query patterns work with SQLite."""
        from agenticflow.capabilities.knowledge_graph import SQLiteGraph
        
        db_path = tmp_path / "query.db"
        graph = SQLiteGraph(db_path)
        
        graph.add_entity("Alice", "Person")
        graph.add_entity("Bob", "Person")
        graph.add_entity("Acme", "Company")
        graph.add_relationship("Alice", "works_at", "Acme")
        graph.add_relationship("Bob", "works_at", "Acme")
        
        # Find who works at Acme
        results = graph.query("? -works_at-> Acme")
        assert len(results) == 2
        sources = {r["source"] for r in results}
        assert sources == {"Alice", "Bob"}
        
        graph.close()
    
    def test_stats(self, tmp_path):
        """Test stats with SQLite."""
        from agenticflow.capabilities.knowledge_graph import SQLiteGraph
        
        db_path = tmp_path / "stats.db"
        graph = SQLiteGraph(db_path)
        
        graph.add_entity("Alice", "Person")
        graph.add_entity("Acme", "Company")
        graph.add_relationship("Alice", "works_at", "Acme")
        
        stats = graph.stats()
        assert stats["entities"] == 2
        assert stats["relationships"] == 1
        
        graph.close()
    
    def test_clear(self, tmp_path):
        """Test clearing SQLite graph."""
        from agenticflow.capabilities.knowledge_graph import SQLiteGraph
        
        db_path = tmp_path / "clear.db"
        graph = SQLiteGraph(db_path)
        
        graph.add_entity("Alice", "Person")
        graph.add_relationship("Alice", "knows", "Bob")
        graph.clear()
        
        assert graph.stats()["entities"] == 0
        assert graph.stats()["relationships"] == 0
        
        graph.close()


class TestJSONFileGraph:
    """Tests for JSONFileGraph backend."""
    
    def test_auto_save(self, tmp_path):
        """Test that changes are auto-saved."""
        from agenticflow.capabilities.knowledge_graph import JSONFileGraph
        import json
        
        file_path = tmp_path / "auto.json"
        graph = JSONFileGraph(file_path, auto_save=True)
        
        graph.add_entity("Alice", "Person", {"role": "engineer"})
        
        # Check file was written
        assert file_path.exists()
        data = json.loads(file_path.read_text())
        assert len(data["entities"]) == 1
        assert data["entities"][0]["id"] == "Alice"
    
    def test_no_auto_save(self, tmp_path):
        """Test manual save when auto_save is False."""
        from agenticflow.capabilities.knowledge_graph import JSONFileGraph
        
        file_path = tmp_path / "manual.json"
        graph = JSONFileGraph(file_path, auto_save=False)
        
        graph.add_entity("Alice", "Person")
        
        # File shouldn't exist yet (or should be empty if created on init)
        if file_path.exists():
            import json
            data = json.loads(file_path.read_text())
            assert len(data.get("entities", [])) == 0
        
        # Now save manually
        graph.save()
        
        import json
        data = json.loads(file_path.read_text())
        assert len(data["entities"]) == 1


class TestKnowledgeGraphBackends:
    """Tests for KnowledgeGraph with different backends."""
    
    def test_memory_backend(self):
        """Test default memory backend."""
        kg = KnowledgeGraph(backend="memory")
        kg.graph.add_entity("Alice", "Person")
        
        assert kg.graph.get_entity("Alice") is not None
        assert kg._backend == "memory"
    
    def test_sqlite_backend(self, tmp_path):
        """Test SQLite backend."""
        db_path = tmp_path / "kg.db"
        kg = KnowledgeGraph(backend="sqlite", path=db_path)
        
        kg.graph.add_entity("Alice", "Person", {"role": "engineer"})
        
        alice = kg.graph.get_entity("Alice")
        assert alice is not None
        assert alice.attributes["role"] == "engineer"
        
        kg.close()
    
    def test_json_backend(self, tmp_path):
        """Test JSON file backend."""
        file_path = tmp_path / "kg.json"
        kg = KnowledgeGraph(backend="json", path=file_path)
        
        kg.graph.add_entity("Alice", "Person", {"role": "engineer"})
        
        # Should auto-save
        assert file_path.exists()
    
    def test_sqlite_requires_path(self):
        """Test that SQLite backend requires path."""
        with pytest.raises(ValueError, match="Path required"):
            KnowledgeGraph(backend="sqlite")
    
    def test_json_requires_path(self):
        """Test that JSON backend requires path."""
        with pytest.raises(ValueError, match="Path required"):
            KnowledgeGraph(backend="json")
    
    def test_invalid_backend(self):
        """Test invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            KnowledgeGraph(backend="invalid")
    
    def test_context_manager(self, tmp_path):
        """Test using KnowledgeGraph as context manager."""
        db_path = tmp_path / "context.db"
        
        with KnowledgeGraph(backend="sqlite", path=db_path) as kg:
            kg.graph.add_entity("Alice", "Person")
        
        # Connection should be closed, but file exists
        assert db_path.exists()
    
    def test_memory_save_load(self, tmp_path):
        """Test save/load with memory backend."""
        kg = KnowledgeGraph(backend="memory")
        kg.graph.add_entity("Alice", "Person", {"role": "engineer"})
        kg.graph.add_entity("Acme", "Company")
        kg.graph.add_relationship("Alice", "works_at", "Acme")
        
        # Save
        save_path = tmp_path / "backup.json"
        kg.save(save_path)
        
        # Clear and reload
        kg.clear()
        assert kg.stats()["entities"] == 0
        
        kg.load(save_path)
        assert kg.stats()["entities"] == 2  # Alice and Acme
        assert kg.graph.get_entity("Alice") is not None
        assert kg.graph.get_entity("Acme") is not None
    
    def test_to_dict_includes_backend(self, tmp_path):
        """Test to_dict includes backend info."""
        db_path = tmp_path / "dict.db"
        kg = KnowledgeGraph(backend="sqlite", path=db_path)
        
        info = kg.to_dict()
        assert info["backend"] == "sqlite"
        assert "path" in info
        assert "stats" in info
        
        kg.close()


class TestKnowledgeGraphFromFile:
    """Tests for KnowledgeGraph.from_file() convenience method."""
    
    def test_from_sqlite_file(self, tmp_path):
        """Test loading from .db file."""
        db_path = tmp_path / "test.db"
        
        # Create a database first
        with KnowledgeGraph(backend="sqlite", path=db_path) as kg:
            kg.graph.add_entity("Alice", "Person", {"role": "engineer"})
        
        # Load using from_file
        kg2 = KnowledgeGraph.from_file(db_path)
        
        assert kg2._backend == "sqlite"
        alice = kg2.get_entity("Alice")
        assert alice is not None
        assert alice.attributes["role"] == "engineer"
        
        kg2.close()
    
    def test_from_json_file(self, tmp_path):
        """Test loading from .json file."""
        json_path = tmp_path / "test.json"
        
        # Create a JSON file first
        kg1 = KnowledgeGraph(backend="json", path=json_path)
        kg1.graph.add_entity("Bob", "Person", {"team": "backend"})
        
        # Load using from_file (fresh instance)
        kg2 = KnowledgeGraph.from_file(json_path)
        
        assert kg2._backend == "json"
        bob = kg2.get_entity("Bob")
        assert bob is not None
        assert bob.attributes["team"] == "backend"
    
    def test_from_sqlite3_extension(self, tmp_path):
        """Test .sqlite3 extension is recognized."""
        db_path = tmp_path / "test.sqlite3"
        
        # Create
        with KnowledgeGraph(backend="sqlite", path=db_path) as kg:
            kg.graph.add_entity("Test", "Demo")
        
        # Load
        kg2 = KnowledgeGraph.from_file(db_path)
        assert kg2._backend == "sqlite"
        kg2.close()
    
    def test_from_unknown_extension_raises(self, tmp_path):
        """Test unknown extension raises error."""
        bad_path = tmp_path / "test.xyz"
        bad_path.touch()
        
        with pytest.raises(ValueError, match="Unknown file extension"):
            KnowledgeGraph.from_file(bad_path)
    
    def test_from_file_with_name(self, tmp_path):
        """Test from_file accepts custom name."""
        db_path = tmp_path / "named.db"
        
        with KnowledgeGraph(backend="sqlite", path=db_path) as kg:
            kg.graph.add_entity("Test", "Demo")
        
        kg2 = KnowledgeGraph.from_file(db_path, name="my_kg")
        assert kg2.name == "my_kg"
        kg2.close()


class TestKnowledgeGraphBatchOperations:
    """Tests for batch operations (performance optimization)."""
    
    def test_add_entities_batch_sqlite(self, tmp_path):
        """Test batch entity insertion with SQLite."""
        db_path = tmp_path / "batch.db"
        
        with KnowledgeGraph(backend="sqlite", path=db_path) as kg:
            entities = [
                ("Alice", "Person", {"role": "engineer"}),
                ("Bob", "Person", {"role": "manager"}),
                ("Acme", "Company", {"industry": "tech"}),
            ]
            
            count = kg.add_entities_batch(entities)
            
            assert count == 3
            assert kg.stats()["entities"] == 3
            
            alice = kg.get_entity("Alice")
            assert alice is not None
            assert alice.attributes["role"] == "engineer"
    
    def test_add_relationships_batch_sqlite(self, tmp_path):
        """Test batch relationship insertion with SQLite."""
        db_path = tmp_path / "batch_rel.db"
        
        with KnowledgeGraph(backend="sqlite", path=db_path) as kg:
            # Add entities first
            kg.add_entities_batch([
                ("Alice", "Person", None),
                ("Bob", "Person", None),
                ("Acme", "Company", None),
            ])
            
            # Batch add relationships
            relationships = [
                ("Alice", "works_at", "Acme"),
                ("Bob", "works_at", "Acme"),
                ("Alice", "reports_to", "Bob"),
            ]
            
            count = kg.add_relationships_batch(relationships)
            
            assert count == 3
            assert kg.stats()["relationships"] == 3
            
            alice_rels = kg.get_relationships("Alice", direction="outgoing")
            assert len(alice_rels) == 2
    
    def test_batch_with_memory_backend(self):
        """Test batch operations work with memory backend (uses default implementation)."""
        kg = KnowledgeGraph(backend="memory")
        
        entities = [
            ("E1", "Type1", {"a": 1}),
            ("E2", "Type1", {"a": 2}),
            ("E3", "Type2", {"a": 3}),
        ]
        
        count = kg.add_entities_batch(entities)
        assert count == 3
        assert kg.stats()["entities"] == 3
    
    def test_pagination_memory(self):
        """Test pagination with memory backend."""
        kg = KnowledgeGraph(backend="memory")
        
        # Add 100 entities
        entities = [(f"E_{i}", "Test", {"idx": i}) for i in range(100)]
        kg.add_entities_batch(entities)
        
        # Get first page
        page1 = kg.get_entities(limit=10, offset=0)
        assert len(page1) == 10
        
        # Get second page
        page2 = kg.get_entities(limit=10, offset=10)
        assert len(page2) == 10
        assert page1[0].id != page2[0].id
        
        # Get page beyond data
        page_beyond = kg.get_entities(limit=10, offset=200)
        assert len(page_beyond) == 0
    
    def test_pagination_sqlite(self, tmp_path):
        """Test pagination with SQLite backend."""
        db_path = tmp_path / "paginate.db"
        
        with KnowledgeGraph(backend="sqlite", path=db_path) as kg:
            # Add 100 entities
            entities = [(f"E_{i}", "Test", {"idx": i}) for i in range(100)]
            kg.add_entities_batch(entities)
            
            # Paginate
            page1 = kg.get_entities(limit=20, offset=0)
            assert len(page1) == 20
            
            page3 = kg.get_entities(limit=20, offset=40)
            assert len(page3) == 20
            
            # Filter + paginate
            kg.add_entities_batch([
                ("P1", "Person", None),
                ("P2", "Person", None),
            ])
            
            people = kg.get_entities(entity_type="Person", limit=10)
            assert len(people) == 2