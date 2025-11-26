"""Extended tests for capabilities - KnowledgeGraph convenience methods and CodebaseAnalyzer."""

import pytest
from agenticflow.capabilities import BaseCapability, KnowledgeGraph


class TestKnowledgeGraphConvenienceMethods:
    """Tests for KnowledgeGraph direct graph access methods."""
    
    def test_add_entity_direct(self):
        kg = KnowledgeGraph()
        entity = kg.add_entity("Alice", "Person", {"role": "Engineer"})
        
        assert entity.id == "Alice"
        assert entity.type == "Person"
        assert entity.attributes["role"] == "Engineer"
    
    def test_add_relationship_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("Alice", "Person")
        kg.add_entity("Acme", "Company")
        
        rel = kg.add_relationship("Alice", "works_at", "Acme")
        
        assert rel.source_id == "Alice"
        assert rel.relation == "works_at"
        assert rel.target_id == "Acme"
    
    def test_get_entity_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("Alice", "Person")
        
        entity = kg.get_entity("Alice")
        assert entity is not None
        assert entity.id == "Alice"
        
        assert kg.get_entity("Bob") is None
    
    def test_get_entities_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("Alice", "Person")
        kg.add_entity("Bob", "Person")
        kg.add_entity("Acme", "Company")
        
        all_entities = kg.get_entities()
        assert len(all_entities) == 3
        
        people = kg.get_entities("Person")
        assert len(people) == 2
    
    def test_get_relationships_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("Alice", "Person")
        kg.add_entity("Bob", "Person")
        kg.add_relationship("Alice", "knows", "Bob")
        
        rels = kg.get_relationships("Alice", direction="outgoing")
        assert len(rels) == 1
        assert rels[0].target_id == "Bob"
    
    def test_query_graph_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("Alice", "Person", {"role": "engineer"})
        kg.add_entity("Acme", "Company")
        kg.add_relationship("Alice", "works_at", "Acme")
        
        results = kg.query_graph("Alice -works_at-> ?")
        assert len(results) == 1
        assert results[0]["target"] == "Acme"
    
    def test_find_path_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("A", "Node")
        kg.add_entity("B", "Node")
        kg.add_entity("C", "Node")
        kg.add_relationship("A", "to", "B")
        kg.add_relationship("B", "to", "C")
        
        paths = kg.find_path("A", "C")
        assert paths is not None
        assert ["A", "B", "C"] in paths
    
    def test_remove_entity_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("Alice", "Person")
        
        assert kg.remove_entity("Alice") is True
        assert kg.get_entity("Alice") is None
        assert kg.remove_entity("NonExistent") is False
    
    def test_get_tool_direct(self):
        kg = KnowledgeGraph()
        
        remember_tool = kg.get_tool("remember")
        assert remember_tool is not None
        assert remember_tool.name == "remember"
        
        assert kg.get_tool("nonexistent") is None
    
    def test_stats_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("Alice", "Person")
        kg.add_entity("Bob", "Person")
        kg.add_relationship("Alice", "knows", "Bob")
        
        stats = kg.stats()
        assert stats["entities"] == 2
        assert stats["relationships"] == 1
    
    def test_clear_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("Alice", "Person")
        kg.add_entity("Bob", "Person")
        kg.add_relationship("Alice", "knows", "Bob")
        
        kg.clear()
        
        assert kg.stats()["entities"] == 0
        assert kg.stats()["relationships"] == 0


class TestCodebaseAnalyzer:
    """Tests for CodebaseAnalyzer capability."""
    
    def test_name_and_description(self):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        analyzer = CodebaseAnalyzer()
        
        assert analyzer.name == "codebase_analyzer"
        assert "codebase" in analyzer.description.lower()
    
    def test_tools_provided(self):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        analyzer = CodebaseAnalyzer()
        tools = analyzer.tools
        
        tool_names = [t.name for t in tools]
        assert "find_callers" in tool_names
        assert "find_usages" in tool_names
        assert "find_subclasses" in tool_names
        assert "get_definition" in tool_names
        assert "find_classes" in tool_names
        assert "find_functions" in tool_names
    
    def test_load_single_file(self, tmp_path):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        test_file = tmp_path / "test_module.py"
        test_file.write_text("""
def hello():
    print("Hello")

class Greeter:
    def greet(self, name):
        return f"Hello, {name}"
""")
        
        analyzer = CodebaseAnalyzer()
        stats = analyzer.load_file(test_file)
        
        assert stats["files"] == 1
        assert stats["modules"] == 1
        assert stats["classes"] == 1
        assert stats["functions"] >= 1
    
    def test_load_directory(self, tmp_path):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        (tmp_path / "module_a.py").write_text("""
def func_a():
    pass

class ClassA:
    pass
""")
        (tmp_path / "module_b.py").write_text("""
from module_a import ClassA

class ClassB(ClassA):
    pass
""")
        
        analyzer = CodebaseAnalyzer()
        stats = analyzer.load_directory(tmp_path)
        
        assert stats["files"] == 2
        assert stats["modules"] == 2
        assert stats["classes"] == 2
    
    def test_skips_test_files_by_default(self, tmp_path):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "test_main.py").write_text("def test_main(): pass")
        
        analyzer = CodebaseAnalyzer()
        stats = analyzer.load_directory(tmp_path, exclude_patterns=["**/test_*"])
        
        assert stats["files"] == 1
    
    def test_include_tests_option(self, tmp_path):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "test_main.py").write_text("def test_main(): pass")
        
        analyzer = CodebaseAnalyzer()
        stats = analyzer.load_directory(tmp_path)  # No exclusions - includes all
        
        assert stats["files"] == 2
    
    def test_get_class(self, tmp_path):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        test_file = tmp_path / "models.py"
        test_file.write_text("""
class Person:
    pass

class Employee(Person):
    pass
""")
        
        analyzer = CodebaseAnalyzer()
        analyzer.load_file(test_file)
        
        person = analyzer.get_definition("Person")
        assert person is not None
        assert person.type == "Class"
    
    def test_find_by_name(self, tmp_path):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        test_file = tmp_path / "utils.py"
        test_file.write_text("""
def process_data():
    pass

def process_file():
    pass

def other_func():
    pass
""")
        
        analyzer = CodebaseAnalyzer()
        analyzer.load_file(test_file)
        
        matches = analyzer.find_functions("process")
        assert len(matches) == 2
    
    def test_stats(self, tmp_path):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        test_file = tmp_path / "app.py"
        test_file.write_text("""
import os

class App:
    def run(self):
        pass

def main():
    pass
""")
        
        analyzer = CodebaseAnalyzer()
        analyzer.load_file(test_file)
        
        stats = analyzer.stats()
        assert "loaded_files" in stats
        assert "types" in stats
        assert stats["loaded_files"] == 1
    
    def test_clear(self, tmp_path):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        test_file = tmp_path / "app.py"
        test_file.write_text("class App: pass")
        
        analyzer = CodebaseAnalyzer()
        analyzer.load_file(test_file)
        
        assert analyzer.stats()["entities"] > 0
        
        analyzer.clear()
        
        assert analyzer.stats()["entities"] == 0
        assert analyzer.stats()["loaded_files"] == 0
    
    def test_kg_property(self):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        analyzer = CodebaseAnalyzer()
        
        assert analyzer.kg is not None
        assert isinstance(analyzer.kg, KnowledgeGraph)
    
    def test_inheritance_detection(self, tmp_path):
        from agenticflow.capabilities import CodebaseAnalyzer
        
        test_file = tmp_path / "models.py"
        test_file.write_text("""
class Base:
    pass

class Child(Base):
    pass
""")
        
        analyzer = CodebaseAnalyzer()
        analyzer.load_file(test_file)
        
        child_class = analyzer.get_definition("Child")
        assert child_class is not None
        
        rels = analyzer.kg.get_relationships("class:Child", "inherits", "outgoing")
        assert len(rels) == 1
        assert rels[0].target_id == "class:Base"
