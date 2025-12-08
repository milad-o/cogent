#!/usr/bin/env python3
"""
Demo: Knowledge Graph Capability

Shows how to use the KnowledgeGraph capability to give an agent
persistent memory of entities and relationships. The agent can
drill down through the graph to find answers.

Key features demonstrated:
1. Load knowledge from a data file
2. Multiple storage backends (memory, sqlite, json)
3. Agent uses KG tools to explore and find answers
4. Multi-hop reasoning through relationships
5. Save/load for persistence
6. Clean programmatic API for direct access
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import json
from pathlib import Path
import tempfile

from config import get_model


def load_knowledge_file(kg, filepath: str) -> dict:
    """
    Load entities and relationships from a knowledge file.
    
    File format:
    - entity|name|type|{"attr": "value"}
    - rel|source|relation|target
    - Lines starting with # are comments
    """
    stats = {"entities": 0, "relationships": 0}
    
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split("|")
            if parts[0] == "entity" and len(parts) >= 3:
                name = parts[1]
                etype = parts[2]
                attrs = json.loads(parts[3]) if len(parts) > 3 else {}
                kg.add_entity(name, etype, attrs)
                stats["entities"] += 1
            elif parts[0] == "rel" and len(parts) >= 4:
                source, relation, target = parts[1], parts[2], parts[3]
                # Auto-create entities if they don't exist
                if not kg.get_entity(source):
                    kg.add_entity(source, "Unknown")
                if not kg.get_entity(target):
                    kg.add_entity(target, "Unknown")
                kg.add_relationship(source, relation, target)
                stats["relationships"] += 1
    
    return stats


async def demo():
    from agenticflow import Agent
    from agenticflow.capabilities import KnowledgeGraph
    
    print("=" * 60)
    print("🧠 Knowledge Graph Capability Demo")
    print("=" * 60)
    
    model = get_model()
    
    # === Step 1: Load KnowledgeGraph from file ===
    print("\n📂 Step 1: Load Knowledge from File")
    print("-" * 40)
    
    kg = KnowledgeGraph()
    data_file = Path(__file__).parent / "data" / "company_knowledge.txt"
    
    stats = load_knowledge_file(kg, str(data_file))
    print(f"✅ Loaded from {data_file.name}:")
    print(f"   Entities: {stats['entities']}")
    print(f"   Relationships: {stats['relationships']}")
    print(f"   Graph stats: {kg.stats()}")
    
    # === Step 2: Explore with direct API ===
    print("\n🔍 Step 2: Direct API Exploration")
    print("-" * 40)
    
    # Get all people
    people = kg.get_entities("Person")
    print(f"People in the system: {[p.id for p in people]}")
    
    # Get all projects
    projects = kg.get_entities("Project")
    print(f"Projects: {[p.id for p in projects]}")
    
    # Query: Who works on ETL Pipeline?
    results = kg.query_graph("? -works_on-> ETL Pipeline")
    print(f"\nWho works on ETL Pipeline?")
    for r in results:
        print(f"  → {r}")
    
    # === Step 3: Multi-hop exploration ===
    print("\n🔗 Step 3: Multi-hop Exploration")
    print("-" * 40)
    
    # Find path from Bob to Eve
    path = kg.find_path("Bob Smith", "Eve Wilson")
    print(f"Path from Bob to Eve: {path}")
    
    # Get Bob's relationships
    bob_rels = kg.get_relationships("Bob Smith", direction="outgoing")
    print(f"\nBob Smith's relationships:")
    for rel in bob_rels:
        print(f"  → {rel.relation} → {rel.target_id}")
    
    # === Step 4: Create agent with KG ===
    print("\n🤖 Step 4: Agent with KnowledgeGraph")
    print("-" * 40)
    
    agent = Agent(
        name="CompanyExpert",
        model=model,
        instructions="""You are a company knowledge expert. You have access to a knowledge graph
containing information about employees, teams, projects, and technologies.

Use the available tools to explore the knowledge graph and find answers.
When asked about relationships or connections, drill down through the graph
to trace the path and provide complete answers.""",
        capabilities=[kg],
    )
    
    print(f"Agent tools: {[t.name for t in agent.all_tools]}")
    
    # === Step 5: Agent drills down to find answers ===
    print("\n💬 Step 5: Agent Queries (Drill-down)")
    print("-" * 40)
    
    questions = [
        "Who is working on the ETL Pipeline project and what technologies does it use?",
        "Who does David Lee report to, and what team does that person lead?",
        "What Python experts do we have and what projects are they working on?",
    ]
    
    for q in questions:
        print(f"\n❓ {q}")
        response = await agent.run(q, strategy="dag")
        print(f"💡 {response}")
    
    # === Step 6: Agent updates knowledge ===
    print("\n➕ Step 6: Agent Adds Knowledge")
    print("-" * 40)
    
    response = await agent.run(
        "Remember that Frank Martinez is a new DevOps Engineer who joined Platform Team and is expert in Kubernetes",
        strategy="dag",
    )
    print(f"Response: {response}")
    
    # Verify Frank was added
    frank = kg.get_entity("Frank Martinez")
    if frank:
        print(f"\n✅ Frank added: {frank.type}, attrs={frank.attributes}")
        rels = kg.get_relationships("Frank Martinez", direction="outgoing")
        print(f"   Relationships: {[(r.relation, r.target_id) for r in rels]}")
    
    # === Step 7: Persistence Demo ===
    print("\n💾 Step 7: Persistence Demo")
    print("-" * 40)
    
    # Demo 1: Save memory graph to file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "knowledge_backup.json"
        kg.save(save_path)
        print(f"✅ Saved to: {save_path}")
        print(f"   File size: {save_path.stat().st_size} bytes")
        
        # Create new graph and load
        from agenticflow.capabilities import KnowledgeGraph as KG
        kg2 = KG(backend="memory")
        kg2.load(save_path)
        print(f"✅ Loaded into new graph: {kg2.stats()}")
        
        # Demo 2: SQLite backend (persistent by default)
        db_path = Path(tmpdir) / "knowledge.db"
        with KG(backend="sqlite", path=db_path) as kg_sqlite:
            kg_sqlite.graph.add_entity("Test", "Demo", {"note": "SQLite persists automatically"})
            print(f"\n✅ SQLite backend:")
            print(f"   Path: {db_path}")
            print(f"   Stats: {kg_sqlite.stats()}")
        
        # Reopen to verify persistence
        with KG(backend="sqlite", path=db_path) as kg_sqlite2:
            test = kg_sqlite2.get_entity("Test")
            print(f"   Reloaded: {test.id if test else 'Not found'} - {test.attributes if test else {}}")
        
        # Demo 3: JSON backend (auto-saves)
        json_path = Path(tmpdir) / "knowledge.json"
        kg_json = KG(backend="json", path=json_path, auto_save=True)
        kg_json.graph.add_entity("Auto", "Demo", {"note": "Auto-saved on change"})
        print(f"\n✅ JSON backend (auto-save):")
        print(f"   Path: {json_path}")
        print(f"   File exists: {json_path.exists()}")
    
    print("\n" + "=" * 60)
    print("✅ Summary:")
    print("   - Load knowledge from structured files")
    print("   - Multiple backends: memory, sqlite, json")
    print("   - Memory backend: explicit save/load")
    print("   - SQLite backend: auto-persistent, great for large graphs")
    print("   - JSON backend: auto-save on changes")
    print("   - Clean API: kg.add_entity(), kg.get_relationships()")
    print("   - Multi-hop queries with find_path()")
    print("   - Agents drill down through graph to find answers")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
