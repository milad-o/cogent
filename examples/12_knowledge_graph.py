#!/usr/bin/env python3
"""
Demo: Knowledge Graph Capability

Shows how to use the KnowledgeGraph capability to give an agent
persistent memory of entities and relationships.

The agent can:
- Remember facts about people, companies, projects
- Recall information and relationships
- Query the graph for multi-hop reasoning
- Build knowledge over multiple conversations
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()


async def demo():
    from langchain_openai import ChatOpenAI
    from agenticflow import Agent
    from agenticflow.capabilities import KnowledgeGraph
    
    print("=" * 60)
    print("🧠 Knowledge Graph Capability Demo")
    print("=" * 60)
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create agent with KnowledgeGraph capability
    kg = KnowledgeGraph()
    
    agent = Agent(
        name="KnowledgeAssistant",
        model=model,
        instructions="""You are a helpful assistant with a knowledge graph memory.
        
Use your tools to:
- remember: Store facts about entities (people, companies, projects, etc.)
- recall: Look up what you know about an entity
- connect: Create relationships between entities
- query_knowledge: Query for patterns like "Alice -works_at-> ?"
- list_knowledge: See all known entities

Always use your knowledge tools to store important facts from conversations
and to answer questions about known entities.""",
        capabilities=[kg],
        memory=True,  # Also keep conversation memory
    )
    
    print(f"\n📌 Created agent with KnowledgeGraph capability")
    print(f"   Tools: {[t.name for t in agent.all_tools]}")
    
    # === Demo 1: Building knowledge ===
    print("\n" + "-" * 40)
    print("📝 Demo 1: Building Knowledge")
    print("-" * 40)
    
    # Manually add some facts to the graph
    kg.graph.add_entity("Alice", "Person", {"role": "Senior Engineer", "team": "Backend"})
    kg.graph.add_entity("Bob", "Person", {"role": "Engineering Manager"})
    kg.graph.add_entity("Acme Corp", "Company", {"industry": "Tech", "size": "500+"})
    
    kg.graph.add_relationship("Alice", "works_at", "Acme Corp")
    kg.graph.add_relationship("Bob", "works_at", "Acme Corp")
    kg.graph.add_relationship("Alice", "reports_to", "Bob")
    
    print("Added entities: Alice, Bob, Acme Corp")
    print("Added relationships: Alice works_at Acme, reports_to Bob")
    
    # === Demo 2: Using tools to query ===
    print("\n" + "-" * 40)
    print("🔍 Demo 2: Querying with Tools")
    print("-" * 40)
    
    # Use the tools directly
    recall = next(t for t in kg.tools if t.name == "recall")
    print("\nRecalling Alice:")
    print(recall.invoke({"entity": "Alice"}))
    
    query = next(t for t in kg.tools if t.name == "query_knowledge")
    print("\nWho works at Acme Corp?")
    print(query.invoke({"pattern": "? -works_at-> Acme Corp"}))
    
    print("\nWho does Alice report to?")
    print(query.invoke({"pattern": "Alice -reports_to-> ?"}))
    
    # === Demo 3: Multi-hop reasoning ===
    print("\n" + "-" * 40)
    print("🔗 Demo 3: Multi-hop Reasoning")
    print("-" * 40)
    
    # Add more data for multi-hop
    kg.graph.add_entity("Project X", "Project", {"status": "active", "deadline": "Q1 2025"})
    kg.graph.add_relationship("Alice", "works_on", "Project X")
    kg.graph.add_relationship("Bob", "oversees", "Project X")
    
    print("Added Project X with relationships")
    
    # Find path between entities
    paths = kg.graph.find_path("Alice", "Project X")
    print(f"\nPath from Alice to Project X: {paths}")
    
    # === Demo 4: Agent using knowledge ===
    print("\n" + "-" * 40)
    print("🤖 Demo 4: Agent Using Knowledge")
    print("-" * 40)
    
    # Let the agent use the knowledge graph
    response = await agent.run(
        "What do you know about Alice and her work relationships?",
        strategy="react",
    )
    print(f"\nAgent response:\n{response}")
    
    # === Demo 5: Stats ===
    print("\n" + "-" * 40)
    print("📊 Knowledge Graph Stats")
    print("-" * 40)
    stats = kg.graph.stats()
    print(f"Entities: {stats['entities']}")
    print(f"Relationships: {stats['relationships']}")
    
    list_tool = next(t for t in kg.tools if t.name == "list_knowledge")
    print("\nAll entities:")
    print(list_tool.invoke({"entity_type": ""}))
    
    print("\n" + "=" * 60)
    print("✅ Knowledge Graph capability provides:")
    print("   - Persistent entity storage")
    print("   - Relationship tracking")
    print("   - Pattern-based queries")
    print("   - Multi-hop reasoning")
    print("=" * 60)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY first")
    else:
        asyncio.run(demo())
