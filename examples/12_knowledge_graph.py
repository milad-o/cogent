#!/usr/bin/env python3
"""
Demo: Knowledge Graph Capability

Shows how to use the KnowledgeGraph capability to give an agent
persistent memory of entities and relationships.

Key features demonstrated:
1. Pre-populate a KnowledgeGraph before giving it to an agent
2. Agent's prompt auto-includes the KG tools
3. Multiple agents can share the same KnowledgeGraph
4. Query patterns for relationships
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
    
    # === Step 1: Create and pre-populate the KnowledgeGraph ===
    print("\n📝 Step 1: Pre-populate KnowledgeGraph")
    print("-" * 40)
    
    kg = KnowledgeGraph()
    
    # Add entities with attributes
    kg.graph.add_entity("Alice", "Person", {
        "role": "Senior Engineer",
        "team": "Backend",
        "skills": ["Python", "Go", "Kubernetes"],
    })
    kg.graph.add_entity("Bob", "Person", {
        "role": "Engineering Manager",
        "team": "Backend",
    })
    kg.graph.add_entity("Acme Corp", "Company", {
        "industry": "Tech",
        "founded": 2015,
        "size": "500+",
    })
    kg.graph.add_entity("Project Alpha", "Project", {
        "status": "active",
        "deadline": "2025-Q1",
    })
    
    # Add relationships
    kg.graph.add_relationship("Alice", "works_at", "Acme Corp")
    kg.graph.add_relationship("Bob", "works_at", "Acme Corp")
    kg.graph.add_relationship("Alice", "reports_to", "Bob")
    kg.graph.add_relationship("Alice", "works_on", "Project Alpha")
    kg.graph.add_relationship("Bob", "manages", "Project Alpha")
    
    print(f"✅ Created graph with {kg.graph.stats()}")
    
    # === Step 2: Create agent with the pre-populated KG ===
    print("\n🤖 Step 2: Create Agent with KnowledgeGraph")
    print("-" * 40)
    
    agent = Agent(
        name="KnowledgeAssistant",
        model=model,
        instructions="You are a helpful assistant with access to a knowledge graph about a company and its employees.",
        capabilities=[kg],
    )
    
    print(f"Agent tools: {[t.name for t in agent.all_tools]}")
    
    # === Step 3: Query the knowledge directly ===
    print("\n🔍 Step 3: Direct Queries (clean API)")
    print("-" * 40)
    
    # Use convenience methods - no tool lookup needed!
    print("Recalling Alice:")
    print(kg.recall("Alice"))
    
    print("\nWho works at Acme Corp?")
    print(kg.query("? -works_at-> Acme Corp"))
    
    # === Step 4: Let the agent use the knowledge ===
    print("\n🤖 Step 4: Agent Query")
    print("-" * 40)
    
    response = await agent.run(
        "Tell me about Alice - what's her role, who does she report to, and what project is she on?",
        strategy="dag",
    )
    print(f"Agent response:\n{response}")
    
    # === Step 5: Shared KG between agents ===
    print("\n👥 Step 5: Shared KnowledgeGraph")
    print("-" * 40)
    
    # Create a second agent that shares the same KG
    agent2 = Agent(
        name="SecondAssistant",
        model=model,
        instructions="You help with project information.",
        capabilities=[kg],  # Same KG instance!
    )
    
    # Agent2 can query the same data
    response2 = await agent2.run(
        "Who manages Project Alpha?",
        strategy="dag",
    )
    print(f"Agent2 response: {response2}")
    
    # === Step 6: Agent adds new knowledge ===
    print("\n➕ Step 6: Agent Adds Knowledge")
    print("-" * 40)
    
    # Agent can add new entities via the remember tool
    response = await agent.run(
        "Remember that Charlie is a new intern who reports to Alice",
        strategy="dag",
    )
    print(f"After adding Charlie: {kg.graph.stats()}")
    
    # Verify Charlie was added
    print(f"Charlie info: {kg.recall('Charlie')}")
    
    print("\n" + "=" * 60)
    print("✅ Summary:")
    print("   - Pre-populate KG before creating agent")
    print("   - Tools auto-injected into agent prompt")
    print("   - Multiple agents can share same KG")
    print("   - Agents can query and add to the KG")
    print("=" * 60)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY first")
    else:
        asyncio.run(demo())
