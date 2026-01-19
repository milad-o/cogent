"""
Example: AI Agent Building and Visualizing Knowledge Graph

Demonstrates an AI agent using KnowledgeGraph tools to remember information,
then visualizing what it learned using the .visualize() API.

Shows:
1. Agent learns from text using knowledge graph tools
2. Agent recalls and connects information
3. Visualize the knowledge graph the agent built

Usage:
    uv run python examples/capabilities/kg_agent_viz.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from agenticflow import Agent
from agenticflow.capabilities import KnowledgeGraph
from agenticflow.observability import Observer
from config import get_model


async def main():
    print("=" * 70)
    print("AI Agent Building Knowledge Graph Demo")
    print("=" * 70)

    # Create output directory for artifacts
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")

    # Create knowledge graph capability
    kg = KnowledgeGraph(backend="memory")

    # Create observer to trace tool usage
    observer = Observer.trace()

    # Create agent with knowledge graph
    agent = Agent(
        name="Knowledge Builder",
        model=get_model(),
        instructions="""You are a knowledge extraction agent. 
        
Extract all entities (people, companies, locations, events, projects, etc.) and 
their relationships from the provided text. Capture important attributes for each 
entity and be thorough in identifying connections between them.

When you finish extracting, say "Knowledge extraction complete!" """,
        capabilities=[kg],
        observer=observer,
    )

    # Sample text to extract knowledge from
    company_info = """
    Alice Johnson is a Senior Software Engineer at TechCorp, working in the Platform team.
    She reports to Bob Smith, who is the Engineering Manager of the Platform team.
    Bob has been at TechCorp for 8 years and previously worked at DataSystems Inc.
    
    Alice is currently leading the Migration Project, which aims to move services to Kubernetes.
    The Migration Project started in Q3 2024 and is expected to complete in Q1 2025.
    
    Charlie Davis, a Product Manager, is the stakeholder for the Migration Project.
    Charlie works closely with both Alice and Bob on project planning.
    
    TechCorp is headquartered in San Francisco and has offices in New York and London.
    The company was founded in 2015 and focuses on cloud infrastructure solutions.
    """

    print("\n📄 Text to analyze:")
    print("-" * 70)
    print(company_info)
    print("-" * 70)

    # Agent extracts knowledge
    print("\n🤖 Agent extracting knowledge...")
    print("-" * 70)
    result = await agent.run(
        f"Extract all entities (people, companies, locations, projects, etc.) and their relationships "
        f"from this text into the knowledge graph:\n\n{company_info}"
    )

    # Show tool usage trace
    print("\n🔧 Tool Usage Trace:")
    print("-" * 70)
    from agenticflow.observability import TraceType
    tool_events = observer.events(event_type=TraceType.TOOL_CALLED)
    for event in tool_events:
        tool_name = event.data.get('tool_name', 'unknown')
        args_preview = str(event.data.get('args', {}))[:50]
        print(f"  📌 {tool_name}({args_preview}...)")
    print(f"\nTotal tool calls: {len(tool_events)}")
    print(result)
    print("-" * 70)

    # Show knowledge graph stats
    print("\n📊 Knowledge Graph Statistics:")
    stats = kg.stats()
    print(f"  Entities: {stats['entities']}")
    print(f"  Relationships: {stats['relationships']}")

    # List what was learned
    print("\n📚 Entities stored:")
    entities = kg.get_entities()
    for entity in entities:
        attrs_preview = ", ".join(f"{k}={v}" for k, v in list(entity.attributes.items())[:2])
        print(f"  • {entity.id} ({entity.type}){': ' + attrs_preview if attrs_preview else ''}")

    # Test agent's recall ability
    print("\n🔍 Testing agent's recall ability:")
    print("-" * 70)
    recall_result = await agent.run("What do you know about Alice Johnson?")
    print(recall_result)
    print("-" * 70)

    # Visualize and save the knowledge graph
    print("\n📊 Saving Knowledge Graph Visualizations...")
    
    view = kg.visualize(direction="LR")
    
    # Save diagrams
    try:
        view.save(output_dir / "company_knowledge.mmd")
        view.save(output_dir / "company_knowledge.html")
        view.save(output_dir / "company_knowledge.dot")
        print(f"✓ Saved visualizations to {output_dir}")
        print(f"  • company_knowledge.mmd (Mermaid source)")
        print(f"  • company_knowledge.html (interactive - open in browser!)")
        print(f"  • company_knowledge.dot (Graphviz)")

        
    except Exception as e:
        print(f"✗ Error saving: {e}")

    # Test querying
    print("\n🔎 Testing knowledge queries:")
    print("-" * 70)
    query_result = await agent.run("Who works at TechCorp?")
    print(query_result)
    print("-" * 70)

    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("\nThe agent:")
    print("  1. Extracted entities and relationships from text")
    print("  2. Stored them using knowledge graph tools")
    print("  3. Can recall and query the information")
    print("  4. Generated organized visualization with type-based grouping")
    print(f"\nCheck the saved files in: {output_dir}")
    print("  • company_knowledge.mmd (Mermaid source)")
    print("  • company_knowledge.html (interactive - open in browser!)")
    print("  • company_knowledge.dot (Graphviz)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
