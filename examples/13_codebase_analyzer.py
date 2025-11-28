"""
Example 13: Codebase Analyzer Capability

Demonstrates using the CodebaseAnalyzer capability to parse and query
Python codebases. An agent can drill down through the code structure
to trace data flow and understand how components connect.

Use cases:
- Understanding unfamiliar codebases
- Tracing data flow through functions
- Finding all callers of a function
- Analyzing inheritance hierarchies
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def demo_programmatic():
    """Demo the CodebaseAnalyzer programmatic API."""
    from agenticflow.capabilities import CodebaseAnalyzer

    print("=" * 60)
    print("🔍 CodebaseAnalyzer Programmatic Demo")
    print("=" * 60)

    analyzer = CodebaseAnalyzer()
    print(f"\n✓ Created {analyzer.name}")
    print(f"  Tools: {[t.name for t in analyzer.tools]}")

    # Load the ETL demo project
    etl_path = Path(__file__).parent / "data" / "etl_project"
    print(f"\n📁 Loading ETL project from: {etl_path}")

    stats = analyzer.load_directory(etl_path)
    print(f"\n📊 Loaded:")
    print(f"   Files:     {stats['files']}")
    print(f"   Classes:   {stats['classes']}")
    print(f"   Functions: {stats['functions']}")

    # Find all classes
    print("\n" + "-" * 40)
    print("📦 Classes in ETL project:")
    for cls in analyzer.find_classes():
        bases = cls.attributes.get("bases", [])
        print(f"  • {cls.attributes['name']}" + (f" (extends {bases})" if bases else ""))

    # Find all functions
    print("\n📋 Top-level functions:")
    for func in analyzer.find_functions():
        print(f"  • {func.attributes['name']}()")

    # Get class details
    print("\n" + "-" * 40)
    print("🔎 ETLPipeline class details:")
    pipeline = analyzer.get_definition("ETLPipeline")
    if pipeline:
        print(f"   File: {Path(pipeline.attributes.get('file_path', '')).name}")
        print(f"   Line: {pipeline.attributes.get('lineno')}")
        
        methods = analyzer.get_class_methods("ETLPipeline")
        print(f"   Methods: {[m.attributes['name'] for m in methods]}")

    # Trace: What does DataTransformer call?
    print("\n" + "-" * 40)
    print("🔗 Tracing DataTransformer method calls:")
    transformer = analyzer.get_definition("DataTransformer")
    if transformer:
        # Get methods
        methods = analyzer.get_class_methods("DataTransformer")
        for method in methods:
            print(f"   • {method.attributes['name']}()")
            # Check what it calls
            method_id = method.id
            rels = analyzer.kg.get_relationships(method_id, "calls", direction="outgoing")
            for rel in rels:
                print(f"      → calls: {rel.target_id.replace('callable:', '')}")

    # Knowledge graph stats
    print("\n" + "-" * 40)
    kg_stats = analyzer.kg.stats()
    print(f"📊 Knowledge Graph: {kg_stats['entities']} entities, {kg_stats['relationships']} relationships")
    
    return analyzer


async def demo_agent(analyzer):
    """Demo an agent using CodebaseAnalyzer to answer questions."""
    from agenticflow.models import ChatModel
    from agenticflow import Agent

    print("\n" + "=" * 60)
    print("🤖 Agent with CodebaseAnalyzer Demo")
    print("=" * 60)

    model = ChatModel(model="gpt-4o-mini", temperature=0)

    agent = Agent(
        name="CodeExpert",
        model=model,
        instructions="""You are a code analysis expert. You have access to a parsed codebase
stored in a knowledge graph. Use the available tools to explore the code structure
and answer questions about:
- Classes and their methods
- Function definitions
- What functions call other functions
- Class inheritance

When tracing code flow, follow the relationships to understand how data moves.""",
        capabilities=[analyzer],
    )

    print(f"\nAgent tools: {[t.name for t in agent.all_tools]}")

    # Agent queries
    questions = [
        "What classes are in this codebase and what do they do?",
        "Trace the ETL flow: what happens when ETLPipeline.run() is called?",
        "What methods does DataLoader have?",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        print("-" * 40)
        response = await agent.run(q, strategy="dag")
        print(f"💡 {response}")

    print("\n" + "=" * 60)
    print("✅ Summary:")
    print("   - CodebaseAnalyzer parses Python AST")
    print("   - Builds KG of classes, functions, calls")
    print("   - Agent uses tools to explore and trace")
    print("   - Supports drilling down through code structure")
    print("=" * 60)


def main():
    # First run programmatic demo (no API key needed)
    analyzer = demo_programmatic()

    # Then run agent demo if API key available
    if os.getenv("OPENAI_API_KEY"):
        asyncio.run(demo_agent(analyzer))
    else:
        print("\n⚠️  Set OPENAI_API_KEY to run the agent demo")


if __name__ == "__main__":
    main()
