"""
Example 17: SSIS Package Analysis with Agent + Capability

Demonstrates the SSISAnalyzer capability - a domain-specific tool that gives
agents the ability to understand and analyze SSIS (SQL Server Integration Services)
packages intelligently.

Key concepts:
- Capabilities provide domain-specific tools to agents
- Agent decides which tools to use based on the question
- Observer provides visibility into agent reasoning
- Works with single packages or entire project directories

Usage:
    uv run python examples/17_ssis_analyzer.py
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from pathlib import Path

from config import get_model

from agenticflow import Agent
from agenticflow.capabilities import SSISAnalyzer


async def main():
    """Run agent-powered SSIS analysis demo."""
    print("🤖 Agent-Powered SSIS Analysis\n")
    
    # Check for sample SSIS project
    project_dir = Path(__file__).parent / "data" / "ssis_project"
    
    if not project_dir.exists():
        print(f"❌ Project directory not found: {project_dir}")
        return
    
    # Initialize capability
    analyzer = SSISAnalyzer()
    
    # Load all packages from the project
    packages = sorted(project_dir.glob("*.dtsx"))
    print(f"📁 Loading {len(packages)} packages from {project_dir.name}/")
    for pkg_path in packages:
        stats = analyzer.load_package(str(pkg_path))
        print(f"   ✓ {pkg_path.name}: {stats['tasks']} tasks, {stats['data_flows']} data flows")
    
    stats = analyzer.stats()
    print(f"\n📊 Knowledge Graph: {stats['total_entities']} entities, {stats['total_relationships']} relationships")
    
    # Create agent with capability
    model = get_model()
    agent = Agent(
        name="SSIS Analyst",
        model=model,
        instructions="""You are an expert SSIS analyst. Use your tools to analyze packages,
find dependencies, trace data lineage, and answer questions about ETL architecture.
Always use tools to gather information - don't make assumptions.
Be specific with package names, task names, and connection details.""",
        capabilities=[analyzer],
    )
    
    # Show what tools the agent has
    print("\n🔧 Agent's available tools (from SSISAnalyzer capability):")
    for tool in agent.all_tools:
        print(f"   • {tool.name}")
    
    # Questions that showcase intelligent tool usage
    questions = [
        "What connections does this project use and which packages use each one?",
        "What is the execution order when MainETL runs, including child packages?",
        "If I change the StagingDB connection, what packages and tasks would be affected?",
        "Trace the data flow for customer data from extraction to warehouse.",
    ]
    
    print("\n" + "-" * 70)
    print("💬 Asking questions - watch how the agent uses tools intelligently")
    print("-" * 70)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"❓ Question {i}: {question}")
        print("=" * 70)
        
        # Agent runs the task
        response = await agent.run(question)
        
        # Show answer (truncate if very long)
        print(f"\n💡 Answer:")
        if len(response) > 1000:
            print(f"{response[:1000]}...")
        else:
            print(response)
    
    # Cleanup
    analyzer.close()
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
