"""
Example: Agent-Powered SSIS Project Analysis

This example demonstrates using an AI agent with the SSISAnalyzer capability
to answer challenging questions about an SSIS project - the kind of questions
that data engineers and DBAs typically need to answer for impact analysis,
documentation, and troubleshooting.

Usage:
    # With LLM (requires OPENAI_API_KEY):
    uv run python examples/19_ssis_agent_analysis.py
    
    # Direct tool demo (no LLM required):
    uv run python examples/19_ssis_agent_analysis.py --direct
"""

import asyncio
from pathlib import Path

from config import get_model

from agenticflow import Agent, Flow
from agenticflow.capabilities import SSISAnalyzer


# Challenging questions that require deep analysis
CHALLENGING_QUESTIONS = [
    # Impact Analysis
    "What packages and data flows would be affected if we change the SourceCRM connection?",
    
    # Dependency Understanding  
    "What is the complete execution order when MainETL runs? Show all tasks including child packages.",
    
    # Data Lineage
    "Trace the customer data lineage from source to final destination.",
    
    # Cross-Package Analysis
    "Which packages share the StagingDB connection and what operations do they perform?",
]


async def main():
    """Run agent-powered SSIS analysis."""
    
    print("=" * 70)
    print("🤖 Agent-Powered SSIS Project Analysis")
    print("=" * 70)
    
    # Path to sample SSIS project
    project_dir = Path(__file__).parent / "data" / "ssis_project"
    
    if not project_dir.exists():
        print(f"❌ Project directory not found: {project_dir}")
        return
    
    # Create the SSISAnalyzer capability and load the project
    print(f"\n📁 Loading SSIS project from: {project_dir}")
    ssis = SSISAnalyzer()
    
    for pkg_path in sorted(project_dir.glob("*.dtsx")):
        stats = ssis.load_package(str(pkg_path))
        print(f"   ✓ {pkg_path.name}: {stats['tasks']} tasks, {stats['data_flows']} data flows")
    
    print(f"\n📊 Total: {ssis.stats()['total_entities']} entities, {ssis.stats()['total_relationships']} relationships")
    
    # Create model from config
    model = get_model()
    
    # Create an agent with the SSIS capability
    print("\n🤖 Creating agent with SSISAnalyzer capability...")
    
    agent = Agent(
        name="ssis_analyst",
        model=model,
        instructions="""You are an expert SSIS (SQL Server Integration Services) analyst.

Use your tools to analyze packages, find dependencies, trace data lineage, and answer
questions about the ETL architecture. Always use tools to gather information before
answering - don't make assumptions.

Provide detailed answers with specific package and task names.""",
        capabilities=[ssis],
    )
    
    # Show the dynamically generated prompt
    print("\n📋 Agent's effective system prompt (auto-generated):")
    print("-" * 50)
    effective_prompt = agent.get_effective_system_prompt()
    # Show first 1500 chars
    if effective_prompt:
        preview = effective_prompt[:1500]
        if len(effective_prompt) > 1500:
            preview += "\n... (truncated)"
        print(preview)
    print("-" * 50)
    
    # Ask challenging questions using Flow (which now properly executes tools)
    print("\n" + "=" * 70)
    print("📝 Asking Challenging Questions via Flow")
    print("=" * 70)
    
    # Create a Flow with single agent - this should work!
    flow = Flow(
        name="ssis_analysis",
        agents=[agent],
        topology="pipeline",
    )
    
    for i, question in enumerate(CHALLENGING_QUESTIONS, 1):
        print(f"\n{'─' * 70}")
        print(f"❓ Question {i}: {question}")
        print("─" * 70)
        
        try:
            # Use Flow.run() - now properly executes tools via agent.run()
            result = await flow.run(question)
            
            # Get the final thought from results
            if result.results:
                answer = result.results[-1].get('thought', '')
                # Clean up response
                answer = str(answer).replace("FINAL ANSWER:", "").strip()
                print(f"\n💡 Answer:\n{answer[:1500]}")
                if len(answer) > 1500:
                    print("... (truncated)")
            else:
                print("\n💡 No result returned")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    # Cleanup
    ssis.close()
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


async def demo_without_llm():
    """Demonstrate the capability tools directly without an LLM."""
    
    print("=" * 70)
    print("🔧 Direct Tool Demonstration (No LLM Required)")
    print("=" * 70)
    
    project_dir = Path(__file__).parent / "data" / "ssis_project"
    
    # Create and load
    ssis = SSISAnalyzer()
    for pkg_path in sorted(project_dir.glob("*.dtsx")):
        ssis.load_package(str(pkg_path))
    
    print("\n" + "─" * 70)
    print("❓ Q1: What packages would be affected if we change SourceCRM connection?")
    print("─" * 70)
    
    # Find packages using SourceCRM
    print("\n🔍 Analyzing SourceCRM usage...")
    connections = ssis.find_connections()
    source_crm = [c for c in connections if c['attributes']['name'] == 'SourceCRM']
    
    if source_crm:
        pkg = source_crm[0]['id'].split('.')[0]
        print(f"   SourceCRM is defined in package: {pkg}")
        
        # What tasks are in that package?
        tasks = ssis.find_tasks()
        pkg_tasks = [t for t in tasks if t['id'].startswith(pkg + '.')]
        print(f"   Tasks that would be affected:")
        for task in pkg_tasks:
            print(f"      - {task['id'].split('.')[-1]} ({task['type']})")
        
        # Who calls this package?
        deps = ssis.find_package_dependencies(pkg)
        if deps['called_by']:
            print(f"   Called by: {deps['called_by']}")
    
    print("\n" + "─" * 70)
    print("❓ Q2: What is the execution order when MainETL runs?")
    print("─" * 70)
    
    print("\n🔍 Analyzing MainETL execution flow...")
    
    # Get MainETL execution order
    main_order = ssis.get_execution_order("MainETL")
    print("   MainETL execution order:")
    for i, task_id in enumerate(main_order, 1):
        task_name = task_id.split(".")[-1]
        
        # Check if it's an Execute Package task
        tasks = ssis.find_tasks()
        task_info = next((t for t in tasks if t['id'] == task_id), None)
        
        if task_info and task_info['type'] == 'ExecutePackageTask':
            child_pkg = task_info['attributes'].get('child_package', '')
            child_name = Path(child_pkg).stem if child_pkg else 'Unknown'
            print(f"      {i}. {task_name} → calls {child_name}")
            
            # Get child package execution order
            child_order = ssis.get_execution_order(child_name)
            for j, child_task_id in enumerate(child_order, 1):
                child_task_name = child_task_id.split(".")[-1]
                print(f"         {i}.{j}. {child_task_name}")
        else:
            task_type = task_info['type'] if task_info else 'Unknown'
            print(f"      {i}. {task_name} [{task_type}]")
    
    print("\n" + "─" * 70)
    print("❓ Q3: Trace customer data lineage")
    print("─" * 70)
    
    print("\n🔍 Tracing customer data flow...")
    
    # Find customer-related data flows
    data_flows = ssis.find_data_flows()
    customer_flows = [df for df in data_flows if 'customer' in df['id'].lower()]
    
    for flow in customer_flows:
        print(f"\n   Data Flow: {flow['id']}")
        
        # Get components in this flow
        all_entities = ssis.kg.graph.get_all_entities()
        components = [e for e in all_entities if e.attributes.get('data_flow') == flow['id']]
        
        sources = [c for c in components if 'source' in c.type.lower()]
        transforms = [c for c in components if c.type in ['DerivedColumn', 'Lookup', 'ConditionalSplit']]
        destinations = [c for c in components if 'destination' in c.type.lower()]
        
        if sources:
            print(f"      📥 Sources: {[s.attributes['name'] for s in sources]}")
        if transforms:
            print(f"      🔄 Transforms: {[t.attributes['name'] for t in transforms]}")
        if destinations:
            print(f"      📤 Destinations: {[d.attributes['name'] for d in destinations]}")
    
    # Also check LoadWarehouse for customer dimension
    print("\n   Following to LoadWarehouse...")
    load_flows = [df for df in data_flows if 'DimCustomer' in df['id'] or 'customer' in df['id'].lower()]
    for flow in load_flows:
        if 'LoadWarehouse' in flow['id']:
            print(f"      Final destination: {flow['id']}")
    
    print("\n" + "─" * 70)
    print("❓ Q4: What staging tables are used?")
    print("─" * 70)
    
    print("\n🔍 Finding staging tables...")
    
    tables = [e for e in ssis.kg.graph.get_all_entities() if e.type == 'Table']
    staging_tables = [t for t in tables if 'staging' in t.id.lower() or 'stg' in t.id.lower()]
    
    if staging_tables:
        for table in staging_tables:
            print(f"   - {table.id}")
            usages = ssis.find_table_usage(table.id)
            for usage in usages:
                print(f"      └─ {usage['id']}")
    else:
        print("   No explicit staging tables found in SQL statements.")
        print("   Checking for staging-related connections...")
        staging_conns = [c for c in connections if 'staging' in c['attributes']['name'].lower()]
        for conn in staging_conns:
            print(f"   - {conn['id']} ({conn['attributes']['connection_type']})")
    
    print("\n" + "─" * 70)
    print("❓ Q5: Which packages share StagingDB connection?")
    print("─" * 70)
    
    print("\n🔍 Finding StagingDB usage across packages...")
    
    staging_conns = [c for c in connections if c['attributes']['name'] == 'StagingDB']
    packages_using_staging = set()
    
    for conn in staging_conns:
        pkg = conn['id'].split('.')[0]
        packages_using_staging.add(pkg)
        print(f"\n   📦 {pkg}:")
        
        # What does this package do?
        pkg_info = next((p for p in ssis.find_packages() if p['id'] == pkg), None)
        if pkg_info:
            desc = pkg_info['attributes'].get('description', 'No description')
            print(f"      Description: {desc}")
        
        # What tasks use it?
        pkg_tasks = [t for t in ssis.find_tasks() if t['id'].startswith(pkg + '.')]
        print(f"      Tasks: {len(pkg_tasks)}")
        for task in pkg_tasks[:3]:
            print(f"         - {task['id'].split('.')[-1]} ({task['type']})")
        if len(pkg_tasks) > 3:
            print(f"         ... and {len(pkg_tasks) - 3} more")
    
    print(f"\n   Summary: {len(packages_using_staging)} packages share StagingDB: {sorted(packages_using_staging)}")
    
    # Cleanup
    ssis.close()
    
    print("\n" + "=" * 70)
    print("Direct Tool Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    if "--direct" in sys.argv or "-d" in sys.argv:
        # Run without LLM
        asyncio.run(demo_without_llm())
    else:
        # Try with agent (requires LLM)
        try:
            asyncio.run(main())
        except Exception as e:
            print(f"\n⚠️  Agent mode requires an LLM. Error: {e}")
            print("   Running direct tool demonstration instead...\n")
            asyncio.run(demo_without_llm())
