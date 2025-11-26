"""
Example: Analyze a Directory of SSIS Packages

This example demonstrates how to analyze a full SSIS project
directory containing multiple related packages:

1. Load all packages from a directory
2. Discover cross-package dependencies  
3. Build project-wide data lineage
4. Analyze the overall ETL architecture
5. Find impact of changes across packages

This is typical of real-world SSIS projects where you have:
- Master orchestration packages that call child packages
- Data extraction packages for different source systems
- Transformation and loading packages
"""

from pathlib import Path

from agenticflow.capabilities import SSISAnalyzer


def main():
    """Analyze a directory of SSIS packages."""
    
    print("=" * 70)
    print("SSIS Project Analysis - Multi-Package Directory")
    print("=" * 70)
    
    # Path to sample SSIS project
    project_dir = Path(__file__).parent / "data" / "ssis_project"
    
    if not project_dir.exists():
        print(f"❌ Project directory not found: {project_dir}")
        print("   Run this example from the examples directory.")
        return
    
    # List packages before loading
    print(f"\n📁 Project directory: {project_dir}")
    packages = list(project_dir.glob("*.dtsx"))
    print(f"   Found {len(packages)} packages:")
    for pkg in sorted(packages):
        print(f"   - {pkg.name}")
    
    # Create analyzer
    analyzer = SSISAnalyzer()
    
    # Load entire directory
    print("\n📦 Loading all packages...")
    for pkg_path in sorted(packages):
        stats = analyzer.load_package(str(pkg_path))
        print(f"   ✓ Loaded {pkg_path.name}: {stats}")
    
    # Overall statistics
    print("\n📊 Project-Wide Statistics:")
    stats = analyzer.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # List all packages with descriptions
    print("\n📋 Package Inventory:")
    for pkg in analyzer.find_packages():
        pkg_name = pkg['id']
        desc = pkg['attributes'].get('description', '(no description)')
        print(f"\n   📦 {pkg_name}")
        print(f"      Description: {desc}")
        
        # Count tasks in this package
        all_tasks = analyzer.find_tasks()
        pkg_tasks = [t for t in all_tasks if t['id'].startswith(pkg_name + ".")]
        print(f"      Tasks: {len(pkg_tasks)}")
    
    # Connections across all packages
    print("\n" + "-" * 70)
    print("🔌 All Connection Managers:")
    connections = analyzer.find_connections()
    conn_by_type: dict[str, list[str]] = {}
    for conn in connections:
        conn_type = conn['attributes']['connection_type']
        if conn_type not in conn_by_type:
            conn_by_type[conn_type] = []
        conn_by_type[conn_type].append(conn['attributes']['name'])
    
    for conn_type, names in sorted(conn_by_type.items()):
        print(f"   {conn_type}:")
        for name in sorted(set(names)):
            count = names.count(name)
            suffix = f" (x{count})" if count > 1 else ""
            print(f"      - {name}{suffix}")
    
    # Package Dependencies - The orchestration view
    print("\n" + "-" * 70)
    print("🔗 Package Dependencies (Execute Package Tasks):")
    
    # Find which packages call others
    dependency_map: dict[str, list[str]] = {}
    all_packages = [p['id'] for p in analyzer.find_packages()]
    
    for pkg in all_packages:
        deps = analyzer.find_package_dependencies(pkg)
        if deps['calls']:
            dependency_map[pkg] = deps['calls']
    
    if dependency_map:
        for caller, callees in dependency_map.items():
            print(f"\n   {caller}")
            for callee in callees:
                print(f"      └─> calls: {callee}")
    else:
        print("   No cross-package dependencies found.")
    
    # Build execution tree from main package
    print("\n" + "-" * 70)
    print("📈 Execution Flow (starting from MainETL):")
    
    main_pkg = "MainETL"
    if main_pkg in all_packages:
        _print_execution_tree(analyzer, main_pkg, indent=1, visited=set())
    else:
        print(f"   Main package '{main_pkg}' not found.")
    
    # Task breakdown by type
    print("\n" + "-" * 70)
    print("⚙️ Task Breakdown by Type:")
    
    all_tasks = analyzer.find_tasks()
    tasks_by_type: dict[str, list[str]] = {}
    for task in all_tasks:
        task_type = task['type']
        if task_type not in tasks_by_type:
            tasks_by_type[task_type] = []
        tasks_by_type[task_type].append(task['id'])
    
    for task_type, tasks in sorted(tasks_by_type.items(), key=lambda x: -len(x[1])):
        print(f"\n   {task_type} ({len(tasks)}):")
        for task_id in tasks[:5]:  # Show first 5
            task_name = task_id.split(".")[-1]
            pkg_name = task_id.split(".")[0]
            print(f"      - {task_name} [{pkg_name}]")
        if len(tasks) > 5:
            print(f"      ... and {len(tasks) - 5} more")
    
    # Data Flow Analysis
    print("\n" + "-" * 70)
    print("🔄 Data Flow Analysis:")
    
    data_flows = analyzer.find_data_flows()
    for flow in data_flows:
        pkg_name = flow['id'].split(".")[0]
        flow_name = flow['id'].split(".")[-1]
        print(f"\n   📊 {flow_name} [{pkg_name}]")
        
        # Find components in this data flow
        all_entities = analyzer.kg.graph.get_all_entities()
        components = [
            e for e in all_entities
            if e.attributes.get("data_flow") == flow["id"]
        ]
        
        # Group by component type
        sources = [c for c in components if "source" in c.type.lower()]
        transforms = [c for c in components if c.type in 
                     ["DerivedColumn", "Lookup", "ConditionalSplit", "Sort", 
                      "Aggregate", "Merge", "UnionAll", "MulticastTransform"]]
        destinations = [c for c in components if "destination" in c.type.lower()]
        
        if sources:
            print(f"      Sources: {', '.join(c.attributes['name'] for c in sources)}")
        if transforms:
            print(f"      Transforms: {', '.join(c.attributes['name'] for c in transforms)}")
        if destinations:
            print(f"      Destinations: {', '.join(c.attributes['name'] for c in destinations)}")
    
    # Tables referenced across the project
    print("\n" + "-" * 70)
    print("📊 Table Usage Across Project:")
    
    all_entities = analyzer.kg.graph.get_all_entities()
    tables = [e for e in all_entities if e.type == "Table"]
    
    if tables:
        for table in sorted(tables, key=lambda t: t.id):
            print(f"\n   {table.id}")
            usages = analyzer.find_table_usage(table.id)
            for usage in usages:
                task_name = usage['id'].split(".")[-1]
                pkg_name = usage['id'].split(".")[0]
                print(f"      └─ {task_name} [{pkg_name}]")
    else:
        print("   No table references found.")
    
    # Impact Analysis Example
    print("\n" + "-" * 70)
    print("🎯 Impact Analysis Example:")
    print("   Question: What happens if 'SourceCRM' connection changes?")
    
    # Find all entities that use SourceCRM connection
    # First, find the SourceCRM connection entity
    source_crm_id = "ExtractCustomers.SourceCRM"
    
    # Get relationships involving SourceCRM (incoming = what uses it)
    impacted = []
    incoming_rels = analyzer.kg.graph.get_relationships(
        source_crm_id, direction="incoming"
    )
    for rel in incoming_rels:
        impacted.append(rel.source_id)
    
    # Also check by searching for entities that reference it
    all_entities = analyzer.kg.graph.get_all_entities()
    for entity in all_entities:
        for attr_val in entity.attributes.values():
            if isinstance(attr_val, str) and 'SourceCRM' in attr_val:
                if entity.id not in impacted:
                    impacted.append(entity.id)
    
    if impacted:
        print(f"\n   Impacted components ({len(impacted)}):")
        for item in sorted(impacted):
            print(f"      - {item}")
    else:
        print("   No direct impacts found (connection may be indirectly used)")
    
    # Query capabilities
    print("\n" + "-" * 70)
    print("🔎 Knowledge Graph Query Examples:")
    
    print("\n   Query: Find all Execute Package tasks")
    results = analyzer.kg.get_entities(entity_type="ExecutePackageTask")
    for r in results:
        print(f"      - {r.id}")
    
    print("\n   Query: Find all OLEDB connections")
    all_conns = analyzer.kg.get_entities(entity_type="ConnectionManager")
    oledb_conns = [c for c in all_conns if c.attributes.get("connection_type") == "OLEDB"]
    for r in oledb_conns:
        print(f"      - {r.attributes['name']}")
    
    # Cleanup
    analyzer.close()
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("")
    print("This example demonstrated:")
    print("  ✓ Loading multiple SSIS packages from a directory")
    print("  ✓ Cross-package dependency discovery")
    print("  ✓ Connection manager inventory")
    print("  ✓ Task breakdown and data flow analysis")
    print("  ✓ Table usage tracking")
    print("  ✓ Impact analysis queries")
    print("=" * 70)


def _print_execution_tree(
    analyzer: SSISAnalyzer, 
    package_name: str, 
    indent: int = 0, 
    visited: set | None = None
) -> None:
    """Recursively print execution tree starting from a package."""
    if visited is None:
        visited = set()
    
    if package_name in visited:
        print("   " * indent + f"↻ {package_name} (circular reference)")
        return
    
    visited.add(package_name)
    
    # Get execution order for this package
    order = analyzer.get_execution_order(package_name)
    
    if not order:
        print("   " * indent + f"📦 {package_name} (no execution order)")
        return
    
    print("   " * indent + f"📦 {package_name}")
    
    for i, task_id in enumerate(order, 1):
        task_name = task_id.split(".")[-1]
        is_last = i == len(order)
        prefix = "└─" if is_last else "├─"
        
        # Check if this is an Execute Package task
        all_tasks = analyzer.find_tasks()
        task_info = next((t for t in all_tasks if t['id'] == task_id), None)
        
        if task_info and task_info['type'] == 'ExecutePackageTask':
            child_pkg = task_info['attributes'].get('child_package', 'Unknown')
            # Clean up path to get package name
            child_name = Path(child_pkg).stem if '\\' in child_pkg or '/' in child_pkg else child_pkg
            print("   " * indent + f"   {prefix} {i}. {task_name}")
            _print_execution_tree(analyzer, child_name, indent + 2, visited)
        else:
            task_type = task_info['type'] if task_info else "Unknown"
            print("   " * indent + f"   {prefix} {i}. {task_name} [{task_type}]")


if __name__ == "__main__":
    main()
