"""
Example: SSIS Package Analysis with Knowledge Graph

This example demonstrates how to:
1. Load SSIS packages into a knowledge graph
2. Query tasks, connections, and data flows
3. Trace data lineage through transformations
4. Find package dependencies
5. Analyze execution order

The SSISAnalyzer parses .dtsx files (XML) and builds a knowledge graph
that can be queried for impact analysis, documentation, and debugging.
"""

import asyncio
from pathlib import Path

# For demo purposes, we'll create a sample package inline
SAMPLE_PACKAGE = '''<?xml version="1.0"?>
<DTS:Executable xmlns:DTS="www.microsoft.com/SqlServer/Dts"
  DTS:ObjectName="ETL_CustomerData">
  <DTS:Property DTS:Name="ObjectName">ETL_CustomerData</DTS:Property>
  <DTS:Property DTS:Name="Description">ETL pipeline for customer data</DTS:Property>
  
  <DTS:ConnectionManagers>
    <DTS:ConnectionManager DTS:CreationName="OLEDB" DTS:ObjectName="SourceDB">
      <DTS:Property DTS:Name="ObjectName">SourceDB</DTS:Property>
    </DTS:ConnectionManager>
    <DTS:ConnectionManager DTS:CreationName="OLEDB" DTS:ObjectName="DestDB">
      <DTS:Property DTS:Name="ObjectName">DestDB</DTS:Property>
    </DTS:ConnectionManager>
    <DTS:ConnectionManager DTS:CreationName="FLATFILE" DTS:ObjectName="LogFile">
      <DTS:Property DTS:Name="ObjectName">LogFile</DTS:Property>
    </DTS:ConnectionManager>
  </DTS:ConnectionManagers>
  
  <DTS:Variables>
    <DTS:Variable>
      <DTS:Property DTS:Name="ObjectName">ProcessDate</DTS:Property>
      <DTS:Property DTS:Name="Namespace">User</DTS:Property>
    </DTS:Variable>
  </DTS:Variables>
  
  <DTS:Executables>
    <!-- Step 1: Truncate staging table -->
    <DTS:Executable DTS:CreationName="Microsoft.ExecuteSQLTask" DTS:ObjectName="Truncate Staging">
      <DTS:Property DTS:Name="ObjectName">Truncate Staging</DTS:Property>
      <DTS:ObjectData>
        <SQLTask:SqlTaskData xmlns:SQLTask="www.microsoft.com/sqlserver/dts/tasks/sqltask"
          SQLTask:SqlStatementSource="TRUNCATE TABLE staging.Customers" />
      </DTS:ObjectData>
    </DTS:Executable>
    
    <!-- Step 2: Load data from source to staging -->
    <DTS:Executable DTS:CreationName="Microsoft.Pipeline" DTS:ObjectName="Load to Staging">
      <DTS:Property DTS:Name="ObjectName">Load to Staging</DTS:Property>
      <DTS:ObjectData>
        <pipeline>
          <components>
            <component name="Source - Customers" componentClassID="Microsoft.OLEDBSource" />
            <component name="Derived - CleanData" componentClassID="Microsoft.DerivedColumn" />
            <component name="Lookup - ValidateRegion" componentClassID="Microsoft.Lookup" />
            <component name="Destination - Staging" componentClassID="Microsoft.OLEDBDestination" />
          </components>
          <paths>
            <path name="Source to Derived" 
              startId="Load to Staging\\Source - Customers.Outputs[Output]"
              endId="Load to Staging\\Derived - CleanData.Inputs[Input]" />
            <path name="Derived to Lookup"
              startId="Load to Staging\\Derived - CleanData.Outputs[Output]"
              endId="Load to Staging\\Lookup - ValidateRegion.Inputs[Input]" />
            <path name="Lookup to Dest"
              startId="Load to Staging\\Lookup - ValidateRegion.Outputs[Match Output]"
              endId="Load to Staging\\Destination - Staging.Inputs[Input]" />
          </paths>
        </pipeline>
      </DTS:ObjectData>
    </DTS:Executable>
    
    <!-- Step 3: Merge to final table -->
    <DTS:Executable DTS:CreationName="Microsoft.ExecuteSQLTask" DTS:ObjectName="Merge to Final">
      <DTS:Property DTS:Name="ObjectName">Merge to Final</DTS:Property>
      <DTS:ObjectData>
        <SQLTask:SqlTaskData xmlns:SQLTask="www.microsoft.com/sqlserver/dts/tasks/sqltask"
          SQLTask:SqlStatementSource="MERGE INTO dbo.Customers AS target USING staging.Customers AS source ON target.CustomerID = source.CustomerID" />
      </DTS:ObjectData>
    </DTS:Executable>
    
    <!-- Step 4: Call child package for reporting -->
    <DTS:Executable DTS:CreationName="Microsoft.ExecutePackageTask" DTS:ObjectName="Run Reports">
      <DTS:Property DTS:Name="ObjectName">Run Reports</DTS:Property>
      <DTS:ObjectData>
        <ExecutePackageTask PackageName="\\Reports\\GenerateCustomerReport.dtsx" />
      </DTS:ObjectData>
    </DTS:Executable>
  </DTS:Executables>
  
  <DTS:PrecedenceConstraints>
    <DTS:PrecedenceConstraint>
      <DTS:Property DTS:Name="From">Truncate Staging</DTS:Property>
      <DTS:Property DTS:Name="To">Load to Staging</DTS:Property>
    </DTS:PrecedenceConstraint>
    <DTS:PrecedenceConstraint>
      <DTS:Property DTS:Name="From">Load to Staging</DTS:Property>
      <DTS:Property DTS:Name="To">Merge to Final</DTS:Property>
    </DTS:PrecedenceConstraint>
    <DTS:PrecedenceConstraint>
      <DTS:Property DTS:Name="From">Merge to Final</DTS:Property>
      <DTS:Property DTS:Name="To">Run Reports</DTS:Property>
    </DTS:PrecedenceConstraint>
  </DTS:PrecedenceConstraints>
</DTS:Executable>
'''


def main():
    """Demonstrate SSIS package analysis."""
    from agenticflow.capabilities import SSISAnalyzer
    import tempfile
    
    print("=" * 60)
    print("SSIS Package Analyzer Demo")
    print("=" * 60)
    
    # Create a temporary package file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".dtsx", delete=False
    ) as f:
        f.write(SAMPLE_PACKAGE)
        pkg_path = f.name
    
    # Create analyzer (can use sqlite for persistence)
    analyzer = SSISAnalyzer()
    
    # Load the package
    print("\n📦 Loading SSIS package...")
    stats = analyzer.load_package(pkg_path)
    print(f"   Loaded: {stats}")
    
    # Show overall statistics
    print("\n📊 Package Statistics:")
    all_stats = analyzer.stats()
    for key, value in all_stats.items():
        print(f"   {key}: {value}")
    
    # Find all packages
    print("\n📋 Packages:")
    for pkg in analyzer.find_packages():
        print(f"   - {pkg['id']}: {pkg['attributes'].get('description', 'No description')}")
    
    # Find all connections
    print("\n🔌 Connection Managers:")
    for conn in analyzer.find_connections():
        print(f"   - {conn['attributes']['name']} ({conn['attributes']['connection_type']})")
    
    # Find all tasks
    print("\n⚙️ Tasks:")
    for task in analyzer.find_tasks():
        disabled = " [DISABLED]" if task["attributes"].get("disabled") else ""
        print(f"   - {task['id']} ({task['type']}){disabled}")
    
    # Find data flows
    print("\n🔄 Data Flow Tasks:")
    for flow in analyzer.find_data_flows():
        print(f"   - {flow['id']}")
        
        # Show components in this data flow
        all_entities = analyzer.kg.graph.get_all_entities()
        components = [
            e for e in all_entities 
            if e.attributes.get("data_flow") == flow["id"]
        ]
        for comp in components:
            print(f"     └─ {comp.attributes['name']} ({comp.type})")
    
    # Find table usage
    print("\n📊 Tables Referenced:")
    tables = [e for e in analyzer.kg.graph.get_all_entities() if e.type == "Table"]
    for table in tables:
        print(f"   - {table.id}")
        tasks = analyzer.find_table_usage(table.id)
        for task in tasks:
            print(f"     └─ Used by: {task['id']}")
    
    # Show execution order
    print("\n📈 Execution Order:")
    order = analyzer.get_execution_order("ETL_CustomerData")
    for i, task in enumerate(order, 1):
        print(f"   {i}. {task.split('.')[-1]}")
    
    # Package dependencies
    print("\n📦 Package Dependencies:")
    deps = analyzer.find_package_dependencies("ETL_CustomerData")
    print(f"   Calls: {deps['calls'] or '(none)'}")
    print(f"   Called by: {deps['called_by'] or '(none)'}")
    
    # Trace data lineage
    print("\n🔍 Data Lineage (tracing back from Destination):")
    dest = "ETL_CustomerData.Load to Staging.Destination - Staging"
    paths = analyzer.trace_data_lineage(dest)
    if paths:
        for i, path in enumerate(paths, 1):
            print(f"   Path {i}:")
            for j, node in enumerate(path):
                prefix = "   └─> " if j == len(path) - 1 else "   ├─> "
                print(f"      {prefix}{node.split('.')[-1]}")
    else:
        print("   (No lineage paths found)")
    
    # Query the knowledge graph directly
    print("\n🔎 Direct KG Query - What does 'Load to Staging' contain?")
    results = analyzer.kg.query("ETL_CustomerData.Load to Staging -contains-> ?")
    for r in results:
        print(f"   - {r['target']} ({r['target_type']})")
    
    # Clean up
    analyzer.close()
    Path(pkg_path).unlink()
    
    print("\n" + "=" * 60)
    print("Demo complete! The SSISAnalyzer can:")
    print("  • Parse any .dtsx package into a knowledge graph")
    print("  • Trace data lineage through transformations")
    print("  • Find package dependencies")
    print("  • Analyze execution order")
    print("  • Query relationships with the KG")
    print("=" * 60)


if __name__ == "__main__":
    main()
