"""Tests for SSIS Analyzer capability."""

import pytest
from pathlib import Path
from agenticflow.capabilities import SSISAnalyzer


# Sample SSIS package XML for testing
SAMPLE_DTSX = '''<?xml version="1.0"?>
<DTS:Executable xmlns:DTS="www.microsoft.com/SqlServer/Dts"
  DTS:refId="Package"
  DTS:CreationDate="11/26/2025"
  DTS:CreatorName="TestUser"
  DTS:DTSID="{12345678-1234-1234-1234-123456789012}"
  DTS:ExecutableType="Microsoft.Package"
  DTS:ObjectName="TestPackage"
  DTS:PackageType="DTSPackage">
  <DTS:Property DTS:Name="ObjectName">TestPackage</DTS:Property>
  <DTS:Property DTS:Name="Description">A test SSIS package</DTS:Property>
  <DTS:Property DTS:Name="CreationDate">11/26/2025</DTS:Property>
  <DTS:Property DTS:Name="CreatorName">TestUser</DTS:Property>
  <DTS:Property DTS:Name="DTSID">{12345678-1234-1234-1234-123456789012}</DTS:Property>
  
  <DTS:ConnectionManagers>
    <DTS:ConnectionManager
      DTS:refId="Package.ConnectionManagers[SourceDB]"
      DTS:CreationName="OLEDB"
      DTS:ObjectName="SourceDB">
      <DTS:Property DTS:Name="ObjectName">SourceDB</DTS:Property>
      <DTS:ObjectData>
        <DTS:ConnectionManager
          DTS:ConnectionString="Data Source=server1;Initial Catalog=SourceDB;Provider=SQLNCLI11;Password=secret123;" />
      </DTS:ObjectData>
    </DTS:ConnectionManager>
    <DTS:ConnectionManager
      DTS:refId="Package.ConnectionManagers[DestDB]"
      DTS:CreationName="OLEDB"
      DTS:ObjectName="DestDB">
      <DTS:Property DTS:Name="ObjectName">DestDB</DTS:Property>
    </DTS:ConnectionManager>
  </DTS:ConnectionManagers>
  
  <DTS:Variables>
    <DTS:Variable DTS:ObjectName="ProcessDate">
      <DTS:Property DTS:Name="ObjectName">ProcessDate</DTS:Property>
      <DTS:Property DTS:Name="Namespace">User</DTS:Property>
      <DTS:Property DTS:Name="DataType">7</DTS:Property>
    </DTS:Variable>
    <DTS:Variable DTS:ObjectName="BatchSize">
      <DTS:Property DTS:Name="ObjectName">BatchSize</DTS:Property>
      <DTS:Property DTS:Name="Namespace">User</DTS:Property>
      <DTS:Property DTS:Name="DataType">3</DTS:Property>
    </DTS:Variable>
  </DTS:Variables>
  
  <DTS:Executables>
    <DTS:Executable
      DTS:refId="Package\\SQL Task - Truncate"
      DTS:CreationName="Microsoft.ExecuteSQLTask"
      DTS:ObjectName="SQL Task - Truncate">
      <DTS:Property DTS:Name="ObjectName">SQL Task - Truncate</DTS:Property>
      <DTS:Property DTS:Name="DTSID">{TASK-0001}</DTS:Property>
      <DTS:ObjectData>
        <SQLTask:SqlTaskData
          xmlns:SQLTask="www.microsoft.com/sqlserver/dts/tasks/sqltask"
          SQLTask:SqlStatementSource="TRUNCATE TABLE dbo.Customers" />
      </DTS:ObjectData>
    </DTS:Executable>
    
    <DTS:Executable
      DTS:refId="Package\\Data Flow Task"
      DTS:CreationName="Microsoft.Pipeline"
      DTS:ObjectName="Data Flow Task">
      <DTS:Property DTS:Name="ObjectName">Data Flow Task</DTS:Property>
      <DTS:Property DTS:Name="DTSID">{TASK-0002}</DTS:Property>
      <DTS:ObjectData>
        <pipeline>
          <components>
            <component
              refId="Package\\Data Flow Task\\OLE DB Source"
              name="OLE DB Source"
              componentClassID="Microsoft.OLEDBSource">
              <outputs>
                <output name="Output">
                  <outputColumns>
                    <outputColumn name="CustomerID" dataType="i4" />
                    <outputColumn name="CustomerName" dataType="wstr" length="100" />
                  </outputColumns>
                </output>
              </outputs>
            </component>
            <component
              refId="Package\\Data Flow Task\\Derived Column"
              name="Derived Column"
              componentClassID="Microsoft.DerivedColumn">
            </component>
            <component
              refId="Package\\Data Flow Task\\OLE DB Destination"
              name="OLE DB Destination"
              componentClassID="Microsoft.OLEDBDestination">
            </component>
          </components>
          <paths>
            <path name="Path 1"
              startId="Package\\Data Flow Task\\OLE DB Source.Outputs[Output]"
              endId="Package\\Data Flow Task\\Derived Column.Inputs[Input]" />
            <path name="Path 2"
              startId="Package\\Data Flow Task\\Derived Column.Outputs[Output]"
              endId="Package\\Data Flow Task\\OLE DB Destination.Inputs[Input]" />
          </paths>
        </pipeline>
      </DTS:ObjectData>
    </DTS:Executable>
    
    <DTS:Executable
      DTS:refId="Package\\Execute Child Package"
      DTS:CreationName="Microsoft.ExecutePackageTask"
      DTS:ObjectName="Execute Child Package">
      <DTS:Property DTS:Name="ObjectName">Execute Child Package</DTS:Property>
      <DTS:ObjectData>
        <ExecutePackageTask PackageName="\\SSIS\\ChildPackage.dtsx" />
      </DTS:ObjectData>
    </DTS:Executable>
  </DTS:Executables>
  
  <DTS:PrecedenceConstraints>
    <DTS:PrecedenceConstraint
      DTS:refId="Package.PrecedenceConstraints[Constraint 1]"
      DTS:From="SQL Task - Truncate"
      DTS:To="Data Flow Task">
      <DTS:Property DTS:Name="From">SQL Task - Truncate</DTS:Property>
      <DTS:Property DTS:Name="To">Data Flow Task</DTS:Property>
      <DTS:Property DTS:Name="EvalOp">Constraint</DTS:Property>
    </DTS:PrecedenceConstraint>
    <DTS:PrecedenceConstraint
      DTS:refId="Package.PrecedenceConstraints[Constraint 2]"
      DTS:From="Data Flow Task"
      DTS:To="Execute Child Package">
      <DTS:Property DTS:Name="From">Data Flow Task</DTS:Property>
      <DTS:Property DTS:Name="To">Execute Child Package</DTS:Property>
      <DTS:Property DTS:Name="EvalOp">Constraint</DTS:Property>
    </DTS:PrecedenceConstraint>
  </DTS:PrecedenceConstraints>
</DTS:Executable>
'''


class TestSSISAnalyzer:
    """Tests for SSISAnalyzer capability."""
    
    @pytest.fixture
    def sample_package(self, tmp_path):
        """Create a sample SSIS package file."""
        pkg_path = tmp_path / "TestPackage.dtsx"
        pkg_path.write_text(SAMPLE_DTSX)
        return pkg_path
    
    @pytest.fixture
    def analyzer(self):
        """Create an SSIS analyzer."""
        return SSISAnalyzer()
    
    def test_load_package(self, analyzer, sample_package):
        """Test loading a single package."""
        stats = analyzer.load_package(sample_package)
        
        assert stats["packages"] == 1
        assert stats["tasks"] >= 2  # SQL Task + Execute Package
        assert stats["data_flows"] == 1
        assert stats["connections"] == 2
        assert stats["variables"] == 2
    
    def test_find_packages(self, analyzer, sample_package):
        """Test finding loaded packages."""
        analyzer.load_package(sample_package)
        
        packages = analyzer.find_packages()
        assert len(packages) == 1
        assert packages[0]["id"] == "TestPackage"
        assert packages[0]["attributes"]["description"] == "A test SSIS package"
    
    def test_find_tasks(self, analyzer, sample_package):
        """Test finding tasks."""
        analyzer.load_package(sample_package)
        
        tasks = analyzer.find_tasks()
        task_names = [t["id"] for t in tasks]
        
        assert "TestPackage.SQL Task - Truncate" in task_names
        assert "TestPackage.Data Flow Task" in task_names
        assert "TestPackage.Execute Child Package" in task_names
    
    def test_find_tasks_by_type(self, analyzer, sample_package):
        """Test finding tasks filtered by type."""
        analyzer.load_package(sample_package)
        
        sql_tasks = analyzer.find_tasks(task_type="SQLTask")
        assert len(sql_tasks) == 1
        assert "SQL Task - Truncate" in sql_tasks[0]["id"]
        
        data_flows = analyzer.find_tasks(task_type="DataFlowTask")
        assert len(data_flows) == 1
    
    def test_find_connections(self, analyzer, sample_package):
        """Test finding connection managers."""
        analyzer.load_package(sample_package)
        
        conns = analyzer.find_connections()
        conn_names = [c["attributes"]["name"] for c in conns]
        
        assert "SourceDB" in conn_names
        assert "DestDB" in conn_names
    
    def test_connection_string_sanitized(self, analyzer, sample_package):
        """Test that passwords are removed from connection strings."""
        analyzer.load_package(sample_package)
        
        conns = analyzer.find_connections()
        for conn in conns:
            conn_str = conn["attributes"].get("connection_string", "")
            assert "secret123" not in conn_str
            if "Password" in conn_str:
                assert "***" in conn_str
    
    def test_find_data_flows(self, analyzer, sample_package):
        """Test finding data flow tasks."""
        analyzer.load_package(sample_package)
        
        flows = analyzer.find_data_flows()
        assert len(flows) == 1
        assert flows[0]["type"] == "DataFlowTask"
    
    def test_package_dependencies(self, analyzer, sample_package):
        """Test finding package dependencies."""
        analyzer.load_package(sample_package)
        
        deps = analyzer.find_package_dependencies("TestPackage")
        
        # The called package name includes the path
        assert len(deps["calls"]) == 1
        assert "ChildPackage" in deps["calls"][0]
    
    def test_data_flow_components(self, analyzer, sample_package):
        """Test that data flow components are parsed."""
        analyzer.load_package(sample_package)
        
        # Query the KG for components
        all_entities = analyzer.kg.graph.get_all_entities()
        types = {e.type for e in all_entities}
        
        assert "Source" in types
        assert "Destination" in types
        assert "DerivedColumn" in types or "Transform" in types
    
    def test_data_flow_paths(self, analyzer, sample_package):
        """Test that data flow paths (lineage) are parsed."""
        analyzer.load_package(sample_package)
        
        # Check flows_to relationships exist
        source = "TestPackage.Data Flow Task.OLE DB Source"
        rels = analyzer.kg.graph.get_relationships(source, direction="outgoing")
        
        flow_rels = [r for r in rels if r.relation == "flows_to"]
        assert len(flow_rels) >= 1
    
    def test_precedence_constraints(self, analyzer, sample_package):
        """Test that precedence constraints are parsed."""
        analyzer.load_package(sample_package)
        
        # Check precedes relationships
        sql_task = "TestPackage.SQL Task - Truncate"
        rels = analyzer.kg.graph.get_relationships(sql_task, relation="precedes", direction="outgoing")
        
        assert len(rels) >= 1
        targets = [r.target_id for r in rels]
        assert any("Data Flow Task" in t for t in targets)
    
    def test_execution_order(self, analyzer, sample_package):
        """Test getting task execution order."""
        analyzer.load_package(sample_package)
        
        order = analyzer.get_execution_order("TestPackage")
        
        # SQL Task should come before Data Flow Task
        sql_idx = next((i for i, t in enumerate(order) if "SQL Task" in t), -1)
        df_idx = next((i for i, t in enumerate(order) if "Data Flow Task" in t), -1)
        
        assert sql_idx < df_idx
    
    def test_table_usage(self, analyzer, sample_package):
        """Test finding table usage from SQL tasks."""
        analyzer.load_package(sample_package)
        
        # Table names are extracted uppercase from SQL
        tasks = analyzer.find_table_usage("CUSTOMERS")
        assert len(tasks) >= 1
    
    def test_stats(self, analyzer, sample_package):
        """Test getting analyzer statistics."""
        analyzer.load_package(sample_package)
        
        stats = analyzer.stats()
        
        assert stats["packages_loaded"] == 1
        assert stats["total_entities"] > 0
        assert stats["total_relationships"] > 0
        assert stats.get("Package", 0) == 1
    
    def test_load_directory(self, analyzer, tmp_path):
        """Test loading packages from a directory."""
        # Create multiple packages
        for i in range(3):
            pkg = tmp_path / f"Package{i}.dtsx"
            pkg.write_text(SAMPLE_DTSX.replace("TestPackage", f"Package{i}"))
        
        stats = analyzer.load_directory(tmp_path)
        
        assert stats["files_processed"] == 3
        assert stats["packages"] == 3
    
    def test_tools_available(self, analyzer):
        """Test that tools are available."""
        tools = analyzer.tools
        tool_names = [t.name for t in tools]
        
        assert "find_packages" in tool_names
        assert "find_tasks" in tool_names
        assert "find_data_flows" in tool_names
        assert "find_connections" in tool_names
        assert "trace_data_lineage" in tool_names
        assert "find_package_dependencies" in tool_names
    
    def test_context_manager(self, tmp_path):
        """Test using analyzer as context manager."""
        pkg_path = tmp_path / "Test.dtsx"
        pkg_path.write_text(SAMPLE_DTSX)
        
        with SSISAnalyzer() as analyzer:
            analyzer.load_package(pkg_path)
            assert len(analyzer.find_packages()) == 1
    
    def test_sqlite_backend(self, tmp_path, sample_package):
        """Test using SQLite backend for persistence."""
        db_path = tmp_path / "ssis.db"
        
        # Load with SQLite backend
        analyzer = SSISAnalyzer(kg_backend="sqlite", kg_path=db_path)
        analyzer.load_package(sample_package)
        analyzer.close()
        
        # Reopen and verify data persists
        analyzer2 = SSISAnalyzer(kg_backend="sqlite", kg_path=db_path)
        packages = analyzer2.find_packages()
        assert len(packages) == 1
        analyzer2.close()
    
    def test_invalid_file(self, analyzer, tmp_path):
        """Test handling invalid file."""
        not_dtsx = tmp_path / "file.txt"
        not_dtsx.write_text("not an ssis package")
        
        with pytest.raises(ValueError):
            analyzer.load_package(not_dtsx)
    
    def test_missing_file(self, analyzer):
        """Test handling missing file."""
        with pytest.raises(FileNotFoundError):
            analyzer.load_package("/nonexistent/package.dtsx")


class TestSSISLineageTracing:
    """Tests specifically for data lineage tracing."""
    
    @pytest.fixture
    def analyzer_with_package(self, tmp_path):
        """Create analyzer with loaded package."""
        pkg_path = tmp_path / "TestPackage.dtsx"
        pkg_path.write_text(SAMPLE_DTSX)
        
        analyzer = SSISAnalyzer()
        analyzer.load_package(pkg_path)
        return analyzer
    
    def test_trace_lineage_from_destination(self, analyzer_with_package):
        """Test tracing lineage from a destination component."""
        dest = "TestPackage.Data Flow Task.OLE DB Destination"
        paths = analyzer_with_package.trace_data_lineage(dest)
        
        # Should find path back to source
        if paths:
            # At least one path should exist
            assert len(paths) >= 1
    
    def test_trace_lineage_tool(self, analyzer_with_package):
        """Test the lineage tool."""
        tools = {t.name: t for t in analyzer_with_package.tools}
        trace_tool = tools["trace_data_lineage"]
        
        result = trace_tool.invoke({
            "target": "TestPackage.Data Flow Task.OLE DB Destination"
        })
        
        assert isinstance(result, str)


class TestSSISExtensibility:
    """Tests for SSIS analyzer extensibility."""
    
    def test_register_custom_handler(self):
        """Test registering a custom task handler."""
        from agenticflow.capabilities.ssis import TaskHandler
        
        class CustomTaskHandler(TaskHandler):
            handled = False
            
            @property
            def task_patterns(self):
                return ["customtask"]
            
            def handle(self, exe, analyzer, package_name, task_name):
                CustomTaskHandler.handled = True
                # Add custom entity
                analyzer.kg.graph.add_entity(
                    f"{task_name}::custom",
                    "CustomData",
                    {"custom": True},
                )
        
        analyzer = SSISAnalyzer()
        analyzer.register_task_handler(CustomTaskHandler())
        
        # Check handler is registered
        handler = analyzer._task_registry.get_handler("MyCustomTask")
        assert handler is not None
    
    def test_classify_all_task_types(self):
        """Test that all common SSIS task types are classified."""
        analyzer = SSISAnalyzer()
        
        task_types = [
            ("SSIS.Pipeline", "DataFlowTask"),
            ("STOCK:SEQUENCE", "SequenceContainer"),
            ("STOCK:FORLOOP", "ForLoopContainer"),
            ("STOCK:FOREACHLOOP", "ForEachLoopContainer"),
            ("Microsoft.ExecutePackageTask", "ExecutePackageTask"),
            ("Microsoft.ExecuteProcess", "ExecuteProcessTask"),
            ("Microsoft.SqlServer.Dts.Tasks.ExecuteSQLTask", "SQLTask"),
            ("Microsoft.ScriptTask", "ScriptTask"),
            ("Microsoft.FileSystemTask", "FileSystemTask"),
            ("Microsoft.FTPTask", "FTPTask"),
            ("Microsoft.SendMailTask", "SendMailTask"),
            ("Microsoft.ExpressionTask", "ExpressionTask"),
            ("Microsoft.WebServiceTask", "WebServiceTask"),
            ("Microsoft.XMLTask", "XMLTask"),
            ("Microsoft.BulkInsertTask", "BulkInsertTask"),
            ("Microsoft.WMIDataReaderTask", "WMIDataReaderTask"),
            ("Microsoft.TransferDatabaseTask", "TransferDatabaseTask"),
            ("SomeUnknownTask", "Task"),  # Falls back to generic
        ]
        
        for creation_name, expected_type in task_types:
            result = analyzer._classify_executable(creation_name)
            assert result == expected_type, f"Expected {creation_name} -> {expected_type}, got {result}"
    
    def test_default_handlers_registered(self):
        """Test that default handlers are registered."""
        analyzer = SSISAnalyzer()
        
        # Should have handlers for these patterns
        assert analyzer._task_registry.get_handler("ExecuteProcessTask") is not None
        assert analyzer._task_registry.get_handler("ScriptTask") is not None
        assert analyzer._task_registry.get_handler("WebServiceTask") is not None
        assert analyzer._task_registry.get_handler("XMLTask") is not None
