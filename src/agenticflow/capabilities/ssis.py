"""
SSIS Package Analyzer Capability.

Parses Microsoft SQL Server Integration Services (SSIS) packages (.dtsx files)
and stores the structure in a knowledge graph for intelligent querying and
lineage tracing.

SSIS packages are XML-based and contain:
- Control Flow: Tasks, containers, precedence constraints
- Data Flow: Sources, transformations, destinations
- Connection Managers: Database connections, file connections
- Variables and Parameters
- Event Handlers
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from langchain_core.tools import BaseTool, tool

from agenticflow.capabilities.base import BaseCapability
from agenticflow.capabilities.knowledge_graph import KnowledgeGraph

if TYPE_CHECKING:
    from agenticflow.capabilities.knowledge_graph import Entity

logger = logging.getLogger(__name__)

# SSIS XML namespaces
NAMESPACES = {
    "DTS": "www.microsoft.com/SqlServer/Dts",
    "SQLTask": "www.microsoft.com/sqlserver/dts/tasks/sqltask",
}


# =============================================================================
# Task Handler Registry (Extensibility)
# =============================================================================

class TaskHandler(ABC):
    """Base class for custom task handlers."""
    
    @property
    @abstractmethod
    def task_patterns(self) -> list[str]:
        """Patterns to match against CreationName (case-insensitive)."""
        ...
    
    @abstractmethod
    def handle(
        self,
        exe: ET.Element,
        analyzer: "SSISAnalyzer",
        package_name: str,
        task_name: str,
    ) -> None:
        """Handle the task - extract additional info and add to KG."""
        ...


class TaskHandlerRegistry:
    """Registry for task handlers - enables extensibility."""
    
    def __init__(self) -> None:
        self._handlers: list[TaskHandler] = []
    
    def register(self, handler: TaskHandler) -> None:
        """Register a task handler."""
        self._handlers.append(handler)
    
    def get_handler(self, exe_type: str) -> TaskHandler | None:
        """Get handler for a task type."""
        exe_type_lower = exe_type.lower()
        for handler in self._handlers:
            for pattern in handler.task_patterns:
                if pattern.lower() in exe_type_lower:
                    return handler
        return None


# Built-in task handlers
class ExecuteProcessTaskHandler(TaskHandler):
    """Handler for Execute Process Task."""
    
    @property
    def task_patterns(self) -> list[str]:
        return ["executeprocess"]
    
    def handle(
        self,
        exe: ET.Element,
        analyzer: "SSISAnalyzer",
        package_name: str,
        task_name: str,
    ) -> None:
        """Extract process execution details."""
        for obj_data in exe.iter():
            if obj_data.tag.endswith("ObjectData"):
                for child in obj_data:
                    executable = None
                    arguments = None
                    working_dir = None
                    
                    for prop in child.iter():
                        prop_name = prop.get("Name") or prop.tag.split("}")[-1]
                        if "Executable" in prop_name:
                            executable = prop.text
                        elif "Arguments" in prop_name:
                            arguments = prop.text
                        elif "WorkingDirectory" in prop_name:
                            working_dir = prop.text
                    
                    if executable:
                        # Update entity with process details
                        entity = analyzer.kg.graph.get_entity(task_name)
                        if entity:
                            entity.attributes["executable"] = executable
                            entity.attributes["arguments"] = arguments or ""
                            entity.attributes["working_directory"] = working_dir or ""
                        
                        # Add relationship to external executable
                        exe_name = Path(executable).name if executable else "unknown"
                        analyzer.kg.graph.add_entity(
                            exe_name,
                            "ExternalProcess",
                            {"path": executable, "arguments": arguments},
                        )
                        analyzer.kg.graph.add_relationship(
                            task_name, "executes", exe_name
                        )


class ScriptTaskHandler(TaskHandler):
    """Handler for Script Task."""
    
    @property
    def task_patterns(self) -> list[str]:
        return ["scripttask"]
    
    def handle(
        self,
        exe: ET.Element,
        analyzer: "SSISAnalyzer",
        package_name: str,
        task_name: str,
    ) -> None:
        """Extract script task details."""
        for obj_data in exe.iter():
            if obj_data.tag.endswith("ObjectData"):
                for child in obj_data:
                    script_lang = None
                    entry_point = None
                    
                    for prop in child.iter():
                        prop_name = prop.get("Name") or prop.tag.split("}")[-1]
                        if "ScriptLanguage" in prop_name:
                            script_lang = prop.text
                        elif "EntryPoint" in prop_name:
                            entry_point = prop.text
                    
                    entity = analyzer.kg.graph.get_entity(task_name)
                    if entity:
                        entity.attributes["script_language"] = script_lang or "VB"
                        entity.attributes["entry_point"] = entry_point or "Main"


class WebServiceTaskHandler(TaskHandler):
    """Handler for Web Service Task."""
    
    @property
    def task_patterns(self) -> list[str]:
        return ["webservice"]
    
    def handle(
        self,
        exe: ET.Element,
        analyzer: "SSISAnalyzer",
        package_name: str,
        task_name: str,
    ) -> None:
        """Extract web service details."""
        for obj_data in exe.iter():
            if obj_data.tag.endswith("ObjectData"):
                for child in obj_data:
                    wsdl_url = None
                    method_name = None
                    
                    for prop in child.iter():
                        prop_name = prop.get("Name") or prop.tag.split("}")[-1]
                        if "WsdlFile" in prop_name or "WSDL" in prop_name:
                            wsdl_url = prop.text
                        elif "MethodName" in prop_name or "WebMethod" in prop_name:
                            method_name = prop.text
                    
                    if wsdl_url:
                        entity = analyzer.kg.graph.get_entity(task_name)
                        if entity:
                            entity.attributes["wsdl_url"] = wsdl_url
                            entity.attributes["method"] = method_name or ""
                        
                        # Add web service as external entity
                        analyzer.kg.graph.add_entity(
                            wsdl_url,
                            "WebService",
                            {"url": wsdl_url, "method": method_name},
                        )
                        analyzer.kg.graph.add_relationship(
                            task_name, "calls_service", wsdl_url
                        )


class XMLTaskHandler(TaskHandler):
    """Handler for XML Task."""
    
    @property
    def task_patterns(self) -> list[str]:
        return ["xmltask"]
    
    def handle(
        self,
        exe: ET.Element,
        analyzer: "SSISAnalyzer",
        package_name: str,
        task_name: str,
    ) -> None:
        """Extract XML task details."""
        for obj_data in exe.iter():
            if obj_data.tag.endswith("ObjectData"):
                for child in obj_data:
                    operation = None
                    source_type = None
                    
                    for prop in child.iter():
                        prop_name = prop.get("Name") or prop.tag.split("}")[-1]
                        if "OperationType" in prop_name:
                            operation = prop.text
                        elif "SourceType" in prop_name:
                            source_type = prop.text
                    
                    entity = analyzer.kg.graph.get_entity(task_name)
                    if entity:
                        entity.attributes["operation"] = operation or ""
                        entity.attributes["source_type"] = source_type or ""


class SSISAnalyzer(BaseCapability):
    """
    Capability for analyzing SSIS packages (.dtsx files).

    Parses SSIS XML and stores the structure in a knowledge graph:
    - Packages, Tasks, Data Flows
    - Connection Managers
    - Data lineage (source → transform → destination)
    - Package dependencies (Execute Package tasks)
    - Precedence constraints (task execution order)

    Example:
        ```python
        from agenticflow.capabilities import SSISAnalyzer
        
        analyzer = SSISAnalyzer()
        
        # Load a single package
        analyzer.load_package("/path/to/package.dtsx")
        
        # Load all packages in a directory
        analyzer.load_directory("/path/to/ssis/project")
        
        # Query the knowledge graph
        tasks = analyzer.find_tasks()
        lineage = analyzer.trace_data_lineage("CustomerTable")
        deps = analyzer.find_package_dependencies("MainPackage")
        
        # Access underlying KG for complex queries
        analyzer.kg.query("Package1 -contains-> ?")
        
        # Register custom task handler
        analyzer.register_task_handler(MyCustomHandler())
        ```
    """

    name: str = "ssis_analyzer"
    description: str = "Analyze SSIS packages and trace data lineage"
    
    # Default task handler registry
    _default_handlers: list[type[TaskHandler]] = [
        ExecuteProcessTaskHandler,
        ScriptTaskHandler,
        WebServiceTaskHandler,
        XMLTaskHandler,
    ]

    def __init__(
        self,
        kg_backend: str = "memory",
        kg_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SSIS analyzer.
        
        Args:
            kg_backend: Knowledge graph backend ('memory', 'sqlite', 'json')
            kg_path: Path for persistent KG storage (required for sqlite/json)
        """
        super().__init__(**kwargs)
        
        if kg_backend == "memory":
            self._kg = KnowledgeGraph(backend="memory", name="ssis_kg")
        else:
            if not kg_path:
                raise ValueError(f"kg_path required for {kg_backend} backend")
            self._kg = KnowledgeGraph(backend=kg_backend, path=kg_path, name="ssis_kg")
        
        self._tools: list[BaseTool] = []
        self._loaded_packages: set[str] = set()
        
        # Initialize task handler registry with defaults
        self._task_registry = TaskHandlerRegistry()
        for handler_class in self._default_handlers:
            self._task_registry.register(handler_class())
        
        self._build_tools()
    
    def register_task_handler(self, handler: TaskHandler) -> None:
        """
        Register a custom task handler.
        
        Allows extending the analyzer with custom logic for specific task types.
        
        Example:
            ```python
            class MyTaskHandler(TaskHandler):
                @property
                def task_patterns(self):
                    return ["mycustomtask"]
                
                def handle(self, exe, analyzer, package_name, task_name):
                    # Custom parsing logic
                    pass
            
            analyzer.register_task_handler(MyTaskHandler())
            ```
        """
        self._task_registry.register(handler)

    def _build_tools(self) -> None:
        """Build the analyzer tools."""
        self._tools = [
            self._find_packages_tool(),
            self._find_tasks_tool(),
            self._find_data_flows_tool(),
            self._find_connections_tool(),
            self._trace_lineage_tool(),
            self._find_dependencies_tool(),
            self._find_callers_tool(),
        ]

    @property
    def tools(self) -> list[BaseTool]:
        """Get the analyzer tools."""
        return self._tools

    @property
    def kg(self) -> KnowledgeGraph:
        """Access the underlying knowledge graph."""
        return self._kg

    # =========================================================================
    # Loading Methods
    # =========================================================================

    def load_package(self, file_path: str | Path) -> dict[str, int]:
        """
        Load and parse a single SSIS package (.dtsx file).

        Args:
            file_path: Path to the .dtsx file

        Returns:
            Statistics about parsed elements
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Package not found: {path}")

        if path.suffix.lower() != ".dtsx":
            raise ValueError(f"Not an SSIS package: {path}")

        if str(path) in self._loaded_packages:
            logger.debug(f"Package already loaded: {path}")
            return {"skipped": 1}

        stats = {
            "packages": 0,
            "tasks": 0,
            "data_flows": 0,
            "connections": 0,
            "variables": 0,
            "precedence_constraints": 0,
        }

        try:
            tree = ET.parse(path)
            root = tree.getroot()
            
            # Parse the package
            self._parse_package(root, path, stats)
            self._loaded_packages.add(str(path))
            
            logger.info(f"Loaded SSIS package: {path.name} - {stats}")
            return stats

        except ET.ParseError as e:
            logger.error(f"XML parse error in {path}: {e}")
            raise ValueError(f"Invalid SSIS package XML: {e}") from e

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> dict[str, int]:
        """
        Load all SSIS packages from a directory.

        Args:
            directory: Path to directory containing .dtsx files
            recursive: Whether to search subdirectories

        Returns:
            Combined statistics
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        pattern = "**/*.dtsx" if recursive else "*.dtsx"
        files = list(dir_path.glob(pattern))
        
        # Also check for uppercase extension
        files.extend(dir_path.glob(pattern.replace(".dtsx", ".DTSX")))
        
        total_stats: dict[str, int] = {}

        for file_path in files:
            try:
                stats = self.load_package(file_path)
                for key, value in stats.items():
                    total_stats[key] = total_stats.get(key, 0) + value
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                total_stats["errors"] = total_stats.get("errors", 0) + 1

        total_stats["files_processed"] = len(files)
        return total_stats

    # =========================================================================
    # Parsing Methods
    # =========================================================================

    def _parse_package(
        self,
        root: ET.Element,
        path: Path,
        stats: dict[str, int],
    ) -> None:
        """Parse the root package element."""
        # Get package name from DTS:Property with Name="ObjectName"
        package_name = self._get_property(root, "ObjectName") or path.stem
        package_id = self._get_property(root, "DTSID") or package_name
        
        # Create package entity
        self._kg.graph.add_entity(
            package_name,
            "Package",
            {
                "file_path": str(path),
                "dts_id": package_id,
                "description": self._get_property(root, "Description") or "",
                "creation_date": self._get_property(root, "CreationDate") or "",
                "creator_name": self._get_property(root, "CreatorName") or "",
            },
            source=str(path),
        )
        stats["packages"] += 1

        # Parse connection managers
        self._parse_connection_managers(root, package_name, stats)
        
        # Parse variables
        self._parse_variables(root, package_name, stats)
        
        # Parse executables (tasks and containers)
        self._parse_executables(root, package_name, stats)
        
        # Parse precedence constraints
        self._parse_precedence_constraints(root, package_name, stats)

    def _parse_connection_managers(
        self,
        root: ET.Element,
        package_name: str,
        stats: dict[str, int],
    ) -> None:
        """Parse connection managers."""
        # Find ConnectionManagers element
        for conn_mgrs in root.iter():
            if conn_mgrs.tag.endswith("ConnectionManagers"):
                for conn in conn_mgrs:
                    if conn.tag.endswith("ConnectionManager"):
                        conn_name = self._get_property(conn, "ObjectName") or "Unknown"
                        conn_type = self._get_attribute(conn, "CreationName") or "Unknown"
                        
                        # Extract connection string if available
                        conn_string = ""
                        for obj_data in conn.iter():
                            if obj_data.tag.endswith("ObjectData"):
                                for child in obj_data:
                                    conn_string = self._get_attribute(child, "ConnectionString") or ""
                                    break
                        
                        self._kg.graph.add_entity(
                            f"{package_name}.{conn_name}",
                            "ConnectionManager",
                            {
                                "name": conn_name,
                                "connection_type": conn_type,
                                "connection_string": self._sanitize_connection_string(conn_string),
                                "package": package_name,
                            },
                        )
                        
                        self._kg.graph.add_relationship(
                            package_name,
                            "has_connection",
                            f"{package_name}.{conn_name}",
                        )
                        stats["connections"] += 1

    def _parse_variables(
        self,
        root: ET.Element,
        package_name: str,
        stats: dict[str, int],
    ) -> None:
        """Parse package variables."""
        for variables in root.iter():
            if variables.tag.endswith("Variables"):
                for var in variables:
                    if var.tag.endswith("Variable"):
                        var_name = self._get_property(var, "ObjectName") or "Unknown"
                        var_ns = self._get_property(var, "Namespace") or "User"
                        data_type = self._get_property(var, "DataType")
                        
                        full_name = f"{package_name}::{var_ns}::{var_name}"
                        
                        self._kg.graph.add_entity(
                            full_name,
                            "Variable",
                            {
                                "name": var_name,
                                "namespace": var_ns,
                                "data_type": data_type,
                                "package": package_name,
                            },
                        )
                        
                        self._kg.graph.add_relationship(
                            package_name,
                            "has_variable",
                            full_name,
                        )
                        stats["variables"] += 1

    def _parse_executables(
        self,
        root: ET.Element,
        package_name: str,
        stats: dict[str, int],
        parent_name: str | None = None,
    ) -> None:
        """Parse executable tasks and containers."""
        parent = parent_name or package_name
        
        for executables in root.iter():
            if executables.tag.endswith("Executables"):
                for exe in executables:
                    if exe.tag.endswith("Executable"):
                        self._parse_executable(exe, package_name, parent, stats)
                break  # Only process direct children

    def _parse_executable(
        self,
        exe: ET.Element,
        package_name: str,
        parent_name: str,
        stats: dict[str, int],
    ) -> None:
        """Parse a single executable (task or container)."""
        exe_name = self._get_property(exe, "ObjectName") or "Unknown"
        exe_type = self._get_attribute(exe, "CreationName") or "Unknown"
        dts_id = self._get_property(exe, "DTSID") or exe_name
        
        full_name = f"{package_name}.{exe_name}"
        
        # Determine entity type based on CreationName
        entity_type = self._classify_executable(exe_type)
        
        self._kg.graph.add_entity(
            full_name,
            entity_type,
            {
                "name": exe_name,
                "task_type": exe_type,
                "dts_id": dts_id,
                "package": package_name,
                "description": self._get_property(exe, "Description") or "",
                "disabled": self._get_property(exe, "Disabled") == "True",
            },
        )
        
        self._kg.graph.add_relationship(
            parent_name,
            "contains",
            full_name,
        )
        
        if entity_type == "DataFlowTask":
            stats["data_flows"] += 1
            self._parse_data_flow(exe, package_name, full_name, stats)
        else:
            stats["tasks"] += 1
        
        # Parse Execute Package Task to find package dependencies
        if "ExecutePackage" in exe_type:
            self._parse_execute_package(exe, package_name, full_name)
        
        # Parse SQL Task for queries
        if "SQLTask" in exe_type:
            self._parse_sql_task(exe, package_name, full_name)
        
        # Use registered task handlers for additional parsing
        handler = self._task_registry.get_handler(exe_type)
        if handler:
            try:
                handler.handle(exe, self, package_name, full_name)
            except Exception as e:
                logger.warning(f"Task handler failed for {full_name}: {e}")
        
        # Parse event handlers
        self._parse_event_handlers(exe, package_name, full_name, stats)
        
        # Recursively parse nested executables (for containers)
        for child in exe:
            if child.tag.endswith("Executables"):
                for nested_exe in child:
                    if nested_exe.tag.endswith("Executable"):
                        self._parse_executable(nested_exe, package_name, full_name, stats)
    
    def _parse_event_handlers(
        self,
        exe: ET.Element,
        package_name: str,
        task_name: str,
        stats: dict[str, int],
    ) -> None:
        """Parse event handlers attached to a task or package."""
        for child in exe:
            if child.tag.endswith("EventHandlers"):
                for handler in child:
                    if handler.tag.endswith("EventHandler"):
                        event_name = self._get_property(handler, "EventName") or "Unknown"
                        handler_name = f"{task_name}::OnEvent::{event_name}"
                        
                        self._kg.graph.add_entity(
                            handler_name,
                            "EventHandler",
                            {
                                "event_name": event_name,
                                "package": package_name,
                                "parent_task": task_name,
                            },
                        )
                        
                        self._kg.graph.add_relationship(
                            task_name,
                            "handles_event",
                            handler_name,
                        )
                        
                        stats["event_handlers"] = stats.get("event_handlers", 0) + 1
                        
                        # Event handlers can contain their own executables
                        for exe_child in handler:
                            if exe_child.tag.endswith("Executables"):
                                for nested_exe in exe_child:
                                    if nested_exe.tag.endswith("Executable"):
                                        self._parse_executable(
                                            nested_exe, package_name, handler_name, stats
                                        )

    def _parse_data_flow(
        self,
        exe: ET.Element,
        package_name: str,
        task_name: str,
        stats: dict[str, int],
    ) -> None:
        """Parse Data Flow Task components."""
        for obj_data in exe.iter():
            if obj_data.tag.endswith("ObjectData"):
                for pipeline in obj_data:
                    if "pipeline" in pipeline.tag.lower():
                        self._parse_pipeline(pipeline, package_name, task_name)
                        break

    def _parse_pipeline(
        self,
        pipeline: ET.Element,
        package_name: str,
        task_name: str,
    ) -> None:
        """Parse pipeline components (sources, transforms, destinations)."""
        components: dict[str, dict[str, Any]] = {}
        
        # First pass: collect all components
        for comp in pipeline.iter():
            if comp.tag.endswith("component"):
                comp_name = comp.get("name") or "Unknown"
                comp_type = comp.get("componentClassID") or ""
                contact_info = comp.get("contactInfo") or ""
                
                full_name = f"{task_name}.{comp_name}"
                entity_type = self._classify_component(comp_type, contact_info)
                
                self._kg.graph.add_entity(
                    full_name,
                    entity_type,
                    {
                        "name": comp_name,
                        "component_type": comp_type,
                        "package": package_name,
                        "data_flow": task_name,
                    },
                )
                
                self._kg.graph.add_relationship(
                    task_name,
                    "contains",
                    full_name,
                )
                
                # Store for path resolution
                ref_id = comp.get("refId") or ""
                components[ref_id] = {"name": full_name, "type": entity_type}
                
                # Parse column mappings for lineage
                self._parse_component_columns(comp, full_name)
        
        # Second pass: parse paths (data flow connections)
        for path in pipeline.iter():
            if path.tag.endswith("path"):
                start_id = path.get("startId") or ""
                end_id = path.get("endId") or ""
                
                # Extract component names from path IDs
                start_comp = self._extract_component_from_path(start_id, components)
                end_comp = self._extract_component_from_path(end_id, components)
                
                if start_comp and end_comp:
                    self._kg.graph.add_relationship(
                        start_comp,
                        "flows_to",
                        end_comp,
                        {"path_name": path.get("name") or ""},
                    )

    def _parse_component_columns(
        self,
        comp: ET.Element,
        component_name: str,
    ) -> None:
        """Parse component input/output columns for detailed lineage."""
        for outputs in comp.iter():
            if outputs.tag.endswith("outputs"):
                for output in outputs:
                    if output.tag.endswith("output"):
                        output_name = output.get("name") or "Output"
                        for cols in output:
                            if cols.tag.endswith("outputColumns"):
                                for col in cols:
                                    if col.tag.endswith("outputColumn"):
                                        col_name = col.get("name") or "Unknown"
                                        self._kg.graph.add_entity(
                                            f"{component_name}.{output_name}.{col_name}",
                                            "Column",
                                            {
                                                "name": col_name,
                                                "component": component_name,
                                                "data_type": col.get("dataType") or "",
                                                "length": col.get("length") or "",
                                            },
                                        )
                                        self._kg.graph.add_relationship(
                                            component_name,
                                            "outputs_column",
                                            f"{component_name}.{output_name}.{col_name}",
                                        )

    def _parse_execute_package(
        self,
        exe: ET.Element,
        package_name: str,
        task_name: str,
    ) -> None:
        """Parse Execute Package Task for package dependencies."""
        for obj_data in exe.iter():
            if obj_data.tag.endswith("ObjectData"):
                for exec_pkg in obj_data:
                    # Look for PackageName property
                    pkg_path = exec_pkg.get("PackageName") or ""
                    if not pkg_path:
                        # Try to find in child elements
                        for child in exec_pkg:
                            if "PackageName" in child.tag or "Package" in child.tag:
                                pkg_path = child.text or child.get("PackageName") or ""
                                break
                    
                    if pkg_path:
                        # Extract package name from path
                        called_pkg = Path(pkg_path).stem
                        
                        self._kg.graph.add_relationship(
                            package_name,
                            "calls_package",
                            called_pkg,
                            {"via_task": task_name, "package_path": pkg_path},
                        )

    def _parse_sql_task(
        self,
        exe: ET.Element,
        package_name: str,
        task_name: str,
    ) -> None:
        """Parse SQL Task for database objects referenced."""
        for obj_data in exe.iter():
            if obj_data.tag.endswith("ObjectData"):
                for sql_task in obj_data:
                    # Try to find SQL statement from various locations
                    sql_text = ""
                    
                    # Check all attributes (including namespaced ones)
                    for attr_name, attr_value in sql_task.attrib.items():
                        if "SqlStatementSource" in attr_name:
                            sql_text = attr_value
                            break
                    
                    # Also check child elements
                    if not sql_text:
                        for child in sql_task.iter():
                            if "SqlStatementSource" in child.tag:
                                sql_text = child.text or ""
                                break
                            # Check attributes on child elements
                            for attr_name, attr_value in child.attrib.items():
                                if "SqlStatementSource" in attr_name:
                                    sql_text = attr_value
                                    break
                    
                    if sql_text:
                        # Extract table references from SQL
                        tables = self._extract_tables_from_sql(sql_text)
                        for table in tables:
                            self._kg.graph.add_entity(
                                table,
                                "Table",
                                {"name": table, "referenced_by": task_name},
                            )
                            self._kg.graph.add_relationship(
                                task_name,
                                "references_table",
                                table,
                            )

    def _parse_precedence_constraints(
        self,
        root: ET.Element,
        package_name: str,
        stats: dict[str, int],
    ) -> None:
        """Parse precedence constraints (task execution order)."""
        for constraints in root.iter():
            if constraints.tag.endswith("PrecedenceConstraints"):
                for constraint in constraints:
                    if constraint.tag.endswith("PrecedenceConstraint"):
                        from_name = self._get_property(constraint, "From") or ""
                        to_name = self._get_property(constraint, "To") or ""
                        eval_op = self._get_property(constraint, "EvalOp") or "Constraint"
                        
                        if from_name and to_name:
                            # Clean up the names (remove package prefix if present)
                            from_task = f"{package_name}.{from_name.split('\\')[-1]}"
                            to_task = f"{package_name}.{to_name.split('\\')[-1]}"
                            
                            self._kg.graph.add_relationship(
                                from_task,
                                "precedes",
                                to_task,
                                {"evaluation": eval_op},
                            )
                            stats["precedence_constraints"] += 1

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_property(self, element: ET.Element, prop_name: str) -> str | None:
        """Get a DTS property value from an element."""
        for prop in element.iter():
            if prop.tag.endswith("Property"):
                if prop.get("Name") == prop_name or prop.get("{www.microsoft.com/SqlServer/Dts}Name") == prop_name:
                    return prop.text
        return None

    def _get_attribute(self, element: ET.Element, attr_name: str) -> str | None:
        """Get an attribute value, checking with and without namespace."""
        value = element.get(attr_name)
        if not value:
            value = element.get(f"{{www.microsoft.com/SqlServer/Dts}}{attr_name}")
        return value

    def _classify_executable(self, exe_type: str) -> str:
        """Classify an executable into an entity type."""
        exe_type_lower = exe_type.lower()
        
        # Containers
        if "sequence" in exe_type_lower:
            return "SequenceContainer"
        elif "forloop" in exe_type_lower:
            return "ForLoopContainer"
        elif "foreachloop" in exe_type_lower:
            return "ForEachLoopContainer"
        
        # Data Flow
        elif "pipeline" in exe_type_lower or "dataflow" in exe_type_lower:
            return "DataFlowTask"
        
        # Package/Process execution
        elif "executepackage" in exe_type_lower:
            return "ExecutePackageTask"
        elif "executeprocess" in exe_type_lower:
            return "ExecuteProcessTask"
        elif "executedts" in exe_type_lower:
            return "ExecuteDTSTask"
        
        # SQL/Database
        elif "sqltask" in exe_type_lower or "executesql" in exe_type_lower:
            return "SQLTask"
        elif "bulkinsert" in exe_type_lower:
            return "BulkInsertTask"
        elif "transferdatabase" in exe_type_lower:
            return "TransferDatabaseTask"
        elif "transferjobs" in exe_type_lower:
            return "TransferJobsTask"
        elif "transferlogins" in exe_type_lower:
            return "TransferLoginsTask"
        elif "transferobjects" in exe_type_lower:
            return "TransferObjectsTask"
        elif "transfererror" in exe_type_lower:
            return "TransferErrorMessagesTask"
        elif "transferstoredprocedures" in exe_type_lower:
            return "TransferStoredProceduresTask"
        
        # Scripting
        elif "scripttask" in exe_type_lower:
            return "ScriptTask"
        elif "activexscript" in exe_type_lower:
            return "ActiveXScriptTask"
        
        # File operations
        elif "filetask" in exe_type_lower or "filesystem" in exe_type_lower:
            return "FileSystemTask"
        elif "ftptask" in exe_type_lower:
            return "FTPTask"
        
        # Communication
        elif "sendemail" in exe_type_lower or "sendmail" in exe_type_lower:
            return "SendMailTask"
        elif "messagequeue" in exe_type_lower or "msmq" in exe_type_lower:
            return "MessageQueueTask"
        elif "webservice" in exe_type_lower:
            return "WebServiceTask"
        
        # XML
        elif "xmltask" in exe_type_lower:
            return "XMLTask"
        
        # WMI
        elif "wmidatareader" in exe_type_lower:
            return "WMIDataReaderTask"
        elif "wmieventwatcher" in exe_type_lower:
            return "WMIEventWatcherTask"
        
        # Analysis Services
        elif "asprocessing" in exe_type_lower or "analysisservices" in exe_type_lower:
            return "AnalysisServicesTask"
        elif "asdatadef" in exe_type_lower:
            return "ASDataDefinitionTask"
        
        # Data mining
        elif "datamining" in exe_type_lower:
            return "DataMiningTask"
        
        # Expression
        elif "expressiontask" in exe_type_lower:
            return "ExpressionTask"
        
        # Maintenance
        elif "maintenance" in exe_type_lower:
            return "MaintenanceTask"
        elif "backup" in exe_type_lower:
            return "BackupTask"
        elif "shrink" in exe_type_lower:
            return "ShrinkDatabaseTask"
        elif "rebuild" in exe_type_lower or "reorganize" in exe_type_lower:
            return "IndexTask"
        elif "updatestatistics" in exe_type_lower:
            return "UpdateStatisticsTask"
        elif "checkintegrity" in exe_type_lower:
            return "CheckIntegrityTask"
        elif "cleanuphistory" in exe_type_lower:
            return "CleanupHistoryTask"
        
        # CDC (Change Data Capture)
        elif "cdccontrol" in exe_type_lower:
            return "CDCControlTask"
        
        # Hadoop/Big Data
        elif "hadoop" in exe_type_lower:
            return "HadoopTask"
        elif "hive" in exe_type_lower:
            return "HiveTask"
        elif "pig" in exe_type_lower:
            return "PigTask"
        
        # Azure
        elif "azure" in exe_type_lower:
            return "AzureTask"
        
        else:
            return "Task"

    def _classify_component(self, comp_type: str, contact_info: str) -> str:
        """Classify a data flow component into an entity type."""
        combined = (comp_type + contact_info).lower()
        
        if "source" in combined:
            return "Source"
        elif "destination" in combined:
            return "Destination"
        elif "lookup" in combined:
            return "Lookup"
        elif "merge" in combined:
            return "MergeTransform"
        elif "conditional" in combined or "split" in combined:
            return "ConditionalSplit"
        elif "derived" in combined:
            return "DerivedColumn"
        elif "aggregate" in combined:
            return "Aggregate"
        elif "sort" in combined:
            return "Sort"
        elif "union" in combined:
            return "UnionAll"
        elif "multicast" in combined:
            return "Multicast"
        elif "rowcount" in combined:
            return "RowCount"
        elif "convert" in combined:
            return "DataConversion"
        elif "script" in combined:
            return "ScriptComponent"
        else:
            return "Transform"

    def _extract_component_from_path(
        self,
        path_id: str,
        components: dict[str, dict[str, Any]],
    ) -> str | None:
        """Extract component name from a path reference ID."""
        # Path IDs are like "Package\DataFlow\Component.Outputs[Output 0]"
        for ref_id, comp_info in components.items():
            if ref_id and ref_id in path_id:
                return comp_info["name"]
        
        # Try to extract from path format
        if "\\" in path_id:
            parts = path_id.split("\\")
            for part in reversed(parts):
                clean_part = part.split(".")[0]
                for ref_id, comp_info in components.items():
                    if clean_part in comp_info["name"]:
                        return comp_info["name"]
        
        return None

    def _extract_tables_from_sql(self, sql: str) -> list[str]:
        """Extract table names from SQL statement (basic extraction)."""
        tables = []
        sql_upper = sql.upper()
        
        # Patterns for various SQL statements
        # Using \S+ to match table names (handles schema.table and [bracketed] names)
        patterns = [
            r"FROM\s+(\S+)",
            r"JOIN\s+(\S+)",
            r"INTO\s+(\S+)",
            r"UPDATE\s+(\S+)",
            r"TABLE\s+(\S+)",  # Handles TRUNCATE TABLE, DROP TABLE, etc.
            r"INSERT\s+INTO\s+(\S+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql_upper)
            for match in matches:
                # Clean up brackets and schema, get just the table name
                table = match.strip("[]").split(".")[-1].strip("[]")
                # Filter out SQL keywords
                if table and table not in ("SELECT", "FROM", "WHERE", "SET", "VALUES", "("):
                    tables.append(table)
        
        return list(set(tables))

    def _sanitize_connection_string(self, conn_str: str) -> str:
        """Remove sensitive info from connection string."""
        # Remove password
        sanitized = re.sub(
            r"(Password|PWD)\s*=\s*[^;]*",
            r"\1=***",
            conn_str,
            flags=re.IGNORECASE,
        )
        return sanitized

    # =========================================================================
    # Query Methods
    # =========================================================================

    def find_packages(self) -> list[dict[str, Any]]:
        """Find all loaded packages."""
        entities = self._kg.graph.get_all_entities(entity_type="Package")
        return [e.to_dict() for e in entities]

    def find_tasks(
        self,
        package: str | None = None,
        task_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find tasks in the loaded packages.
        
        Args:
            package: Filter by package name
            task_type: Filter by task type (e.g., 'SQLTask', 'DataFlowTask')
        """
        all_entities = self._kg.graph.get_all_entities()
        tasks = []
        
        task_types = {
            "Task", "DataFlowTask", "SQLTask", "ScriptTask", 
            "ExecutePackageTask", "FileSystemTask", "FTPTask",
            "SendMailTask", "ExpressionTask",
        }
        
        for entity in all_entities:
            if entity.type in task_types:
                if package and entity.attributes.get("package") != package:
                    continue
                if task_type and entity.type != task_type:
                    continue
                tasks.append(entity.to_dict())
        
        return tasks

    def find_data_flows(self, package: str | None = None) -> list[dict[str, Any]]:
        """Find all Data Flow tasks."""
        return self.find_tasks(package=package, task_type="DataFlowTask")

    def find_connections(self, package: str | None = None) -> list[dict[str, Any]]:
        """Find all connection managers."""
        entities = self._kg.graph.get_all_entities(entity_type="ConnectionManager")
        if package:
            return [e.to_dict() for e in entities if e.attributes.get("package") == package]
        return [e.to_dict() for e in entities]

    def trace_data_lineage(
        self,
        target: str,
        max_depth: int = 10,
    ) -> list[list[str]]:
        """
        Trace data lineage backwards from a target.
        
        Args:
            target: Target entity name (component, table, or column)
            max_depth: Maximum traversal depth
            
        Returns:
            List of lineage paths (source → ... → target)
        """
        paths = []
        
        def trace_back(entity_id: str, current_path: list[str], depth: int) -> None:
            if depth > max_depth:
                return
            
            # Get incoming relationships
            incoming = self._kg.graph.get_relationships(entity_id, direction="incoming")
            
            # Filter for data flow relationships
            lineage_relations = {"flows_to", "outputs_column", "contains"}
            sources = [
                r for r in incoming 
                if r.relation in lineage_relations
            ]
            
            if not sources:
                # This is a source node
                paths.append(list(reversed(current_path)))
                return
            
            for rel in sources:
                if rel.source_id not in current_path:
                    trace_back(
                        rel.source_id,
                        current_path + [rel.source_id],
                        depth + 1,
                    )
        
        trace_back(target, [target], 0)
        return paths

    def find_package_dependencies(self, package: str) -> dict[str, Any]:
        """
        Find packages called by and calling a package.
        
        Args:
            package: Package name
            
        Returns:
            Dict with 'calls' and 'called_by' lists
        """
        calls = self._kg.graph.get_relationships(
            package, relation="calls_package", direction="outgoing"
        )
        called_by = self._kg.graph.get_relationships(
            package, relation="calls_package", direction="incoming"
        )
        
        return {
            "package": package,
            "calls": [r.target_id for r in calls],
            "called_by": [r.source_id for r in called_by],
        }

    def find_table_usage(self, table: str) -> list[dict[str, Any]]:
        """Find all tasks that reference a table."""
        incoming = self._kg.graph.get_relationships(
            table, relation="references_table", direction="incoming"
        )
        
        results = []
        for rel in incoming:
            task = self._kg.graph.get_entity(rel.source_id)
            if task:
                results.append(task.to_dict())
        
        return results

    def get_execution_order(self, package: str) -> list[str]:
        """
        Get the execution order of tasks in a package.
        
        Uses topological sort based on precedence constraints.
        """
        # Get all tasks in package
        tasks = self.find_tasks(package=package)
        task_names = {t["id"] for t in tasks}
        
        # Build adjacency list from precedence constraints
        adj: dict[str, list[str]] = {name: [] for name in task_names}
        in_degree: dict[str, int] = {name: 0 for name in task_names}
        
        for task_name in task_names:
            rels = self._kg.graph.get_relationships(
                task_name, relation="precedes", direction="outgoing"
            )
            for rel in rels:
                if rel.target_id in task_names:
                    adj[task_name].append(rel.target_id)
                    in_degree[rel.target_id] += 1
        
        # Topological sort (Kahn's algorithm)
        queue = [t for t, d in in_degree.items() if d == 0]
        order = []
        
        while queue:
            task = queue.pop(0)
            order.append(task)
            
            for next_task in adj[task]:
                in_degree[next_task] -= 1
                if in_degree[next_task] == 0:
                    queue.append(next_task)
        
        return order

    def stats(self) -> dict[str, int]:
        """Get statistics about the loaded packages."""
        kg_stats = self._kg.stats()
        
        # Count by type
        all_entities = self._kg.graph.get_all_entities()
        type_counts: dict[str, int] = {}
        for entity in all_entities:
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1
        
        return {
            "total_entities": kg_stats["entities"],
            "total_relationships": kg_stats["relationships"],
            "packages_loaded": len(self._loaded_packages),
            **type_counts,
        }

    # =========================================================================
    # Tool Definitions
    # =========================================================================

    def _find_packages_tool(self) -> BaseTool:
        @tool
        def find_packages() -> str:
            """Find all loaded SSIS packages."""
            packages = self.find_packages()
            if not packages:
                return "No packages loaded."
            return "\n".join(
                f"- {p['id']}: {p['attributes'].get('description', 'No description')}"
                for p in packages
            )
        return find_packages

    def _find_tasks_tool(self) -> BaseTool:
        @tool
        def find_tasks(package: str = "", task_type: str = "") -> str:
            """
            Find tasks in SSIS packages.
            
            Args:
                package: Filter by package name (optional)
                task_type: Filter by type like SQLTask, DataFlowTask (optional)
            """
            tasks = self.find_tasks(
                package=package or None,
                task_type=task_type or None,
            )
            if not tasks:
                return "No tasks found."
            return "\n".join(
                f"- {t['id']} ({t['type']})"
                for t in tasks
            )
        return find_tasks

    def _find_data_flows_tool(self) -> BaseTool:
        @tool
        def find_data_flows(package: str = "") -> str:
            """
            Find Data Flow tasks in SSIS packages.
            
            Args:
                package: Filter by package name (optional)
            """
            flows = self.find_data_flows(package=package or None)
            if not flows:
                return "No data flows found."
            return "\n".join(f"- {f['id']}" for f in flows)
        return find_data_flows

    def _find_connections_tool(self) -> BaseTool:
        @tool
        def find_connections(package: str = "") -> str:
            """
            Find connection managers in SSIS packages.
            
            Args:
                package: Filter by package name (optional)
            """
            conns = self.find_connections(package=package or None)
            if not conns:
                return "No connections found."
            return "\n".join(
                f"- {c['id']}: {c['attributes'].get('connection_type', 'Unknown')}"
                for c in conns
            )
        return find_connections

    def _trace_lineage_tool(self) -> BaseTool:
        @tool
        def trace_data_lineage(target: str, max_depth: int = 10) -> str:
            """
            Trace data lineage backwards from a target.
            
            Args:
                target: Target entity (component, table, column)
                max_depth: Maximum depth to trace
            """
            paths = self.trace_data_lineage(target, max_depth)
            if not paths:
                return f"No lineage found for '{target}'."
            
            result = [f"Lineage for '{target}':"]
            for i, path in enumerate(paths, 1):
                result.append(f"  Path {i}: {' → '.join(path)}")
            return "\n".join(result)
        return trace_data_lineage

    def _find_dependencies_tool(self) -> BaseTool:
        @tool
        def find_package_dependencies(package: str) -> str:
            """
            Find package dependencies (calls and called_by).
            
            Args:
                package: Package name to analyze
            """
            deps = self.find_package_dependencies(package)
            
            result = [f"Dependencies for '{package}':"]
            if deps["calls"]:
                result.append(f"  Calls: {', '.join(deps['calls'])}")
            else:
                result.append("  Calls: (none)")
            
            if deps["called_by"]:
                result.append(f"  Called by: {', '.join(deps['called_by'])}")
            else:
                result.append("  Called by: (none)")
            
            return "\n".join(result)
        return find_package_dependencies

    def _find_callers_tool(self) -> BaseTool:
        @tool
        def find_table_usage(table: str) -> str:
            """
            Find all tasks that reference a database table.
            
            Args:
                table: Table name to search for
            """
            tasks = self.find_table_usage(table)
            if not tasks:
                return f"No tasks reference '{table}'."
            return "\n".join(
                f"- {t['id']} ({t['type']})"
                for t in tasks
            )
        return find_table_usage

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> "SSISAnalyzer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._kg.graph.close()

    def close(self) -> None:
        """Close the analyzer and underlying knowledge graph."""
        self._kg.graph.close()
