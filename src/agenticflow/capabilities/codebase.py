"""
Codebase Analyzer Capability.

A capability for parsing and analyzing Python codebases, storing the
structure in a knowledge graph for intelligent querying.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, tool

from agenticflow.capabilities.base import BaseCapability
from agenticflow.capabilities.knowledge_graph import KnowledgeGraph

if TYPE_CHECKING:
    from agenticflow.capabilities.knowledge_graph import Entity

logger = logging.getLogger(__name__)


class CodebaseAnalyzer(BaseCapability):
    """
    Capability for analyzing Python codebases.

    Parses Python source files using AST and stores the structure
    (modules, classes, functions, imports, calls) in a knowledge graph.

    Example:
        ```python
        analyzer = CodebaseAnalyzer()
        analyzer.load_directory("/path/to/project/src")

        # Query the codebase
        classes = analyzer.find_classes()
        callers = analyzer.find_callers("my_function")
        subclasses = analyzer.find_subclasses("BaseClass")
        ```
    """

    name: str = "codebase_analyzer"
    description: str = "Analyze Python codebases and query code structure"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the codebase analyzer."""
        super().__init__(**kwargs)
        self._kg = KnowledgeGraph(name="codebase_kg")
        self._tools: list[BaseTool] = []
        self._loaded_files: set[str] = set()
        self._build_tools()

    def _build_tools(self) -> None:
        """Build the analyzer tools."""
        self._tools = [
            self._find_classes_tool(),
            self._find_functions_tool(),
            self._find_callers_tool(),
            self._find_usages_tool(),
            self._find_subclasses_tool(),
            self._find_imports_tool(),
            self._get_definition_tool(),
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

    def load_file(self, file_path: str | Path) -> dict[str, int]:
        """
        Load and parse a single Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            Statistics about parsed elements
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.suffix == ".py":
            raise ValueError(f"Not a Python file: {path}")

        if str(path) in self._loaded_files:
            logger.debug(f"File already loaded: {path}")
            return {"skipped": 1}

        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {path}: {e}")
            return {"errors": 1}

        stats = self._process_ast(tree, path)
        self._loaded_files.add(str(path))
        return stats

    def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        exclude_patterns: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Load all Python files from a directory.

        Args:
            directory: Path to the directory
            recursive: Whether to search recursively
            exclude_patterns: Glob patterns to exclude (e.g., ["**/test_*"])

        Returns:
            Aggregate statistics about parsed elements
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        exclude_patterns = exclude_patterns or []
        pattern = "**/*.py" if recursive else "*.py"

        total_stats: dict[str, int] = {
            "files": 0,
            "modules": 0,
            "classes": 0,
            "functions": 0,
            "imports": 0,
            "errors": 0,
        }

        for py_file in dir_path.glob(pattern):
            # Check exclusions
            should_exclude = False
            for excl in exclude_patterns:
                if py_file.match(excl):
                    should_exclude = True
                    break

            if should_exclude:
                continue

            stats = self.load_file(py_file)
            for key, value in stats.items():
                total_stats[key] = total_stats.get(key, 0) + value

        return total_stats

    def _process_ast(self, tree: ast.AST, file_path: Path) -> dict[str, int]:
        """Process an AST and populate the knowledge graph."""
        stats = {"files": 1, "modules": 0, "classes": 0, "functions": 0, "imports": 0}

        # Create module entity
        module_name = file_path.stem
        module_id = f"module:{module_name}"
        self._kg.add_entity(
            module_id,
            "Module",
            {
                "name": module_name,
                "file_path": str(file_path),
                "full_path": str(file_path.absolute()),
            },
        )
        stats["modules"] = 1

        # Process top-level nodes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                self._process_class(node, module_id, file_path)
                stats["classes"] += 1
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                self._process_function(node, module_id, file_path)
                stats["functions"] += 1
            elif isinstance(node, ast.Import | ast.ImportFrom):
                self._process_import(node, module_id)
                stats["imports"] += 1

        return stats

    def _process_class(
        self, node: ast.ClassDef, parent_id: str, file_path: Path
    ) -> None:
        """Process a class definition."""
        class_id = f"class:{node.name}"

        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attr_name(base)}")

        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        self._kg.add_entity(
            class_id,
            "Class",
            {
                "name": node.name,
                "bases": bases,
                "decorators": decorators,
                "lineno": node.lineno,
                "file_path": str(file_path),
                "docstring": ast.get_docstring(node) or "",
            },
        )

        # Relationship to module
        self._kg.add_relationship(parent_id, "contains", class_id)

        # Inheritance relationships
        for base in bases:
            base_id = f"class:{base}"
            # Create base class entity if not exists
            if not self._kg.get_entity(base_id):
                self._kg.add_entity(base_id, "Class", {"name": base, "external": True})
            self._kg.add_relationship(class_id, "inherits", base_id)

        # Process methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                self._process_method(item, class_id, file_path)

    def _process_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, parent_id: str, file_path: Path
    ) -> None:
        """Process a function definition."""
        func_id = f"function:{node.name}"

        # Get parameters
        params = []
        for arg in node.args.args:
            param_info = {"name": arg.arg}
            if arg.annotation:
                param_info["type"] = self._get_annotation_str(arg.annotation)
            params.append(param_info)

        # Get return type
        return_type = None
        if node.returns:
            return_type = self._get_annotation_str(node.returns)

        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        self._kg.add_entity(
            func_id,
            "Function",
            {
                "name": node.name,
                "parameters": params,
                "return_type": return_type,
                "decorators": decorators,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "lineno": node.lineno,
                "file_path": str(file_path),
                "docstring": ast.get_docstring(node) or "",
            },
        )

        self._kg.add_relationship(parent_id, "contains", func_id)

        # Process function calls within the body
        self._process_calls(node, func_id)

    def _process_method(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, class_id: str, file_path: Path
    ) -> None:
        """Process a method definition."""
        method_id = f"method:{class_id.split(':')[1]}.{node.name}"

        # Get parameters (skip self/cls)
        params = []
        args_list = node.args.args[1:] if node.args.args else []
        for arg in args_list:
            param_info = {"name": arg.arg}
            if arg.annotation:
                param_info["type"] = self._get_annotation_str(arg.annotation)
            params.append(param_info)

        # Get return type
        return_type = None
        if node.returns:
            return_type = self._get_annotation_str(node.returns)

        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        # Determine method kind
        method_kind = "method"
        if any(d in ["staticmethod", "classmethod", "property"] for d in decorators):
            if "staticmethod" in decorators:
                method_kind = "staticmethod"
            elif "classmethod" in decorators:
                method_kind = "classmethod"
            elif "property" in decorators:
                method_kind = "property"

        self._kg.add_entity(
            method_id,
            "Method",
            {
                "name": node.name,
                "class": class_id.split(":")[1],
                "parameters": params,
                "return_type": return_type,
                "decorators": decorators,
                "method_kind": method_kind,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "lineno": node.lineno,
                "file_path": str(file_path),
                "docstring": ast.get_docstring(node) or "",
            },
        )

        self._kg.add_relationship(class_id, "has_method", method_id)

        # Process calls within method body
        self._process_calls(node, method_id)

    def _process_import(
        self, node: ast.Import | ast.ImportFrom, module_id: str
    ) -> None:
        """Process an import statement."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_id = f"import:{alias.name}"
                if not self._kg.get_entity(import_id):
                    self._kg.add_entity(
                        import_id,
                        "Import",
                        {"name": alias.name, "alias": alias.asname},
                    )
                self._kg.add_relationship(module_id, "imports", import_id)
        else:
            # ImportFrom
            module_name = node.module or ""
            for alias in node.names:
                full_name = f"{module_name}.{alias.name}" if module_name else alias.name
                import_id = f"import:{full_name}"
                if not self._kg.get_entity(import_id):
                    self._kg.add_entity(
                        import_id,
                        "Import",
                        {
                            "name": alias.name,
                            "from_module": module_name,
                            "alias": alias.asname,
                        },
                    )
                self._kg.add_relationship(module_id, "imports", import_id)

    def _process_calls(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, caller_id: str
    ) -> None:
        """Extract function/method calls from a function body."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                callee_name = self._get_call_name(child)
                if callee_name:
                    # Create a "calls" relationship
                    # We'll create placeholder entities for callees
                    callee_id = f"callable:{callee_name}"
                    if not self._kg.get_entity(callee_id):
                        self._kg.add_entity(
                            callee_id,
                            "Callable",
                            {"name": callee_name, "resolved": False},
                        )
                    self._kg.add_relationship(caller_id, "calls", callee_id)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_attr_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'module.Class')."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Get decorator name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attr_name(node)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            elif isinstance(node.func, ast.Attribute):
                return self._get_attr_name(node.func)
        return "unknown"

    def _get_annotation_str(self, node: ast.expr) -> str:
        """Convert type annotation to string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            base = self._get_annotation_str(node.value)
            if isinstance(node.slice, ast.Tuple):
                args = ", ".join(self._get_annotation_str(e) for e in node.slice.elts)
            else:
                args = self._get_annotation_str(node.slice)
            return f"{base}[{args}]"
        elif isinstance(node, ast.Attribute):
            return self._get_attr_name(node)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type with |
            left = self._get_annotation_str(node.left)
            right = self._get_annotation_str(node.right)
            return f"{left} | {right}"
        return "Any"

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Get the name of a called function/method."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    # =========================================================================
    # Query Methods
    # =========================================================================

    def find_classes(self, pattern: str | None = None) -> list[Entity]:
        """
        Find all classes, optionally matching a pattern.

        Args:
            pattern: Optional name pattern (substring match)

        Returns:
            List of class entities
        """
        classes = self._kg.get_entities("Class")
        if pattern:
            classes = [c for c in classes if pattern.lower() in c.id.lower()]
        return classes

    def find_functions(self, pattern: str | None = None) -> list[Entity]:
        """
        Find all functions, optionally matching a pattern.

        Args:
            pattern: Optional name pattern (substring match)

        Returns:
            List of function entities
        """
        functions = self._kg.get_entities("Function")
        if pattern:
            functions = [f for f in functions if pattern.lower() in f.id.lower()]
        return functions

    def find_callers(self, function_name: str) -> list[Entity]:
        """
        Find all functions/methods that call a given function.

        Args:
            function_name: Name of the function to find callers for

        Returns:
            List of caller entities
        """
        # Find the callable entity
        callee_id = f"callable:{function_name}"
        if not self._kg.get_entity(callee_id):
            return []

        # Find incoming "calls" relationships
        rels = self._kg.get_relationships(callee_id, "calls", direction="incoming")
        callers = []
        for rel in rels:
            caller = self._kg.get_entity(rel.source)
            if caller:
                callers.append(caller)
        return callers

    def find_usages(self, name: str) -> list[dict[str, Any]]:
        """
        Find all usages of a class, function, or import.

        Args:
            name: Name to search for

        Returns:
            List of usage information
        """
        usages = []

        # Check various entity types
        for prefix in ["class:", "function:", "method:", "callable:", "import:"]:
            entity_id = f"{prefix}{name}"
            entity = self._kg.get_entity(entity_id)
            if entity:
                # Get all relationships pointing to this entity
                rels = self._kg.get_relationships(entity_id, direction="incoming")
                for rel in rels:
                    source = self._kg.get_entity(rel.source)
                    if source:
                        usages.append({
                            "used_in": source.id,
                            "type": source.type,
                            "relationship": rel.relation,
                            "file": source.attributes.get("file_path", "unknown"),
                        })

        return usages

    def find_subclasses(self, base_class: str) -> list[Entity]:
        """
        Find all subclasses of a given class.

        Args:
            base_class: Name of the base class

        Returns:
            List of subclass entities
        """
        base_id = f"class:{base_class}"
        if not self._kg.get_entity(base_id):
            return []

        # Find incoming "inherits" relationships
        rels = self._kg.get_relationships(base_id, "inherits", direction="incoming")
        subclasses = []
        for rel in rels:
            subclass = self._kg.get_entity(rel.source)
            if subclass:
                subclasses.append(subclass)
        return subclasses

    def find_imports(self, module_name: str | None = None) -> list[Entity]:
        """
        Find all imports, optionally filtered by module.

        Args:
            module_name: Optional module name filter

        Returns:
            List of import entities
        """
        imports = self._kg.get_entities("Import")
        if module_name:
            imports = [
                i for i in imports
                if module_name.lower() in i.attributes.get("from_module", "").lower()
                or module_name.lower() in i.attributes.get("name", "").lower()
            ]
        return imports

    def get_definition(self, name: str) -> Entity | None:
        """
        Get the definition of a class, function, or method.

        Args:
            name: Name to look up

        Returns:
            Entity with definition info, or None
        """
        for prefix in ["class:", "function:", "method:"]:
            entity = self._kg.get_entity(f"{prefix}{name}")
            if entity:
                return entity
        return None

    def get_class_methods(self, class_name: str) -> list[Entity]:
        """
        Get all methods of a class.

        Args:
            class_name: Name of the class

        Returns:
            List of method entities
        """
        class_id = f"class:{class_name}"
        if not self._kg.get_entity(class_id):
            return []

        rels = self._kg.get_relationships(class_id, "has_method", direction="outgoing")
        methods = []
        for rel in rels:
            method = self._kg.get_entity(rel.target_id)
            if method:
                methods.append(method)
        return methods

    # =========================================================================
    # Tool Implementations
    # =========================================================================

    def _find_classes_tool(self) -> BaseTool:
        analyzer = self

        @tool
        def find_classes(pattern: str = "") -> str:
            """
            Find classes in the codebase.

            Args:
                pattern: Optional name pattern to filter by

            Returns:
                List of matching classes
            """
            classes = analyzer.find_classes(pattern if pattern else None)
            if not classes:
                return "No classes found"

            lines = [f"Found {len(classes)} classes:"]
            for c in classes:
                bases = c.attributes.get("bases", [])
                base_str = f" (extends {', '.join(bases)})" if bases else ""
                lines.append(f"  - {c.attributes['name']}{base_str}")
                lines.append(f"    File: {c.attributes.get('file_path', 'unknown')}")

            return "\n".join(lines)

        return find_classes

    def _find_functions_tool(self) -> BaseTool:
        analyzer = self

        @tool
        def find_functions(pattern: str = "") -> str:
            """
            Find functions in the codebase.

            Args:
                pattern: Optional name pattern to filter by

            Returns:
                List of matching functions
            """
            functions = analyzer.find_functions(pattern if pattern else None)
            if not functions:
                return "No functions found"

            lines = [f"Found {len(functions)} functions:"]
            for f in functions:
                ret_type = f.attributes.get("return_type") or "None"
                lines.append(f"  - {f.attributes['name']}() -> {ret_type}")
                lines.append(f"    File: {f.attributes.get('file_path', 'unknown')}")

            return "\n".join(lines)

        return find_functions

    def _find_callers_tool(self) -> BaseTool:
        analyzer = self

        @tool
        def find_callers(function_name: str) -> str:
            """
            Find all callers of a function.

            Args:
                function_name: Name of the function

            Returns:
                List of callers
            """
            callers = analyzer.find_callers(function_name)
            if not callers:
                return f"No callers found for '{function_name}'"

            lines = [f"Found {len(callers)} callers of '{function_name}':"]
            for c in callers:
                lines.append(f"  - {c.id} ({c.type})")

            return "\n".join(lines)

        return find_callers

    def _find_usages_tool(self) -> BaseTool:
        analyzer = self

        @tool
        def find_usages(name: str) -> str:
            """
            Find all usages of a name (class, function, import).

            Args:
                name: Name to search for

            Returns:
                Usage information
            """
            usages = analyzer.find_usages(name)
            if not usages:
                return f"No usages found for '{name}'"

            lines = [f"Found {len(usages)} usages of '{name}':"]
            for u in usages:
                lines.append(f"  - {u['relationship']} in {u['used_in']}")
                lines.append(f"    File: {u['file']}")

            return "\n".join(lines)

        return find_usages

    def _find_subclasses_tool(self) -> BaseTool:
        analyzer = self

        @tool
        def find_subclasses(base_class: str) -> str:
            """
            Find all subclasses of a class.

            Args:
                base_class: Name of the base class

            Returns:
                List of subclasses
            """
            subclasses = analyzer.find_subclasses(base_class)
            if not subclasses:
                return f"No subclasses found for '{base_class}'"

            lines = [f"Found {len(subclasses)} subclasses of '{base_class}':"]
            for s in subclasses:
                lines.append(f"  - {s.attributes['name']}")
                lines.append(f"    File: {s.attributes.get('file_path', 'unknown')}")

            return "\n".join(lines)

        return find_subclasses

    def _find_imports_tool(self) -> BaseTool:
        analyzer = self

        @tool
        def find_imports(module_name: str = "") -> str:
            """
            Find imports in the codebase.

            Args:
                module_name: Optional module name to filter by

            Returns:
                List of imports
            """
            imports = analyzer.find_imports(module_name if module_name else None)
            if not imports:
                return "No imports found"

            lines = [f"Found {len(imports)} imports:"]
            for i in imports:
                from_mod = i.attributes.get("from_module", "")
                if from_mod:
                    lines.append(f"  - from {from_mod} import {i.attributes['name']}")
                else:
                    lines.append(f"  - import {i.attributes['name']}")

            return "\n".join(lines)

        return find_imports

    def _get_definition_tool(self) -> BaseTool:
        analyzer = self

        @tool
        def get_definition(name: str) -> str:
            """
            Get the definition of a class, function, or method.

            Args:
                name: Name to look up

            Returns:
                Definition information
            """
            entity = analyzer.get_definition(name)
            if not entity:
                return f"No definition found for '{name}'"

            lines = [f"Definition of '{name}' ({entity.type}):"]
            lines.append(f"  File: {entity.attributes.get('file_path', 'unknown')}")
            lines.append(f"  Line: {entity.attributes.get('lineno', 'unknown')}")

            if entity.type == "Class":
                bases = entity.attributes.get("bases", [])
                if bases:
                    lines.append(f"  Bases: {', '.join(bases)}")

            if entity.type in ("Function", "Method"):
                params = entity.attributes.get("parameters", [])
                param_strs = []
                for p in params:
                    ptype = p.get("type", "")
                    param_strs.append(f"{p['name']}: {ptype}" if ptype else p["name"])
                lines.append(f"  Parameters: ({', '.join(param_strs)})")

                ret_type = entity.attributes.get("return_type")
                if ret_type:
                    lines.append(f"  Returns: {ret_type}")

            docstring = entity.attributes.get("docstring", "")
            if docstring:
                lines.append(f"  Docstring: {docstring[:100]}...")

            return "\n".join(lines)

        return get_definition

    def stats(self) -> dict[str, Any]:
        """Get analyzer statistics."""
        kg_stats = self._kg.stats()
        return {
            "loaded_files": len(self._loaded_files),
            "entities": kg_stats["entities"],
            "relationships": kg_stats["relationships"],
            "types": kg_stats.get("type_counts", {}),
        }

    def clear(self) -> None:
        """Clear all loaded data."""
        self._kg.clear()
        self._loaded_files.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base["stats"] = self.stats()
        return base
