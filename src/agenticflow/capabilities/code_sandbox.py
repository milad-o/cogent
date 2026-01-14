"""
CodeSandbox capability - safe Python code execution.

Provides sandboxed code execution with resource limits and security controls,
enabling agents to run and test Python code safely.

Example:
    ```python
    from agenticflow import Agent
    from agenticflow.capabilities import CodeSandbox
    
    agent = Agent(
        name="Coder",
        model=model,
        capabilities=[CodeSandbox()],
    )
    
    # Agent can now execute Python code
    await agent.run("Write and run a function that calculates fibonacci(10)")
    ```
"""

from __future__ import annotations

import ast
import contextlib
import io
import signal
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agenticflow.tools.base import BaseTool, tool

from agenticflow.capabilities.base import BaseCapability


@dataclass
class ExecutionResult:
    """Result of code execution."""
    
    success: bool
    output: str
    error: str | None = None
    return_value: Any = None
    execution_time_ms: float = 0
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "return_value": repr(self.return_value) if self.return_value is not None else None,
            "execution_time_ms": self.execution_time_ms,
            "executed_at": self.executed_at.isoformat(),
        }


class TimeoutError(Exception):
    """Raised when execution times out."""
    pass


class SecurityError(Exception):
    """Raised when code violates security constraints."""
    pass


# Dangerous modules/functions that should be blocked
BLOCKED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "urllib",
    "requests",
    "httpx",
    "ftplib",
    "smtplib",
    "telnetlib",
    "pickle",
    "shelve",
    "marshal",
    "ctypes",
    "multiprocessing",
    "threading",
    "_thread",
    "signal",
    "resource",
    "pty",
    "tty",
    "termios",
    "fcntl",
    "pipes",
    "posix",
    "pwd",
    "grp",
    "spwd",
    "crypt",
    "select",
    "selectors",
    "mmap",
    "code",
    "codeop",
    "importlib",
    "__import__",
    "builtins",
    "__builtins__",
}

# Dangerous built-in functions
BLOCKED_BUILTINS = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "breakpoint",
    "memoryview",
    "exit",
    "quit",
}


class CodeValidator(ast.NodeVisitor):
    """AST visitor to validate code security."""
    
    def __init__(
        self,
        blocked_imports: set[str],
        blocked_builtins: set[str],
        allow_imports: bool = False,
    ):
        self.blocked_imports = blocked_imports
        self.blocked_builtins = blocked_builtins
        self.allow_imports = allow_imports
        self.errors: list[str] = []
    
    def visit_Import(self, node: ast.Import) -> None:
        if not self.allow_imports:
            for alias in node.names:
                self.errors.append(f"Import not allowed: {alias.name}")
        else:
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in self.blocked_imports:
                    self.errors.append(f"Blocked import: {alias.name}")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not self.allow_imports:
            self.errors.append(f"Import not allowed: from {node.module}")
        else:
            if node.module:
                module = node.module.split(".")[0]
                if module in self.blocked_imports:
                    self.errors.append(f"Blocked import: from {node.module}")
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        # Check for blocked built-in calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.blocked_builtins:
                self.errors.append(f"Blocked function: {node.func.id}()")
        
        # Check for __dunder__ method calls
        if isinstance(node.func, ast.Attribute):
            if node.func.attr.startswith("__") and node.func.attr.endswith("__"):
                if node.func.attr not in ("__init__", "__str__", "__repr__", "__len__"):
                    self.errors.append(f"Blocked dunder method: {node.func.attr}")
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block access to dangerous attributes
        dangerous_attrs = {"__class__", "__bases__", "__subclasses__", "__mro__", "__code__", "__globals__"}
        if node.attr in dangerous_attrs:
            self.errors.append(f"Blocked attribute access: {node.attr}")
        self.generic_visit(node)


class CodeSandbox(BaseCapability):
    """
    CodeSandbox capability for safe Python code execution.
    
    Provides a sandboxed environment for executing Python code with:
    - Timeout limits
    - Import restrictions
    - Built-in function restrictions
    - Output capture
    
    Args:
        timeout: Maximum execution time in seconds (default: 5)
        max_output_length: Maximum output length in characters (default: 10000)
        allow_imports: Whether to allow any imports (default: False)
        allowed_imports: Set of allowed module names if allow_imports=True
        name: Capability name (default: "code_sandbox")
    
    Example:
        ```python
        # Basic sandbox (no imports allowed)
        sandbox = CodeSandbox()
        
        # Allow specific safe imports
        sandbox = CodeSandbox(
            allow_imports=True,
            allowed_imports={"math", "random", "datetime", "json", "re"},
        )
        ```
    
    Security Notes:
        - No file system access
        - No network access
        - No subprocess execution
        - Limited built-in functions
        - Timeout protection
    """
    
    # Safe imports that can be allowed
    SAFE_IMPORTS = {
        "math",
        "random",
        "datetime",
        "json",
        "re",
        "collections",
        "itertools",
        "functools",
        "operator",
        "string",
        "textwrap",
        "unicodedata",
        "decimal",
        "fractions",
        "statistics",
        "heapq",
        "bisect",
        "array",
        "copy",
        "pprint",
        "enum",
        "dataclasses",
        "typing",
        "abc",
        "time",  # Safe for datetime operations, strftime needs it
    }
    
    def __init__(
        self,
        timeout: int = 5,
        max_output_length: int = 10000,
        allow_imports: bool = False,
        allowed_imports: set[str] | None = None,
        name: str = "code_sandbox",
    ):
        self._name = name
        self._timeout = timeout
        self._max_output_length = max_output_length
        self._allow_imports = allow_imports
        
        # Build blocked imports set
        if allow_imports and allowed_imports:
            self._blocked_imports = BLOCKED_IMPORTS - allowed_imports
        elif allow_imports:
            self._blocked_imports = BLOCKED_IMPORTS - self.SAFE_IMPORTS
        else:
            self._blocked_imports = BLOCKED_IMPORTS
        
        self._blocked_builtins = BLOCKED_BUILTINS
        self._tools_cache: list[BaseTool] | None = None
        
        # Execution history
        self._history: list[ExecutionResult] = []
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return "Safe Python code execution sandbox"
    
    @property
    def tools(self) -> list[BaseTool]:
        if self._tools_cache is None:
            self._tools_cache = [
                self._execute_tool(),
                self._execute_function_tool(),
            ]
        return self._tools_cache
    
    @property
    def history(self) -> list[ExecutionResult]:
        """Get execution history."""
        return self._history
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def validate(self, code: str) -> list[str]:
        """
        Validate code for security issues.
        
        Args:
            code: Python code to validate
            
        Returns:
            List of security violation messages (empty if valid)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"Syntax error: {e}"]
        
        validator = CodeValidator(
            blocked_imports=self._blocked_imports,
            blocked_builtins=self._blocked_builtins,
            allow_imports=self._allow_imports,
        )
        validator.visit(tree)
        
        return validator.errors
    
    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code in sandbox.
        
        Args:
            code: Python code to execute
            
        Returns:
            ExecutionResult with output and status
        """
        import time
        start_time = time.perf_counter()
        
        # Validate first
        errors = self.validate(code)
        if errors:
            result = ExecutionResult(
                success=False,
                output="",
                error=f"Security violation: {'; '.join(errors)}",
            )
            self._history.append(result)
            return result
        
        # Create restricted globals
        safe_builtins = {
            k: v for k, v in __builtins__.items()
            if k not in self._blocked_builtins
        } if isinstance(__builtins__, dict) else {
            k: getattr(__builtins__, k) 
            for k in dir(__builtins__) 
            if not k.startswith("_") and k not in self._blocked_builtins
        }
        
        # Add safe functions
        safe_builtins.update({
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "pow": pow,
            "divmod": divmod,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "frozenset": frozenset,
            "type": type,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "callable": callable,
            "repr": repr,
            "ascii": ascii,
            "chr": chr,
            "ord": ord,
            "hex": hex,
            "oct": oct,
            "bin": bin,
            "format": format,
            "slice": slice,
            "all": all,
            "any": any,
            "iter": iter,
            "next": next,
            "id": id,
            "hash": hash,
            "object": object,
            "staticmethod": staticmethod,
            "classmethod": classmethod,
            "property": property,
            "super": super,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "RuntimeError": RuntimeError,
            "StopIteration": StopIteration,
            "ZeroDivisionError": ZeroDivisionError,
            "True": True,
            "False": False,
            "None": None,
        })
        
        restricted_globals = {"__builtins__": safe_builtins}
        
        # Add allowed imports
        if self._allow_imports:
            allowed_modules: dict[str, Any] = {}
            for module_name in self.SAFE_IMPORTS - self._blocked_imports:
                try:
                    allowed_modules[module_name] = __import__(module_name)
                    restricted_globals[module_name] = allowed_modules[module_name]
                except ImportError:
                    pass
            
            # Provide a safe __import__ that only allows pre-approved modules
            def safe_import(
                name: str,
                globals_dict: dict | None = None,
                locals_dict: dict | None = None,
                fromlist: tuple = (),
                level: int = 0,
            ) -> Any:
                if name not in allowed_modules:
                    raise ImportError(f"Import of '{name}' is not allowed")
                return allowed_modules[name]
            
            # Add to both builtins and globals so import statement can find it
            safe_builtins["__import__"] = safe_import
            restricted_globals["__import__"] = safe_import
        
        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result_value = None
        error_msg = None
        
        try:
            # Set timeout using signal (Unix only, main thread only)
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Execution timed out after {self._timeout} seconds")
            
            # Only use signal on Unix systems AND in main thread
            import threading
            use_signal = (
                hasattr(signal, "SIGALRM") and 
                threading.current_thread() is threading.main_thread()
            )
            if use_signal:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self._timeout)
            
            try:
                with contextlib.redirect_stdout(stdout_capture), \
                     contextlib.redirect_stderr(stderr_capture):
                    # Execute the code
                    exec(compile(code, "<sandbox>", "exec"), restricted_globals)
                    
                    # Try to get a return value if code defines a result
                    if "_result" in restricted_globals:
                        result_value = restricted_globals["_result"]
                    elif "result" in restricted_globals:
                        result_value = restricted_globals["result"]
            finally:
                if use_signal:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                    
        except TimeoutError as e:
            error_msg = str(e)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        
        # Collect output
        output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        if stderr_output:
            output += f"\n[stderr]\n{stderr_output}"
        
        # Truncate if needed
        if len(output) > self._max_output_length:
            output = output[:self._max_output_length] + "\n[Output truncated...]"
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        result = ExecutionResult(
            success=error_msg is None,
            output=output,
            error=error_msg,
            return_value=result_value,
            execution_time_ms=execution_time,
        )
        
        self._history.append(result)
        return result
    
    def execute_function(
        self,
        code: str,
        function_name: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """
        Execute a function defined in code.
        
        Args:
            code: Python code defining the function
            function_name: Name of the function to call
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            ExecutionResult with function return value
        """
        args = args or []
        kwargs = kwargs or {}
        
        # Build code that calls the function and stores result
        call_args = ", ".join([repr(a) for a in args])
        call_kwargs = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
        all_args = ", ".join(filter(None, [call_args, call_kwargs]))
        
        full_code = f"{code}\n\n_result = {function_name}({all_args})"
        
        return self.execute(full_code)
    
    def clear_history(self) -> int:
        """Clear execution history. Returns count cleared."""
        count = len(self._history)
        self._history.clear()
        return count
    
    # =========================================================================
    # Tool Generation
    # =========================================================================
    
    def _execute_tool(self) -> BaseTool:
        sandbox = self
        
        @tool
        def execute_python(code: str) -> str:
            """
            Execute Python code in a secure sandbox.
            
            The sandbox has restrictions:
            - No file system access (no open, os, pathlib)
            - No network access (no requests, urllib, socket)
            - No dangerous operations (no eval, exec, subprocess)
            - Timeout protection (5 seconds max)
            
            To return a value, assign it to a variable named 'result':
                result = some_calculation()
            
            Args:
                code: Python code to execute
            
            Returns:
                Execution output and any errors
            """
            exec_result = sandbox.execute(code)
            
            lines = []
            if exec_result.success:
                lines.append("✓ Execution successful")
                if exec_result.output:
                    lines.append(f"\nOutput:\n{exec_result.output}")
                if exec_result.return_value is not None:
                    lines.append(f"\nResult: {repr(exec_result.return_value)}")
            else:
                lines.append("✗ Execution failed")
                lines.append(f"\nError: {exec_result.error}")
            
            lines.append(f"\n[Executed in {exec_result.execution_time_ms:.2f}ms]")
            
            return "\n".join(lines)
        
        return execute_python
    
    def _execute_function_tool(self) -> BaseTool:
        sandbox = self
        
        @tool
        def run_function(code: str, function_name: str, arguments: str = "[]") -> str:
            """
            Define and run a Python function with arguments.
            
            Args:
                code: Python code defining the function
                function_name: Name of the function to call
                arguments: JSON array of arguments, e.g., "[1, 2, 3]" or '["hello"]'
            
            Returns:
                Function return value and any output
            
            Example:
                code: "def add(a, b): return a + b"
                function_name: "add"
                arguments: "[3, 5]"
                -> Returns: 8
            """
            import json
            
            try:
                args = json.loads(arguments) if arguments else []
            except json.JSONDecodeError:
                return f"Error: Invalid JSON arguments: {arguments}"
            
            if not isinstance(args, list):
                args = [args]
            
            exec_result = sandbox.execute_function(code, function_name, args)
            
            lines = []
            if exec_result.success:
                lines.append("✓ Function executed successfully")
                if exec_result.output:
                    lines.append(f"\nOutput:\n{exec_result.output}")
                if exec_result.return_value is not None:
                    lines.append(f"\nReturn value: {repr(exec_result.return_value)}")
                else:
                    lines.append("\nReturn value: None")
            else:
                lines.append("✗ Function execution failed")
                lines.append(f"\nError: {exec_result.error}")
            
            return "\n".join(lines)
        
        return run_function
