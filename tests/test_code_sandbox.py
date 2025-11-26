"""Tests for CodeSandbox capability."""

import pytest

from agenticflow.capabilities.code_sandbox import (
    CodeSandbox,
    CodeValidator,
    ExecutionResult,
    BLOCKED_IMPORTS,
    BLOCKED_BUILTINS,
)


class TestExecutionResult:
    """Test ExecutionResult dataclass."""
    
    def test_create_success(self):
        """Test creating a successful result."""
        result = ExecutionResult(
            success=True,
            output="Hello",
            return_value=42,
        )
        
        assert result.success is True
        assert result.output == "Hello"
        assert result.return_value == 42
        assert result.error is None
    
    def test_create_failure(self):
        """Test creating a failed result."""
        result = ExecutionResult(
            success=False,
            output="",
            error="Some error",
        )
        
        assert result.success is False
        assert result.error == "Some error"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        result = ExecutionResult(
            success=True,
            output="test",
            return_value=[1, 2, 3],
            execution_time_ms=10.5,
        )
        
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "test"
        assert d["return_value"] == "[1, 2, 3]"
        assert d["execution_time_ms"] == 10.5


class TestCodeValidator:
    """Test code validation."""
    
    def test_valid_simple_code(self):
        """Test valid simple code passes."""
        validator = CodeValidator(BLOCKED_IMPORTS, BLOCKED_BUILTINS, allow_imports=False)
        
        import ast
        tree = ast.parse("x = 1 + 2\nprint(x)")
        validator.visit(tree)
        
        assert validator.errors == []
    
    def test_blocks_import(self):
        """Test import is blocked when not allowed."""
        validator = CodeValidator(BLOCKED_IMPORTS, BLOCKED_BUILTINS, allow_imports=False)
        
        import ast
        tree = ast.parse("import math")
        validator.visit(tree)
        
        assert len(validator.errors) == 1
        assert "Import not allowed" in validator.errors[0]
    
    def test_blocks_dangerous_import(self):
        """Test dangerous imports are blocked even when imports allowed."""
        validator = CodeValidator(BLOCKED_IMPORTS, BLOCKED_BUILTINS, allow_imports=True)
        
        import ast
        tree = ast.parse("import os")
        validator.visit(tree)
        
        assert len(validator.errors) == 1
        assert "Blocked import" in validator.errors[0]
    
    def test_blocks_from_import(self):
        """Test from import is blocked."""
        validator = CodeValidator(BLOCKED_IMPORTS, BLOCKED_BUILTINS, allow_imports=True)
        
        import ast
        tree = ast.parse("from subprocess import run")
        validator.visit(tree)
        
        assert len(validator.errors) == 1
        assert "Blocked import" in validator.errors[0]
    
    def test_blocks_eval(self):
        """Test eval is blocked."""
        validator = CodeValidator(BLOCKED_IMPORTS, BLOCKED_BUILTINS, allow_imports=False)
        
        import ast
        tree = ast.parse("eval('1+1')")
        validator.visit(tree)
        
        assert len(validator.errors) == 1
        assert "Blocked function: eval" in validator.errors[0]
    
    def test_blocks_exec(self):
        """Test exec is blocked."""
        validator = CodeValidator(BLOCKED_IMPORTS, BLOCKED_BUILTINS, allow_imports=False)
        
        import ast
        tree = ast.parse("exec('x=1')")
        validator.visit(tree)
        
        assert len(validator.errors) == 1
        assert "Blocked function: exec" in validator.errors[0]
    
    def test_blocks_open(self):
        """Test open is blocked."""
        validator = CodeValidator(BLOCKED_IMPORTS, BLOCKED_BUILTINS, allow_imports=False)
        
        import ast
        tree = ast.parse("open('file.txt')")
        validator.visit(tree)
        
        assert len(validator.errors) == 1
        assert "Blocked function: open" in validator.errors[0]
    
    def test_blocks_dangerous_attributes(self):
        """Test dangerous attribute access is blocked."""
        validator = CodeValidator(BLOCKED_IMPORTS, BLOCKED_BUILTINS, allow_imports=False)
        
        import ast
        tree = ast.parse("x.__class__.__bases__")
        validator.visit(tree)
        
        assert len(validator.errors) >= 1
        assert any("__class__" in e or "__bases__" in e for e in validator.errors)


class TestCodeSandboxInit:
    """Test CodeSandbox initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        sandbox = CodeSandbox()
        
        assert sandbox.name == "code_sandbox"
        assert sandbox._timeout == 5
        assert sandbox._allow_imports is False
    
    def test_custom_settings(self):
        """Test custom settings."""
        sandbox = CodeSandbox(
            timeout=10,
            max_output_length=5000,
            allow_imports=True,
            name="custom_sandbox",
        )
        
        assert sandbox.name == "custom_sandbox"
        assert sandbox._timeout == 10
        assert sandbox._max_output_length == 5000
        assert sandbox._allow_imports is True
    
    def test_allowed_imports(self):
        """Test custom allowed imports."""
        sandbox = CodeSandbox(
            allow_imports=True,
            allowed_imports={"math", "json"},
        )
        
        # math and json should not be blocked
        assert "math" not in sandbox._blocked_imports
        assert "json" not in sandbox._blocked_imports
        # os should still be blocked
        assert "os" in sandbox._blocked_imports


class TestCodeSandboxValidation:
    """Test code validation."""
    
    def test_validate_valid_code(self):
        """Test validation of valid code."""
        sandbox = CodeSandbox()
        errors = sandbox.validate("x = 1 + 2")
        
        assert errors == []
    
    def test_validate_syntax_error(self):
        """Test validation catches syntax errors."""
        sandbox = CodeSandbox()
        errors = sandbox.validate("def foo(")
        
        assert len(errors) == 1
        assert "Syntax error" in errors[0]
    
    def test_validate_blocked_import(self):
        """Test validation catches blocked imports."""
        sandbox = CodeSandbox()
        errors = sandbox.validate("import os")
        
        assert len(errors) == 1
        assert "Import not allowed" in errors[0]
    
    def test_validate_blocked_function(self):
        """Test validation catches blocked functions."""
        sandbox = CodeSandbox()
        errors = sandbox.validate("eval('1+1')")
        
        assert len(errors) == 1
        assert "Blocked function" in errors[0]


class TestCodeSandboxExecution:
    """Test code execution."""
    
    def test_execute_simple(self):
        """Test simple code execution."""
        sandbox = CodeSandbox()
        result = sandbox.execute("print('Hello')")
        
        assert result.success is True
        assert "Hello" in result.output
    
    def test_execute_math(self):
        """Test math operations."""
        sandbox = CodeSandbox()
        result = sandbox.execute("result = 2 + 3 * 4")
        
        assert result.success is True
        assert result.return_value == 14
    
    def test_execute_with_result(self):
        """Test capturing result variable."""
        sandbox = CodeSandbox()
        result = sandbox.execute("result = [i**2 for i in range(5)]")
        
        assert result.success is True
        assert result.return_value == [0, 1, 4, 9, 16]
    
    def test_execute_with_print(self):
        """Test output capture."""
        sandbox = CodeSandbox()
        result = sandbox.execute("for i in range(3): print(i)")
        
        assert result.success is True
        assert "0" in result.output
        assert "1" in result.output
        assert "2" in result.output
    
    def test_execute_function_definition(self):
        """Test defining and calling functions."""
        sandbox = CodeSandbox()
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

result = factorial(5)
"""
        result = sandbox.execute(code)
        
        assert result.success is True
        assert result.return_value == 120
    
    def test_execute_class_definition(self):
        """Test defining and using classes."""
        sandbox = CodeSandbox()
        code = """
class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1
        return self.count

c = Counter()
c.increment()
c.increment()
result = c.count
"""
        result = sandbox.execute(code)
        
        assert result.success is True
        assert result.return_value == 2
    
    def test_execute_blocked_import(self):
        """Test blocked import fails."""
        sandbox = CodeSandbox()
        result = sandbox.execute("import os")
        
        assert result.success is False
        assert "Security violation" in result.error
    
    def test_execute_blocked_open(self):
        """Test blocked open fails."""
        sandbox = CodeSandbox()
        result = sandbox.execute("f = open('test.txt')")
        
        assert result.success is False
        assert "Security violation" in result.error
    
    def test_execute_blocked_eval(self):
        """Test blocked eval fails."""
        sandbox = CodeSandbox()
        result = sandbox.execute("eval('1+1')")
        
        assert result.success is False
        assert "Security violation" in result.error
    
    def test_execute_exception(self):
        """Test exception handling."""
        sandbox = CodeSandbox()
        result = sandbox.execute("x = 1/0")
        
        assert result.success is False
        assert "ZeroDivisionError" in result.error
    
    def test_execute_with_allowed_imports(self):
        """Test execution with allowed imports."""
        sandbox = CodeSandbox(allow_imports=True)
        result = sandbox.execute("import math\nresult = math.sqrt(16)")
        
        assert result.success is True
        assert result.return_value == 4.0
    
    def test_execute_truncates_output(self):
        """Test output truncation."""
        sandbox = CodeSandbox(max_output_length=100)
        result = sandbox.execute("print('x' * 200)")
        
        assert result.success is True
        assert len(result.output) <= 150  # Some buffer for truncation message
        assert "truncated" in result.output.lower()


class TestCodeSandboxExecuteFunction:
    """Test function execution."""
    
    def test_execute_function(self):
        """Test execute_function."""
        sandbox = CodeSandbox()
        result = sandbox.execute_function(
            "def add(a, b): return a + b",
            "add",
            args=[3, 5],
        )
        
        assert result.success is True
        assert result.return_value == 8
    
    def test_execute_function_no_args(self):
        """Test function with no arguments."""
        sandbox = CodeSandbox()
        result = sandbox.execute_function(
            "def greet(): return 'Hello!'",
            "greet",
        )
        
        assert result.success is True
        assert result.return_value == "Hello!"
    
    def test_execute_function_with_kwargs(self):
        """Test function with keyword arguments."""
        sandbox = CodeSandbox()
        result = sandbox.execute_function(
            "def greet(name, greeting='Hello'): return f'{greeting}, {name}!'",
            "greet",
            args=["World"],
            kwargs={"greeting": "Hi"},
        )
        
        assert result.success is True
        assert result.return_value == "Hi, World!"


class TestCodeSandboxTools:
    """Test tool generation."""
    
    def test_get_tools(self):
        """Test getting tools."""
        sandbox = CodeSandbox()
        tools = sandbox.tools
        
        names = [t.name for t in tools]
        assert "execute_python" in names
        assert "run_function" in names
    
    def test_tools_cached(self):
        """Test tools are cached."""
        sandbox = CodeSandbox()
        tools1 = sandbox.tools
        tools2 = sandbox.tools
        
        assert tools1 is tools2
    
    def test_execute_python_tool(self):
        """Test execute_python tool."""
        sandbox = CodeSandbox()
        tools = {t.name: t for t in sandbox.tools}
        
        result = tools["execute_python"].invoke({"code": "print('test')"})
        
        assert "successful" in result.lower()
        assert "test" in result
    
    def test_execute_python_tool_error(self):
        """Test execute_python tool with error."""
        sandbox = CodeSandbox()
        tools = {t.name: t for t in sandbox.tools}
        
        result = tools["execute_python"].invoke({"code": "import os"})
        
        assert "failed" in result.lower()
        assert "Security" in result
    
    def test_run_function_tool(self):
        """Test run_function tool."""
        sandbox = CodeSandbox()
        tools = {t.name: t for t in sandbox.tools}
        
        result = tools["run_function"].invoke({
            "code": "def double(x): return x * 2",
            "function_name": "double",
            "arguments": "[5]",
        })
        
        assert "successful" in result.lower()
        assert "10" in result
    
    def test_run_function_tool_invalid_json(self):
        """Test run_function tool with invalid JSON."""
        sandbox = CodeSandbox()
        tools = {t.name: t for t in sandbox.tools}
        
        result = tools["run_function"].invoke({
            "code": "def f(): pass",
            "function_name": "f",
            "arguments": "not json",
        })
        
        assert "Invalid JSON" in result


class TestCodeSandboxHistory:
    """Test execution history."""
    
    def test_history_tracking(self):
        """Test execution history is tracked."""
        sandbox = CodeSandbox()
        
        sandbox.execute("x = 1")
        sandbox.execute("y = 2")
        
        assert len(sandbox.history) == 2
    
    def test_clear_history(self):
        """Test clearing history."""
        sandbox = CodeSandbox()
        
        sandbox.execute("x = 1")
        sandbox.execute("y = 2")
        
        count = sandbox.clear_history()
        
        assert count == 2
        assert len(sandbox.history) == 0


class TestCodeSandboxSecurity:
    """Security-focused tests."""
    
    def test_no_file_access(self):
        """Test file access is blocked."""
        sandbox = CodeSandbox()
        
        # Various file access attempts
        tests = [
            "open('test.txt')",
            "open('/etc/passwd')",
        ]
        
        for code in tests:
            result = sandbox.execute(code)
            assert result.success is False, f"Should block: {code}"
    
    def test_no_network_access(self):
        """Test network access is blocked."""
        sandbox = CodeSandbox()
        
        tests = [
            "import socket",
            "import urllib",
            "import requests",
            "import httpx",
        ]
        
        for code in tests:
            result = sandbox.execute(code)
            assert result.success is False, f"Should block: {code}"
    
    def test_no_subprocess(self):
        """Test subprocess is blocked."""
        sandbox = CodeSandbox()
        
        tests = [
            "import subprocess",
            "import os",
        ]
        
        for code in tests:
            result = sandbox.execute(code)
            assert result.success is False, f"Should block: {code}"
    
    def test_no_code_injection(self):
        """Test code injection is blocked."""
        sandbox = CodeSandbox()
        
        tests = [
            "eval('__import__(\"os\")')",
            "exec('import os')",
            "compile('import os', '', 'exec')",
        ]
        
        for code in tests:
            result = sandbox.execute(code)
            assert result.success is False, f"Should block: {code}"
    
    def test_no_attribute_escape(self):
        """Test attribute-based escapes are blocked."""
        sandbox = CodeSandbox()
        
        tests = [
            "().__class__.__bases__[0].__subclasses__()",
            "''.__class__.__mro__",
        ]
        
        for code in tests:
            result = sandbox.execute(code)
            assert result.success is False, f"Should block: {code}"
    
    def test_builtin_functions_available(self):
        """Test safe built-in functions work."""
        sandbox = CodeSandbox()
        
        code = """
result = {
    'len': len([1,2,3]),
    'sum': sum([1,2,3]),
    'max': max(1,2,3),
    'min': min(1,2,3),
    'sorted': sorted([3,1,2]),
    'range': list(range(3)),
    'zip': list(zip([1,2], [3,4])),
    'map': list(map(lambda x: x*2, [1,2])),
    'filter': list(filter(lambda x: x>1, [1,2,3])),
}
"""
        result = sandbox.execute(code)
        
        assert result.success is True
        assert result.return_value["len"] == 3
        assert result.return_value["sum"] == 6
        assert result.return_value["sorted"] == [1, 2, 3]
