"""
Example 16: CodeSandbox Capability

This example demonstrates the CodeSandbox capability that provides
safe Python code execution with resource limits and security restrictions.
Perfect for code analysis, testing, and exploration agents.

Features:
- Safe code execution with restricted builtins
- Configurable timeout and output limits
- Blocked dangerous operations (file/network/subprocess)
- Optional safe imports (math, json, datetime, etc.)
- Execution history tracking
- Function execution with arguments
"""

import asyncio

from agenticflow import Agent
from agenticflow.capabilities import CodeSandbox
from agenticflow.executors import ExecutionStrategy


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ============================================================
# 1. BASIC CODE EXECUTION
# ============================================================

def basic_execution_demo():
    """Demonstrate basic code execution."""
    separator("Basic Code Execution")
    
    sandbox = CodeSandbox()
    
    # Simple arithmetic
    result = sandbox.execute("result = 2 + 3 * 4")
    print(f"2 + 3 * 4 = {result.return_value}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")
    
    # With print statements
    result = sandbox.execute("""
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)
average = total / len(numbers)
print(f"Sum: {total}, Average: {average}")
result = {"sum": total, "average": average}
""")
    print(f"Output: {result.output.strip()}")
    print(f"Return value: {result.return_value}")
    
    # List comprehension
    result = sandbox.execute("""
squares = [x**2 for x in range(10)]
result = squares
""")
    print(f"Squares 0-9: {result.return_value}")


# ============================================================
# 2. FUNCTION EXECUTION
# ============================================================

def function_execution_demo():
    """Demonstrate function execution with arguments."""
    separator("Function Execution")
    
    sandbox = CodeSandbox()
    
    # Define and call a function
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""
    
    # Execute function with different arguments
    for n in [5, 10, 20]:
        result = sandbox.execute_function(code, "fibonacci", args=[n])
        print(f"fibonacci({n}) = {result.return_value}")
    
    # Function with kwargs
    code_with_kwargs = """
def greet(name, greeting="Hello", punctuation="!"):
    return f"{greeting}, {name}{punctuation}"
"""
    
    result = sandbox.execute_function(
        code_with_kwargs, 
        "greet",
        args=["World"],
        kwargs={"greeting": "Hi", "punctuation": "?"}
    )
    print(f"Greeting: {result.return_value}")


# ============================================================
# 3. ALLOWED IMPORTS
# ============================================================

def imports_demo():
    """Demonstrate safe imports."""
    separator("Safe Imports")
    
    # Create sandbox with imports enabled
    sandbox = CodeSandbox(allow_imports=True)
    
    # Math operations
    result = sandbox.execute("""
import math
result = {
    "pi": math.pi,
    "sqrt_2": math.sqrt(2),
    "sin_45": math.sin(math.radians(45)),
    "factorial_10": math.factorial(10)
}
""")
    print("Math results:")
    for key, value in result.return_value.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # JSON processing
    result = sandbox.execute("""
import json
data = {"name": "AgenticFlow", "version": "0.1.0", "features": ["agents", "tools", "graphs"]}
result = json.dumps(data, indent=2)
""")
    print(f"\nJSON output:\n{result.return_value}")
    
    # Datetime
    result = sandbox.execute("""
import datetime
now = datetime.datetime.now()
result = now.strftime("%Y-%m-%d %H:%M:%S")
""")
    print(f"\nCurrent datetime: {result.return_value}")
    
    # Statistics
    result = sandbox.execute("""
import statistics
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
result = {
    "mean": statistics.mean(data),
    "median": statistics.median(data),
    "stdev": statistics.stdev(data)
}
""")
    print(f"\nStatistics: {result.return_value}")


# ============================================================
# 4. SECURITY RESTRICTIONS
# ============================================================

def security_demo():
    """Demonstrate security restrictions."""
    separator("Security Restrictions")
    
    sandbox = CodeSandbox()
    
    # Try blocked operations
    blocked_operations = [
        ("File access", "open('test.txt', 'w')"),
        ("System import", "import os"),
        ("Subprocess", "import subprocess"),
        ("Eval", "eval('print(1)')"),
        ("Exec", "exec('print(1)')"),
        ("Compile", "compile('print(1)', '', 'exec')"),
    ]
    
    for name, code in blocked_operations:
        result = sandbox.execute(code)
        status = "✓ BLOCKED" if not result.success else "✗ ALLOWED (unexpected)"
        print(f"{name}: {status}")
    
    # Even with imports enabled, dangerous modules are blocked
    sandbox_with_imports = CodeSandbox(allow_imports=True)
    
    print("\nWith imports enabled:")
    for name, code in blocked_operations[:3]:  # File, os, subprocess
        result = sandbox_with_imports.execute(code)
        status = "✓ BLOCKED" if not result.success else "✗ ALLOWED (unexpected)"
        print(f"  {name}: {status}")


# ============================================================
# 5. RESOURCE LIMITS
# ============================================================

def resource_limits_demo():
    """Demonstrate resource limits."""
    separator("Resource Limits")
    
    # Timeout - need allow_imports to access time module
    sandbox = CodeSandbox(timeout=1, allow_imports=True)
    print("Testing timeout (1 second limit)...")
    
    result = sandbox.execute("""
import time
time.sleep(2)  # This will timeout
result = "completed"
""")
    print(f"Timeout test: {'✓ Timed out' if not result.success else '✗ Completed'}")
    if result.error:
        print(f"Error: {result.error.split(chr(10))[0]}")
    
    # Output truncation
    sandbox = CodeSandbox(max_output_length=100)
    result = sandbox.execute("print('x' * 500)")
    print(f"\nOutput truncation test:")
    print(f"  Output length: {len(result.output)}")
    print(f"  Truncated: {'✓ Yes' if 'truncated' in result.output.lower() else '✗ No'}")


# ============================================================
# 6. EXECUTION HISTORY
# ============================================================

def history_demo():
    """Demonstrate execution history."""
    separator("Execution History")
    
    sandbox = CodeSandbox()
    
    # Execute several pieces of code
    codes = [
        "result = 1 + 1",
        "result = 2 * 3",
        "result = 10 / 2",
        "result = 2 ** 10",
    ]
    
    for code in codes:
        sandbox.execute(code)
    
    print(f"Execution history ({len(sandbox.history)} items):")
    for i, entry in enumerate(sandbox.history, 1):
        print(f"  {i}. success={entry.success}, result={entry.return_value}, time={entry.execution_time_ms:.2f}ms")
    
    # Clear history
    sandbox.clear_history()
    print(f"\nAfter clear: {len(sandbox.history)} items")


# ============================================================
# 7. USING TOOLS
# ============================================================

def tools_demo():
    """Demonstrate using sandbox as tools."""
    separator("CodeSandbox Tools")
    
    sandbox = CodeSandbox(allow_imports=True)
    tools = sandbox.tools
    
    print(f"Available tools: {[t.name for t in tools]}")
    print()
    
    # Find execute_python tool
    execute_tool = next(t for t in tools if t.name == "execute_python")
    
    # Use the tool directly
    tool_result = execute_tool.invoke({"code": """
import math
primes = []
for n in range(2, 50):
    if all(n % i != 0 for i in range(2, int(math.sqrt(n)) + 1)):
        primes.append(n)
print(f"Found {len(primes)} primes: {primes}")
result = primes
"""})
    
    print(f"Tool result:\n{tool_result}")


# ============================================================
# 8. AGENT WITH CODESANDBOX
# ============================================================

async def agent_demo():
    """Demonstrate agent using CodeSandbox for computation."""
    separator("Agent with CodeSandbox")
    
    sandbox = CodeSandbox(allow_imports=True, timeout=5)
    
    agent = Agent(
        name="Compute Agent",
        instructions="""You are a computation agent that helps users with calculations.
Use the execute_python tool to run Python code for calculations.
Always show your code and explain the results clearly.

Safe imports available: math, json, datetime, statistics, random, decimal, fractions, itertools, functools, collections.""",
        capabilities=[sandbox],
        strategy=ExecutionStrategy.REACT,
        max_iterations=5,
    )
    
    # Test computation
    print("Asking agent to calculate prime factors...")
    
    response = await agent.run(
        "Calculate and list all prime factors of 360. Show the code you use."
    )
    
    print(f"\nAgent response:\n{response.output}")


# ============================================================
# RUN ALL DEMOS
# ============================================================

if __name__ == "__main__":
    # Synchronous demos
    basic_execution_demo()
    function_execution_demo()
    imports_demo()
    security_demo()
    resource_limits_demo()
    history_demo()
    tools_demo()
    
    # Async agent demo (optional - requires OPENAI_API_KEY)
    import os
    if os.environ.get("OPENAI_API_KEY"):
        asyncio.run(agent_demo())
    else:
        print("\n" + "="*60)
        print("  Skipping agent demo (set OPENAI_API_KEY to enable)")
        print("="*60)
