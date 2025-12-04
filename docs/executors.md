# Executors Module

The `agenticflow.executors` module defines execution strategies - HOW agents process tasks and execute tool calls.

## Overview

Executors determine the execution pattern for agent tasks:
- **NativeExecutor**: High-performance parallel execution (default)
- **SequentialExecutor**: Sequential tool execution for ordered tasks
- **TreeSearchExecutor**: LATS-style Monte Carlo tree search (best accuracy)

```python
from agenticflow.executors import run, NativeExecutor, TreeSearchExecutor

# Standalone execution (no Agent needed)
result = await run("Search for Python tutorials", tools=[search])

# With Agent
from agenticflow import Agent

agent = Agent(name="Worker", model=model, tools=[search])
executor = NativeExecutor(agent)
result = await executor.execute("Research and calculate metrics")
```

## Standalone Execution

The `run()` function provides the fastest path for simple tasks:

```python
from agenticflow.executors import run
from agenticflow.tools import tool

@tool
def search(query: str) -> str:
    '''Search the web.'''
    return f"Results for {query}"

@tool
def calculate(expression: str) -> float:
    '''Calculate a math expression.'''
    return eval(expression)

# Simple execution
result = await run(
    "Search for Python tutorials and calculate 2+2",
    tools=[search, calculate],
)

# With options
result = await run(
    "Complex research task",
    tools=[search, calculate],
    model="gpt-4o",
    system_prompt="You are a research assistant.",
    max_iterations=20,
    max_tool_calls=50,
    resilience=True,  # Auto-retry on rate limits
    verbose=True,     # Print retry info
)
```

### Using Different Providers

```python
from agenticflow.models.anthropic import AnthropicChat
from agenticflow.models.gemini import GeminiChat

# With Anthropic
result = await run(
    "Explain quantum computing",
    model=AnthropicChat(model="claude-sonnet-4-20250514"),
)

# With Gemini
result = await run(
    "Write a poem",
    model=GeminiChat(model="gemini-2.0-flash"),
)
```

---

## NativeExecutor

The default high-performance executor with parallel tool execution:

```python
from agenticflow.executors import NativeExecutor
from agenticflow import Agent

agent = Agent(name="Worker", model=model, tools=[search, calculate])
executor = NativeExecutor(agent)

result = await executor.execute("Research and calculate metrics")
```

### Key Features

1. **Parallel Tool Execution**: Multiple tool calls execute concurrently
2. **Cached Model Binding**: Zero-overhead tool binding
3. **LLM Resilience**: Automatic retry for rate limits
4. **Minimal Overhead**: Direct asyncio loop (no graph framework)

### Configuration

```python
executor = NativeExecutor(
    agent,
    max_iterations=10,        # Max LLM call iterations
    max_tool_calls=20,        # Max total tool calls
    parallel_tool_calls=True, # Execute tools in parallel
)
```

---

## SequentialExecutor

For tasks requiring ordered tool execution:

```python
from agenticflow.executors import SequentialExecutor

executor = SequentialExecutor(agent)
result = await executor.execute("Step 1, then Step 2, then Step 3")
```

### Use Cases

- Tasks with dependencies between steps
- Debugging (easier to trace)
- When tool order matters

---

## TreeSearchExecutor

LATS (Language Agent Tree Search) implementation using Monte Carlo Tree Search for maximum accuracy:

```python
from agenticflow.executors import TreeSearchExecutor

executor = TreeSearchExecutor(
    agent,
    max_iterations=50,      # MCTS iterations
    max_depth=10,           # Maximum tree depth
    exploration_weight=1.4, # UCB1 exploration constant
    n_expansions=3,         # Children per expansion
)

result = await executor.execute("Solve this complex problem")

# Access search results
print(f"Nodes explored: {result.nodes_explored}")
print(f"Best path: {result.best_path}")
print(f"Reflections: {result.reflections}")
```

### How It Works

Based on "Language Agent Tree Search Unifies Reasoning Acting and Planning" (Zhou et al., 2024):

1. **Selection**: Use UCB1 to select promising nodes
2. **Expansion**: Generate multiple possible actions
3. **Simulation**: Execute actions and evaluate
4. **Backpropagation**: Update path values
5. **Reflection**: Learn from failed paths

### TreeSearchResult

```python
@dataclass
class TreeSearchResult:
    answer: str              # Final answer
    success: bool            # Whether successful path found
    best_path: list[Node]    # Best path through tree
    nodes_explored: int      # Total nodes explored
    iterations: int          # MCTS iterations performed
    reflections: list[str]   # Reflections from failed paths
```

### SearchNode

Each node in the search tree:

```python
@dataclass
class SearchNode:
    id: str
    parent: SearchNode | None
    action: dict[str, Any]    # Tool call or thought
    observation: str          # Action result
    state: NodeState          # PENDING, EXPANDED, TERMINAL, etc.
    value: float              # Estimated value (0.0-1.0)
    visits: int               # Visit count for UCB1
    children: list[SearchNode]
    depth: int
    reflection: str | None    # Self-reflection if failed
    
    def ucb1_score(self, exploration_weight=1.414) -> float:
        """UCB1 score balancing exploitation and exploration."""
        ...
    
    def backpropagate(self, value: float) -> None:
        """Update values up the tree."""
        ...
```

### When to Use TreeSearch

| Scenario | Best Executor |
|----------|---------------|
| Simple tool use | `NativeExecutor` |
| Speed critical | `run()` or `NativeExecutor` |
| Complex reasoning | `TreeSearchExecutor` |
| Multi-step planning | `TreeSearchExecutor` |
| Maximum accuracy needed | `TreeSearchExecutor` |

---

## Execution Strategy Enum

```python
from agenticflow.executors import ExecutionStrategy

class ExecutionStrategy(Enum):
    NATIVE = "native"           # NativeExecutor (default)
    SEQUENTIAL = "sequential"   # SequentialExecutor
    TREE_SEARCH = "tree_search" # TreeSearchExecutor
```

---

## Factory Function

Create executors by strategy name:

```python
from agenticflow.executors import create_executor, ExecutionStrategy

# By enum
executor = create_executor(
    agent,
    strategy=ExecutionStrategy.TREE_SEARCH,
)

# By string
executor = create_executor(
    agent,
    strategy="tree_search",
    max_iterations=50,
)
```

---

## Data Models

### ToolCall

```python
from agenticflow.executors import ToolCall

@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    result: Any = None
    error: str | None = None
```

### ExecutionPlan

```python
from agenticflow.executors import ExecutionPlan

@dataclass
class ExecutionPlan:
    steps: list[str]
    tool_calls: list[ToolCall]
    estimated_iterations: int
```

### CompletionCheck

```python
from agenticflow.executors import CompletionCheck

@dataclass
class CompletionCheck:
    is_complete: bool
    final_answer: str | None
    needs_more_work: bool
    reason: str
```

---

## Exports

```python
from agenticflow.executors import (
    # Strategy enum
    ExecutionStrategy,
    # Data classes
    ToolCall,
    ExecutionPlan,
    CompletionCheck,
    # Tree search classes
    SearchNode,
    NodeState,
    TreeSearchResult,
    # Base
    BaseExecutor,
    # Executors
    NativeExecutor,
    SequentialExecutor,
    TreeSearchExecutor,
    # Standalone execution
    run,
    # Factory
    create_executor,
)
```
