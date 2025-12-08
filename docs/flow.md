# Flow Module

The `agenticflow.flow` module provides the main entry point for orchestrating multiple agents using simple coordination patterns.

## Overview

A Flow coordinates multiple agents using topologies (Supervisor, Pipeline, Mesh, Hierarchical):

```python
from agenticflow import Agent, Flow
from agenticflow.models import ChatModel

model = ChatModel(model="gpt-4o")

researcher = Agent(name="researcher", model=model)
writer = Agent(name="writer", model=model)
editor = Agent(name="editor", model=model)

# Pipeline: research → write → edit
flow = Flow(
    name="content-team",
    agents=[researcher, writer, editor],
    topology="pipeline",
)

result = await flow.run("Create a blog post about AI")
print(result.output)
```

---

## Creating Flows

### Basic Flow

```python
from agenticflow import Flow

flow = Flow(
    name="my-flow",
    agents=[agent1, agent2, agent3],
    topology="pipeline",  # or "supervisor", "mesh", "hierarchical"
)
```

### Flow with Observability

```python
flow = Flow(
    name="my-flow",
    agents=[agent1, agent2, agent3],
    topology="pipeline",
    verbose=True,  # Enable progress output
)

# Or with specific verbosity level
flow = Flow(
    name="my-flow",
    agents=agents,
    topology="pipeline",
    verbose="debug",  # "minimal", "verbose", "debug", "trace"
)
```

---

## Topology Types

### Pipeline

Sequential processing where each agent's output feeds the next:

```python
flow = Flow(
    name="content-pipeline",
    agents=[researcher, writer, editor],
    topology="pipeline",
)

# Execution: researcher → writer → editor
result = await flow.run("Create documentation for our API")
```

### Supervisor

One coordinator agent manages and delegates to workers:

```python
flow = Flow(
    name="project-team",
    agents=[manager, analyst, developer, tester],
    topology="supervisor",
    # First agent is the coordinator by default
)

# manager coordinates analyst, developer, tester
result = await flow.run("Build a login feature")
```

### Mesh

All agents collaborate through multiple rounds:

```python
flow = Flow(
    name="review-panel",
    agents=[analyst1, analyst2, analyst3],
    topology="mesh",
    max_rounds=3,  # Maximum collaboration rounds
)

result = await flow.run("Evaluate this product proposal")
```

### Hierarchical

Tree structure with delegation levels:

```python
flow = Flow(
    name="organization",
    agents=[ceo, cto, cfo, dev1, dev2, acc1],
    topology="hierarchical",
    hierarchy={
        "ceo": ["cto", "cfo"],
        "cto": ["dev1", "dev2"],
        "cfo": ["acc1"],
    },
)

result = await flow.run("Plan Q4 strategy")
```

---

## FlowConfig

Advanced configuration options:

```python
from agenticflow.flow import FlowConfig

config = FlowConfig(
    max_rounds=3,           # For mesh topology
    parallel=True,          # Enable parallel execution
    timeout_seconds=300,    # Overall timeout
    fail_fast=True,         # Stop on first error
)

flow = Flow(
    name="my-flow",
    agents=agents,
    topology="pipeline",
    config=config,
)
```

---

## Running Flows

### Basic Execution

```python
result = await flow.run("Your task description")

print(result.output)       # Final output
print(result.success)      # True if successful
print(result.duration_ms)  # Execution time
```

### With Memory

```python
from agenticflow.memory import Memory

memory = Memory()

result = await flow.run(
    "Create a report",
    memory=memory,
)

# Access shared state after execution
findings = await memory.recall("findings")
```

### With Context

```python
from agenticflow import RunContext
from dataclasses import dataclass

@dataclass
class ProjectContext(RunContext):
    project_id: str
    deadline: str
    priority: str

result = await flow.run(
    "Complete the project",
    context=ProjectContext(
        project_id="proj-123",
        deadline="2024-03-01",
        priority="high",
    ),
)
```

### With Observer

```python
from agenticflow.observability import Observer

observer = Observer.debug()

result = await flow.run(
    "Execute task",
    observer=observer,
)
```

---

## FlowResult

The result returned from flow execution:

```python
result = await flow.run("Task")

# Core properties
result.output          # Final output string
result.success         # True if completed successfully
result.duration_ms     # Total execution time
result.iterations      # Number of iterations/rounds

# Agent-specific results
result.agent_outputs   # Dict[agent_name, output]
for agent, output in result.agent_outputs.items():
    print(f"{agent}: {output}")

# Metadata
result.metadata        # Additional execution metadata
result.trace_id        # Trace ID for debugging
```

---

## Visualization

Generate diagrams of your flow:

```python
# Get graph view
view = flow.graph()

# Render in various formats
print(view.mermaid())   # Mermaid diagram
print(view.ascii())     # ASCII art

# Save to file
view.save("flow.png")
view.save("flow.svg")

# Open in browser
view.open()
```

---

## Event Bus Integration

Access the flow's event bus for custom event handling:

```python
from agenticflow.observability import EventBus, ConsoleEventHandler

# Custom event bus
bus = EventBus()
bus.subscribe_all(ConsoleEventHandler())

flow = Flow(
    name="my-flow",
    agents=agents,
    topology="pipeline",
    event_bus=bus,
)

# Events will be published to your bus
result = await flow.run("Task")

# Query event history
events = bus.get_history(limit=100)
```

---

## Tool Registry

Share tools across all agents in a flow:

```python
from agenticflow.tools import ToolRegistry

registry = ToolRegistry()
registry.register(search_tool)
registry.register(analyze_tool)

flow = Flow(
    name="my-flow",
    agents=agents,
    topology="pipeline",
    tool_registry=registry,  # All agents can access these tools
)
```

---

## Error Handling

```python
try:
    result = await flow.run("Task")
    if not result.success:
        print(f"Flow failed: {result.error}")
except Exception as e:
    print(f"Flow error: {e}")
```

### Fail Fast vs Continue

```python
# Stop on first agent error
flow = Flow(
    agents=agents,
    topology="pipeline",
    config=FlowConfig(fail_fast=True),
)

# Continue despite errors
flow = Flow(
    agents=agents,
    topology="pipeline",
    config=FlowConfig(fail_fast=False),
)
```

---

## Verbosity Levels

The `verbose` parameter controls output:

| Value | Description |
|-------|-------------|
| `False` | No output (silent) |
| `True` | Basic progress (agent start/complete) |
| `"minimal"` | Same as `True` |
| `"verbose"` | Show agent outputs/thoughts |
| `"debug"` | Include tool calls |
| `"trace"` | Maximum detail + execution graph |

```python
# Examples
flow = Flow(agents=agents, topology="pipeline", verbose=False)    # Silent
flow = Flow(agents=agents, topology="pipeline", verbose=True)     # Progress
flow = Flow(agents=agents, topology="pipeline", verbose="debug")  # Full debug
```

---

## API Reference

### Flow Class

```python
Flow(
    name: str,                          # Flow identifier
    agents: list[Agent],                # Agents to coordinate
    topology: str = "supervisor",       # Coordination pattern
    config: FlowConfig | None = None,   # Advanced config
    verbose: VerbosityLevel = False,    # Output verbosity
    event_bus: EventBus | None = None,  # Custom event bus
    tool_registry: ToolRegistry | None = None,  # Shared tools
)
```

### FlowConfig

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_rounds` | `int` | Max rounds for mesh (default: 3) |
| `parallel` | `bool` | Enable parallel execution |
| `timeout_seconds` | `int` | Overall timeout |
| `fail_fast` | `bool` | Stop on first error |

### FlowResult

| Property | Type | Description |
|----------|------|-------------|
| `output` | `str` | Final output |
| `success` | `bool` | Whether flow succeeded |
| `duration_ms` | `float` | Execution time |
| `iterations` | `int` | Number of iterations |
| `agent_outputs` | `dict` | Per-agent outputs |
| `metadata` | `dict` | Additional metadata |

### Topology Values

| Value | Description |
|-------|-------------|
| `"supervisor"` | One coordinator, many workers |
| `"pipeline"` | Sequential A → B → C |
| `"mesh"` | All collaborate in rounds |
| `"hierarchical"` | Tree delegation |
