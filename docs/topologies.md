# Topologies Module

The `agenticflow.topologies` module provides multi-agent coordination patterns for complex workflows.

## Overview

Topologies define how agents collaborate:
- **Supervisor** - One agent coordinates and delegates to workers
- **Pipeline** - Sequential processing A → B → C
- **Mesh** - All agents collaborate in rounds until consensus
- **Hierarchical** - Tree structure with delegation levels

```python
from agenticflow import Agent
from agenticflow.topologies import Supervisor, Pipeline, Mesh

model = ChatModel(model="gpt-4o")
researcher = Agent(name="researcher", model=model)
writer = Agent(name="writer", model=model)
editor = Agent(name="editor", model=model)

# Supervisor pattern
topology = Supervisor(
    coordinator=researcher,
    workers=[writer, editor],
)
result = await topology.run("Create a blog post about AI")
```

---

## Supervisor

One coordinator agent manages and delegates to worker agents:

```
      ┌─────────────┐
      │ Coordinator │
      └──────┬──────┘
             │ delegates
    ┌────────┼────────┐
    ▼        ▼        ▼
┌────────┐ ┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│ │Worker 3│
└────────┘ └────────┘ └────────┘
```

### Basic Usage

```python
from agenticflow import Agent
from agenticflow.topologies import Supervisor

# Create agents
coordinator = Agent(
    name="manager",
    model=model,
    instructions="You coordinate work. Delegate to specialists.",
)
analyst = Agent(name="analyst", model=model, tools=[analyze_data])
writer = Agent(name="writer", model=model, tools=[write_content])

# Create supervisor topology
topology = Supervisor(
    coordinator=coordinator,
    workers=[analyst, writer],
)

result = await topology.run("Analyze sales data and write a summary")
print(result.output)
```

### With Agent Roles

```python
from agenticflow.topologies import Supervisor, AgentConfig

topology = Supervisor(
    coordinator=AgentConfig(
        agent=coordinator,
        role="project manager",
        can_finish=True,
    ),
    workers=[
        AgentConfig(agent=analyst, role="data analyst"),
        AgentConfig(agent=writer, role="content writer"),
        AgentConfig(agent=reviewer, role="quality reviewer"),
    ],
)
```

### Convenience Function

```python
from agenticflow.topologies import supervisor

# Quick setup without AgentConfig
topology = supervisor(
    coordinator=coordinator,
    workers=[analyst, writer, reviewer],
)
```

---

## Pipeline

Sequential processing where each agent's output feeds the next:

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Stage 1  │ ─► │ Stage 2  │ ─► │ Stage 3  │
│ Research │    │  Write   │    │  Edit    │
└──────────┘    └──────────┘    └──────────┘
```

### Basic Usage

```python
from agenticflow.topologies import Pipeline

researcher = Agent(name="researcher", model=model, tools=[search])
writer = Agent(name="writer", model=model)
editor = Agent(name="editor", model=model)

topology = Pipeline(stages=[researcher, writer, editor])

result = await topology.run("Create an article about quantum computing")
# Research → Write → Edit
```

### With Stage Configuration

```python
from agenticflow.topologies import Pipeline, AgentConfig

topology = Pipeline(
    stages=[
        AgentConfig(
            agent=researcher,
            role="research specialist",
            output_format="structured findings",
        ),
        AgentConfig(
            agent=writer,
            role="content writer",
            input_template="Based on: {previous_output}\n\nWrite article.",
        ),
        AgentConfig(
            agent=editor,
            role="editor",
            input_template="Edit this draft:\n{previous_output}",
        ),
    ],
)
```

### Convenience Function

```python
from agenticflow.topologies import pipeline

topology = pipeline(stages=[researcher, writer, editor])
```

### Parallel Stages

Execute multiple agents in parallel within a stage:

```python
topology = Pipeline(
    stages=[
        researcher,
        [analyst1, analyst2],  # Run in parallel
        writer,
    ],
)
```

---

## Mesh

All agents collaborate through multiple rounds:

```
        Round 1          Round 2          Round 3
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  A ←→ B     │  │  A ←→ B     │  │  A ←→ B     │
    │   ↘ ↙      │  │   ↘ ↙      │  │   ↘ ↙      │
    │    C        │  │    C        │  │    C        │
    └─────────────┘  └─────────────┘  └─────────────┘
```

### Basic Usage

```python
from agenticflow.topologies import Mesh

analyst1 = Agent(name="business_analyst", model=model)
analyst2 = Agent(name="technical_analyst", model=model)
analyst3 = Agent(name="financial_analyst", model=model)

topology = Mesh(
    agents=[analyst1, analyst2, analyst3],
    max_rounds=3,
)

result = await topology.run("Evaluate this product idea: AI-powered fitness app")
```

### Configuration Options

```python
from agenticflow.topologies import Mesh, AgentConfig

topology = Mesh(
    agents=[
        AgentConfig(agent=analyst1, role="business perspective"),
        AgentConfig(agent=analyst2, role="technical feasibility"),
        AgentConfig(agent=analyst3, role="financial analysis"),
    ],
    max_rounds=5,
    consensus_threshold=0.8,  # Stop when 80% agree
    round_format="structured",  # Each agent sees structured summary
)
```

### Convenience Function

```python
from agenticflow.topologies import mesh

topology = mesh(
    agents=[analyst1, analyst2, analyst3],
    max_rounds=3,
)
```

---

## Context Strategies

Prevent context explosion in multi-round topologies with pluggable strategies.

**Problem:** With 5 agents × 10 rounds, you'd pass 25,000+ tokens of history!

```python
from agenticflow.topologies import (
    Mesh,
    SlidingWindowStrategy,
    SummarizationStrategy,
    StructuredHandoffStrategy,
    BlackboardStrategy,
    RetrievalStrategy,
)
```

### Strategy Overview

| Strategy | Description | LLM Calls |
|----------|-------------|-----------|
| `SlidingWindowStrategy` | Keep only last N rounds | None |
| `SummarizationStrategy` | LLM-compress older rounds | Yes (cached) |
| `RetrievalStrategy` | VectorStore + semantic search | None |
| `StructuredHandoffStrategy` | Extract decisions/findings as JSON | Yes |
| `BlackboardStrategy` | Read/write specific Memory keys | None |

### SlidingWindowStrategy (Default)

Keep only the last N rounds of history:

```python
mesh = Mesh(
    agents=[analyst1, analyst2],
    max_rounds=10,
    context_strategy=SlidingWindowStrategy(max_rounds=3),  # Only last 3 rounds
)
```

### SummarizationStrategy

Summarize older rounds, keep recent ones in full:

```python
mesh = Mesh(
    agents=[analyst1, analyst2],
    context_strategy=SummarizationStrategy(
        model=model,
        keep_full_rounds=2,     # Keep last 2 rounds in full
        max_summary_tokens=200,  # Summary size
    ),
)
```

### StructuredHandoffStrategy

Extract structured data (decisions, findings) instead of raw text:

```python
mesh = Mesh(
    agents=[analyst1, analyst2],
    context_strategy=StructuredHandoffStrategy(
        model=model,
        max_items_per_category=5,
    ),
)
# Passes: {"decisions": [...], "findings": [...], "questions": [...]}
```

### RetrievalStrategy

Store outputs in VectorStore, retrieve only relevant context:

```python
from agenticflow.vectorstore import VectorStore

vs = VectorStore()
mesh = Mesh(
    agents=[analyst1, analyst2],
    context_strategy=RetrievalStrategy(vectorstore=vs, k=5),
)
```

### BlackboardStrategy

Use shared Memory as a blackboard:

```python
from agenticflow.memory import Memory

memory = Memory()
mesh = Mesh(
    agents=[analyst1, analyst2],
    context_strategy=BlackboardStrategy(
        memory=memory,
        keys=["decisions", "findings"],
    ),
)
```

### Example

See [examples/topologies/context_strategies.py](../examples/topologies/context_strategies.py) for a complete demo.

---

## Hierarchical

Tree structure with delegation at each level:

```
                ┌──────────┐
                │   CEO    │
                └────┬─────┘
           ┌─────────┼─────────┐
           ▼         ▼         ▼
      ┌────────┐ ┌────────┐ ┌────────┐
      │  CTO   │ │  CFO   │ │  CMO   │
      └───┬────┘ └───┬────┘ └───┬────┘
       ┌──┴──┐    ┌──┴──┐    ┌──┴──┐
       ▼     ▼    ▼     ▼    ▼     ▼
     Dev1  Dev2  Acc1  Acc2  Mkt1  Mkt2
```

### Basic Usage

```python
from agenticflow.topologies import Hierarchical, AgentConfig

# Define hierarchy
topology = Hierarchical(
    root=AgentConfig(agent=ceo, role="CEO"),
    children={
        "CEO": [
            AgentConfig(agent=cto, role="CTO"),
            AgentConfig(agent=cfo, role="CFO"),
        ],
        "CTO": [
            AgentConfig(agent=dev1, role="Developer"),
            AgentConfig(agent=dev2, role="Developer"),
        ],
        "CFO": [
            AgentConfig(agent=accountant, role="Accountant"),
        ],
    },
)

result = await topology.run("Plan Q4 product launch")
```

### With Max Depth

```python
topology = Hierarchical(
    root=ceo_config,
    children=hierarchy,
    max_depth=3,  # Limit delegation depth
)
```

---

## AgentConfig

Configure agent behavior in topologies:

```python
from agenticflow.topologies import AgentConfig

config = AgentConfig(
    agent=my_agent,           # The Agent instance
    role="data analyst",      # Role description for context
    can_finish=True,          # Can complete the workflow
    input_template=None,      # Custom input formatting
    output_format=None,       # Expected output format
    max_iterations=10,        # Max iterations for this agent
)
```

---

## TopologyResult

All topologies return a `TopologyResult`:

```python
result = await topology.run("Task description")

print(result.output)         # Final output string
print(result.success)        # True if completed successfully
print(result.iterations)     # Number of iterations/rounds
print(result.agent_outputs)  # Dict of agent → output
print(result.duration_ms)    # Execution time
print(result.metadata)       # Additional metadata
```

---

## With Flow

Use topologies through the Flow interface:

```python
from agenticflow import Flow

# Supervisor via Flow
flow = Flow(
    name="content-team",
    agents=[coordinator, writer, editor],
    topology="supervisor",
)

# Pipeline via Flow
flow = Flow(
    name="content-pipeline",
    agents=[researcher, writer, editor],
    topology="pipeline",
)

# Mesh via Flow
flow = Flow(
    name="review-panel",
    agents=[analyst1, analyst2, analyst3],
    topology="mesh",
    max_rounds=3,
)

result = await flow.run("Create content about AI")
```

---

## Visualization

Generate visual diagrams of topologies:

```python
topology = Supervisor(coordinator=coord, workers=[w1, w2, w3])

# Get graph view
view = topology.graph()

# Render in various formats
print(view.mermaid())   # Mermaid diagram
print(view.ascii())     # ASCII art
print(view.dot())       # Graphviz DOT

# Save to file
view.save("topology.png")
view.save("topology.svg")
```

---

## Team Memory

Share state between agents during execution:

```python
from agenticflow.memory import Memory

# Shared memory
team_memory = Memory()

result = await topology.run(
    "Complete the project",
    memory=team_memory,
)

# Access shared state after execution
findings = await team_memory.recall("findings")
decisions = await team_memory.recall("decisions")
```

---

## API Reference

### Topology Classes

| Class | Description |
|-------|-------------|
| `Supervisor` | One coordinator delegates to workers |
| `Pipeline` | Sequential stage-by-stage processing |
| `Mesh` | All agents collaborate in rounds |
| `Hierarchical` | Tree structure with levels |

### Convenience Functions

| Function | Description |
|----------|-------------|
| `supervisor(coordinator, workers)` | Create Supervisor topology |
| `pipeline(stages)` | Create Pipeline topology |
| `mesh(agents, max_rounds)` | Create Mesh topology |

### Configuration

| Class | Description |
|-------|-------------|
| `AgentConfig` | Configure agent behavior in topology |
| `TopologyConfig` | Global topology configuration |
| `TopologyResult` | Result from topology execution |
| `TopologyType` | Enum of topology types |

### Context Strategies

| Class | Description |
|-------|-------------|
| `SlidingWindowStrategy` | Keep only last N rounds |
| `SummarizationStrategy` | LLM-compress older rounds |
| `RetrievalStrategy` | VectorStore + semantic search |
| `StructuredHandoffStrategy` | Extract decisions/findings as JSON |
| `BlackboardStrategy` | Read/write specific Memory keys |
| `CompositeStrategy` | Combine multiple strategies |
