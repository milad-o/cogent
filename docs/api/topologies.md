# Topologies API Reference

Topologies define how multiple agents collaborate and communicate. AgenticFlow provides prebuilt patterns and a flexible policy system for custom coordination.

## Overview

```python
from agenticflow import (
    # Topology classes
    BaseTopology,
    SupervisorTopology,
    PipelineTopology,
    MeshTopology,
    HierarchicalTopology,
    CustomTopology,
    TopologyFactory,
    
    # Configuration
    TopologyConfig,
    TopologyState,
    
    # Policies
    TopologyPolicy,
    AgentPolicy,
    HandoffRule,
    HandoffCondition,
    AcceptancePolicy,
    ExecutionMode,
    HandoffStrategy,
)
```

---

## Quick Start

```python
import asyncio
from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    EventBus,
    SupervisorTopology,
    TopologyConfig,
)

# Create agents
event_bus = EventBus()

supervisor = Agent(
    AgentConfig(
        name="Supervisor",
        role=AgentRole.SUPERVISOR,
        model="gpt-4o",
        system_prompt="You coordinate research tasks.",
    ),
    event_bus=event_bus,
)

researcher = Agent(
    AgentConfig(
        name="Researcher",
        role=AgentRole.WORKER,
        model="gpt-4o",
        system_prompt="You research topics thoroughly.",
    ),
    event_bus=event_bus,
)

writer = Agent(
    AgentConfig(
        name="Writer",
        role=AgentRole.WORKER,
        model="gpt-4o",
        system_prompt="You write clear reports.",
    ),
    event_bus=event_bus,
)

# Create topology
topology = SupervisorTopology(
    config=TopologyConfig(name="research-team"),
    agents=[supervisor, researcher, writer],
    supervisor_name="Supervisor",
)

# Run
async def main():
    result = await topology.run("Research AI trends and write a summary")
    for r in result.results:
        print(f"[{r['agent']}]: {r['thought'][:100]}...")

asyncio.run(main())
```

---

## Prebuilt Topologies

### SupervisorTopology

One coordinator manages multiple workers. The supervisor delegates tasks and workers report back.

```
┌────────────┐
│ Supervisor │
└─────┬──────┘
      │ delegate
   ┌──┴──┐
   │     │
┌──▼──┐ ┌▼────┐
│Work1│ │Work2│
└─────┘ └─────┘
```

```python
from agenticflow import SupervisorTopology, TopologyConfig

topology = SupervisorTopology(
    config=TopologyConfig(
        name="customer-support",
        max_iterations=20,
    ),
    agents=[manager, tech_agent, billing_agent],
    supervisor_name="Manager",  # Optional, defaults to first agent
)

# With parallel workers
topology = SupervisorTopology(
    config=TopologyConfig(name="parallel-team"),
    agents=[supervisor, worker1, worker2, worker3],
    supervisor_name="Supervisor",
)

# Run parallel fan-out
results = await topology.run_parallel(
    "Analyze this data from 3 perspectives",
    agent_names=["Worker1", "Worker2", "Worker3"],
)
```

### PipelineTopology

Sequential processing through stages. Each agent processes and passes to the next.

```
┌──────────┐    ┌────────┐    ┌────────┐
│Researcher│───►│ Writer │───►│ Editor │
└──────────┘    └────────┘    └────────┘
```

```python
from agenticflow import PipelineTopology, TopologyConfig

topology = PipelineTopology(
    config=TopologyConfig(name="content-pipeline"),
    agents=[researcher, writer, editor],
    stages=["Researcher", "Writer", "Editor"],  # Optional, defaults to agent order
    allow_skip=False,   # Allow skipping stages
    allow_repeat=False, # Allow going back
)

# Run the pipeline
result = await topology.run("Create an article about Python async")
```

### MeshTopology

All agents can communicate freely with each other. Best for collaborative problem-solving.

```
┌────────┐
│Analyst1│◄──────┐
└───┬────┘       │
    │       ┌────▼───┐
    └──────►│Analyst2│
            └────┬───┘
                 │
            ┌────▼───┐
    ┌──────►│Analyst3│
    │       └────────┘
    │            │
└───┴────────────┘
```

```python
from agenticflow import MeshTopology, TopologyConfig

topology = MeshTopology(
    config=TopologyConfig(
        name="collaborative-team",
        max_iterations=15,
    ),
    agents=[analyst1, analyst2, analyst3],
)

result = await topology.run("Discuss the pros and cons of microservices")
```

### HierarchicalTopology

Agents organized in levels. Higher levels coordinate, lower levels execute.

```
        ┌─────┐
        │ CEO │
        └──┬──┘
     ┌─────┼─────┐
     │     │     │
  ┌──▼─┐ ┌─▼──┐ ┌▼───┐
  │Mgr1│ │Mgr2│ │Mgr3│
  └──┬─┘ └─┬──┘ └─┬──┘
     │     │      │
  ┌──▼─┐ ┌─▼──┐ ┌─▼──┐
  │Dev1│ │Dev2│ │Dev3│
  └────┘ └────┘ └────┘
```

```python
from agenticflow import HierarchicalTopology, TopologyConfig

topology = HierarchicalTopology(
    config=TopologyConfig(name="organization"),
    agents=[ceo, manager1, manager2, dev1, dev2, dev3],
    levels=[
        ["CEO"],                        # Level 0 - Top
        ["Manager1", "Manager2"],       # Level 1 - Middle
        ["Dev1", "Dev2", "Dev3"],       # Level 2 - Bottom
    ],
)

result = await topology.run("Plan and implement the new feature")
```

---

## TopologyConfig

Configuration for topology behavior.

```python
from agenticflow import TopologyConfig, HandoffStrategy

config = TopologyConfig(
    name="my-topology",                          # Required name
    description="A custom multi-agent system",    # Optional description
    max_iterations=100,                          # Max agent steps
    handoff_strategy=HandoffStrategy.AUTOMATIC,  # Handoff mode
    enable_memory=True,                          # Enable memory features
    enable_checkpointing=True,                   # Enable state checkpoints
    recursion_limit=50,                          # LangGraph recursion limit
    metadata={"version": "1.0"},                 # Custom metadata
)
```

### HandoffStrategy

```python
from agenticflow import HandoffStrategy

class HandoffStrategy(Enum):
    COMMAND = "command"       # Explicit LangGraph Command handoffs
    INTERRUPT = "interrupt"   # Human-in-the-loop with interrupts
    AUTOMATIC = "automatic"   # Let topology decide routing
    BROADCAST = "broadcast"   # Send to all agents
```

---

## TopologyState

Shared state across all agents in a topology.

```python
from agenticflow import TopologyState

@dataclass
class TopologyState:
    messages: list[dict[str, Any]]     # Conversation history
    current_agent: str | None          # Currently active agent
    task: str                          # The task being processed
    context: dict[str, Any]            # Shared context
    iteration: int                     # Current iteration count
    completed: bool                    # Whether task is complete
    error: str | None                  # Error message if failed
    results: list[dict[str, Any]]      # Agent results

# Access state
state = TopologyState.from_dict({"task": "Analyze data", "iteration": 0})
data = state.to_dict()
```

---

## Policy System

Policies define the rules for agent communication and handoffs.

### TopologyPolicy

Complete handoff policy for a topology.

```python
from agenticflow import TopologyPolicy, HandoffCondition, ExecutionMode

# Create custom policy
policy = TopologyPolicy(
    entry_point="Gateway",                    # First agent to receive task
    default_acceptance=AcceptancePolicy.ACCEPT_ALL,
    allow_self_handoff=False,
    execution_mode=ExecutionMode.SEQUENTIAL,
    parallel_groups=[["Worker1", "Worker2"]], # Groups that can run in parallel
)

# Add rules
policy.add_rule("Gateway", "Validator", label="validate")
policy.add_rule("Validator", "Processor", condition=HandoffCondition.ON_SUCCESS)
policy.add_rule("Validator", "Gateway", condition=HandoffCondition.ON_FAILURE)
policy.add_rule("Processor", "Storage", label="store")

# Use factory methods
policy = TopologyPolicy.supervisor("Supervisor", ["Worker1", "Worker2"])
policy = TopologyPolicy.pipeline(["Stage1", "Stage2", "Stage3"])
policy = TopologyPolicy.mesh(["Agent1", "Agent2", "Agent3"])
policy = TopologyPolicy.hierarchical([["Top"], ["Mid1", "Mid2"], ["Bot1", "Bot2"]])
```

### AgentPolicy

Policy for individual agent behavior.

```python
from agenticflow import AgentPolicy, AcceptancePolicy

policy = AgentPolicy(
    agent_name="Worker1",
    acceptance=AcceptancePolicy.ACCEPT_LISTED,  # Only accept from specific agents
    accept_from=["Supervisor"],                 # Accept from supervisor only
    reject_from=[],                             # Reject none (for REJECT_LISTED)
    can_send_to=["Supervisor"],                 # Can only report to supervisor
    can_finish=False,                           # Cannot end the workflow
)

# Custom acceptance function
policy = AgentPolicy(
    agent_name="ConditionalAgent",
    acceptance=AcceptancePolicy.CONDITIONAL,
    acceptance_fn=lambda source, state: state.get("priority") == "high",
)
```

### HandoffRule

Rules for individual handoffs.

```python
from agenticflow import HandoffRule, HandoffCondition

rule = HandoffRule(
    source="Validator",
    target="Processor",
    condition=HandoffCondition.ON_SUCCESS,
    label="validated",
    priority=10,  # Higher priority rules are checked first
)

# With state transformation
rule = HandoffRule(
    source="Analyzer",
    target="Reporter",
    transform=lambda state: {
        **state,
        "analysis_complete": True,
        "summary": state.get("analysis", "")[:500],
    },
)

# With custom condition
rule = HandoffRule(
    source="Router",
    target="HighPriorityHandler",
    condition=HandoffCondition.CONDITIONAL,
    condition_fn=lambda state: state.get("priority", 0) > 5,
)
```

### HandoffCondition

When handoffs are allowed.

```python
from agenticflow import HandoffCondition

class HandoffCondition(Enum):
    ALWAYS = "always"           # Always allow
    ON_SUCCESS = "on_success"   # Only on successful completion
    ON_FAILURE = "on_failure"   # Only on failure/error
    ON_REQUEST = "on_request"   # Only when explicitly requested
    CONDITIONAL = "conditional" # Custom condition function
```

### AcceptancePolicy

How agents accept incoming tasks.

```python
from agenticflow import AcceptancePolicy

class AcceptancePolicy(Enum):
    ACCEPT_ALL = "accept_all"       # Accept from anyone
    ACCEPT_LISTED = "accept_listed" # Only from specific agents
    REJECT_LISTED = "reject_listed" # From all except specific
    CONDITIONAL = "conditional"     # Custom function
```

### ExecutionMode

How agents execute within a topology.

```python
from agenticflow import ExecutionMode

class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"  # One at a time (default)
    PARALLEL = "parallel"      # Independent agents concurrently
    FAN_OUT = "fan_out"        # Supervisor fans to workers
    FAN_IN = "fan_in"          # Workers converge to coordinator
```

---

## BaseTopology Methods

All topology classes inherit from `BaseTopology`.

### `run(task, context=None, thread_id=None, on_step=None) -> TopologyState`

Run the topology on a task.

```python
# Basic run
result = await topology.run("Analyze the quarterly report")

# With context
result = await topology.run(
    "Review the code",
    context={"files": ["main.py", "utils.py"], "focus": "security"},
)

# With step callback
def on_step(state):
    agent = state.get("current_agent")
    if agent:
        print(f"[{agent}] processing...")

result = await topology.run("Complex task", on_step=on_step)

# With thread ID for checkpointing
result = await topology.run(
    "Long task",
    thread_id="conversation-123",
)
```

### `stream(task, context=None, thread_id=None) -> AsyncIterator[dict]`

Stream topology execution, yielding state after each step.

```python
async for state in topology.stream("Analyze this data"):
    agent = state.get("current_agent")
    if agent and state.get("results"):
        latest = state["results"][-1]
        print(f"[{agent}]: {latest['thought'][:100]}...")
```

### `run_parallel(task, context=None, agent_names=None, merge_strategy="combine", on_agent_complete=None) -> dict`

Run multiple agents in parallel on the same task.

```python
# Run all agents in parallel
results = await topology.run_parallel("Analyze from different perspectives")

# Specify which agents
results = await topology.run_parallel(
    "Research this topic",
    agent_names=["Researcher1", "Researcher2", "Researcher3"],
)

# With progress callback
results = await topology.run_parallel(
    "Analyze data",
    on_agent_complete=lambda name, r: print(f"✓ {name} done"),
)

# Merge strategies
results = await topology.run_parallel(task, merge_strategy="combine")  # List of all results
results = await topology.run_parallel(task, merge_strategy="first")    # First to complete
results = await topology.run_parallel(task, merge_strategy="vote")     # Most common result

# Result structure
print(results["results"])  # Dict[agent_name, result]
print(results["timing"])   # Dict[agent_name, duration_ms]
print(results["errors"])   # Dict[agent_name, error_message]
print(results["merged"])   # Combined result based on strategy
```

### `resume(thread_id, human_input) -> TopologyState`

Resume execution after a human-in-the-loop interrupt.

```python
# After an interrupt
result = await topology.resume(
    thread_id="conversation-123",
    human_input="Yes, proceed with option A",
)
```

### `handoff(target, state_update=None, resume_value=None) -> Command`

Create a handoff command (for use in custom node functions).

```python
from langgraph.types import Command

# In a custom node
def custom_node(state):
    if should_delegate:
        return topology.handoff(
            "Worker",
            state_update={"delegated": True},
        )
    return state
```

### `request_human_input(question, state) -> Any`

Request human input using interrupt.

```python
# In a custom node
def approval_node(state):
    if needs_approval:
        response = topology.request_human_input(
            "Do you approve this action?",
            state,
        )
        return {"approved": response == "yes"}
    return state
```

### Visualization

#### `draw_mermaid(**kwargs) -> str`

Generate a Mermaid diagram of the topology.

```python
diagram = topology.draw_mermaid(
    theme="forest",      # default, forest, dark, neutral, base
    direction="TB",      # TB, TD, BT, LR, RL
    title="My Topology",
    show_tools=True,
    show_roles=True,
)
print(diagram)
```

#### `draw_mermaid_png(**kwargs) -> bytes`

Generate a PNG image of the topology diagram.

```python
png_bytes = topology.draw_mermaid_png(theme="dark")
with open("topology.png", "wb") as f:
    f.write(png_bytes)
```

---

## CustomTopology

For complex custom patterns using the policy system.

```python
from agenticflow import (
    Agent,
    AgentConfig,
    BaseTopology,
    TopologyConfig,
    TopologyPolicy,
    AgentPolicy,
    HandoffRule,
    HandoffCondition,
    ExecutionMode,
    EventBus,
)

# Define agents
event_bus = EventBus()
agents = [
    Agent(AgentConfig(name="Gateway", model="gpt-4o"), event_bus),
    Agent(AgentConfig(name="Validator", model="gpt-4o"), event_bus),
    Agent(AgentConfig(name="Processor", model="gpt-4o"), event_bus),
    Agent(AgentConfig(name="ErrorHandler", model="gpt-4o"), event_bus),
    Agent(AgentConfig(name="Storage", model="gpt-4o"), event_bus),
]

# Create custom policy
policy = TopologyPolicy(entry_point="Gateway")

# Define flow
policy.add_rule("Gateway", "Validator", label="validate")
policy.add_rule("Validator", "Processor", condition=HandoffCondition.ON_SUCCESS, label="valid")
policy.add_rule("Validator", "ErrorHandler", condition=HandoffCondition.ON_FAILURE, label="invalid")
policy.add_rule("ErrorHandler", "Gateway", label="retry")
policy.add_rule("Processor", "Storage", label="store")

# Agent policies
policy.add_agent_policy(AgentPolicy(
    agent_name="Gateway",
    can_send_to=["Validator"],
    can_finish=False,
))
policy.add_agent_policy(AgentPolicy(
    agent_name="Validator",
    can_send_to=["Processor", "ErrorHandler"],
    can_finish=False,
))
policy.add_agent_policy(AgentPolicy(
    agent_name="Storage",
    can_send_to=[],
    can_finish=True,
))

# Create topology with custom policy
topology = BaseTopology(
    config=TopologyConfig(name="validation-pipeline"),
    agents=agents,
    policy=policy,
)

result = await topology.run("Process this data")
```

---

## TopologyFactory

Create topologies dynamically using the factory.

```python
from agenticflow import TopologyFactory

# Create topology by type
topology = TopologyFactory.create(
    topology_type="supervisor",  # supervisor, pipeline, mesh, hierarchical
    config=TopologyConfig(name="dynamic-team"),
    agents=[supervisor, worker1, worker2],
    supervisor_name="Supervisor",
)

# Or use specific factory methods
topology = TopologyFactory.supervisor(config, agents, supervisor_name="Boss")
topology = TopologyFactory.pipeline(config, agents, stages=["A", "B", "C"])
topology = TopologyFactory.mesh(config, agents)
topology = TopologyFactory.hierarchical(config, agents, levels=[["Top"], ["Bot1", "Bot2"]])
```

---

## Complete Example: Research Pipeline

```python
import asyncio
from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    EventBus,
    PipelineTopology,
    TopologyConfig,
    ProgressTracker,
    OutputConfig,
)

async def main():
    event_bus = EventBus()
    
    # Create specialized agents
    researcher = Agent(
        AgentConfig(
            name="Researcher",
            role=AgentRole.SPECIALIST,
            model="gpt-4o",
            system_prompt="""You are a thorough researcher. 
            Gather key facts and insights about the topic.
            Be comprehensive but concise.""",
        ),
        event_bus=event_bus,
    )
    
    analyst = Agent(
        AgentConfig(
            name="Analyst",
            role=AgentRole.SPECIALIST,
            model="gpt-4o",
            system_prompt="""You are a critical analyst.
            Analyze the research, identify patterns and insights.
            Provide balanced analysis with pros and cons.""",
        ),
        event_bus=event_bus,
    )
    
    writer = Agent(
        AgentConfig(
            name="Writer",
            role=AgentRole.WORKER,
            model="gpt-4o",
            system_prompt="""You are a skilled technical writer.
            Take the research and analysis to create a clear,
            well-structured report. Use headers and bullet points.""",
        ),
        event_bus=event_bus,
    )
    
    editor = Agent(
        AgentConfig(
            name="Editor",
            role=AgentRole.CRITIC,
            model="gpt-4o",
            system_prompt="""You are a meticulous editor.
            Review the report for clarity, accuracy, and flow.
            Make improvements and provide the final polished version.""",
        ),
        event_bus=event_bus,
    )
    
    # Create pipeline
    topology = PipelineTopology(
        config=TopologyConfig(
            name="research-pipeline",
            description="Research → Analyze → Write → Edit",
            max_iterations=10,
        ),
        agents=[researcher, analyst, writer, editor],
        stages=["Researcher", "Analyst", "Writer", "Editor"],
    )
    
    # Generate diagram
    print("Pipeline Structure:")
    print(topology.draw_mermaid(direction="LR"))
    print()
    
    # Run with progress tracking
    tracker = ProgressTracker(OutputConfig.verbose())
    
    def on_step(state):
        agent = state.get("current_agent")
        if agent:
            tracker.step(f"step-{state['iteration']}", f"{agent} processing")
    
    with tracker:
        tracker.start_workflow("research", "Research Pipeline")
        
        result = await topology.run(
            "Write a comprehensive report on the future of AI agents in software development",
            on_step=on_step,
        )
        
        tracker.success("Pipeline complete!")
    
    # Show results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for r in result.results:
        print(f"\n[{r['agent']}]")
        print("-" * 40)
        print(r['thought'][:500] + "..." if len(r['thought']) > 500 else r['thought'])

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Next Steps

- [Agents](agents.md) - Agent configuration and resilience
- [Events](events.md) - Event-driven communication
- [Observability](observability.md) - Progress tracking and metrics
