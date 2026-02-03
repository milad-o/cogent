# Subagents: Native Delegation Support

**Status:** Production Ready (v0.x.x+)

Subagents enable true multi-agent coordination where a coordinator agent delegates tasks to specialist agents while preserving full metadata (tokens, duration, delegation chain) for accurate cost tracking and observability.

---

## Overview

### The Problem

When using `agent.as_tool()`, Response metadata is lost because tools return strings:

```python
# ❌ Old approach - loses metadata
specialist = Agent(name="specialist", model="gpt-4o")
coordinator = Agent(
    name="coordinator",
    model="gpt-4o",
    tools=[specialist.as_tool()],  # Response[T] → string
)

response = await coordinator.run("Analyze this data")
# Token count only includes coordinator, not specialist ❌
```

### The Solution

Native `subagents=` parameter preserves Response metadata through executor interception:

```python
# ✅ New approach - preserves metadata
specialist = Agent(name="specialist", model="gpt-4o")
coordinator = Agent(
    name="coordinator",
    model="gpt-4o",
    subagents={"specialist": specialist},  # Full Response[T] preserved
)

response = await coordinator.run("Analyze this data")
# Token count includes coordinator + specialist ✅
print(f"Total tokens: {response.metadata.tokens.total_tokens}")
print(f"Subagent calls: {len(response.subagent_responses)}")
```

---

## How It Works

1. **LLM Perspective:** Subagents appear as regular tools with a `task` parameter
2. **Executor Interception:** Executor detects subagent tools and routes to SubagentRegistry
3. **Metadata Preservation:** Full `Response[T]` objects are cached, not just strings
4. **Automatic Aggregation:** Tokens, duration, and delegation chain are aggregated automatically

**Key Principle:** Zero LLM behavior changes - uses existing tool calling mechanism.

---

## Quick Start

### Basic Example

```python
from cogent import Agent

# Create specialist agents
data_analyst = Agent(
    name="data_analyst",
    model="gpt-4o-mini",
    instructions="Analyze data and provide statistical insights.",
)

market_researcher = Agent(
    name="market_researcher",
    model="gpt-4o-mini",
    instructions="Research market trends and competitive landscape.",
)

# Create coordinator with subagents
coordinator = Agent(
    name="coordinator",
    model="gpt-4o-mini",
    instructions="""Coordinate research tasks:
- Use data_analyst for numerical analysis
- Use market_researcher for market trends
Synthesize their findings.""",
    subagents={
        "data_analyst": data_analyst,
        "market_researcher": market_researcher,
    },
)

# Run task - coordinator will delegate automatically
response = await coordinator.run(
    "Analyze Q4 2025 e-commerce growth: 18% YoY to $1.2T globally, "
    "mobile is 65% of total. What are the key insights?"
)

print(response.content)
print(f"Total tokens: {response.metadata.tokens.total_tokens}")
```

### Accessing Metadata

```python
# Token aggregation (coordinator + all subagents)
total_tokens = response.metadata.tokens.total_tokens
print(f"Total cost: {total_tokens}")

# Individual subagent responses
for sub_resp in response.subagent_responses:
    print(f"{sub_resp.metadata.agent}: {sub_resp.metadata.tokens.total_tokens} tokens")

# Delegation chain
for delegation in response.metadata.delegation_chain:
    print(f"{delegation['agent']} - {delegation['tokens']} tokens - {delegation['duration']:.2f}s")
```

---

## API Reference

### Agent Constructor

```python
Agent(
    name: str,
    model: str | BaseChatModel,
    subagents: dict[str, Agent] | None = None,
    **kwargs
)
```

**Parameters:**
- `subagents`: Dictionary mapping tool names to Agent instances
  - Keys become tool names the LLM can call
  - Values are the specialist agents to delegate to

**Example:**
```python
coordinator = Agent(
    name="coordinator",
    model="gpt-4o",
    subagents={
        "analyst": analyst_agent,
        "researcher": researcher_agent,
    },
)
```

### Response Metadata

```python
@dataclass
class Response[T]:
    content: T
    metadata: ResponseMetadata
    subagent_responses: list[Response] | None  # NEW: Responses from delegated subagents
    # ... other fields
```

```python
@dataclass
class ResponseMetadata:
    agent: str
    model: str
    tokens: TokenUsage
    duration: float
    delegation_chain: list[dict] | None  # NEW: Chain of delegations
    # ... other fields
```

**Delegation Chain Structure:**
```python
{
    "agent": "analyst",           # Subagent name
    "model": "gpt-4o-mini",       # Model used
    "tokens": 150,                # Total tokens
    "duration": 2.5,              # Seconds
}
```

---

## Best Practices

### 1. Clear Agent Responsibilities

```python
# ✅ GOOD: Specific, non-overlapping responsibilities
data_cleaner = Agent(
    name="data_cleaner",
    instructions="Clean and normalize messy data. Fix formatting, handle nulls.",
)

data_validator = Agent(
    name="data_validator",
    instructions="Validate data quality. Check for errors, inconsistencies.",
)

# ❌ BAD: Overlapping, vague responsibilities
helper1 = Agent(name="helper1", instructions="Help with data stuff")
helper2 = Agent(name="helper2", instructions="Also help with data")
```

### 2. Descriptive Naming

```python
# ✅ GOOD: Names that indicate purpose
subagents={
    "sql_generator": sql_agent,
    "data_visualizer": viz_agent,
    "report_writer": report_agent,
}

# ❌ BAD: Generic names
subagents={
    "agent1": sql_agent,
    "helper": viz_agent,
    "assistant": report_agent,
}
```

### 3. Coordinator Instructions

```python
# ✅ GOOD: Explicit delegation guidelines
coordinator = Agent(
    instructions="""You coordinate ETL tasks:
- Use data_analyst to understand CSV structure and issues
- Use data_cleaner to design transformation rules
- Use sql_generator to create database schema
Synthesize their work into a complete ETL plan.""",
    subagents={...},
)

# ❌ BAD: Vague instructions
coordinator = Agent(
    instructions="You have some helpers. Use them if you want.",
    subagents={...},
)
```

### 4. Observability

Always use an Observer to see delegation flow:

```python
from cogent import Agent, Observer

observer = Observer(level="progress")

coordinator = Agent(
    name="coordinator",
    model="gpt-4o",
    subagents={...},
    observer=observer,  # Shows [subagent-decision], [subagent-call], etc.
)
```

### 5. Context Propagation

Context automatically propagates through delegation:

```python
from cogent import RunContext

ctx = RunContext(
    thread_id="session-123",
    user_id="user-456",
    metadata={"department": "analytics"},
)

response = await coordinator.run("Analyze data", context=ctx)
# All subagents receive the same context automatically
```

---

## Advanced Patterns

### Nested Subagents

Subagents can have their own subagents:

```python
# Specialist with sub-specialists
data_analyst = Agent(
    name="data_analyst",
    model="gpt-4o",
    subagents={
        "statistician": statistician_agent,
        "visualizer": viz_agent,
    },
)

# Top-level coordinator
coordinator = Agent(
    name="coordinator",
    model="gpt-4o",
    subagents={
        "data_analyst": data_analyst,  # Has its own subagents
        "report_writer": writer_agent,
    },
)
```

### Conditional Delegation

The LLM decides when to delegate:

```python
coordinator = Agent(
    instructions="""Analyze requests:
- For simple questions, answer directly
- For complex analysis, delegate to data_analyst
- For market research, delegate to market_researcher
Use your judgment on which tasks need specialist help.""",
    subagents={
        "data_analyst": analyst,
        "market_researcher": researcher,
    },
)

# LLM may or may not delegate based on complexity
response1 = await coordinator.run("What is 2+2?")  # Answers directly
response2 = await coordinator.run("Analyze Q4 sales trends")  # Delegates to analyst
```

### Mixed Tools and Subagents

Subagents and regular tools work together:

```python
from cogent import tool

@tool
def search_database(query: str) -> str:
    """Search internal database."""
    return database.search(query)

coordinator = Agent(
    name="coordinator",
    model="gpt-4o",
    tools=[search_database],  # Regular tool
    subagents={
        "analyst": analyst_agent,  # Subagent
    },
)

# LLM can use both:
# 1. Call search_database tool → get data
# 2. Delegate to analyst → analyze data
```

---

## Migration Guide

### From `agent.as_tool()`

**Before:**
```python
specialist = Agent(name="specialist", model="gpt-4o")

coordinator = Agent(
    name="coordinator",
    model="gpt-4o",
    tools=[specialist.as_tool()],
)

response = await coordinator.run("Do research")
# ❌ Only coordinator tokens counted
```

**After:**
```python
specialist = Agent(name="specialist", model="gpt-4o")

coordinator = Agent(
    name="coordinator",
    model="gpt-4o",
    subagents={"specialist": specialist},  # Changed from tools to subagents
)

response = await coordinator.run("Do research")
# ✅ Coordinator + specialist tokens counted
```

**Migration Steps:**

1. Change `tools=[agent.as_tool()]` → `subagents={"name": agent}`
2. Access metadata via `response.subagent_responses`
3. Token counts now include all subagents automatically

---

## Troubleshooting

### Subagent not being called

**Problem:** LLM ignores subagent tools

**Solutions:**
- Make coordinator instructions explicit about delegation
- Use descriptive subagent names (e.g., "data_analyst" not "helper")
- Add descriptions to subagent Agent configs

```python
specialist = Agent(
    name="specialist",
    model="gpt-4o",
    description="Expert in data analysis and statistics",  # Helps LLM understand when to call
)
```

### Token counts seem wrong

**Problem:** Tokens don't match expectations

**Debug:**
```python
print(f"Coordinator: {response.metadata.tokens.total_tokens}")
for sub in response.subagent_responses:
    print(f"{sub.metadata.agent}: {sub.metadata.tokens.total_tokens}")
print(f"Delegation chain: {response.metadata.delegation_chain}")
```

### Subagent errors

**Problem:** Subagent fails during execution

**Behavior:** Error returned to coordinator as tool result, LLM can retry or handle

```python
# LLM sees error message and can:
# 1. Retry with different parameters
# 2. Try different subagent
# 3. Handle error in response
```

---

## Performance Considerations

### Memory Usage

Each subagent maintains its own conversation history if `conversation=True`:

```python
# ✅ GOOD: Disable conversation for stateless subagents
data_cleaner = Agent(
    name="data_cleaner",
    model="gpt-4o-mini",
    conversation=False,  # Saves memory
)

# ❌ BAD: Unnecessary conversation history
data_cleaner = Agent(
    name="data_cleaner",
    model="gpt-4o-mini",
    conversation=True,  # Wastes memory if not needed
)
```

### Parallel Execution

Subagents execute in parallel when LLM calls multiple at once:

```python
# LLM decides to call both in one turn
# → Both execute in parallel automatically
# → Results returned together
```

### Model Selection

Use appropriate models for each role:

```python
# ✅ GOOD: Match model to task complexity
coordinator = Agent(
    name="coordinator",
    model="gpt-4o",  # Complex orchestration
    subagents={
        "summarizer": Agent(model="gpt-4o-mini"),  # Simple task
        "analyst": Agent(model="gpt-4o"),  # Complex analysis
    },
)
```

---

## Examples

See:
- [`examples/basics/simple_delegation.py`](../examples/basics/simple_delegation.py) - Minimal example
- [`examples/advanced/subagent_coordinator.py`](../examples/advanced/subagent_coordinator.py) - Full-featured coordinator
- [`examples/advanced/single_vs_multi_etl.py`](../examples/advanced/single_vs_multi_etl.py) - Before/after comparison

---

## Comparison: Single Agent vs Multi-Agent

| Aspect | Single Agent | Multi-Agent (Subagents) |
|--------|--------------|-------------------------|
| **Memory** | Natural 4-layer memory | Must share via RunContext |
| **Complexity** | Simple, one config | More setup, multiple configs |
| **Specialization** | Generalist approach | Focused specialists |
| **Token cost** | Usually lower | Higher (multiple calls) |
| **Observability** | One agent trace | Full delegation chain |
| **Best for** | Linear workflows | Complex coordination |

**Rule of Thumb:** Start with a single agent. Add subagents when you need:
- Clear specialist roles (SQL expert, data cleaner, etc.)
- Separation of concerns (analysis vs presentation)
- Delegated decision-making (coordinator decides who handles what)

---

## FAQ

**Q: Can subagents have different models?**  
A: Yes! Each agent can use a different model.

**Q: Do subagents share conversation history?**  
A: No. Each agent has its own conversation if `conversation=True`. Use RunContext to share state.

**Q: Can I mix `subagents=` and `tools=`?**  
A: Yes! They work together seamlessly.

**Q: Are token counts accurate?**  
A: Yes - coordinator + all subagent tokens are aggregated automatically.

**Q: Can subagents call each other?**  
A: Not directly. But nested subagents work (subagent has its own subagents).

**Q: What if a subagent fails?**  
A: Error is returned to coordinator as a tool result. The LLM can handle it.

**Q: How deep can I nest?**  
A: No hard limit, but 2-3 levels max is recommended for clarity.

**Q: Does this work with all models?**  
A: Yes - any model that supports tool calling (OpenAI, Anthropic, Gemini, etc.)

---

## Related Documentation

- [Agent Documentation](./agent.md)
- [Tool Building](./tool-building.md)
- [Context Layer](./context.md)
- [Observability](./observability.md)
