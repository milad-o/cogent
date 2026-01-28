# Agentive Context Control (ACC)

**Bounded memory for long conversations with drift prevention.**

## Overview

ACC (Agentic Context Compression) maintains bounded internal state instead of unbounded transcript replay. Based on [arXiv:2601.11653](https://arxiv.org/abs/2601.11653), it prevents:

- **Context drift** — Maintains constraints and entities across turns
- **Memory poisoning** — Verifies artifacts before committing
- **Context overflow** — Bounded state regardless of conversation length

## Quick Start

Enable ACC with `acc=True` on Agent or Memory:

```python
from cogent import Agent
from cogent.memory import Memory

# Option 1: Enable on Agent
agent = Agent(name="Assistant", model="gpt-4o", acc=True)

# Option 2: Enable on Memory
memory = Memory(acc=True)
agent = Agent(name="Assistant", model="gpt-4o", memory=memory)

# Use thread_id to persist context across turns
await agent.run("My name is Alice", thread_id="session-1")
await agent.run("I prefer dark mode", thread_id="session-1")
await agent.run("What's my name?", thread_id="session-1")  # Remembers!
```

## Custom Bounds

For fine-grained control, pass custom bounds directly to `AgentCognitiveCompressor`:

```python
from cogent import Agent
from cogent.memory import Memory
from cogent.memory.acc import AgentCognitiveCompressor

# Create ACC with custom bounds
acc = AgentCognitiveCompressor(
    max_constraints=10,  # Rules, guidelines (default: 10)
    max_entities=30,     # Facts, knowledge (default: 50)
    max_actions=20,      # Past actions (default: 30)
    max_context=15,      # Relevant context (default: 20)
)

# Pass to Agent or Memory
agent = Agent(name="Assistant", model="gpt-4o", acc=acc)
# OR
memory = Memory(acc=acc)
agent = Agent(name="Assistant", model="gpt-4o", memory=memory)

# Access state for monitoring
print(f"Entities: {len(acc.state.entities)}/{acc.state.max_entities}")
print(f"Actions: {len(acc.state.actions)}/{acc.state.max_actions}")
```

## When to Use ACC

| Use ACC When | Don't Use When |
|--------------|-------------------|
| Long conversations (>10 turns) | Short, stateless queries |
| Need to prevent drift | Simple Q&A |
| Bounded memory is critical | Need full transcript replay |
| Multi-turn workflows | One-off operations |

## How ACC Works

ACC maintains bounded internal state with four categories:

| Category | Purpose | Default Max |
|----------|---------|-------------|
| **Constraints** | Rules, guidelines, requirements | 10 |
| **Entities** | Facts, knowledge, data | 50 |
| **Actions** | What worked/failed | 30 |
| **Context** | Relevant snippets | 20 |

Total: ~110 items regardless of conversation length.

```python
from cogent.memory.acc import BoundedMemoryState

# View state contents
state = BoundedMemoryState()
print(state.constraints)  # List of constraints
print(state.entities)     # List of entities
print(state.actions)      # List of actions
print(state.context)      # List of context items
```

## ACC vs SemanticCache

| Feature | ACC | SemanticCache |
|---------|-----|---------------|
| **Purpose** | Bounded conversation context | Cache tool outputs |
| **Matching** | Structured memory extraction | Semantic similarity |
| **Use Case** | Long conversations | Expensive tool calls |
| **Thread-aware** | Yes (thread_id) | No |

**Use together:** ACC for conversation context, SemanticCache for tool output caching.

## Best Practices

1. **Always use thread_id** — Required for context persistence across turns
2. **Set appropriate bounds** — Smaller bounds = less context but faster
3. **Scope per user/session** — Use unique thread_id per conversation
4. **Monitor state** — Check entity/action counts for debugging

## Examples

See working examples:
- [examples/advanced/acc.py](../examples/advanced/acc.py) — ACC usage patterns
- [examples/advanced/content_review.py](../examples/advanced/content_review.py) — ACC with Memory integration

## API Reference

### BoundedMemoryState

```python
class BoundedMemoryState:
    def __init__(
        self,
        max_constraints: int = 10,
        max_entities: int = 50,
        max_actions: int = 30,
        max_context: int = 20,
    ):
        """Initialize bounded state with category limits."""
    
    @property
    def constraints(self) -> list[str]: ...
    @property
    def entities(self) -> list[str]: ...
    @property
    def actions(self) -> list[str]: ...
    @property
    def context(self) -> list[str]: ...
```

### AgentCognitiveCompressor

```python
class AgentCognitiveCompressor:
    def __init__(
        self,
        state: BoundedMemoryState,
        forget_gate: SemanticForgetGate | None = None,
    ):
        """Initialize ACC with bounded state."""
    
    async def update_from_turn(
        self,
        user_message: str,
        assistant_message: str,
        tool_calls: list[dict],
        current_task: str,
    ) -> None:
        """Update memory state from a conversation turn."""
```

## Further Reading

- [Memory System](memory.md) — Overview of all memory components
- [Semantic Cache](memory.md#semantic-cache) — Similarity-based caching
- [Agent Configuration](agent.md) — Configuring agents with ACC
