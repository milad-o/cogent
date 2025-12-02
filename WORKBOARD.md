# Spawning Agents - Workboard

## Vision
Supervisor agents can spawn ephemeral specialist agents on-demand for parallel task execution.

## Core Concepts

### SpawningConfig
```python
SpawningConfig(
    max_concurrent=10,      # Max parallel spawns
    max_depth=3,            # Recursive spawn depth
    max_total_spawns=50,    # Total spawns per task
    ephemeral=True,         # Auto-cleanup after task
    inherit_tools=False,    # Spawns inherit parent tools
    inherit_model=True,     # Spawns use parent's model
)
```

### AgentSpec (Optional - for predefined specialists)
```python
AgentSpec(
    role="researcher",
    system_prompt="You are a thorough researcher...",
    tools=[web_search, fetch_page],  # Specific tools for this role
    model=None,  # Use parent's model if None
)
```

### Dynamic Role Creation
If no AgentSpec provided, LLM dynamically creates:
- Role name
- System prompt for the specialist
- Selects from available tools

## Tasks

- [x] 1. Create `AgentSpec` dataclass
- [x] 2. Create `SpawningConfig` dataclass  
- [x] 3. Create `SpawnManager` class
- [x] 4. Add `spawn_agent` tool (LLM-callable)
- [x] 5. Add spawning support to Agent
- [x] 6. Add spawn events to observability
- [x] 7. Add `parallel_map` method
- [x] 8. Add `MockChatModel` for testing
- [x] 9. Write tests (22 passing)
- [ ] 10. Create example
- [x] 11. Export from __init__.py

## API Design

```python
# Option 1: Predefined specialists
supervisor = Agent(
    name="Supervisor",
    model=model,
    spawning=SpawningConfig(
        max_concurrent=10,
        specs={
            "researcher": AgentSpec(
                system_prompt="You research topics thoroughly",
                tools=[web_search],
            ),
            "coder": AgentSpec(
                system_prompt="You write clean code",
                tools=[filesystem, sandbox],
            ),
        },
    ),
)

# Option 2: Dynamic (LLM decides roles)
supervisor = Agent(
    name="Supervisor", 
    model=model,
    spawning=SpawningConfig(
        max_concurrent=10,
        available_tools=[web_search, filesystem],  # Pool for spawns
    ),
)

# LLM spawns via tool:
# spawn_agent(role="researcher", task="Research X", tools=["web_search"])
```

## Events
- `AGENT_SPAWNED` - New agent spawned
- `AGENT_SPAWN_COMPLETED` - Spawned agent finished
- `AGENT_SPAWN_FAILED` - Spawned agent failed
- `AGENT_DESPAWNED` - Agent cleaned up

## File Structure
```
src/agenticflow/agent/
├── spawning.py      # SpawningConfig, AgentSpec, SpawnedAgent
├── base.py          # Add spawning support to Agent
```
