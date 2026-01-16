# Migration Guide: v1.7.0 → v1.8.0

## Overview

Version 1.8.0 reorganizes the module structure for better separation of concerns. **No breaking changes** — backward compatibility is maintained through compatibility modules.

## What Changed

### Module Relocations

Several modules were moved to better reflect their architectural role:

| Old Location | New Location | What |
|-------------|--------------|------|
| `agenticflow.context` | `agenticflow.core.context` | `RunContext`, `EMPTY_CONTEXT` |
| `agenticflow.flow.utils` | `agenticflow.core.utils` | `IdempotencyGuard`, `RetryBudget`, `emit_later`, `jittered_delay`, `Stopwatch` |
| `agenticflow.flow.threading` | *(inlined into `flow/reactive.py`)* | `thread_id_from_data()` |

### Removed Modules

- **`agenticflow.core.models`** — Unused deprecated module (replaced by `agenticflow.models`)

## Migration Path

### Option 1: Update Imports (Recommended)

Update your imports to use the new locations:

```python
# Old (still works via compatibility layer)
from agenticflow.context import RunContext

# New (recommended)
from agenticflow.core.context import RunContext
```

```python
# Old
from agenticflow.flow.utils import IdempotencyGuard, RetryBudget

# New
from agenticflow.core.utils import IdempotencyGuard, RetryBudget
# or
from agenticflow.core import IdempotencyGuard, RetryBudget
```

### Option 2: Use Main Package Exports (Easiest)

The main package still exports everything:

```python
# This continues to work
from agenticflow import RunContext, EMPTY_CONTEXT
```

### Option 3: No Changes Required

If you import from `agenticflow.reactive`, a compatibility module maintains backward compatibility:

```python
# This still works (via reactive.py compatibility module)
from agenticflow.reactive import ReactiveFlow, react_to, skill
```

## Affected Imports

If you have any of these imports, consider updating them:

```python
# Context imports
from agenticflow.context import RunContext  # → agenticflow.core.context

# Utility imports (if importing directly)
from agenticflow.flow.utils import IdempotencyGuard  # → agenticflow.core.utils
from agenticflow.flow.threading import thread_id_from_data  # → use ReactiveFlow.thread_by_data()
```

## New Exports from `core/`

The `core` module now exports reactive utilities:

```python
from agenticflow.core import (
    # Context
    RunContext,
    EMPTY_CONTEXT,
    
    # Reactive utilities
    IdempotencyGuard,
    RetryBudget,
    emit_later,
    jittered_delay,
    Stopwatch,
    
    # Existing core exports
    generate_id,
    now_utc,
    TaskStatus,
    AgentStatus,
    # ... etc
)
```

## Architecture Clarification

The module structure now clearly reflects architectural layers:

### `core/` — Foundational Primitives
- Enums (`TaskStatus`, `AgentStatus`, etc.)
- Messages (`BaseMessage`, `HumanMessage`, etc.)
- Utilities (`generate_id`, `now_utc`, etc.)
- **NEW**: Context (`RunContext`)
- **NEW**: Reactive primitives (`IdempotencyGuard`, `RetryBudget`, etc.)

### `flow/` — Event-Driven Orchestration
- `ReactiveFlow` — Event-driven multi-agent flows
- `triggers` — Event reaction system
- `skills` — Event-triggered behaviors
- `patterns` — Flow patterns (pipeline, supervisor, mesh)
- Flow-specific logic only

### `agent/` — Agent Core
- Agent class and configuration
- Tool execution
- Memory management
- Streaming and output handling

## Testing

All 1,333 tests pass in v1.8.0. If you encounter issues:

1. Check import statements match new locations
2. Use compatibility imports from main package
3. File an issue on GitHub if you find breaking changes

## Questions?

If you have questions about the migration or encounter issues, please:
- Review the [CHANGELOG.md](CHANGELOG.md) for detailed changes
- Check the [documentation](docs/)
- Open an issue on GitHub
