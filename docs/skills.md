# Skills

Skills are event-triggered behavioral specializations that dynamically modify an agent's context, prompts, and tools based on incoming events.

Unlike tools (which are code-based functions), skills are prompt-driven specializations that inject temporary capabilities into agents when matching event patterns occur.

## Overview

**Skills provide:**
- Dynamic prompt injection based on events
- Temporary tool availability during specific contexts
- Priority-based layering of multiple skills
- Conditional activation with event filters
- Context enrichment for domain-specific knowledge

**Use Skills when you need:**
- Role-based expertise that activates on demand
- Context-aware behavior changes
- Temporary tool access for specific scenarios
- Event-driven prompt engineering

---

## Creating Skills

Use the `skill()` function to define event-triggered specializations:

```python
from cogent import skill, tool
from cogent.events import has_data

@tool
def run_python(code: str) -> str:
    """Execute Python code and return output."""
    # Implementation here
    return "Output: ..."

@tool
def lint_code(code: str) -> str:
    """Lint Python code and report issues."""
    return "✓ Linting passed"

# Define a Python expert skill
python_skill = skill(
    "python_expert",
    on="code.write",
    when=has_data("language", "python"),
    prompt="""You are a Python expert. Follow these guidelines:
    - Use type hints on all functions
    - Follow PEP 8 conventions
    - Include docstrings with Args/Returns
    - Prefer composition over inheritance
    - Use modern Python 3.13+ features""",
    tools=[run_python, lint_code],
    priority=10,
)

# Define a debugging skill
debugger_skill = skill(
    "debugger",
    on="error.*",
    prompt="""You are a debugging specialist. Analyze errors systematically:
    1. Read logs to understand what happened
    2. Inspect relevant variables
    3. Form a hypothesis about root cause
    4. Propose fix with confidence level""",
    tools=[read_logs, inspect_variables],
    priority=20,
)
```

---

## Skill Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique skill identifier |
| `on` | `str \| EventPattern` | Event pattern to match (e.g., `"code.*"`, `"error.network"`) |
| `when` | `Callable[[Event], bool]` | Optional condition filter for fine-grained control |
| `prompt` | `str` | Prompt text injected into agent context when skill activates |
| `tools` | `list[Callable]` | Tools temporarily added to agent during skill activation |
| `context_enricher` | `Callable[[Event, dict], dict]` | Function to enrich context with dynamic data |
| `priority` | `int` | Higher priority skills apply first (default: 0) |

---

## Registering Skills

Skills are registered on a `Flow` and automatically activate when matching events occur:

```python
from cogent import Flow, Agent, react_to

# Create flow
flow = Flow()

# Register skills
flow.register_skill(python_skill)
flow.register_skill(debugger_skill)

# Register agents that react to events
coder = Agent(
    name="coder",
    model="gpt4",
    system_prompt="You are a helpful coding assistant.",
)
flow.register(coder, [react_to("code.*")])

debugger = Agent(
    name="debugger",
    model="gpt4",
    system_prompt="You are a debugging expert.",
)
flow.register(debugger, [react_to("error.*")])

# When events fire, matching skills inject prompts/tools automatically
result = await flow.run(
    "Write a fibonacci function",
    initial_event="code.write",
    initial_data={"language": "python"},
)
# The coder agent receives python_skill prompt and tools
```

---

## Event Pattern Matching

Skills activate when their `on` pattern matches the event type:

```python
# Exact match
skill("exact", on="code.write", ...)

# Wildcard match (all subtypes)
skill("wildcard", on="code.*", ...)

# Root wildcard (all events)
skill("global", on="*", ...)
```

### Conditional Activation

Use `when` parameter for fine-grained control:

```python
from cogent.events import has_data, matches

# Activate only for Python code
python_skill = skill(
    "python_expert",
    on="code.write",
    when=has_data("language", "python"),
    ...
)

# Activate only for high-priority errors
critical_skill = skill(
    "critical_handler",
    on="error.*",
    when=lambda e: e.data.get("severity") == "critical",
    ...
)

# Combine multiple conditions
advanced_skill = skill(
    "advanced",
    on="task.*",
    when=lambda e: e.data.get("priority") > 5 and e.data.get("assigned"),
    ...
)
```

---

## Context Enrichers

Context enrichers allow you to inject dynamic data into the agent's execution context based on the triggering event:

```python
def enrich_with_history(event, context):
    """Add ticket history to context."""
    ticket_id = event.data.get("ticket_id")
    context["history"] = fetch_ticket_history(ticket_id)
    context["related_tickets"] = find_related_tickets(ticket_id)
    return context

support_skill = skill(
    "support",
    on="ticket.*",
    prompt="You are a support specialist. Use the ticket history to provide context-aware assistance.",
    context_enricher=enrich_with_history,
)
```

```python
def enrich_with_codebase(event, context):
    """Add relevant code files to context."""
    repo = event.data.get("repository")
    file_path = event.data.get("file")
    context["codebase"] = load_relevant_files(repo, file_path)
    context["dependencies"] = get_dependencies(file_path)
    return context

code_review_skill = skill(
    "code_reviewer",
    on="code.review",
    prompt="Review code with attention to dependencies and related files.",
    context_enricher=enrich_with_codebase,
    priority=15,
)
```

---

## Priority and Layering

When multiple skills match the same event, they are applied in priority order (highest first):

```python
# Base coding skill (low priority)
base_skill = skill(
    "base_coder",
    on="code.*",
    prompt="Write clean, maintainable code.",
    priority=1,
)

# Language-specific skill (medium priority)
python_skill = skill(
    "python_expert",
    on="code.write",
    when=has_data("language", "python"),
    prompt="Use Python best practices and type hints.",
    priority=10,
)

# Framework-specific skill (high priority)
fastapi_skill = skill(
    "fastapi_expert",
    on="code.write",
    when=has_data("framework", "fastapi"),
    prompt="Follow FastAPI patterns: Pydantic models, dependency injection, async.",
    priority=20,
)

# For a code.write event with framework=fastapi and language=python:
# All three skills activate and prompts are injected in order:
# 1. fastapi_skill (priority 20)
# 2. python_skill (priority 10)
# 3. base_skill (priority 1)
```

---

## Skill Composition Patterns

### Domain Expert Skills

Create specialized knowledge domains:

```python
security_skill = skill(
    "security_expert",
    on="code.review",
    when=has_data("check_security", True),
    prompt="""Review code for security vulnerabilities:
    - SQL injection risks
    - XSS vulnerabilities
    - Authentication/authorization flaws
    - Sensitive data exposure""",
    tools=[security_scanner, vulnerability_checker],
)

performance_skill = skill(
    "performance_expert",
    on="code.optimize",
    prompt="""Optimize code for performance:
    - Identify bottlenecks
    - Suggest algorithm improvements
    - Recommend caching strategies
    - Analyze time/space complexity""",
    tools=[profiler, benchmark_runner],
)
```

### Workflow Stage Skills

Skills for different stages of a process:

```python
planning_skill = skill(
    "planner",
    on="task.plan",
    prompt="Break down the task into actionable steps with clear success criteria.",
    priority=10,
)

implementation_skill = skill(
    "implementer",
    on="task.execute",
    prompt="Implement the planned steps systematically, testing as you go.",
    tools=[execute_code, run_tests],
    priority=10,
)

review_skill = skill(
    "reviewer",
    on="task.review",
    prompt="Review implementation against requirements, check edge cases.",
    tools=[lint_checker, test_runner],
    priority=10,
)
```

### Contextual Tool Access

Provide tools only when needed:

```python
# Database tools only during data operations
db_skill = skill(
    "database_access",
    on="data.*",
    prompt="You have database access. Query efficiently and handle transactions properly.",
    tools=[query_db, update_db, begin_transaction, rollback],
)

# External API tools only during integration work
api_skill = skill(
    "api_integration",
    on="integration.*",
    prompt="You can make external API calls. Handle rate limits and errors gracefully.",
    tools=[call_api, check_rate_limit, retry_request],
)
```

---

## Complete Example

Here's a complete example showing skills in action:

```python
from cogent import Agent, Flow, Observer, react_to, skill, tool

# Define tools
@tool
def run_python(code: str) -> str:
    """Execute Python code."""
    # Implementation
    return "Execution result..."

@tool
def lint_code(code: str) -> str:
    """Lint Python code."""
    return "✓ No issues found"

@tool
def read_logs(path: str) -> str:
    """Read log files."""
    return "Log contents..."

@tool
def inspect_vars(var_name: str) -> str:
    """Inspect variable values."""
    return f"{var_name} = ..."

# Define skills
python_skill = skill(
    "python_expert",
    on="code.write",
    when=lambda e: e.data.get("language") == "python",
    prompt="You are a Python expert. Write type-annotated, PEP 8 compliant code.",
    tools=[run_python, lint_code],
    priority=10,
)

debug_skill = skill(
    "debugger",
    on="error.*",
    prompt="You are debugging. Be systematic: read logs → inspect vars → hypothesize → fix.",
    tools=[read_logs, inspect_vars],
    priority=20,
)

# Set up flow
async def main():
    observer = Observer(level="progress")
    flow = Flow(observer=observer)
    
    # Register skills
    flow.register_skill(python_skill)
    flow.register_skill(debug_skill)
    
    # Register agents
    coder = Agent(
        name="coder",
        model="gpt4",
        system_prompt="You are a helpful coding assistant.",
    )
    flow.register(coder, [react_to("code.write")])
    
    debugger_agent = Agent(
        name="debugger_agent",
        model="gpt4",
        system_prompt="You are a debugging expert.",
    )
    flow.register(debugger_agent, [react_to("error.*")])
    
    # Run with code.write event → python_skill activates
    result = await flow.run(
        "Write a function to calculate fibonacci numbers",
        initial_event="code.write",
        initial_data={"language": "python"},
    )
    print(result.output)
    
    # Run with error event → debug_skill activates
    result = await flow.run(
        "Investigate the API connection failure",
        initial_event="error.network",
        initial_data={"endpoint": "/api/v1/users"},
    )
    print(result.output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## Best Practices

### 1. **Keep Skills Focused**
Each skill should have a single, clear purpose:

```python
# ✅ Good: Focused skill
python_skill = skill(
    "python_expert",
    on="code.write",
    prompt="Python best practices: type hints, PEP 8, docstrings.",
    ...
)

# ❌ Bad: Too broad
everything_skill = skill(
    "do_everything",
    on="*",
    prompt="Do everything perfectly in all domains...",
    ...
)
```

### 2. **Use Appropriate Priorities**
Higher priority for more specific skills:

```python
general_skill = skill("general", on="code.*", priority=1)
language_skill = skill("python", on="code.*", priority=10)
framework_skill = skill("fastapi", on="code.*", priority=20)
```

### 3. **Combine When Conditions Wisely**
Use `when` for conditions that can't be expressed in event patterns:

```python
skill(
    "senior_dev",
    on="code.review",
    when=lambda e: e.data.get("complexity") == "high",
    ...
)
```

### 4. **Enrich Context Efficiently**
Don't over-fetch in context enrichers:

```python
# ✅ Good: Fetch only what's needed
def enrich(event, context):
    context["recent_history"] = fetch_last_10_events()
    return context

# ❌ Bad: Fetching too much
def enrich(event, context):
    context["all_history"] = fetch_entire_database()  # Too much!
    return context
```

### 5. **Document Skill Activation**
Use clear names and docstrings:

```python
@dataclass
class PythonExpertSkill:
    """Activates on Python code write events.
    
    Injects Python best practices and provides linting/execution tools.
    """
```

---

## See Also

- **[Flow Documentation](flow.md)** - Complete Flow system guide
- **[Events Documentation](events.md)** - Event patterns and matching
- **[Tools Documentation](tool-building.md)** - Creating and using tools
- **[Examples: skills.py](../examples/advanced/skills.py)** - Working example code

---

## API Reference

### `skill()`

```python
def skill(
    name: str,
    on: str | EventPattern,
    *,
    when: Callable[[Event], bool] | None = None,
    prompt: str | None = None,
    tools: list[Callable] | None = None,
    context_enricher: Callable[[Event, dict], dict] | None = None,
    priority: int = 0,
) -> Skill:
    """Create an event-triggered behavioral specialization.
    
    Args:
        name: Unique identifier for the skill
        on: Event pattern to match (e.g., "code.*", "error.network")
        when: Optional filter function for conditional activation
        prompt: Prompt text injected into agent context
        tools: Tools made available during skill activation
        context_enricher: Function to add dynamic context
        priority: Application order (higher = applied first)
    
    Returns:
        Skill instance ready for registration
    """
```

### `Flow.register_skill()`

```python
def register_skill(self, skill: Skill) -> None:
    """Register a skill on the flow.
    
    Args:
        skill: Skill to register
    """
```
