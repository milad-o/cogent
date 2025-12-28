# Tool Resilience & Recovery

AgenticFlow provides production-grade resilience features that automatically handle transient failures, prevent cascading failures, and enable graceful degradation.

## Quick Start

```python
from agenticflow import Agent
from agenticflow.agent.resilience import ResilienceConfig

# Default: 3 retries with exponential backoff (ENABLED by default)
agent = Agent(name="MyAgent", model=model, tools=[...])

# Aggressive: 5 retries for flaky services
agent = Agent(
    name="ReliableAgent",
    model=model,
    tools=[...],
    resilience=ResilienceConfig.aggressive()
)

# Fast-fail: No retries for time-sensitive operations
agent = Agent(
    name="FastAgent",
    model=model,
    tools=[...],
    resilience=ResilienceConfig.fast_fail()
)
```

## Default Configuration

**Agents have resilience ENABLED by default** with sensible production settings:

- **Max Retries**: 3 attempts per tool call
- **Strategy**: Exponential backoff with jitter
- **Timeout**: 60 seconds per tool call  
- **Circuit Breaker**: Enabled (protects against cascading failures)
- **Failure Learning**: Enabled (learns from patterns)

## When Resilience Applies

### ✅ Direct Tool Calls (`agent.act()`)

The resilience layer automatically retries failed tool calls:

```python
# Resilience applies here - automatic retry with exponential backoff
result = await agent.act(
    tool_name="web_search",
    args={"query": "Python tutorials"},
    use_resilience=True  # Default
)
```

**Behavior**: Tool failures are retried transparently. The LLM never sees transient errors.

### 🤖 LLM-Driven Execution (`agent.run()`)

When the LLM decides which tools to call, resilience **does NOT** apply by default:

```python
# LLM sees tool errors and decides retry strategy
result = await agent.run("Search for Python tutorials")
```

**Behavior**: Tool errors are returned to the LLM, which can decide whether to retry, try a different approach, or give up.

**Why?** This gives the LLM flexibility to adapt its strategy based on errors rather than blindly retrying.

### 🔧 Workflows & Programmatic Usage

Use `.act()` in workflows and scripts to get automatic resilience:

```python
# In a workflow - resilience applies to each tool call
async def data_pipeline(agent: Agent):
    raw_data = await agent.act("fetch_data", {"source": "api"})
    cleaned = await agent.act("clean_data", {"data": raw_data})
    result = await agent.act("analyze_data", {"data": cleaned})
    return result
```

## Resilience Features

### 1. Automatic Retry with Exponential Backoff

```python
from agenticflow.agent.resilience import RetryPolicy, RetryStrategy

policy = RetryPolicy(
    max_retries=5,
    strategy=RetryStrategy.EXPONENTIAL_JITTER,
    base_delay=0.5,  # Start with 0.5s delay
    max_delay=30.0,  # Cap at 30s
    jitter_factor=0.3,  # Add 30% randomness
)

config = ResilienceConfig(retry_policy=policy)
agent = Agent(name="Agent", model=model, resilience=config)
```

**Available Strategies**:
- `EXPONENTIAL_JITTER`: Exponential backoff with random jitter (recommended)
- `EXPONENTIAL`: Pure exponential backoff (2^attempt * base_delay)
- `LINEAR`: Linear increase (attempt * base_delay)
- `FIXED`: Fixed delay between retries
- `NONE`: No backoff (fail fast)

### 2. Circuit Breaker

Prevents cascading failures by temporarily disabling failing tools:

```python
config = ResilienceConfig(
    circuit_breaker_enabled=True,  # Default
    circuit_breaker_config=CircuitBreaker(
        failure_threshold=3,  # Open after 3 failures
        reset_timeout=30.0,   # Wait 30s before testing recovery
    )
)
```

**States**:
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Blocking requests after repeated failures
- **HALF_OPEN**: Testing if service recovered

### 3. Fallback Tool Chains

Gracefully degrade to backup tools when primary fails:

```python
agent = Agent(
    name="Agent",
    model=model,
    tools=[primary_search, backup_search],
    resilience=ResilienceConfig(fallback_enabled=True),
)

# Register fallback chain
agent.config = agent.config.with_fallbacks({
    "primary_search": ["backup_search"]
})
agent._setup_resilience()  # Re-init resilience with new config

# If primary_search fails after retries, backup_search is tried
result = await agent.act("primary_search", {"query": "test"})
```

### 4. Failure Learning

Learn from failures to adapt future behavior:

```python
config = ResilienceConfig(
    learning_enabled=True,  # Default
)
```

The resilience layer tracks:
- Common failure patterns
- Error message patterns (rate limits, server errors, etc.)
- Recovery methods that worked
- Tool reliability over time

## Pre-Built Configurations

### Default / Balanced

```python
ResilienceConfig()  # or ResilienceConfig.balanced()
```

- 3 retries
- Exponential backoff with jitter
- 60s timeout
- Circuit breaker enabled
- Failure learning enabled

**Use for**: General purpose workflows, balanced reliability/latency

### Aggressive

```python
ResilienceConfig.aggressive()
```

- 5 retries
- Exponential backoff with jitter
- 120s timeout
- Circuit breaker enabled
- Failure learning enabled
- Fallback enabled

**Use for**: Flaky external APIs, critical data retrieval, background jobs

### Fast-Fail

```python
ResilienceConfig.fast_fail()
```

- 0 retries
- 10s timeout
- No circuit breaker
- No learning
- No fallbacks

**Use for**: Real-time paths, user-facing operations, testing

## Custom Configuration

Full control over all resilience parameters:

```python
from agenticflow.agent.resilience import (
    ResilienceConfig,
    RetryPolicy,
    RetryStrategy,
    CircuitBreaker,
    RecoveryAction,
)

custom_config = ResilienceConfig(
    retry_policy=RetryPolicy(
        max_retries=4,
        strategy=RetryStrategy.EXPONENTIAL_JITTER,
        base_delay=1.0,
        max_delay=60.0,
        jitter_factor=0.25,
    ),
    circuit_breaker_enabled=True,
    circuit_breaker_config=CircuitBreaker(
        failure_threshold=5,
        success_threshold=2,
        reset_timeout=60.0,
    ),
    fallback_enabled=True,
    learning_enabled=True,
    timeout_seconds=90.0,
    on_failure=RecoveryAction.SKIP,  # or RETRY, FALLBACK, ABORT, ADAPT
)

agent = Agent(name="Agent", model=model, resilience=custom_config)
```

## Per-Tool Configuration

Override resilience settings for specific tools:

```python
config = ResilienceConfig()

# Add per-tool overrides
config.tool_configs["expensive_tool"] = {
    "timeout_seconds": 300.0,  # 5 min timeout
    "retry_policy": RetryPolicy.no_retry(),  # Don't retry
}

config.tool_configs["flaky_api"] = {
    "retry_policy": RetryPolicy.aggressive(),  # More retries
}
```

## Monitoring Resilience

Track retries and recovery with observability:

```python
from agenticflow.observability import Observer, ObservabilityLevel

observer = Observer(
    level=ObservabilityLevel.DEBUG,  # See retry events
    show_timestamps=True,
    show_duration=True,
)
agent.add_observer(observer)

# Output shows retry attempts:
# [12:00:01] 🔄 Retry 1/3 for web_search in 1.2s
# [12:00:03] 🔄 Retry 2/3 for web_search in 2.5s
# [12:00:06] ✅ Recovered via retry
```

## Example: Production API Integration

```python
from agenticflow import Agent
from agenticflow.agent.resilience import ResilienceConfig, RetryPolicy
from agenticflow.tools import tool

@tool
async def external_api_call(query: str) -> dict:
    """Call external API (may have transient failures)."""
    # API implementation
    pass

# Production agent with resilience
agent = Agent(
    name="APIAgent",
    model=model,
    tools=[external_api_call],
    resilience=ResilienceConfig(
        retry_policy=RetryPolicy(
            max_retries=5,
            base_delay=1.0,
            max_delay=30.0,
        ),
        circuit_breaker_enabled=True,
        learning_enabled=True,
        timeout_seconds=120.0,
    ),
)

# Direct call - automatic retry on transient failures
result = await agent.act(
    tool_name="external_api_call",
    args={"query": "production data"},
)
```

## Best Practices

1. **Use `.act()` for reliability**: Direct tool calls get automatic resilience
2. **Choose the right config**: Default for most cases, Aggressive for flaky services, Fast-fail for testing
3. **Monitor with Observer**: Track retry patterns and adjust configuration
4. **Set appropriate timeouts**: Balance between giving tools time and failing fast
5. **Use fallbacks for critical paths**: Define backup tools for important operations
6. **Test resilience**: Verify retry behavior with simulated failures
7. **Circuit breakers prevent cascades**: Let failing services recover instead of hammering them

## Troubleshooting

### Tool keeps failing despite retries

Check:
- Tool signature matches between primary and fallback
- Timeout is sufficient for tool execution
- Error is retryable (not ValueError, TypeError, etc.)
- Circuit breaker hasn't opened

### Resilience not applying

Ensure:
- Using `agent.act()` not `agent.run()` for direct resilience
- `use_resilience=True` (default)
- Agent has resilience config set
- Tool is registered with agent

### Too many retries slowing down

Consider:
- Reduce `max_retries`
- Use `Fast-fail` config for testing
- Check if error is actually retryable
- Increase `base_delay` to space out retries

## See Also

- [tool_resilience.py](./tool_resilience.py) - Complete working examples
- [docs/observability.md](../../docs/observability.md) - Monitoring and tracing
- [docs/tools.md](../../docs/tools.md) - Creating tools
