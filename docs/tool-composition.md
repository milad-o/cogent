# Tool Composition Patterns

**Research basis:** [arXiv:2601.11327](https://arxiv.org/abs/2601.11327) - Tools are the primary value driver in agentic systems. Composition makes tools exponentially more powerful than isolated execution.

Tool composition is the art of combining multiple tools to solve complex tasks. cogent provides flexible execution strategies that enable powerful composition patterns while maintaining simplicity.

## Quick Start

```python
from cogent import Agent
from cogent.executors import NativeExecutor, SequentialExecutor
from cogent.tools import tool

# Pattern 1: Parallel execution (default)
agent = Agent(model="gpt-4o-mini", tools=[tool1, tool2, tool3])
result = await agent.arun("Use all three tools")  # Runs in parallel

# Pattern 2: Sequential execution
agent = Agent(model="gpt-4o-mini", tools=[step1, step2, step3])
executor = SequentialExecutor(agent)
result = await executor.execute("Do step1, then step2, then step3")  # Runs sequentially
```

## Execution Strategies

cogent provides two built-in execution strategies:

| Strategy | Tool Execution | Use Case |
|----------|----------------|----------|
| **NativeExecutor** (default) | Parallel when LLM batches tools<br>Sequential when LLM calls one at a time | Most tasks - flexible, performant |
| **SequentialExecutor** | Always sequential, even if LLM batches | Strict ordering, debugging, stateful tools |

**Key insight:** With NativeExecutor, the **LLM decides** whether to call tools in parallel (batched in one turn) or sequentially (one per turn). SequentialExecutor **forces** sequential execution regardless of what the LLM wants.

### NativeExecutor (Default)

**Parallel tool execution when LLM requests multiple tools** - Maximum performance.

```python
from cogent import Agent

agent = Agent(
    model="gpt-4o-mini",
    tools=[fetch_weather, fetch_news, fetch_stock]
)

# If LLM requests multiple tools in one turn, they run concurrently
result = await agent.run("Get weather, news, and stock data")
```

**Execution behavior:**
- **Parallel**: When LLM requests multiple tools in one turn (e.g., `fetch_weather`, `fetch_news`, `fetch_stock` all at once)
- **Sequential**: When LLM naturally calls tools one at a time across multiple turns
- **Flexible**: LLM decides based on task requirements and prompt

**When to use:**
- Default choice for most tasks
- Tools may or may not be independent
- Want maximum performance when tools can run parallel
- Trust LLM to determine execution order

**Performance:**
- 3 tools in one turn × 0.5s each = **~0.5s** (parallel execution via asyncio.gather)
- 3 tools across 3 turns × 0.5s each = **~1.5s + LLM overhead** (sequential, LLM decides)

**Features:**
- Direct asyncio loop (no graph overhead)
- Configurable concurrency limit (default: 20 concurrent tools)
- LLM resilience with automatic retry for rate limits
- Tool call batching (up to 50 tools per LLM turn)

### SequentialExecutor

**Forces sequential tool execution** - One tool at a time, always.

```python
from cogent import Agent
from cogent.executors import SequentialExecutor

agent = Agent(
    model="gpt-4o-mini",
    tools=[search_company, analyze_sentiment, generate_report]
)

executor = SequentialExecutor(agent)
result = await executor.execute("""
    1. Search for company info
    2. Analyze sentiment of what you found
    3. Generate report using both pieces
""")
```

**Execution behavior:**
- **Always sequential**: Even if LLM requests multiple tools, they execute one at a time
- **Deterministic order**: Tools execute in the order LLM requested them
- **Forced serialization**: No parallelism, even for independent tools

**When to use:**
- **Strict ordering required**: When you MUST guarantee tools execute in a specific sequence
- **Debugging**: To see exactly what order tools run in
- **Stateful tools**: Tools that modify shared state (databases, files, etc.)
- **Resource constraints**: External API has strict rate limits

**Difference from NativeExecutor:**
```python
# NativeExecutor - LLM requests 3 tools → all 3 run in parallel
# SequentialExecutor - LLM requests 3 tools → they run 1, 2, 3 sequentially

# If LLM naturally calls one tool per turn, both executors behave the same!
```

**Trade-offs:**
- Slower (no parallelism)
- More predictable execution order
- Easier to debug
- Better for stateful operations

## Composition Patterns

### Pattern 1: Parallel Execution

**Independent tools that can run concurrently.**

```python
from cogent import Agent
from cogent.tools import tool

@tool
async def fetch_weather(city: str) -> str:
    """Fetch weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@tool
async def fetch_news(topic: str) -> str:
    """Fetch news about a topic."""
    return f"Latest {topic} news: AI breakthroughs"

@tool
async def fetch_stock(symbol: str) -> str:
    """Fetch stock price."""
    return f"Stock {symbol}: $150.25 (+2.3%)"

agent = Agent(
    model="gpt-4o-mini",
    tools=[fetch_weather, fetch_news, fetch_stock]
)

# NativeExecutor (default) runs these in parallel
result = await agent.arun("""
    Get me:
    1. Weather in San Francisco
    2. Latest AI news  
    3. Stock price for AAPL
""")
```

**Result:** All three tools execute concurrently (~0.5s total).

### Pattern 2: Sequential Execution

**Dependent tools that need ordered execution.**

```python
from cogent import Agent
from cogent.executors import SequentialExecutor
from cogent.tools import tool

@tool
async def search_company(name: str) -> str:
    """Search for company information."""
    return f"{name} - Tech company, founded 2020, AI specialization"

@tool
async def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text."""
    return "Sentiment: POSITIVE (innovative, strong fundamentals)"

@tool
async def generate_report(company_info: str, sentiment: str) -> str:
    """Generate investment report."""
    return f"""
Investment Report:
- Company: {company_info}
- Sentiment: {sentiment}
- Recommendation: BUY
"""

agent = Agent(
    name="sequential_demo",
    model="gpt-4o-mini",
    tools=[search_company, analyze_sentiment, generate_report]
)

# Option 1: Use SequentialExecutor to FORCE sequential execution
executor = SequentialExecutor(agent)
result = await executor.execute("""
    Research ACME Corp:
    1. Search for company info
    2. Analyze sentiment
    3. Generate report
    
    Execute in strict order.
""")

# Option 2: Use NativeExecutor with clear prompting (LLM will likely call sequentially)
result = await agent.run("""
    Research ACME Corp step by step:
    1. First search for company info
    2. Then analyze the sentiment of what you found
    3. Finally generate a report using both pieces
    
    Do these one at a time, in order.
""")
```

**Result:** 
- **SequentialExecutor**: Tools execute sequentially: search → analyze → report (forced)
- **NativeExecutor with prompt**: LLM likely calls tools one per turn sequentially (natural)

**When to use each:**
- **SequentialExecutor**: Need to guarantee sequential execution regardless of LLM behavior
- **NativeExecutor + prompt**: Trust LLM to decide, get parallel execution when possible

### Pattern 3: Conditional Logic

**If/else branching based on tool results.**

```python
from cogent import Agent
from cogent.tools import tool

@tool
async def check_inventory(product: str) -> str:
    """Check product inventory."""
    if "laptop" in product.lower():
        return f"{product}: OUT OF STOCK (restock: 2024-03-15)"
    return f"{product}: IN STOCK (23 units available)"

@tool
async def backorder(product: str) -> str:
    """Place item on backorder."""
    return f"Backorder created for {product}"

@tool
async def complete_purchase(product: str) -> str:
    """Complete purchase."""
    return f"Purchase complete: {product} - Total: $999"

agent = Agent(
    model="gpt-4o-mini",
    tools=[check_inventory, backorder, complete_purchase]
)

result = await agent.arun("""
    Try to purchase a 'Gaming Laptop'.
    
    First check inventory. Then:
    - If IN STOCK: complete the purchase
    - If OUT OF STOCK: place a backorder
""")
```

**Result:** Agent uses if/else logic:
- Calls `check_inventory`
- Sees "OUT OF STOCK"
- Calls `backorder` (not `complete_purchase`)

### Pattern 4: Error Recovery

**Fallback chains for resilient execution.**

```python
from cogent import Agent
from cogent.tools import tool

@tool
async def premium_api(query: str) -> str:
    """Premium API (might fail)."""
    raise RuntimeError("Quota exceeded")

@tool
async def fallback_api(query: str) -> str:
    """Fallback API (more reliable)."""
    return f"Fallback result for {query}"

@tool
async def cached_data(query: str) -> str:
    """Cached data (always works)."""
    return f"Cached result for {query}"

agent = Agent(
    model="gpt-4o-mini",
    tools=[premium_api, fallback_api, cached_data],
    system_prompt="""
Strategy: Try premium_api first. If it fails, use fallback_api.
If that fails too, use cached_data as last resort.
"""
)

result = await agent.arun("Get data about quantum computing")
```

**Result:** Agent tries premium_api → fails → uses fallback_api.

### Pattern 5: Mixed Strategy

**Parallel data gathering + sequential processing.**

```python
from cogent import Agent

# Phase 1: Parallel data gathering
# Phase 2: Sequential report generation

agent = Agent(
    model="gpt-4o-mini",
    tools=[
        fetch_weather,  # Parallel
        fetch_news,     # Parallel
        fetch_stock,    # Parallel
        generate_report # Sequential (after gathering)
    ]
)

result = await agent.arun("""
    Create a market summary:
    
    1. Gather data in parallel:
       - Weather in NYC
       - Latest tech news
       - TSLA stock price
    
    2. Generate summary report using all data
""")
```

**Result:**
- **Phase 1:** Three tools run in parallel (~0.5s)
- **Phase 2:** Report generated with all results (~0.5s)
- **Total:** ~1.0s (vs ~2.0s fully sequential)

### Pattern 6: Real-World Composition

**Combining capabilities with custom tools.**

```python
from cogent import Agent
from cogent.capabilities import WebSearch
from cogent.tools import tool

@tool
async def summarize_findings(research: str) -> str:
    """Create executive summary."""
    return f"""
EXECUTIVE SUMMARY:
- Research: {len(research)} chars
- Key topics: AI, productivity
- Recommendation: Implement AI tools
"""

agent = Agent(
    model="gpt-4o-mini",
    capabilities=[WebSearch()],
    tools=[summarize_findings]
)

result = await agent.arun("""
    Research top 3 AI productivity tools of 2024.
    Then create an executive summary.
""")
```

**Result:**
- WebSearch performs parallel searches
- `summarize_findings` processes results
- Clean separation of concerns

## Configuration

### NativeExecutor Configuration

```python
from cogent.executors import NativeExecutor

executor = NativeExecutor(
    agent,
    max_tool_calls_per_turn=50,  # Max tools per LLM response
    max_concurrent_tools=20,      # Max concurrent tool execution
    resilience=True               # Enable LLM retry on rate limits
)
```

**Parameters:**
- `max_tool_calls_per_turn` (default: 50)
  - Prevents runaway tool execution
  - LLM can request up to 50 tools in one response
  
- `max_concurrent_tools` (default: 20)
  - Controls asyncio.Semaphore for tool execution
  - Prevents overwhelming external APIs
  
- `resilience` (default: True)
  - Automatic retry for LLM rate limits
  - Uses RetryPolicy.aggressive() by default
  - Can be configured via agent's resilience_config

### SequentialExecutor Configuration

```python
from cogent.executors import SequentialExecutor

executor = SequentialExecutor(
    agent,
    max_tool_calls_per_turn=50,  # Same as NativeExecutor
    max_concurrent_tools=1,       # Always 1 (sequential)
    resilience=True               # LLM retry enabled
)
```

SequentialExecutor inherits from NativeExecutor but forces tools to run one at a time.

## Factory Pattern

Create executors using the factory:

```python
from cogent.executors import create_executor, ExecutionStrategy

# Parallel (default)
executor = create_executor(agent, ExecutionStrategy.NATIVE)

# Sequential
executor = create_executor(agent, ExecutionStrategy.SEQUENTIAL)
```

## Standalone Execution

For quick tasks without creating an Agent:

```python
from cogent.executors import run
from cogent.tools import tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"

# Standalone execution - no Agent needed
result = await run(
    "Search for Python tutorials",
    tools=[search],
    model="gpt-4o-mini",
    max_iterations=25
)
```

**Benefits:**
- No Agent class overhead
- Fastest execution path
- Great for simple tasks

## Performance Tips

### 1. Use Parallel Execution by Default

```python
# ✅ GOOD - Default NativeExecutor
agent = Agent(model="gpt-4o-mini", tools=[...])
result = await agent.arun(task)
```

**Why:** Most tools are independent. Parallel execution is 2-3× faster.

### 2. Only Use Sequential When Needed

```python
# ❌ BAD - Unnecessary sequential
executor = SequentialExecutor(agent)
result = await executor.execute("Get weather and news")  # Independent!

# ✅ GOOD - Sequential only when dependent
executor = SequentialExecutor(agent)
result = await executor.execute("Search, then analyze, then report")
```

**Why:** Sequential execution is slower. Only use for dependencies.

### 3. Configure Concurrency Limits

```python
# ✅ GOOD - Tune for your use case
executor = NativeExecutor(
    agent,
    max_concurrent_tools=10  # Lower if external API has rate limits
)
```

**Why:** Prevents overwhelming external services with 20 concurrent calls.

### 4. Use Resilience for Production

```python
# ✅ GOOD - Production config
executor = NativeExecutor(
    agent,
    resilience=True  # Automatic LLM retry
)
```

**Why:** Handles transient LLM failures (rate limits, timeouts) automatically.

## Error Handling

### LLM Errors

Both executors include automatic retry for LLM failures:

```python
from cogent.agent.resilience import RetryPolicy

# Configure via agent's resilience_config
agent = Agent(
    model="gpt-4o-mini",
    tools=[...],
    # Resilience applied at executor level
)

executor = NativeExecutor(agent, resilience=True)
```

**Automatic retry for:**
- Rate limit errors (429)
- Timeout errors
- Transient API failures

**Retry policy:**
- Max 5 attempts
- Exponential backoff: 1s, 2s, 4s, 8s, 16s
- Total timeout: 120s

### Tool Errors

Tool errors are captured and returned as ToolMessage:

```python
@tool
async def risky_operation() -> str:
    """Might fail."""
    raise ValueError("Something went wrong")

# Agent receives error message and can retry or use fallback
```

**Best practice:** Let the agent handle tool errors via fallback patterns (see Pattern 4).

## Testing

### Testing Parallel Execution

```python
import pytest
from cogent import Agent
from cogent.tools import tool

@tool
async def mock_tool(value: str) -> str:
    return f"Processed: {value}"

@pytest.mark.asyncio
async def test_parallel_execution():
    agent = Agent(
        model="gpt-4o-mini",
        tools=[mock_tool]
    )
    
    result = await agent.arun("Process A, B, and C in parallel")
    
    assert "Processed: A" in result
    assert "Processed: B" in result
    assert "Processed: C" in result
```

### Testing Sequential Execution

```python
@pytest.mark.asyncio
async def test_sequential_execution():
    call_order = []
    
    @tool
    async def step1() -> str:
        call_order.append(1)
        return "Step 1 done"
    
    @tool
    async def step2() -> str:
        call_order.append(2)
        return "Step 2 done"
    
    agent = Agent(model="gpt-4o-mini", tools=[step1, step2])
    executor = SequentialExecutor(agent)
    
    await executor.execute("Do step1 then step2")
    
    assert call_order == [1, 2]  # Sequential order preserved
```

## Best Practices

### ✅ DO

1. **Use parallel by default** - NativeExecutor is faster for most tasks
2. **Use prompts to guide execution** - Tell LLM when order matters ("do step by step")
3. **Configure concurrency** - Tune `max_concurrent_tools` for external APIs
4. **Enable resilience** - Use `resilience=True` in production
5. **Test composition** - Unit test your tool chains
6. **Use fallbacks** - Implement error recovery patterns
7. **Let LLM decide** - Trust the model to determine parallel vs sequential when possible

### ❌ DON'T

1. **Don't use SequentialExecutor unnecessarily** - Only when you need guaranteed sequential execution
2. **Don't hardcode execution order** - Let the LLM decide when possible via prompts
3. **Don't ignore tool errors** - Implement fallback patterns
4. **Don't skip concurrency limits** - Can overwhelm external services
5. **Don't disable resilience in production** - Transient failures are common
6. **Don't assume parallel = faster** - If LLM calls tools one at a time, both executors are the same speed

## Examples

See [examples/advanced/tool_composition.py](../examples/advanced/tool_composition.py) for complete runnable examples of all patterns.

## Further Reading

- [Executors API Reference](./executors.md)
- [Tools Guide](./tools.md)
- [Error Handling](./resilience.md)
- [arXiv:2601.11327](https://arxiv.org/abs/2601.11327) - Research on tool composition
