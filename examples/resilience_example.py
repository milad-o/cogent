#!/usr/bin/env python3
"""
Resilience Example - Intelligent retry, circuit breaker, and fallback for agents.

This example demonstrates AgenticFlow's production-grade resilience features:

1. **Retry with Exponential Backoff**: Automatic retry on transient failures
2. **Circuit Breaker**: Prevent cascading failures by blocking failing tools
3. **Fallback Tools**: Graceful degradation to alternative tools
4. **Failure Learning**: Adapt behavior based on past failures

Agents don't give up easily - they adapt, learn, and recover!
"""

import asyncio
import random
import time

from agenticflow import Agent, AgentConfig, EventBus, ToolRegistry
from agenticflow.agents.resilience import (
    CircuitBreaker,
    CircuitState,
    FailureMemory,
    FallbackRegistry,
    ResilienceConfig,
    RetryPolicy,
    RetryStrategy,
    ToolResilience,
)
from agenticflow.core.enums import AgentRole
from agenticflow.observability.progress import OutputConfig, ProgressTracker


# =============================================================================
# Simulated Tools (some are flaky!)
# =============================================================================

class FlakySearchTool:
    """A search tool that fails randomly (simulates API issues)."""
    
    name = "web_search"
    description = "Search the web for information (flaky - fails sometimes)"
    
    def __init__(self, failure_rate: float = 0.7):
        self.failure_rate = failure_rate
        self.call_count = 0
    
    def invoke(self, args: dict) -> str:
        self.call_count += 1
        query = args.get("query", "")
        
        # Simulate network latency
        time.sleep(random.uniform(0.1, 0.3))
        
        # Fail randomly
        if random.random() < self.failure_rate:
            error_types = [
                ConnectionError("Connection reset by peer"),
                TimeoutError("Request timed out after 30s"),
                RuntimeError("API rate limit exceeded (429)"),
            ]
            raise random.choice(error_types)
        
        return f"Search results for '{query}': 10 articles found about {query}"


class CachedSearchTool:
    """A cached search fallback (always works, but less fresh)."""
    
    name = "cached_search"
    description = "Search cached results (always works, fallback for web_search)"
    
    def invoke(self, args: dict) -> str:
        query = args.get("query", "")
        time.sleep(0.05)  # Fast
        return f"Cached results for '{query}': 5 cached articles (may be outdated)"


class LocalSearchTool:
    """A local search fallback (always works, limited results)."""
    
    name = "local_search"
    description = "Search local documents (always works, limited scope)"
    
    def invoke(self, args: dict) -> str:
        query = args.get("query", "")
        time.sleep(0.02)  # Very fast
        return f"Local results for '{query}': 2 local documents found"


class AlwaysFailsTool:
    """A tool that always fails (for testing circuit breaker)."""
    
    name = "always_fails"
    description = "A tool that always fails"
    
    def invoke(self, args: dict) -> str:
        raise RuntimeError("This tool always fails!")


class ReliableTool:
    """A reliable tool that always works."""
    
    name = "calculate"
    description = "Perform calculations"
    
    def invoke(self, args: dict) -> str:
        expr = args.get("expr", "0")
        try:
            result = eval(expr)  # Simple calculator (don't use eval in production!)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"


# =============================================================================
# Example 1: Basic Retry with Exponential Backoff
# =============================================================================

async def example_basic_retry():
    """Demonstrate basic retry functionality."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Retry with Exponential Backoff")
    print("=" * 70)
    
    # Create tools
    flaky_search = FlakySearchTool(failure_rate=0.6)  # 60% failure rate
    
    # Setup registry
    registry = ToolRegistry()
    registry.register(flaky_search)
    
    # Configure aggressive retry
    resilience_config = ResilienceConfig(
        retry_policy=RetryPolicy(
            max_retries=5,
            strategy=RetryStrategy.EXPONENTIAL_JITTER,
            base_delay=0.5,
            max_delay=5.0,
        ),
        timeout_seconds=10.0,
    )
    
    # Create agent with resilience
    config = AgentConfig(
        name="ResilientResearcher",
        role=AgentRole.WORKER,
        tools=["web_search"],
        resilience_config=resilience_config,
    )
    
    event_bus = EventBus()
    agent = Agent(config=config, event_bus=event_bus, tool_registry=registry)
    
    # Create progress tracker
    tracker = ProgressTracker(OutputConfig.verbose())
    
    # Try the flaky tool with resilience
    tracker.start("Testing retry with exponential backoff")
    
    try:
        result = await agent.act(
            "web_search",
            {"query": "Python async programming"},
            tracker=tracker,
        )
        tracker.update(f"Success! Result: {result[:80]}...")
        tracker.complete()
        print(f"\n✅ Succeeded after {flaky_search.call_count} attempts")
    except Exception as e:
        tracker.error(str(e))
        print(f"\n❌ Failed after {flaky_search.call_count} attempts: {e}")


# =============================================================================
# Example 2: Fallback Tools
# =============================================================================

async def example_fallback_tools():
    """Demonstrate fallback tool chain."""
    print("\n" + "=" * 70)
    print("Example 2: Fallback Tool Chain")
    print("=" * 70)
    
    # Create tools
    flaky_search = FlakySearchTool(failure_rate=0.9)  # Very flaky!
    cached_search = CachedSearchTool()
    local_search = LocalSearchTool()
    
    # Setup registry
    registry = ToolRegistry()
    registry.register(flaky_search)
    registry.register(cached_search)
    registry.register(local_search)
    
    # Configure with fallbacks
    resilience_config = ResilienceConfig(
        retry_policy=RetryPolicy(max_retries=2),  # Try twice before fallback
        fallback_enabled=True,
        timeout_seconds=5.0,
    )
    
    # Create agent with fallbacks
    config = AgentConfig(
        name="FallbackResearcher",
        role=AgentRole.WORKER,
        tools=["web_search", "cached_search", "local_search"],
        resilience_config=resilience_config,
        fallback_tools={
            "web_search": ["cached_search", "local_search"],
        },
    )
    
    event_bus = EventBus()
    agent = Agent(config=config, event_bus=event_bus, tool_registry=registry)
    
    # Create progress tracker
    tracker = ProgressTracker(OutputConfig.verbose())
    
    # Try the tool (will likely fallback)
    tracker.start("Testing fallback chain: web_search → cached_search → local_search")
    
    try:
        result = await agent.act(
            "web_search",
            {"query": "circuit breaker pattern"},
            tracker=tracker,
        )
        tracker.update(f"Result: {result}")
        tracker.complete()
        print(f"\n✅ Got result (via fallback if needed)")
    except Exception as e:
        tracker.error(str(e))
        print(f"\n❌ All fallbacks exhausted: {e}")


# =============================================================================
# Example 3: Circuit Breaker
# =============================================================================

async def example_circuit_breaker():
    """Demonstrate circuit breaker pattern."""
    print("\n" + "=" * 70)
    print("Example 3: Circuit Breaker Pattern")
    print("=" * 70)
    
    # Create a tool that always fails
    always_fails = AlwaysFailsTool()
    cached_search = CachedSearchTool()
    
    # Setup registry
    registry = ToolRegistry()
    registry.register(always_fails)
    registry.register(cached_search)
    
    # Configure with circuit breaker
    resilience_config = ResilienceConfig(
        retry_policy=RetryPolicy(max_retries=1),
        circuit_breaker_enabled=True,
        circuit_breaker_config=CircuitBreaker(
            failure_threshold=3,  # Open after 3 failures
            reset_timeout=5.0,    # Try again after 5 seconds
        ),
        fallback_enabled=True,
        timeout_seconds=5.0,
    )
    
    config = AgentConfig(
        name="CircuitBreakerDemo",
        role=AgentRole.WORKER,
        tools=["always_fails", "cached_search"],
        resilience_config=resilience_config,
        fallback_tools={
            "always_fails": ["cached_search"],
        },
    )
    
    event_bus = EventBus()
    agent = Agent(config=config, event_bus=event_bus, tool_registry=registry)
    
    print("\nCalling failing tool multiple times to trigger circuit breaker...\n")
    
    for i in range(6):
        print(f"\n--- Attempt {i + 1} ---")
        
        # Check circuit status before call
        status = agent.get_circuit_status("always_fails")
        print(f"Circuit state: {status.get('state', 'unknown')}")
        
        try:
            result = await agent.act(
                "always_fails",
                {"input": f"test {i}"},
            )
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        await asyncio.sleep(0.5)
    
    # Show final circuit status
    print("\n--- Final Circuit Status ---")
    all_status = agent.get_circuit_status()
    for tool_name, status in all_status.items():
        print(f"  {tool_name}: {status}")
    
    # Reset circuit
    print("\n--- Resetting Circuit ---")
    agent.reset_circuit("always_fails")
    print(f"Circuit after reset: {agent.get_circuit_status('always_fails')}")


# =============================================================================
# Example 4: Failure Learning
# =============================================================================

async def example_failure_learning():
    """Demonstrate failure learning and suggestions."""
    print("\n" + "=" * 70)
    print("Example 4: Failure Learning and Suggestions")
    print("=" * 70)
    
    # Create failure memory directly to demonstrate learning
    memory = FailureMemory()
    
    # Simulate some failures
    print("\nSimulating tool failures...\n")
    
    failures = [
        ("web_search", TimeoutError("Request timed out"), {"query": "test1"}),
        ("web_search", TimeoutError("Request timed out"), {"query": "test2"}),
        ("web_search", ConnectionError("Connection refused"), {"query": "test3"}),
        ("api_call", RuntimeError("Rate limit exceeded"), {"endpoint": "/data"}),
        ("api_call", RuntimeError("Rate limit exceeded"), {"endpoint": "/users"}),
        ("api_call", RuntimeError("Rate limit exceeded"), {"endpoint": "/items"}),
    ]
    
    for tool, error, args in failures:
        memory.record_failure(tool, error, args)
        print(f"  Recorded failure: {tool} - {type(error).__name__}")
    
    # Record some successes too
    memory.record_success("web_search")
    memory.record_success("web_search")
    memory.record_success("api_call")
    
    # Get suggestions
    print("\n--- Failure Analysis ---\n")
    
    for tool in ["web_search", "api_call"]:
        suggestions = memory.get_suggestions(tool)
        print(f"Tool: {tool}")
        print(f"  Failure rate: {suggestions['failure_rate']:.1%}")
        print(f"  Total calls: {suggestions['total_calls']}")
        print(f"  Common errors: {suggestions['common_errors']}")
        print("  Recommendations:")
        for rec in suggestions["recommendations"]:
            print(f"    - {rec}")
        print()
    
    # Check if pattern should be avoided
    print("--- Pattern Avoidance ---")
    print(f"Should avoid web_search with similar args: {memory.should_avoid('web_search', {'query': 'test'})}")


# =============================================================================
# Example 5: Resilience Policies Comparison
# =============================================================================

async def example_policy_comparison():
    """Compare different resilience policies."""
    print("\n" + "=" * 70)
    print("Example 5: Resilience Policy Comparison")
    print("=" * 70)
    
    policies = {
        "Aggressive (never give up)": ResilienceConfig.aggressive(),
        "Balanced (default)": ResilienceConfig.balanced(),
        "Fast-fail (testing)": ResilienceConfig.fast_fail(),
    }
    
    print("\n--- Policy Configurations ---\n")
    
    for name, config in policies.items():
        print(f"{name}:")
        print(f"  Retry: max_retries={config.retry_policy.max_retries}, strategy={config.retry_policy.strategy.value}")
        print(f"  Circuit breaker: {config.circuit_breaker_enabled}")
        print(f"  Fallback: {config.fallback_enabled}")
        print(f"  Learning: {config.learning_enabled}")
        print(f"  Timeout: {config.timeout_seconds}s")
        print(f"  On failure: {config.on_failure.value}")
        print()
    
    # Custom policy example
    print("--- Creating Custom Policy ---\n")
    
    custom_policy = ResilienceConfig(
        retry_policy=RetryPolicy(
            max_retries=10,
            strategy=RetryStrategy.EXPONENTIAL_JITTER,
            base_delay=0.1,
            max_delay=30.0,
            jitter_factor=0.5,
        ),
        circuit_breaker_enabled=True,
        circuit_breaker_config=CircuitBreaker(
            failure_threshold=5,
            success_threshold=3,
            reset_timeout=60.0,
        ),
        fallback_enabled=True,
        learning_enabled=True,
        timeout_seconds=120.0,
    )
    
    print("Custom 'Ultra-Resilient' Policy:")
    print(f"  Max retries: {custom_policy.retry_policy.max_retries}")
    print(f"  Base delay: {custom_policy.retry_policy.base_delay}s")
    print(f"  Max delay: {custom_policy.retry_policy.max_delay}s")
    print(f"  Circuit threshold: {custom_policy.circuit_breaker_config.failure_threshold} failures")
    print(f"  Reset timeout: {custom_policy.circuit_breaker_config.reset_timeout}s")


# =============================================================================
# Example 6: Retry Delay Calculation
# =============================================================================

def example_retry_delays():
    """Show how retry delays are calculated."""
    print("\n" + "=" * 70)
    print("Example 6: Retry Delay Calculation")
    print("=" * 70)
    
    strategies = [
        ("Fixed", RetryPolicy(strategy=RetryStrategy.FIXED, base_delay=1.0)),
        ("Linear", RetryPolicy(strategy=RetryStrategy.LINEAR, base_delay=1.0)),
        ("Exponential", RetryPolicy(strategy=RetryStrategy.EXPONENTIAL, base_delay=1.0)),
        ("Exponential+Jitter", RetryPolicy(strategy=RetryStrategy.EXPONENTIAL_JITTER, base_delay=1.0, jitter_factor=0.25)),
    ]
    
    print("\n--- Delay by Attempt ---\n")
    print(f"{'Strategy':<20} | {'Attempt 1':>10} | {'Attempt 2':>10} | {'Attempt 3':>10} | {'Attempt 4':>10} | {'Attempt 5':>10}")
    print("-" * 85)
    
    for name, policy in strategies:
        delays = [f"{policy.get_delay(i):.2f}s" for i in range(1, 6)]
        print(f"{name:<20} | {delays[0]:>10} | {delays[1]:>10} | {delays[2]:>10} | {delays[3]:>10} | {delays[4]:>10}")
    
    print("\nNote: Exponential+Jitter has random variation for anti-thundering-herd")


# =============================================================================
# Example 7: Full Agent with Resilience
# =============================================================================

async def example_full_agent():
    """Complete example with agent using all resilience features."""
    print("\n" + "=" * 70)
    print("Example 7: Full Resilient Agent Demo")
    print("=" * 70)
    
    # Create tools
    flaky_search = FlakySearchTool(failure_rate=0.5)
    cached_search = CachedSearchTool()
    local_search = LocalSearchTool()
    calculator = ReliableTool()
    
    # Setup registry
    registry = ToolRegistry()
    registry.register(flaky_search)
    registry.register(cached_search)
    registry.register(local_search)
    registry.register(calculator)
    
    # Create fully resilient agent
    config = AgentConfig(
        name="SuperResilientAgent",
        role=AgentRole.SPECIALIST,
        description="An agent that never gives up!",
        tools=["web_search", "cached_search", "local_search", "calculate"],
        resilience_config=ResilienceConfig.aggressive(),
        fallback_tools={
            "web_search": ["cached_search", "local_search"],
        },
    )
    
    event_bus = EventBus()
    agent = Agent(config=config, event_bus=event_bus, tool_registry=registry)
    
    print(f"\nAgent: {agent.name}")
    print(f"Resilience: AGGRESSIVE (never give up)")
    print(f"Tools: {config.tools}")
    print(f"Fallbacks: {config.fallback_tools}")
    
    # Progress tracker
    tracker = ProgressTracker(OutputConfig.verbose())
    
    # Run multiple tool calls
    print("\n--- Running Tool Calls ---\n")
    
    tracker.start("Running multiple resilient tool calls")
    
    # Call 1: Flaky search (may retry/fallback)
    print("\n1. Web search (may need retry/fallback):")
    try:
        result = await agent.act("web_search", {"query": "resilience patterns"}, tracker=tracker)
        print(f"   Result: {result[:60]}...")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Call 2: Reliable calculator
    print("\n2. Calculation (always works):")
    result = await agent.act("calculate", {"expr": "2 ** 10"}, tracker=tracker)
    print(f"   Result: {result}")
    
    # Call 3: Parallel calls with resilience
    print("\n3. Parallel calls with resilience:")
    results = await agent.act_many([
        ("web_search", {"query": "async programming"}),
        ("calculate", {"expr": "sum(range(100))"}),
        ("cached_search", {"query": "Python tips"}),
    ], tracker=tracker)
    
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"   [{i}] Error: {r}")
        else:
            print(f"   [{i}] {str(r)[:50]}...")
    
    tracker.complete()
    
    # Show failure suggestions
    print("\n--- Failure Analysis ---")
    suggestions = agent.get_failure_suggestions("web_search")
    if suggestions:
        print(f"web_search failure rate: {suggestions['failure_rate']:.1%}")
        if suggestions["recommendations"]:
            print("Recommendations:")
            for rec in suggestions["recommendations"]:
                print(f"  - {rec}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all examples."""
    print("=" * 70)
    print("AgenticFlow Resilience Examples")
    print("Intelligent retry, circuit breaker, and fallback for agents")
    print("=" * 70)
    
    # Run examples
    await example_basic_retry()
    await example_fallback_tools()
    await example_circuit_breaker()
    await example_failure_learning()
    await example_policy_comparison()
    example_retry_delays()
    await example_full_agent()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
