"""
Tool Resilience & Recovery - Automatic retry, circuit breakers, and failure learning.

Demonstrates AgenticFlow's tool resilience features:
- Automatic retry with exponential backoff
- Circuit breaker to prevent cascading failures
- Fallback tool chains for graceful degradation
- Default vs custom resilience configurations

Run:
    uv run ./examples/observability/tool_resilience.py
"""

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent
from agenticflow.agent.resilience import (
    ResilienceConfig,
    RetryPolicy,
    RetryStrategy,
    RecoveryAction,
)
from agenticflow.observability import Observer, ObservabilityLevel
from agenticflow.tools.base import tool


# =============================================================================
# Simulated Tools with Failure Scenarios
# =============================================================================

_api_call_attempts = 0
_flaky_search_attempts = 0


@tool
def unreliable_api(query: str) -> str:
    """Call an unreliable external API that fails randomly."""
    global _api_call_attempts
    _api_call_attempts += 1
    
    if _api_call_attempts < 3:
        raise RuntimeError(f"API connection timeout (attempt {_api_call_attempts})")
    
    return f"API Result: {query} (succeeded after {_api_call_attempts} attempts)"


@tool
def flaky_search(query: str) -> str:
    """Search that fails 70% of the time."""
    global _flaky_search_attempts
    _flaky_search_attempts += 1
    
    if random.random() < 0.7:
        raise ConnectionError(f"Search service unavailable (attempt {_flaky_search_attempts})")
    
    return f"Search results for '{query}': Found 5 relevant articles"


@tool
def always_fails(query: str) -> str:
    """Tool that always fails - triggers circuit breaker."""
    raise RuntimeError("Service permanently down")


@tool
def reliable_backup(query: str) -> str:
    """Reliable fallback tool that always works."""
    return f"Fallback Result: {query} (using backup service)"


# =============================================================================
# Demo Functions
# =============================================================================

async def demo_default_resilience():
    """Demo 1: Default resilience with direct .act() call."""
    print("\n" + "=" * 80)
    print("Demo 1: Default Resilience (agent.act)")
    print("=" * 80)
    
    global _api_call_attempts
    _api_call_attempts = 0
    
    model = get_model()
    agent = Agent(
        name="DefaultAgent",
        model=model,
        tools=[unreliable_api],
    )
    
    print(f"Config: retry_on_error={agent.config.retry_on_error}, max_retries={agent.config.max_retries}")
    print("Calling unreliable_api (fails 2x, succeeds on 3rd attempt)...\n")
    
    try:
        result = await agent.act(
            tool_name="unreliable_api",
            args={"query": "test data"},
            use_resilience=True,
        )
        print(f"Result: {result}")
        print(f"Attempts: {_api_call_attempts}\n")
    except Exception as e:
        print(f"Failed: {e}")
        print(f"Attempts: {_api_call_attempts}\n")


async def demo_aggressive_resilience():
    """Demo 2: Aggressive resilience with custom configuration."""
    print("\n" + "=" * 80)
    print("Demo 2: Aggressive Resilience Config")
    print("=" * 80)
    
    global _flaky_search_attempts
    _flaky_search_attempts = 0
    
    model = get_model()
    aggressive_config = ResilienceConfig.aggressive()
    
    print(f"Config: max_retries={aggressive_config.retry_policy.max_retries}, "
          f"strategy={aggressive_config.retry_policy.strategy.value}, "
          f"timeout={aggressive_config.timeout_seconds}s")
    print("Calling flaky_search (70% failure rate)...\n")
    
    agent = Agent(
        name="AggressiveAgent",
        model=model,
        tools=[flaky_search],
        resilience=aggressive_config,
    )
    
    try:
        result = await agent.act(
            tool_name="flaky_search",
            args={"query": "Python async programming tutorials"},
            use_resilience=True,
        )
        print(f"Result: {result}")
        print(f"Attempts: {_flaky_search_attempts}\n")
    except Exception as e:
        print(f"Failed: {e}")
        print(f"Attempts: {_flaky_search_attempts}\n")


async def demo_custom_retry_policy():
    """Demo 3: Custom retry policy with specific strategy."""
    print("\n" + "=" * 80)
    print("Demo 3: Custom Retry Policy")
    print("=" * 80)
    
    global _api_call_attempts
    _api_call_attempts = 0
    
    model = get_model()
    custom_policy = RetryPolicy(
        max_retries=5,
        strategy=RetryStrategy.EXPONENTIAL,
        base_delay=0.5,
        max_delay=10.0,
        jitter_factor=0.3,
    )
    
    custom_config = ResilienceConfig(
        retry_policy=custom_policy,
        timeout_seconds=60.0,
        circuit_breaker_enabled=True,
        fallback_enabled=False,
        learning_enabled=True,
    )
    
    print(f"Policy: max_retries={custom_policy.max_retries}, "
          f"strategy={custom_policy.strategy.value}, "
          f"delays={custom_policy.base_delay}s-{custom_policy.max_delay}s")
    print("Calling unreliable_api...\n")
    
    agent = Agent(
        name="CustomRetryAgent",
        model=model,
        tools=[unreliable_api],
        resilience=custom_config,
    )
    
    try:
        result = await agent.act(
            tool_name="unreliable_api",
            args={"query": "important data"},
            use_resilience=True,
        )
        print(f"Result: {result}")
        print(f"Attempts: {_api_call_attempts}\n")
    except Exception as e:
        print(f"Failed: {e}")
        print(f"Attempts: {_api_call_attempts}\n")


async def demo_fallback_tools():
    """Demo 4: Fallback tool chains for graceful degradation."""
    print("\n" + "=" * 80)
    print("Demo 4: Fallback Tool Chains")
    print("=" * 80)
    
    model = get_model()
    fallback_config = ResilienceConfig(
        retry_policy=RetryPolicy(max_retries=2),
        circuit_breaker_enabled=True,
        fallback_enabled=True,
        learning_enabled=True,
    )
    
    agent = Agent(
        name="FallbackAgent",
        model=model,
        tools=[always_fails, reliable_backup],
        resilience=fallback_config,
    )
    
    agent.config = agent.config.with_fallbacks({"always_fails": ["reliable_backup"]})
    agent._setup_resilience()
    
    print("Fallback chain: always_fails -> reliable_backup")
    print("Calling always_fails...\n")
    
    try:
        result = await agent.act(
            tool_name="always_fails",
            args={"query": "process this"},
            use_resilience=True,
        )
        print(f"Result: {result}\n")
    except Exception as e:
        print(f"Failed: {e}\n")


async def demo_fast_fail():
    """Demo 5: Fast-fail configuration for critical paths."""
    print("\n" + "=" * 80)
    print("Demo 5: Fast-Fail Config")
    print("=" * 80)
    
    model = get_model()
    fast_fail_config = ResilienceConfig.fast_fail()
    
    print(f"Config: max_retries={fast_fail_config.retry_policy.max_retries}, "
          f"timeout={fast_fail_config.timeout_seconds}s")
    print("Calling always_fails...\n")
    
    agent = Agent(
        name="FastFailAgent",
        model=model,
        tools=[always_fails],
        resilience=fast_fail_config,
    )
    
    try:
        result = await agent.act(
            tool_name="always_fails",
            args={"query": "critical operation"},
            use_resilience=True,
        )
        print(f"Result: {result}\n")
    except Exception as e:
        print(f"Failed (expected): {type(e).__name__}: {e}\n")


async def demo_resilience_comparison():
    """Demo 6: Side-by-side comparison of resilience strategies."""
    print("\n" + "=" * 80)
    print("Demo 6: Configuration Comparison")
    print("=" * 80)
    
    configs = {
        "Default": ResilienceConfig(),
        "Aggressive": ResilienceConfig.aggressive(),
        "Balanced": ResilienceConfig.balanced(),
        "Fast-Fail": ResilienceConfig.fast_fail(),
    }
    
    print("┌─────────────────┬──────────┬──────────────┬──────────┬─────────┐")
    print("│ Configuration   │ Retries  │ Strategy     │ Circuit  │ Learning│")
    print("├─────────────────┼──────────┼──────────────┼──────────┼─────────┤")
    
    for name, config in configs.items():
        retries = config.retry_policy.max_retries
        strategy = config.retry_policy.strategy.value[:12]
        circuit = "YES" if config.circuit_breaker_enabled else "NO "
        learning = "YES" if config.learning_enabled else "NO "
        print(f"│ {name:<15} │ {retries:^8} │ {strategy:<12} │ {circuit:^8} │ {learning:^7} │")
    
    print("└─────────────────┴──────────┴──────────────┴──────────┴─────────┘\n")


async def main():
    """Run all resilience demos."""
    print("=" * 80)
    print("Tool Resilience Examples")
    print("=" * 80)
    
    await demo_default_resilience()
    await demo_aggressive_resilience()
    await demo_custom_retry_policy()
    await demo_fallback_tools()
    await demo_fast_fail()
    await demo_resilience_comparison()
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("Resilience applies when using agent.act() directly.")
    print("Default: 3 retries, exponential backoff, 60s timeout.")
    print("Configs: aggressive(), balanced(), fast_fail(), or custom.")
    print("See RESILIENCE_README.md for details.\n")


if __name__ == "__main__":
    asyncio.run(main())
