"""
Tests for the resilience module.
"""

import asyncio
import time

import pytest

from agenticflow.agents.resilience import (
    CircuitBreaker,
    CircuitState,
    ExecutionResult,
    FailureMemory,
    FailureRecord,
    FallbackConfig,
    FallbackRegistry,
    RecoveryAction,
    ResilienceConfig,
    RetryPolicy,
    RetryStrategy,
    ToolResilience,
)


# =============================================================================
# RetryPolicy Tests
# =============================================================================


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_default_policy(self):
        """Test default retry policy values."""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.strategy == RetryStrategy.EXPONENTIAL_JITTER
        assert policy.base_delay == 1.0
        assert policy.max_delay == 60.0

    def test_no_retry_policy(self):
        """Test no-retry policy."""
        policy = RetryPolicy.no_retry()
        assert policy.max_retries == 0
        assert policy.strategy == RetryStrategy.NONE

    def test_aggressive_policy(self):
        """Test aggressive retry policy."""
        policy = RetryPolicy.aggressive()
        assert policy.max_retries == 5
        assert policy.base_delay == 0.5

    def test_conservative_policy(self):
        """Test conservative retry policy."""
        policy = RetryPolicy.conservative()
        assert policy.max_retries == 2
        assert policy.strategy == RetryStrategy.FIXED

    def test_fixed_delay(self):
        """Test fixed delay calculation."""
        policy = RetryPolicy(strategy=RetryStrategy.FIXED, base_delay=2.0)
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(5) == 2.0

    def test_linear_delay(self):
        """Test linear delay calculation."""
        policy = RetryPolicy(strategy=RetryStrategy.LINEAR, base_delay=1.0)
        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 3.0

    def test_exponential_delay(self):
        """Test exponential delay calculation."""
        policy = RetryPolicy(strategy=RetryStrategy.EXPONENTIAL, base_delay=1.0)
        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 4.0
        assert policy.get_delay(4) == 8.0

    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=5.0,
        )
        assert policy.get_delay(10) == 5.0

    def test_exponential_jitter_delay(self):
        """Test exponential with jitter has variation."""
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL_JITTER,
            base_delay=1.0,
            jitter_factor=0.5,
        )
        # Get multiple delays - they should vary due to jitter
        delays = [policy.get_delay(2) for _ in range(10)]
        # At least some should be different (jitter introduces randomness)
        unique_delays = set(round(d, 3) for d in delays)
        assert len(unique_delays) > 1

    def test_should_retry_max_retries(self):
        """Test that should_retry respects max_retries."""
        policy = RetryPolicy(max_retries=3)
        error = RuntimeError("test")
        assert policy.should_retry(error, 1) is True
        assert policy.should_retry(error, 2) is True
        assert policy.should_retry(error, 3) is False

    def test_should_retry_no_retry_strategy(self):
        """Test that NONE strategy doesn't retry."""
        policy = RetryPolicy(strategy=RetryStrategy.NONE, max_retries=5)
        error = RuntimeError("test")
        assert policy.should_retry(error, 1) is False

    def test_should_retry_retryable_exceptions(self):
        """Test retryable exceptions."""
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(TimeoutError("timeout"), 1) is True
        assert policy.should_retry(ConnectionError("conn"), 1) is True

    def test_should_retry_non_retryable_exceptions(self):
        """Test non-retryable exceptions."""
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(ValueError("invalid"), 1) is False
        assert policy.should_retry(TypeError("type"), 1) is False
        assert policy.should_retry(PermissionError("perm"), 1) is False


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self):
        """Test that circuit starts closed."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_open is False
        assert breaker.can_execute() is True

    def test_opens_after_failures(self):
        """Test that circuit opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        # Record failures
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
        
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
        
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open is True
        assert breaker.can_execute() is False

    def test_success_resets_failure_count(self):
        """Test that success resets failure count."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()  # Reset
        breaker.record_failure()
        breaker.record_failure()
        
        # Should still be closed (3 failures not consecutive)
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_after_timeout(self):
        """Test transition to half-open after timeout."""
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
        
        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        
        # Wait for reset timeout
        time.sleep(0.15)
        
        # Should be half-open now
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.can_execute() is True

    def test_half_open_to_closed_on_success(self):
        """Test transition from half-open to closed on success."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            success_threshold=2,
            reset_timeout=0.1,
        )
        
        # Open circuit
        breaker.record_failure()
        breaker.record_failure()
        
        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Record successes
        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN
        
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Test transition from half-open back to open on failure."""
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
        
        # Open circuit
        breaker.record_failure()
        breaker.record_failure()
        
        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Fail again
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_reset(self):
        """Test manual reset."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        # Open circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        
        # Reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    def test_force_open(self):
        """Test force open."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        
        breaker.force_open()
        assert breaker.state == CircuitState.OPEN

    def test_get_status(self):
        """Test get_status returns useful info."""
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure()
        
        status = breaker.get_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 1
        assert status["failure_threshold"] == 3


# =============================================================================
# FailureMemory Tests
# =============================================================================


class TestFailureMemory:
    """Tests for FailureMemory."""

    def test_record_failure(self):
        """Test recording failures."""
        memory = FailureMemory()
        
        memory.record_failure("tool1", ValueError("test"), {"arg": "value"})
        
        assert len(memory._records) == 1
        assert memory._tool_stats["tool1"]["failures"] == 1

    def test_record_success(self):
        """Test recording successes."""
        memory = FailureMemory()
        
        memory.record_success("tool1")
        memory.record_success("tool1")
        
        assert memory._tool_stats["tool1"]["successes"] == 2

    def test_failure_rate(self):
        """Test failure rate calculation."""
        memory = FailureMemory()
        
        # 2 failures, 2 successes = 50% failure rate
        memory.record_failure("tool1", ValueError("e1"), {})
        memory.record_failure("tool1", ValueError("e2"), {})
        memory.record_success("tool1")
        memory.record_success("tool1")
        
        assert memory.get_failure_rate("tool1") == 0.5

    def test_failure_rate_no_data(self):
        """Test failure rate with no data."""
        memory = FailureMemory()
        assert memory.get_failure_rate("unknown") == 0.0

    def test_common_errors(self):
        """Test getting common errors."""
        memory = FailureMemory()
        
        memory.record_failure("tool1", ValueError("v1"), {})
        memory.record_failure("tool1", ValueError("v2"), {})
        memory.record_failure("tool1", TypeError("t1"), {})
        
        errors = memory.get_common_errors("tool1")
        assert len(errors) == 2
        assert errors[0] == ("ValueError", 2)
        assert errors[1] == ("TypeError", 1)

    def test_should_avoid(self):
        """Test should_avoid for repeated failures."""
        memory = FailureMemory(learning_threshold=3)
        
        args = {"query": "test"}
        
        # 2 failures - should not avoid yet
        memory.record_failure("tool1", ValueError("e1"), args)
        memory.record_failure("tool1", ValueError("e2"), args)
        assert memory.should_avoid("tool1", args) is False
        
        # 3rd failure - should avoid now
        memory.record_failure("tool1", ValueError("e3"), args)
        assert memory.should_avoid("tool1", args) is True

    def test_record_recovery(self):
        """Test recording recovery."""
        memory = FailureMemory()
        
        memory.record_failure("tool1", ValueError("e"), {})
        memory.record_recovery("tool1", "retry")
        
        assert memory._records[-1].recovered is True
        assert memory._records[-1].recovery_method == "retry"

    def test_get_suggestions(self):
        """Test getting suggestions."""
        memory = FailureMemory()
        
        # High failure rate
        for _ in range(6):
            memory.record_failure("tool1", TimeoutError("timeout"), {})
        for _ in range(2):
            memory.record_success("tool1")
        
        suggestions = memory.get_suggestions("tool1")
        
        assert suggestions["tool"] == "tool1"
        assert suggestions["failure_rate"] == 0.75
        assert suggestions["total_calls"] == 8
        assert len(suggestions["recommendations"]) > 0

    def test_clear(self):
        """Test clearing memory."""
        memory = FailureMemory()
        
        memory.record_failure("tool1", ValueError("e"), {})
        memory.record_success("tool1")
        
        memory.clear()
        
        assert len(memory._records) == 0
        assert len(memory._tool_stats) == 0


# =============================================================================
# FallbackRegistry Tests
# =============================================================================


class TestFallbackRegistry:
    """Tests for FallbackRegistry."""

    def test_register_and_get_fallback(self):
        """Test registering and getting fallbacks."""
        registry = FallbackRegistry()
        
        registry.register("tool1", ["fallback1", "fallback2"])
        
        assert registry.get_fallback("tool1", 0) == "fallback1"
        assert registry.get_fallback("tool1", 1) == "fallback2"
        assert registry.get_fallback("tool1", 2) is None

    def test_get_all_fallbacks(self):
        """Test getting all fallbacks."""
        registry = FallbackRegistry()
        registry.register("tool1", ["fb1", "fb2", "fb3"])
        
        assert registry.get_all_fallbacks("tool1") == ["fb1", "fb2", "fb3"]
        assert registry.get_all_fallbacks("unknown") == []

    def test_has_fallback(self):
        """Test checking if fallback exists."""
        registry = FallbackRegistry()
        registry.register("tool1", ["fb1"])
        
        assert registry.has_fallback("tool1") is True
        assert registry.has_fallback("tool2") is False

    def test_transform_args(self):
        """Test argument transformation."""
        registry = FallbackRegistry()
        
        def transformer(fallback_name: str, args: dict) -> dict:
            return {**args, "source": fallback_name}
        
        registry.register("tool1", ["fb1"], transform_args=transformer)
        
        result = registry.transform_args("tool1", "fb1", {"query": "test"})
        assert result == {"query": "test", "source": "fb1"}

    def test_transform_args_no_transformer(self):
        """Test args pass through without transformer."""
        registry = FallbackRegistry()
        registry.register("tool1", ["fb1"])
        
        args = {"query": "test"}
        result = registry.transform_args("tool1", "fb1", args)
        assert result == args


# =============================================================================
# ResilienceConfig Tests
# =============================================================================


class TestResilienceConfig:
    """Tests for ResilienceConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ResilienceConfig()
        
        assert config.circuit_breaker_enabled is True
        assert config.fallback_enabled is True
        assert config.learning_enabled is True
        assert config.timeout_seconds == 60.0

    def test_aggressive_config(self):
        """Test aggressive configuration."""
        config = ResilienceConfig.aggressive()
        
        assert config.retry_policy.max_retries == 5
        assert config.on_failure == RecoveryAction.ADAPT

    def test_fast_fail_config(self):
        """Test fast-fail configuration."""
        config = ResilienceConfig.fast_fail()
        
        assert config.retry_policy.max_retries == 0
        assert config.circuit_breaker_enabled is False
        assert config.fallback_enabled is False
        assert config.on_failure == RecoveryAction.ABORT

    def test_tool_specific_overrides(self):
        """Test tool-specific configuration overrides."""
        config = ResilienceConfig(
            timeout_seconds=30.0,
            tool_configs={
                "slow_tool": {"timeout_seconds": 120.0},
                "fast_tool": {
                    "timeout_seconds": 5.0,
                    "retry_policy": RetryPolicy.no_retry(),
                },
            },
        )
        
        assert config.get_timeout("normal_tool") == 30.0
        assert config.get_timeout("slow_tool") == 120.0
        assert config.get_timeout("fast_tool") == 5.0


# =============================================================================
# ToolResilience Tests
# =============================================================================


class TestToolResilience:
    """Tests for ToolResilience."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful tool execution."""
        resilience = ToolResilience()
        
        async def success_fn(args: dict) -> str:
            return f"Success: {args.get('input', '')}"
        
        result = await resilience.execute(
            tool_fn=success_fn,
            tool_name="test_tool",
            args={"input": "hello"},
        )
        
        assert result.success is True
        assert result.result == "Success: hello"
        assert result.attempts == 1
        assert result.used_fallback is False

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on transient failure."""
        config = ResilienceConfig(
            retry_policy=RetryPolicy(
                max_retries=3,
                strategy=RetryStrategy.FIXED,
                base_delay=0.01,
            ),
        )
        resilience = ToolResilience(config=config)
        
        call_count = 0
        
        def flaky_fn(args: dict) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timeout")
            return "Success"
        
        result = await resilience.execute(
            tool_fn=flaky_fn,
            tool_name="flaky",
            args={},
        )
        
        assert result.success is True
        assert result.attempts == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks(self):
        """Test circuit breaker blocks calls."""
        config = ResilienceConfig(
            retry_policy=RetryPolicy.no_retry(),
            circuit_breaker_enabled=True,
            circuit_breaker_config=CircuitBreaker(failure_threshold=2),
            fallback_enabled=False,
        )
        resilience = ToolResilience(config=config)
        
        def fail_fn(args: dict) -> str:
            raise RuntimeError("fail")
        
        # First two calls fail and open circuit
        await resilience.execute(fail_fn, "test", {})
        await resilience.execute(fail_fn, "test", {})
        
        # Third call should be blocked by circuit
        result = await resilience.execute(fail_fn, "test", {})
        
        assert result.success is False
        assert result.circuit_state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_fallback_execution(self):
        """Test fallback tool execution."""
        registry = FallbackRegistry()
        registry.register("primary", ["fallback"])
        
        config = ResilienceConfig(
            retry_policy=RetryPolicy.no_retry(),
            fallback_enabled=True,
        )
        
        resilience = ToolResilience(config=config, fallback_registry=registry)
        
        def primary_fn(args: dict) -> str:
            raise RuntimeError("primary failed")
        
        def fallback_fn(args: dict) -> str:
            return "Fallback result"
        
        def get_tool(name: str):
            if name == "fallback":
                return fallback_fn
            return None
        
        result = await resilience.execute(
            tool_fn=primary_fn,
            tool_name="primary",
            args={},
            fallback_fn=get_tool,
        )
        
        assert result.success is True
        assert result.result == "Fallback result"
        assert result.used_fallback is True
        assert result.tool_used == "fallback"

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test timeout handling."""
        config = ResilienceConfig(
            retry_policy=RetryPolicy.no_retry(),
            timeout_seconds=0.1,
        )
        resilience = ToolResilience(config=config)
        
        async def slow_fn(args: dict) -> str:
            await asyncio.sleep(1.0)
            return "result"
        
        result = await resilience.execute(
            tool_fn=slow_fn,
            tool_name="slow",
            args={},
        )
        
        assert result.success is False
        assert "timed out" in str(result.error).lower()

    def test_circuit_status(self):
        """Test getting circuit status."""
        resilience = ToolResilience()
        
        # No breaker yet
        status = resilience.get_circuit_status("test")
        assert status == {"state": "no_breaker"}
        
        # Create breaker by checking
        resilience._get_circuit_breaker("test")
        status = resilience.get_circuit_status("test")
        assert status["state"] == "closed"

    def test_reset_circuits(self):
        """Test resetting circuits."""
        resilience = ToolResilience()
        
        # Create and open a breaker
        breaker = resilience._get_circuit_breaker("test")
        breaker.force_open()
        assert breaker.state == CircuitState.OPEN
        
        # Reset
        resilience.reset_circuit("test")
        assert breaker.state == CircuitState.CLOSED

    def test_failure_suggestions(self):
        """Test getting failure suggestions."""
        config = ResilienceConfig(learning_enabled=True)
        resilience = ToolResilience(config=config)
        
        # Record some failures
        resilience.failure_memory.record_failure("tool", TimeoutError("t"), {})
        resilience.failure_memory.record_failure("tool", TimeoutError("t"), {})
        resilience.failure_memory.record_success("tool")
        
        suggestions = resilience.get_failure_suggestions("tool")
        
        assert suggestions is not None
        assert suggestions["tool"] == "tool"
        assert suggestions["failure_rate"] > 0


# =============================================================================
# ExecutionResult Tests
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_success_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            result="data",
            tool_used="tool1",
            attempts=1,
        )
        
        assert result.success is True
        assert result.result == "data"
        assert result.error is None

    def test_failure_result(self):
        """Test failed execution result."""
        error = RuntimeError("failed")
        result = ExecutionResult(
            success=False,
            error=error,
            tool_used="tool1",
            attempts=3,
            recovery_action=RecoveryAction.ABORT,
        )
        
        assert result.success is False
        assert result.error == error
        assert result.recovery_action == RecoveryAction.ABORT

    def test_fallback_result(self):
        """Test fallback execution result."""
        result = ExecutionResult(
            success=True,
            result="fallback data",
            tool_used="fallback_tool",
            used_fallback=True,
            fallback_chain=["fb1", "fb2"],
        )
        
        assert result.used_fallback is True
        assert result.tool_used == "fallback_tool"
        assert len(result.fallback_chain) == 2
