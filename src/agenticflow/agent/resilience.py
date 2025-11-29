"""
Resilience - intelligent retry, recovery, and fault tolerance for agents.

This module provides production-grade resilience patterns:

1. **RetryPolicy**: Configurable retry with exponential backoff
2. **CircuitBreaker**: Disable failing tools temporarily to prevent cascading failures
3. **FallbackRegistry**: Define alternative tools when primary fails
4. **ToolResilience**: Unified resilience layer wrapping tool execution
5. **FailureMemory**: Learn from failures to adapt future behavior

Agents don't give up easily - they adapt, learn, and recover!
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from agenticflow.core.utils import generate_id, now_utc

if TYPE_CHECKING:
    from agenticflow.observability.progress import ProgressTracker


class RetryStrategy(Enum):
    """Retry backoff strategies."""
    
    NONE = "none"  # No retry
    FIXED = "fixed"  # Fixed delay between retries
    LINEAR = "linear"  # Linearly increasing delay
    EXPONENTIAL = "exponential"  # Exponential backoff (recommended)
    EXPONENTIAL_JITTER = "exponential_jitter"  # Exponential with random jitter (best)


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class RecoveryAction(Enum):
    """Actions to take on failure."""
    
    RETRY = "retry"  # Retry the same tool
    FALLBACK = "fallback"  # Try a fallback tool
    SKIP = "skip"  # Skip and continue (return error result)
    ABORT = "abort"  # Abort entire execution
    ADAPT = "adapt"  # Let agent decide (re-think)


@dataclass
class RetryPolicy:
    """
    Configurable retry policy with backoff strategies.
    
    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retry)
        strategy: Backoff strategy
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter_factor: Random jitter factor (0.0-1.0)
        retryable_exceptions: Exception types that trigger retry
        non_retryable_exceptions: Exception types that should NOT retry
        
    Example:
        ```python
        # Aggressive retry for flaky APIs
        policy = RetryPolicy(
            max_retries=5,
            strategy=RetryStrategy.EXPONENTIAL_JITTER,
            base_delay=1.0,
            max_delay=30.0,
        )
        
        # Conservative retry for expensive operations
        policy = RetryPolicy(
            max_retries=2,
            strategy=RetryStrategy.FIXED,
            base_delay=5.0,
        )
        
        # No retry (fail fast)
        policy = RetryPolicy.no_retry()
        ```
    """
    
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    jitter_factor: float = 0.25  # 25% random jitter
    retryable_exceptions: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        OSError,
    )
    non_retryable_exceptions: tuple[type[Exception], ...] = (
        ValueError,
        TypeError,
        PermissionError,
        KeyError,
    )
    # Message patterns that indicate retryable errors (rate limits, server errors)
    retryable_patterns: tuple[str, ...] = (
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "429",
        "resource exhausted",
        "resourceexhausted",
        "quota exceeded",
        "quota_exceeded",
        "temporarily unavailable",
        "service unavailable",
        "503",
        "502",
        "500",
        "server error",
        "overloaded",
        "capacity",
        "try again",
        "retry",
    )
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry attempt.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds before next retry
        """
        if self.strategy == RetryStrategy.NONE:
            return 0.0
        
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.strategy in (RetryStrategy.EXPONENTIAL, RetryStrategy.EXPONENTIAL_JITTER):
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay
        
        # Apply max cap
        delay = min(delay, self.max_delay)
        
        # Add jitter for exponential_jitter
        if self.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            jitter = delay * self.jitter_factor * random.random()
            delay = delay + jitter
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number (1-based)
            
        Returns:
            True if should retry, False otherwise
        """
        # attempt is 1-based, max_retries means we can retry that many times
        # So if max_retries=5, we allow attempts 1,2,3,4,5 to retry
        if attempt > self.max_retries:
            return False
        
        if self.strategy == RetryStrategy.NONE:
            return False
        
        # Check non-retryable first (takes precedence)
        if isinstance(exception, self.non_retryable_exceptions):
            return False
        
        # Check if explicitly retryable
        if isinstance(exception, self.retryable_exceptions):
            return True
        
        # Check error message for retryable patterns (rate limits, server errors)
        error_msg = str(exception).lower()
        error_type = type(exception).__name__.lower()
        
        # Check for rate limit and server error patterns
        if any(pattern in error_msg or pattern in error_type for pattern in self.retryable_patterns):
            return True
        
        # Default: retry most runtime errors, but check for non-retryable patterns
        if isinstance(exception, (RuntimeError, Exception)):
            non_retryable_patterns = [
                "not found",
                "unauthorized",
                "forbidden",
                "invalid argument",
                "invalid_argument",
                "not supported",
                "not implemented",
                "authentication",
                "permission denied",
            ]
            if any(pattern in error_msg for pattern in non_retryable_patterns):
                return False
            return True
        
        return False
    
    @classmethod
    def no_retry(cls) -> RetryPolicy:
        """Create a policy with no retry (fail fast)."""
        return cls(max_retries=0, strategy=RetryStrategy.NONE)
    
    @classmethod
    def aggressive(cls) -> RetryPolicy:
        """Create an aggressive retry policy for flaky services."""
        return cls(
            max_retries=5,
            strategy=RetryStrategy.EXPONENTIAL_JITTER,
            base_delay=0.5,
            max_delay=30.0,
            jitter_factor=0.3,
        )
    
    @classmethod
    def conservative(cls) -> RetryPolicy:
        """Create a conservative retry policy for expensive operations."""
        return cls(
            max_retries=2,
            strategy=RetryStrategy.FIXED,
            base_delay=5.0,
            max_delay=10.0,
        )
    
    @classmethod
    def default(cls) -> RetryPolicy:
        """Create the default balanced retry policy."""
        return cls()


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    When a tool fails repeatedly, the circuit "opens" and blocks
    further calls, allowing the service to recover. After a reset
    timeout, it enters "half-open" state to test recovery.
    
    States:
        CLOSED: Normal operation, calls pass through
        OPEN: Failing, calls blocked (returns error immediately)
        HALF_OPEN: Testing recovery, limited calls allowed
    
    Attributes:
        failure_threshold: Failures before opening circuit
        success_threshold: Successes in half-open before closing
        reset_timeout: Seconds before moving from open to half-open
        half_open_max_calls: Max calls allowed in half-open state
        
    Example:
        ```python
        breaker = CircuitBreaker(
            failure_threshold=3,  # Open after 3 failures
            reset_timeout=30.0,   # Wait 30s before testing
        )
        
        if breaker.can_execute():
            try:
                result = await tool.invoke(args)
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
        ```
    """
    
    failure_threshold: int = 3
    success_threshold: int = 2
    reset_timeout: float = 30.0  # seconds
    half_open_max_calls: int = 3
    
    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, repr=False)
    _failure_count: int = field(default=0, repr=False)
    _success_count: int = field(default=0, repr=False)
    _last_failure_time: float | None = field(default=None, repr=False)
    _half_open_calls: int = field(default=0, repr=False)
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout transitions."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            if time.time() - self._last_failure_time >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0
        return self._state
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking calls)."""
        return self.state == CircuitState.OPEN
    
    def can_execute(self) -> bool:
        """
        Check if a call should be allowed through.
        
        Returns:
            True if call should proceed, False if blocked
        """
        state = self.state
        
        if state == CircuitState.CLOSED:
            return True
        
        if state == CircuitState.OPEN:
            return False
        
        # HALF_OPEN: Allow limited calls to test recovery
        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        
        return True
    
    def record_success(self) -> None:
        """Record a successful call."""
        state = self.state
        
        if state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                # Recovery confirmed, close circuit
                self._state = CircuitState.CLOSED
                self._reset_counters()
        elif state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed call."""
        state = self.state
        self._last_failure_time = time.time()
        
        if state == CircuitState.HALF_OPEN:
            # Failed during recovery test, reopen circuit
            self._state = CircuitState.OPEN
            self._failure_count = self.failure_threshold
        elif state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                # Too many failures, open circuit
                self._state = CircuitState.OPEN
    
    def _reset_counters(self) -> None:
        """Reset internal counters."""
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = None
    
    def reset(self) -> None:
        """Manually reset the circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._reset_counters()
    
    def force_open(self) -> None:
        """Manually open the circuit (for testing/maintenance)."""
        self._state = CircuitState.OPEN
        self._last_failure_time = time.time()
    
    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status as dict."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "reset_timeout": self.reset_timeout,
            "seconds_until_half_open": (
                max(0, self.reset_timeout - (time.time() - self._last_failure_time))
                if self._state == CircuitState.OPEN and self._last_failure_time
                else None
            ),
        }


@dataclass
class FailureRecord:
    """Record of a tool failure for learning."""
    
    tool_name: str
    error_type: str
    error_message: str
    args_hash: str  # Hash of args that caused failure
    timestamp: float
    attempt: int
    recovered: bool = False
    recovery_method: str | None = None  # retry, fallback, adapt


@dataclass
class FailureMemory:
    """
    Learn from failures to adapt future behavior.
    
    Tracks failure patterns and success/failure ratios to:
    - Identify problematic argument patterns
    - Suggest better retry strategies
    - Recommend fallback tools
    - Avoid repeating known failures
    
    Example:
        ```python
        memory = FailureMemory()
        
        # Record a failure
        memory.record_failure("web_search", ValueError("Rate limited"), {"query": "..."})
        
        # Check before calling
        if memory.should_avoid("web_search", {"query": "..."}):
            # Try alternative approach
            pass
        
        # Get recommendations
        suggestions = memory.get_suggestions("web_search")
        ```
    """
    
    max_records: int = 1000
    learning_threshold: int = 3  # Failures before marking pattern as problematic
    
    _records: list[FailureRecord] = field(default_factory=list, repr=False)
    _tool_stats: dict[str, dict[str, int]] = field(default_factory=dict, repr=False)
    _pattern_failures: dict[str, int] = field(default_factory=dict, repr=False)  # tool:error_type -> count
    
    def record_failure(
        self,
        tool_name: str,
        exception: Exception,
        args: dict[str, Any],
        attempt: int = 1,
    ) -> None:
        """Record a failure for learning."""
        error_type = type(exception).__name__
        error_message = str(exception)
        args_hash = self._hash_args(args)
        
        record = FailureRecord(
            tool_name=tool_name,
            error_type=error_type,
            error_message=error_message,
            args_hash=args_hash,
            timestamp=time.time(),
            attempt=attempt,
        )
        
        self._records.append(record)
        
        # Trim old records
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]
        
        # Update stats
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {"successes": 0, "failures": 0}
        self._tool_stats[tool_name]["failures"] += 1
        
        # Track error patterns
        pattern_key = f"{tool_name}:{error_type}"
        self._pattern_failures[pattern_key] = self._pattern_failures.get(pattern_key, 0) + 1
    
    def record_success(self, tool_name: str) -> None:
        """Record a successful call."""
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {"successes": 0, "failures": 0}
        self._tool_stats[tool_name]["successes"] += 1
    
    def record_recovery(
        self,
        tool_name: str,
        recovery_method: str,
    ) -> None:
        """Mark the last failure as recovered."""
        for record in reversed(self._records):
            if record.tool_name == tool_name and not record.recovered:
                record.recovered = True
                record.recovery_method = recovery_method
                break
    
    def get_failure_rate(self, tool_name: str) -> float:
        """Get failure rate for a tool (0.0 to 1.0)."""
        stats = self._tool_stats.get(tool_name)
        if not stats:
            return 0.0
        
        total = stats["successes"] + stats["failures"]
        if total == 0:
            return 0.0
        
        return stats["failures"] / total
    
    def get_common_errors(self, tool_name: str) -> list[tuple[str, int]]:
        """Get most common error types for a tool."""
        errors = [
            (k.split(":", 1)[1], v)
            for k, v in self._pattern_failures.items()
            if k.startswith(f"{tool_name}:")
        ]
        return sorted(errors, key=lambda x: x[1], reverse=True)
    
    def should_avoid(self, tool_name: str, args: dict[str, Any]) -> bool:
        """
        Check if this tool+args combination should be avoided.
        
        Returns True if this pattern has failed repeatedly.
        """
        args_hash = self._hash_args(args)
        
        # Count recent failures with same tool and args pattern
        recent_failures = sum(
            1 for r in self._records[-100:]  # Check last 100 records
            if r.tool_name == tool_name
            and r.args_hash == args_hash
            and not r.recovered
        )
        
        return recent_failures >= self.learning_threshold
    
    def get_suggestions(self, tool_name: str) -> dict[str, Any]:
        """Get suggestions for improving tool reliability."""
        stats = self._tool_stats.get(tool_name, {"successes": 0, "failures": 0})
        failure_rate = self.get_failure_rate(tool_name)
        common_errors = self.get_common_errors(tool_name)
        
        suggestions = {
            "tool": tool_name,
            "failure_rate": failure_rate,
            "total_calls": stats["successes"] + stats["failures"],
            "common_errors": common_errors[:5],
            "recommendations": [],
        }
        
        # Generate recommendations
        if failure_rate > 0.5:
            suggestions["recommendations"].append(
                "High failure rate - consider using a fallback tool"
            )
        
        if common_errors:
            top_error = common_errors[0][0]
            if "timeout" in top_error.lower():
                suggestions["recommendations"].append(
                    "Frequent timeouts - consider increasing timeout or using exponential backoff"
                )
            elif "rate" in top_error.lower() or "limit" in top_error.lower():
                suggestions["recommendations"].append(
                    "Rate limiting detected - add delays between calls or use a different API key"
                )
            elif "connection" in top_error.lower():
                suggestions["recommendations"].append(
                    "Connection issues - check network or use aggressive retry policy"
                )
        
        return suggestions
    
    def _hash_args(self, args: dict[str, Any]) -> str:
        """Create a hash of args for pattern matching."""
        # Simple hash based on keys and value types/lengths
        parts = []
        for k, v in sorted(args.items()):
            if isinstance(v, str):
                parts.append(f"{k}:str:{len(v)//100}")  # Bucket by length
            elif isinstance(v, (int, float)):
                parts.append(f"{k}:num")
            elif isinstance(v, (list, tuple)):
                parts.append(f"{k}:list:{len(v)}")
            else:
                parts.append(f"{k}:{type(v).__name__}")
        return "|".join(parts)
    
    def clear(self) -> None:
        """Clear all recorded failures."""
        self._records.clear()
        self._tool_stats.clear()
        self._pattern_failures.clear()


@dataclass
class FallbackConfig:
    """Configuration for a fallback tool."""
    
    primary_tool: str
    fallback_tools: list[str]  # Ordered list of fallbacks to try
    transform_args: Callable[[str, dict], dict] | None = None  # Transform args for fallback


class FallbackRegistry:
    """
    Registry of fallback tools for graceful degradation.
    
    When a primary tool fails, the registry provides alternative
    tools that can accomplish similar goals.
    
    Example:
        ```python
        registry = FallbackRegistry()
        
        # Register fallback chain
        registry.register(
            "web_search",
            ["cached_search", "local_search", "llm_search"],
        )
        
        # Get next fallback
        fallback = registry.get_fallback("web_search", attempt=1)
        # Returns "cached_search"
        ```
    """
    
    def __init__(self) -> None:
        self._fallbacks: dict[str, FallbackConfig] = {}
    
    def register(
        self,
        primary_tool: str,
        fallback_tools: list[str],
        transform_args: Callable[[str, dict], dict] | None = None,
    ) -> None:
        """
        Register fallback tools for a primary tool.
        
        Args:
            primary_tool: Name of the primary tool
            fallback_tools: Ordered list of fallback tool names
            transform_args: Optional function to transform args for fallback
        """
        self._fallbacks[primary_tool] = FallbackConfig(
            primary_tool=primary_tool,
            fallback_tools=fallback_tools,
            transform_args=transform_args,
        )
    
    def get_fallback(
        self,
        tool_name: str,
        attempt: int = 0,
    ) -> str | None:
        """
        Get the fallback tool for a given attempt number.
        
        Args:
            tool_name: Name of the failing tool
            attempt: Current fallback attempt (0-indexed)
            
        Returns:
            Name of fallback tool, or None if no more fallbacks
        """
        config = self._fallbacks.get(tool_name)
        if not config or attempt >= len(config.fallback_tools):
            return None
        return config.fallback_tools[attempt]
    
    def get_all_fallbacks(self, tool_name: str) -> list[str]:
        """Get all fallback tools for a primary tool."""
        config = self._fallbacks.get(tool_name)
        return config.fallback_tools if config else []
    
    def transform_args(
        self,
        primary_tool: str,
        fallback_tool: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Transform arguments for a fallback tool.
        
        Args:
            primary_tool: Original tool name
            fallback_tool: Fallback tool being used
            args: Original arguments
            
        Returns:
            Transformed arguments for the fallback tool
        """
        config = self._fallbacks.get(primary_tool)
        if config and config.transform_args:
            return config.transform_args(fallback_tool, args)
        return args
    
    def has_fallback(self, tool_name: str) -> bool:
        """Check if a tool has fallbacks registered."""
        return tool_name in self._fallbacks


@dataclass
class ResilienceConfig:
    """
    Complete resilience configuration for an agent.
    
    Combines retry policy, circuit breaker, fallbacks, and
    failure memory into a unified configuration.
    
    Attributes:
        retry_policy: Retry configuration
        circuit_breaker_enabled: Whether to use circuit breakers
        circuit_breaker_config: Default circuit breaker settings
        fallback_enabled: Whether to try fallback tools
        learning_enabled: Whether to learn from failures
        timeout_seconds: Default timeout for tool calls
        on_failure: Default action when all retries exhausted
        
    Example:
        ```python
        # Production config
        config = ResilienceConfig(
            retry_policy=RetryPolicy.aggressive(),
            circuit_breaker_enabled=True,
            fallback_enabled=True,
            learning_enabled=True,
            timeout_seconds=30.0,
        )
        
        # Fast-fail config for testing
        config = ResilienceConfig(
            retry_policy=RetryPolicy.no_retry(),
            circuit_breaker_enabled=False,
            timeout_seconds=5.0,
            on_failure=RecoveryAction.ABORT,
        )
        ```
    """
    
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy.default)
    circuit_breaker_enabled: bool = True
    circuit_breaker_config: CircuitBreaker = field(default_factory=CircuitBreaker)
    fallback_enabled: bool = True
    learning_enabled: bool = True
    timeout_seconds: float = 60.0
    on_failure: RecoveryAction = RecoveryAction.SKIP
    
    # Per-tool overrides
    tool_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def get_retry_policy(self, tool_name: str) -> RetryPolicy:
        """Get retry policy for a specific tool (with overrides)."""
        if tool_name in self.tool_configs:
            overrides = self.tool_configs[tool_name]
            if "retry_policy" in overrides:
                return overrides["retry_policy"]
        return self.retry_policy
    
    def get_timeout(self, tool_name: str) -> float:
        """Get timeout for a specific tool (with overrides)."""
        if tool_name in self.tool_configs:
            overrides = self.tool_configs[tool_name]
            if "timeout_seconds" in overrides:
                return overrides["timeout_seconds"]
        return self.timeout_seconds
    
    @classmethod
    def aggressive(cls) -> ResilienceConfig:
        """Create aggressive resilience config (never give up)."""
        return cls(
            retry_policy=RetryPolicy.aggressive(),
            circuit_breaker_enabled=True,
            fallback_enabled=True,
            learning_enabled=True,
            timeout_seconds=120.0,
            on_failure=RecoveryAction.ADAPT,
        )
    
    @classmethod
    def balanced(cls) -> ResilienceConfig:
        """Create balanced resilience config (default)."""
        return cls()
    
    @classmethod
    def fast_fail(cls) -> ResilienceConfig:
        """Create fast-fail config for testing."""
        return cls(
            retry_policy=RetryPolicy.no_retry(),
            circuit_breaker_enabled=False,
            fallback_enabled=False,
            learning_enabled=False,
            timeout_seconds=10.0,
            on_failure=RecoveryAction.ABORT,
        )


@dataclass
class ExecutionResult:
    """Result of a resilient tool execution."""
    
    success: bool
    result: Any = None
    error: Exception | None = None
    tool_used: str = ""  # May differ from requested if fallback used
    attempts: int = 1
    total_time_ms: float = 0.0
    used_fallback: bool = False
    fallback_chain: list[str] = field(default_factory=list)
    circuit_state: CircuitState | None = None
    recovery_action: RecoveryAction | None = None


class ToolResilience:
    """
    Unified resilience layer for tool execution.
    
    Wraps tool calls with:
    - Timeout enforcement
    - Retry with exponential backoff
    - Circuit breaker protection
    - Fallback tool execution
    - Failure learning and adaptation
    
    Example:
        ```python
        resilience = ToolResilience(
            config=ResilienceConfig.aggressive(),
            fallback_registry=fallback_registry,
        )
        
        # Execute with full resilience
        result = await resilience.execute(
            tool_fn=tool.invoke,
            tool_name="web_search",
            args={"query": "Python tutorials"},
        )
        
        if result.success:
            print(result.result)
        else:
            print(f"Failed after {result.attempts} attempts: {result.error}")
        ```
    """
    
    def __init__(
        self,
        config: ResilienceConfig | None = None,
        fallback_registry: FallbackRegistry | None = None,
        tracker: ProgressTracker | None = None,
    ) -> None:
        self.config = config or ResilienceConfig()
        self.fallback_registry = fallback_registry or FallbackRegistry()
        self.tracker = tracker
        
        # Per-tool circuit breakers
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        
        # Failure memory for learning
        self.failure_memory = FailureMemory() if self.config.learning_enabled else None
    
    def _get_circuit_breaker(self, tool_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a tool."""
        if tool_name not in self._circuit_breakers:
            self._circuit_breakers[tool_name] = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_config.failure_threshold,
                success_threshold=self.config.circuit_breaker_config.success_threshold,
                reset_timeout=self.config.circuit_breaker_config.reset_timeout,
            )
        return self._circuit_breakers[tool_name]
    
    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit progress event if tracker available."""
        if self.tracker:
            if event_type == "retry":
                self.tracker.update(
                    f"🔄 Retry {data.get('attempt', '?')}/{data.get('max_retries', '?')} "
                    f"for {data.get('tool', 'tool')} in {data.get('delay', 0):.1f}s"
                )
            elif event_type == "circuit_open":
                self.tracker.update(
                    f"⚡ Circuit breaker OPEN for {data.get('tool', 'tool')} - "
                    f"blocking calls for {data.get('reset_timeout', 30)}s"
                )
            elif event_type == "fallback":
                self.tracker.update(
                    f"↩️ Falling back: {data.get('from_tool', '?')} → {data.get('to_tool', '?')}"
                )
            elif event_type == "recovery":
                self.tracker.update(
                    f"✅ Recovered via {data.get('method', 'retry')}"
                )
    
    async def execute(
        self,
        tool_fn: Callable[..., Any],
        tool_name: str,
        args: dict[str, Any],
        fallback_fn: Callable[[str], Callable[..., Any]] | None = None,
        tool_obj: Any | None = None,
    ) -> ExecutionResult:
        """
        Execute a tool with full resilience.
        
        Args:
            tool_fn: The tool function to call
            tool_name: Name of the tool (for tracking)
            args: Arguments to pass to the tool
            fallback_fn: Optional function to get fallback tool by name
            tool_obj: Optional tool object (for ainvoke support)
            
        Returns:
            ExecutionResult with success/failure details
        """
        start_time = time.time()
        result = ExecutionResult(success=False, tool_used=tool_name)
        
        # Check circuit breaker
        if self.config.circuit_breaker_enabled:
            breaker = self._get_circuit_breaker(tool_name)
            if breaker.is_open:
                result.circuit_state = breaker.state
                self._emit("circuit_open", {
                    "tool": tool_name,
                    "reset_timeout": breaker.reset_timeout,
                })
                
                # Try fallback if circuit is open
                if self.config.fallback_enabled and fallback_fn:
                    return await self._try_fallbacks(
                        tool_name, args, fallback_fn, result, start_time
                    )
                
                result.error = RuntimeError(f"Circuit breaker open for {tool_name}")
                result.recovery_action = RecoveryAction.SKIP
                return result
        
        # Check failure memory for known bad patterns
        if self.failure_memory and self.failure_memory.should_avoid(tool_name, args):
            # Try fallback for known bad patterns
            if self.config.fallback_enabled and fallback_fn:
                self._emit("fallback", {
                    "from_tool": tool_name,
                    "to_tool": "alternative",
                    "reason": "known_failure_pattern",
                })
                return await self._try_fallbacks(
                    tool_name, args, fallback_fn, result, start_time
                )
        
        # Execute with retry
        retry_policy = self.config.get_retry_policy(tool_name)
        timeout = self.config.get_timeout(tool_name)
        last_error: Exception | None = None
        
        for attempt in range(retry_policy.max_retries + 1):
            result.attempts = attempt + 1
            
            try:
                # Execute with timeout
                call_result = await asyncio.wait_for(
                    self._execute_tool(tool_fn, args, tool_obj=tool_obj),
                    timeout=timeout,
                )
                
                # Success!
                result.success = True
                result.result = call_result
                result.total_time_ms = (time.time() - start_time) * 1000
                
                # Update circuit breaker and memory
                if self.config.circuit_breaker_enabled:
                    breaker = self._get_circuit_breaker(tool_name)
                    breaker.record_success()
                
                if self.failure_memory:
                    self.failure_memory.record_success(tool_name)
                    if attempt > 0:
                        self.failure_memory.record_recovery(tool_name, "retry")
                        self._emit("recovery", {"method": "retry", "attempts": attempt + 1})
                
                return result
                
            except asyncio.TimeoutError as e:
                last_error = TimeoutError(f"Tool {tool_name} timed out after {timeout}s")
            except Exception as e:
                last_error = e
            
            # Record failure
            if self.config.circuit_breaker_enabled:
                breaker = self._get_circuit_breaker(tool_name)
                breaker.record_failure()
                result.circuit_state = breaker.state
            
            if self.failure_memory and last_error:
                self.failure_memory.record_failure(
                    tool_name, last_error, args, attempt + 1
                )
            
            # Check if we should retry
            if last_error and retry_policy.should_retry(last_error, attempt + 1):
                delay = retry_policy.get_delay(attempt + 1)
                self._emit("retry", {
                    "tool": tool_name,
                    "attempt": attempt + 1,
                    "max_retries": retry_policy.max_retries,
                    "delay": delay,
                    "error": str(last_error),
                })
                await asyncio.sleep(delay)
            else:
                break
        
        # All retries exhausted, try fallbacks
        result.error = last_error
        
        if self.config.fallback_enabled and fallback_fn:
            fallback_result = await self._try_fallbacks(
                tool_name, args, fallback_fn, result, start_time
            )
            if fallback_result.success:
                return fallback_result
        
        # Complete failure
        result.total_time_ms = (time.time() - start_time) * 1000
        result.recovery_action = self.config.on_failure
        return result
    
    async def _execute_tool(
        self,
        tool_fn: Callable[..., Any],
        args: dict[str, Any],
        tool_obj: Any | None = None,
    ) -> Any:
        """Execute tool function, handling both sync and async.
        
        Args:
            tool_fn: The invoke method or callable
            args: Arguments dict for the tool
            tool_obj: Optional tool object (for ainvoke detection)
        """
        # Check if the tool object has ainvoke (async tool interface)
        if tool_obj is not None and hasattr(tool_obj, "ainvoke"):
            # Tool with async interface - use ainvoke for proper async support
            return await tool_obj.ainvoke(args)
        elif inspect.iscoroutinefunction(tool_fn):
            return await tool_fn(args)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: tool_fn(args))
    
    async def _try_fallbacks(
        self,
        primary_tool: str,
        args: dict[str, Any],
        fallback_fn: Callable[[str], Callable[..., Any]],
        result: ExecutionResult,
        start_time: float,
    ) -> ExecutionResult:
        """Try fallback tools in order."""
        fallbacks = self.fallback_registry.get_all_fallbacks(primary_tool)
        
        for i, fallback_name in enumerate(fallbacks):
            self._emit("fallback", {
                "from_tool": primary_tool,
                "to_tool": fallback_name,
                "attempt": i + 1,
            })
            
            result.fallback_chain.append(fallback_name)
            
            try:
                fallback_tool = fallback_fn(fallback_name)
                if fallback_tool is None:
                    continue
                
                # Transform args for fallback
                transformed_args = self.fallback_registry.transform_args(
                    primary_tool, fallback_name, args
                )
                
                # Execute fallback with timeout (no retry for fallbacks)
                timeout = self.config.get_timeout(fallback_name)
                call_result = await asyncio.wait_for(
                    self._execute_tool(fallback_tool, transformed_args),
                    timeout=timeout,
                )
                
                # Fallback succeeded!
                result.success = True
                result.result = call_result
                result.tool_used = fallback_name
                result.used_fallback = True
                result.total_time_ms = (time.time() - start_time) * 1000
                
                if self.failure_memory:
                    self.failure_memory.record_recovery(primary_tool, f"fallback:{fallback_name}")
                
                self._emit("recovery", {
                    "method": "fallback",
                    "tool": fallback_name,
                })
                
                return result
                
            except Exception as e:
                # Fallback failed, try next
                result.error = e
                continue
        
        # All fallbacks failed
        result.total_time_ms = (time.time() - start_time) * 1000
        result.recovery_action = self.config.on_failure
        return result
    
    def get_circuit_status(self, tool_name: str) -> dict[str, Any]:
        """Get circuit breaker status for a tool."""
        if tool_name not in self._circuit_breakers:
            return {"state": "no_breaker"}
        return self._circuit_breakers[tool_name].get_status()
    
    def get_all_circuit_status(self) -> dict[str, dict[str, Any]]:
        """Get circuit breaker status for all tools."""
        return {
            name: breaker.get_status()
            for name, breaker in self._circuit_breakers.items()
        }
    
    def reset_circuit(self, tool_name: str) -> None:
        """Reset circuit breaker for a tool."""
        if tool_name in self._circuit_breakers:
            self._circuit_breakers[tool_name].reset()
    
    def reset_all_circuits(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._circuit_breakers.values():
            breaker.reset()
    
    def get_failure_suggestions(self, tool_name: str) -> dict[str, Any] | None:
        """Get suggestions for a failing tool based on learned patterns."""
        if self.failure_memory:
            return self.failure_memory.get_suggestions(tool_name)
        return None


@dataclass
class ModelExecutionResult:
    """Result of a resilient model (LLM) execution."""
    
    success: bool
    result: Any = None
    error: Exception | None = None
    attempts: int = 1
    total_time_ms: float = 0.0
    recovery_action: RecoveryAction | None = None
    error_type: str | None = None  # e.g., "rate_limit", "timeout", "api_error"


class ModelResilience:
    """
    Resilience layer specifically for LLM model calls.
    
    Provides retry with exponential backoff for API errors like:
    - Rate limits (429, ResourceExhausted)
    - Server errors (500, 502, 503)
    - Timeouts
    - Connection errors
    
    Designed to wrap model.ainvoke() calls with intelligent retry.
    
    Example:
        ```python
        from agenticflow.agent.resilience import ModelResilience, RetryPolicy
        
        # Create resilience with aggressive retry for rate limits
        resilience = ModelResilience(
            retry_policy=RetryPolicy(
                max_retries=5,
                base_delay=2.0,  # Start with 2s delay
                max_delay=60.0,  # Cap at 60s
            ),
            on_retry=lambda attempt, error, delay: print(
                f"⚠️ Retry {attempt}: {error} (waiting {delay:.1f}s)"
            ),
        )
        
        # Wrap LLM call
        result = await resilience.execute(
            model.ainvoke,
            messages,
        )
        
        if result.success:
            response = result.result
        else:
            raise result.error
        ```
    """
    
    def __init__(
        self,
        retry_policy: RetryPolicy | None = None,
        timeout_seconds: float = 120.0,
        on_retry: Callable[[int, Exception, float], None] | None = None,
        on_error: Callable[[Exception, int], None] | None = None,
        on_success: Callable[[int], None] | None = None,
        tracker: "ProgressTracker | None" = None,
    ) -> None:
        """Initialize ModelResilience.
        
        Args:
            retry_policy: Retry configuration. Defaults to aggressive policy.
            timeout_seconds: Timeout for each LLM call.
            on_retry: Callback(attempt, error, delay) before each retry.
            on_error: Callback(error, attempts) on final failure.
            on_success: Callback(attempts) on success.
            tracker: Optional progress tracker for observability.
        """
        self.retry_policy = retry_policy or RetryPolicy.aggressive()
        self.timeout = timeout_seconds
        self.on_retry = on_retry
        self.on_error = on_error
        self.on_success = on_success
        self.tracker = tracker
        
        # Stats
        self._total_calls = 0
        self._total_retries = 0
        self._total_failures = 0
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for reporting."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if any(p in error_str or p in error_type for p in ["rate", "429", "resourceexhausted", "quota"]):
            return "rate_limit"
        if any(p in error_str for p in ["timeout", "timed out"]):
            return "timeout"
        if any(p in error_str for p in ["connection", "network"]):
            return "connection"
        if any(p in error_str for p in ["500", "502", "503", "server error", "unavailable"]):
            return "server_error"
        if any(p in error_str for p in ["invalid", "malformed"]):
            return "invalid_request"
        return "api_error"
    
    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit progress event if tracker is available."""
        if not self.tracker:
            return
        
        if event_type == "retry":
            attempt = data.get("attempt", 1)
            max_retries = data.get("max_retries", 3)
            delay = data.get("delay", 0)
            error = data.get("error", "unknown")
            error_type = data.get("error_type", "error")
            
            emoji = "🔄" if error_type != "rate_limit" else "⏳"
            self.tracker.update(
                f"{emoji} LLM retry {attempt}/{max_retries} ({error_type}) - "
                f"waiting {delay:.1f}s..."
            )
        elif event_type == "error":
            error_type = data.get("error_type", "error")
            attempts = data.get("attempts", 1)
            self.tracker.update(
                f"❌ LLM failed after {attempts} attempts ({error_type})"
            )
        elif event_type == "success":
            attempts = data.get("attempts", 1)
            if attempts > 1:
                self.tracker.update(
                    f"✅ LLM succeeded after {attempts} attempts"
                )
    
    async def execute(
        self,
        model_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> ModelExecutionResult:
        """
        Execute an LLM call with resilience.
        
        Args:
            model_fn: The model method to call (e.g., model.ainvoke).
            *args: Positional arguments for the model method.
            **kwargs: Keyword arguments for the model method.
            
        Returns:
            ModelExecutionResult with success/failure details.
            
        Example:
            ```python
            result = await resilience.execute(model.ainvoke, messages)
            if result.success:
                ai_message = result.result
            ```
        """
        self._total_calls += 1
        start_time = time.time()
        result = ModelExecutionResult(success=False)
        last_error: Exception | None = None
        
        for attempt in range(self.retry_policy.max_retries + 1):
            result.attempts = attempt + 1
            
            try:
                # Execute with timeout
                call_result = await asyncio.wait_for(
                    model_fn(*args, **kwargs),
                    timeout=self.timeout,
                )
                
                # Success!
                result.success = True
                result.result = call_result
                result.total_time_ms = (time.time() - start_time) * 1000
                
                # Callbacks and tracking
                if self.on_success:
                    self.on_success(attempt + 1)
                
                if attempt > 0:
                    self._emit("success", {"attempts": attempt + 1})
                
                return result
                
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"LLM call timed out after {self.timeout}s")
            except Exception as e:
                last_error = e
            
            # Classify error and check if retryable
            error_type = self._classify_error(last_error)
            result.error_type = error_type
            
            # Check if we should retry
            if last_error and self.retry_policy.should_retry(last_error, attempt + 1):
                delay = self.retry_policy.get_delay(attempt + 1)
                self._total_retries += 1
                
                # Callbacks
                if self.on_retry:
                    self.on_retry(attempt + 1, last_error, delay)
                
                self._emit("retry", {
                    "attempt": attempt + 1,
                    "max_retries": self.retry_policy.max_retries,
                    "delay": delay,
                    "error": str(last_error)[:100],
                    "error_type": error_type,
                })
                
                await asyncio.sleep(delay)
            else:
                break
        
        # All retries exhausted
        result.error = last_error
        result.total_time_ms = (time.time() - start_time) * 1000
        result.recovery_action = RecoveryAction.ABORT
        self._total_failures += 1
        
        # Callbacks
        if self.on_error:
            self.on_error(last_error, result.attempts)
        
        self._emit("error", {
            "attempts": result.attempts,
            "error_type": result.error_type,
        })
        
        return result
    
    @property
    def stats(self) -> dict[str, int]:
        """Get resilience statistics."""
        return {
            "total_calls": self._total_calls,
            "total_retries": self._total_retries,
            "total_failures": self._total_failures,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_calls = 0
        self._total_retries = 0
        self._total_failures = 0
