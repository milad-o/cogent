"""
Tests for the interceptors module.
"""

import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

from agenticflow.interceptors.base import (
    Interceptor,
    InterceptContext,
    InterceptResult,
    Phase,
    StopExecution,
    run_interceptors,
)
from agenticflow.interceptors.budget import (
    BudgetGuard,
    BudgetExhaustedError,
    ExitBehavior,
)
from agenticflow.interceptors.context import (
    ContextCompressor,
    TokenLimiter,
    _estimate_tokens,
    _messages_to_text,
)
from agenticflow.interceptors.security import (
    PIIShield,
    PIIAction,
    ContentFilter,
)
from agenticflow.interceptors.ratelimit import RateLimiter, ThrottleInterceptor
from agenticflow.interceptors.audit import Auditor, AuditEvent, AuditTraceType


# =============================================================================
# Test InterceptResult
# =============================================================================

class TestInterceptResult:
    """Test InterceptResult factory methods."""
    
    def test_ok(self):
        result = InterceptResult.ok()
        assert result.proceed is True
        assert result.modified_messages is None
        assert result.skip_action is False
    
    def test_stop(self):
        result = InterceptResult.stop("Stopped!")
        assert result.proceed is False
        assert result.final_response == "Stopped!"
    
    def test_stop_no_response(self):
        result = InterceptResult.stop()
        assert result.proceed is False
        assert result.final_response is None
    
    def test_skip(self):
        result = InterceptResult.skip()
        assert result.proceed is True
        assert result.skip_action is True
    
    def test_modify_messages(self):
        new_msgs = [{"role": "user", "content": "hello"}]
        result = InterceptResult.modify_messages(new_msgs)
        assert result.proceed is True
        assert result.modified_messages == new_msgs
    
    def test_modify_args(self):
        new_args = {"key": "value"}
        result = InterceptResult.modify_args(new_args)
        assert result.proceed is True
        assert result.modified_tool_args == new_args


# =============================================================================
# Test InterceptContext
# =============================================================================

class TestInterceptContext:
    """Test InterceptContext creation."""
    
    def test_create_minimal(self):
        mock_agent = MagicMock()
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_RUN,
            task="Test task",
            messages=[],
        )
        assert ctx.agent is mock_agent
        assert ctx.phase == Phase.PRE_RUN
        assert ctx.task == "Test task"
        assert ctx.messages == []
        assert ctx.state == {}
        assert ctx.model_calls == 0
        assert ctx.tool_calls == 0
    
    def test_create_with_tool_info(self):
        mock_agent = MagicMock()
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_ACT,
            task="Test",
            messages=[],
            tool_name="search",
            tool_args={"query": "hello"},
        )
        assert ctx.tool_name == "search"
        assert ctx.tool_args == {"query": "hello"}
    
    def test_shared_state(self):
        mock_agent = MagicMock()
        state = {"counter": 0}
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_RUN,
            task="Test",
            messages=[],
            state=state,
        )
        ctx.state["counter"] = 1
        assert state["counter"] == 1  # Same dict


# =============================================================================
# Test Phase Enum
# =============================================================================

class TestPhase:
    """Test Phase enum."""
    
    def test_all_phases_exist(self):
        assert Phase.PRE_RUN.value == "pre_run"
        assert Phase.PRE_THINK.value == "pre_think"
        assert Phase.POST_THINK.value == "post_think"
        assert Phase.PRE_ACT.value == "pre_act"
        assert Phase.POST_ACT.value == "post_act"
        assert Phase.POST_RUN.value == "post_run"
        assert Phase.ON_ERROR.value == "on_error"


# =============================================================================
# Test Interceptor Base Class
# =============================================================================

class TestInterceptorBase:
    """Test Interceptor base class."""
    
    @pytest.mark.asyncio
    async def test_default_implementations_return_ok(self):
        """All default phase methods should return ok()."""
        
        class EmptyInterceptor(Interceptor):
            pass
        
        interceptor = EmptyInterceptor()
        mock_agent = MagicMock()
        
        for phase in Phase:
            ctx = InterceptContext(
                agent=mock_agent,
                phase=phase,
                task="Test",
                messages=[],
            )
            result = await interceptor.intercept(ctx)
            assert result.proceed is True
    
    @pytest.mark.asyncio
    async def test_dispatch_routes_to_phase_method(self):
        """intercept() should route to the correct phase method."""
        
        class TrackingInterceptor(Interceptor):
            def __init__(self):
                self.called_phases = []
            
            async def pre_run(self, ctx):
                self.called_phases.append("pre_run")
                return InterceptResult.ok()
            
            async def pre_think(self, ctx):
                self.called_phases.append("pre_think")
                return InterceptResult.ok()
        
        interceptor = TrackingInterceptor()
        mock_agent = MagicMock()
        
        ctx1 = InterceptContext(agent=mock_agent, phase=Phase.PRE_RUN, task="", messages=[])
        await interceptor.intercept(ctx1)
        
        ctx2 = InterceptContext(agent=mock_agent, phase=Phase.PRE_THINK, task="", messages=[])
        await interceptor.intercept(ctx2)
        
        assert interceptor.called_phases == ["pre_run", "pre_think"]
    
    def test_name_property(self):
        class MyInterceptor(Interceptor):
            pass
        
        assert MyInterceptor().name == "MyInterceptor"


# =============================================================================
# Test run_interceptors
# =============================================================================

class TestRunInterceptors:
    """Test the run_interceptors function."""
    
    @pytest.mark.asyncio
    async def test_empty_list_returns_ok(self):
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.PRE_RUN, task="", messages=[])
        result = await run_interceptors([], ctx)
        assert result.proceed is True
    
    @pytest.mark.asyncio
    async def test_single_interceptor(self):
        class SimpleInterceptor(Interceptor):
            async def pre_run(self, ctx):
                return InterceptResult.ok()
        
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.PRE_RUN, task="", messages=[])
        result = await run_interceptors([SimpleInterceptor()], ctx)
        assert result.proceed is True
    
    @pytest.mark.asyncio
    async def test_stop_halts_chain(self):
        class Stopper(Interceptor):
            async def pre_run(self, ctx):
                return InterceptResult.stop("Stopped!")
        
        class NeverCalled(Interceptor):
            def __init__(self):
                self.called = False
            
            async def pre_run(self, ctx):
                self.called = True
                return InterceptResult.ok()
        
        never = NeverCalled()
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.PRE_RUN, task="", messages=[])
        
        result = await run_interceptors([Stopper(), never], ctx)
        
        assert result.proceed is False
        assert result.final_response == "Stopped!"
        assert never.called is False
    
    @pytest.mark.asyncio
    async def test_modifications_accumulate(self):
        class ModifyMessages(Interceptor):
            async def pre_run(self, ctx):
                return InterceptResult.modify_messages([{"role": "system", "content": "modified"}])
        
        class ModifyTask(Interceptor):
            async def pre_run(self, ctx):
                return InterceptResult(proceed=True, modified_task="new task")
        
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.PRE_RUN, task="old", messages=[])
        
        result = await run_interceptors([ModifyMessages(), ModifyTask()], ctx)
        
        assert result.modified_messages == [{"role": "system", "content": "modified"}]
        assert result.modified_task == "new task"
    
    @pytest.mark.asyncio
    async def test_stop_execution_exception(self):
        class RaiseStop(Interceptor):
            async def pre_run(self, ctx):
                raise StopExecution("Emergency stop!", "test reason")
        
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.PRE_RUN, task="", messages=[])
        
        result = await run_interceptors([RaiseStop()], ctx)
        
        assert result.proceed is False
        assert result.final_response == "Emergency stop!"


# =============================================================================
# Test BudgetGuard
# =============================================================================

class TestBudgetGuard:
    """Test BudgetGuard interceptor."""
    
    def test_default_values(self):
        guard = BudgetGuard()
        assert guard.model_calls == 0  # unlimited
        assert guard.tool_calls == 0   # unlimited
        assert guard.exit_behavior == ExitBehavior.STOP
    
    def test_custom_values(self):
        guard = BudgetGuard(model_calls=10, tool_calls=50, exit_behavior="error")
        assert guard.model_calls == 10
        assert guard.tool_calls == 50
        assert guard.exit_behavior == ExitBehavior.ERROR
    
    @pytest.mark.asyncio
    async def test_pre_run_resets_counters(self):
        guard = BudgetGuard()
        guard._current_model_calls = 5
        guard._current_tool_calls = 10
        
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.PRE_RUN, task="", messages=[])
        
        await guard.pre_run(ctx)
        
        assert guard._current_model_calls == 0
        assert guard._current_tool_calls == 0
    
    @pytest.mark.asyncio
    async def test_post_think_increments_model_counter(self):
        guard = BudgetGuard(model_calls=10)
        
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.POST_THINK, task="", messages=[])
        
        await guard.post_think(ctx)
        
        assert guard._current_model_calls == 1
    
    @pytest.mark.asyncio
    async def test_post_act_increments_tool_counter(self):
        guard = BudgetGuard(tool_calls=10)
        
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.POST_ACT, task="", messages=[])
        
        await guard.post_act(ctx)
        
        assert guard._current_tool_calls == 1
    
    @pytest.mark.asyncio
    async def test_model_limit_stops_execution(self):
        guard = BudgetGuard(model_calls=2)
        guard._current_model_calls = 2  # Already at limit
        
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.PRE_THINK, task="", messages=[])
        
        result = await guard.pre_think(ctx)
        
        assert result.proceed is False
        assert "call limit" in result.final_response.lower()
    
    @pytest.mark.asyncio
    async def test_tool_limit_stops_execution(self):
        guard = BudgetGuard(tool_calls=5)
        guard._current_tool_calls = 5  # Already at limit
        
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.PRE_ACT, task="", messages=[])
        
        result = await guard.pre_act(ctx)
        
        assert result.proceed is False
        assert "call limit" in result.final_response.lower()
    
    @pytest.mark.asyncio
    async def test_error_exit_behavior_raises(self):
        guard = BudgetGuard(model_calls=1, exit_behavior=ExitBehavior.ERROR)
        guard._current_model_calls = 1  # At limit
        
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.PRE_THINK, task="", messages=[])
        
        with pytest.raises(BudgetExhaustedError) as exc_info:
            await guard.pre_think(ctx)
        
        assert exc_info.value.call_type == "model"
        assert exc_info.value.limit == 1
    
    def test_budget_remaining_unlimited(self):
        guard = BudgetGuard()  # No limits
        assert guard.model_budget_remaining is None
        assert guard.tool_budget_remaining is None
    
    def test_budget_remaining_limited(self):
        guard = BudgetGuard(model_calls=10, tool_calls=20)
        guard._current_model_calls = 3
        guard._current_tool_calls = 5
        
        assert guard.model_budget_remaining == 7
        assert guard.tool_budget_remaining == 15


# =============================================================================
# Test StopExecution Exception
# =============================================================================

class TestStopExecution:
    """Test StopExecution exception."""
    
    def test_basic(self):
        exc = StopExecution("Response text")
        assert exc.response == "Response text"
        assert exc.reason == ""
    
    def test_with_reason(self):
        exc = StopExecution("Response", "Budget exhausted")
        assert exc.response == "Response"
        assert exc.reason == "Budget exhausted"
        assert str(exc) == "Budget exhausted"


# =============================================================================
# Integration Test: Multiple Interceptors
# =============================================================================

class TestInterceptorChaining:
    """Test chaining multiple interceptors."""
    
    @pytest.mark.asyncio
    async def test_order_matters(self):
        """Interceptors should run in order."""
        order = []
        
        class First(Interceptor):
            async def pre_run(self, ctx):
                order.append(1)
                return InterceptResult.ok()
        
        class Second(Interceptor):
            async def pre_run(self, ctx):
                order.append(2)
                return InterceptResult.ok()
        
        class Third(Interceptor):
            async def pre_run(self, ctx):
                order.append(3)
                return InterceptResult.ok()
        
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.PRE_RUN, task="", messages=[])
        
        await run_interceptors([First(), Second(), Third()], ctx)
        
        assert order == [1, 2, 3]
    
    @pytest.mark.asyncio
    async def test_shared_state(self):
        """Interceptors should share state."""
        
        class Writer(Interceptor):
            async def pre_run(self, ctx):
                ctx.state["value"] = 42
                return InterceptResult.ok()
        
        class Reader(Interceptor):
            def __init__(self):
                self.read_value = None
            
            async def pre_run(self, ctx):
                self.read_value = ctx.state.get("value")
                return InterceptResult.ok()
        
        reader = Reader()
        mock_agent = MagicMock()
        ctx = InterceptContext(agent=mock_agent, phase=Phase.PRE_RUN, task="", messages=[])
        
        await run_interceptors([Writer(), reader], ctx)
        
        assert reader.read_value == 42


# =============================================================================
# Test ContextCompressor
# =============================================================================

class TestContextCompressor:
    """Test ContextCompressor interceptor."""
    
    def test_default_values(self):
        compressor = ContextCompressor()
        assert compressor.threshold_tokens == 8000
        assert compressor.keep_recent == 4
    
    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold_tokens must be at least 1000"):
            ContextCompressor(threshold_tokens=500)
    
    def test_invalid_keep_recent(self):
        with pytest.raises(ValueError, match="keep_recent must be at least 1"):
            ContextCompressor(keep_recent=0)
    
    @pytest.mark.asyncio
    async def test_no_compression_under_threshold(self):
        """Should not compress when under token limit."""
        compressor = ContextCompressor(threshold_tokens=8000)
        
        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "Hello"}]  # Very short
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_THINK,
            task="test",
            messages=messages,
        )
        
        result = await compressor.pre_think(ctx)
        
        assert result.proceed is True
        assert result.modified_messages is None


class TestTokenLimiter:
    """Test TokenLimiter interceptor."""
    
    def test_default_values(self):
        limiter = TokenLimiter()
        assert limiter.max_tokens == 16000
    
    @pytest.mark.asyncio
    async def test_under_limit_continues(self):
        limiter = TokenLimiter(max_tokens=1000)
        
        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "Hi"}]  # ~1 token
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_THINK,
            task="test",
            messages=messages,
        )
        
        result = await limiter.pre_think(ctx)
        
        assert result.proceed is True
    
    @pytest.mark.asyncio
    async def test_over_limit_stops(self):
        limiter = TokenLimiter(max_tokens=5)  # Very low limit
        
        mock_agent = MagicMock()
        # This message is ~60+ chars / 4 = ~15+ tokens, well above limit of 5
        messages = [{"role": "user", "content": "This is a much longer message that definitely exceeds limit tokens"}]
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_THINK,
            task="test",
            messages=messages,
        )
        
        result = await limiter.pre_think(ctx)
        
        assert result.proceed is False
        assert "limit reached" in result.final_response.lower()


class TestTokenEstimation:
    """Test token estimation helpers."""
    
    def test_estimate_tokens(self):
        # 4 chars ≈ 1 token
        assert _estimate_tokens("1234") == 1
        assert _estimate_tokens("12345678") == 2
    
    def test_messages_to_text(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        text = _messages_to_text(messages)
        assert "Hello" in text
        assert "Hi there" in text


# =============================================================================
# Test PIIShield
# =============================================================================

class TestPIIShield:
    """Test PIIShield interceptor."""
    
    def test_default_values(self):
        shield = PIIShield()
        assert "all" in shield.patterns
        assert shield.action == PIIAction.MASK
    
    def test_scan_email(self):
        shield = PIIShield(patterns=["email"])
        matches = shield._scan_text("Contact me at test@example.com please")
        assert len(matches) == 1
        assert matches[0][0] == "email"
        assert matches[0][1] == "test@example.com"
    
    def test_scan_phone(self):
        shield = PIIShield(patterns=["phone_us"])
        matches = shield._scan_text("Call me at 555-123-4567")
        assert len(matches) == 1
        assert matches[0][0] == "phone_us"
    
    def test_scan_ssn(self):
        shield = PIIShield(patterns=["ssn"])
        matches = shield._scan_text("SSN: 123-45-6789")
        assert len(matches) == 1
        assert matches[0][0] == "ssn"
    
    def test_scan_credit_card(self):
        shield = PIIShield(patterns=["credit_card"])
        matches = shield._scan_text("Card: 4111111111111111")  # Test Visa
        assert len(matches) == 1
        assert matches[0][0] == "credit_card"
    
    def test_mask_text(self):
        shield = PIIShield(patterns=["email"])
        masked, detections = shield._mask_text("Email: user@test.com")
        assert "[EMAIL_REDACTED]" in masked
        assert "user@test.com" not in masked
        assert len(detections) == 1
    
    def test_mask_multiple(self):
        shield = PIIShield(patterns=["email", "phone_us"])
        text = "Email: a@b.com, Phone: 555-111-2222"
        masked, detections = shield._mask_text(text)
        assert "[EMAIL_REDACTED]" in masked
        assert "[PHONE_US_REDACTED]" in masked
        assert len(detections) == 2
    
    @pytest.mark.asyncio
    async def test_mask_action_modifies_messages(self):
        shield = PIIShield(patterns=["email"], action=PIIAction.MASK)
        
        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "My email is test@example.com"}]
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_THINK,
            task="test",
            messages=messages,
        )
        
        result = await shield.pre_think(ctx)
        
        assert result.proceed is True
        assert result.modified_messages is not None
        assert "[EMAIL_REDACTED]" in result.modified_messages[0]["content"]
    
    @pytest.mark.asyncio
    async def test_block_action_stops(self):
        shield = PIIShield(patterns=["ssn"], action=PIIAction.BLOCK)
        
        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "My SSN is 123-45-6789"}]
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_THINK,
            task="test",
            messages=messages,
        )
        
        with pytest.raises(StopExecution):
            await shield.pre_think(ctx)
    
    @pytest.mark.asyncio
    async def test_warn_action_continues(self):
        shield = PIIShield(patterns=["email"], action=PIIAction.WARN)
        
        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "Email: test@test.com"}]
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_THINK,
            task="test",
            messages=messages,
        )
        
        result = await shield.pre_think(ctx)
        
        assert result.proceed is True
        # Detections tracked but messages not modified
        assert result.modified_messages is None
        assert len(ctx.state["pii_shield"]["detections"]) > 0
    
    @pytest.mark.asyncio
    async def test_no_pii_passes_through(self):
        shield = PIIShield(patterns=["email", "ssn"])
        
        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_THINK,
            task="test",
            messages=messages,
        )
        
        result = await shield.pre_think(ctx)
        
        assert result.proceed is True
        assert result.modified_messages is None
    
    def test_custom_pattern(self):
        shield = PIIShield(
            patterns=[],
            custom_patterns={"employee_id": r"EMP-\d{6}"},
        )
        matches = shield._scan_text("Employee ID: EMP-123456")
        assert len(matches) == 1
        assert matches[0][0] == "employee_id"


# =============================================================================
# Test ContentFilter
# =============================================================================

class TestContentFilter:
    """Test ContentFilter interceptor."""
    
    def test_blocked_words(self):
        filter_ = ContentFilter(blocked_words=["password", "secret"])
        matches = filter_._check_text("The password is secret123")
        assert len(matches) >= 1  # At least "password" matches
    
    def test_case_insensitive(self):
        filter_ = ContentFilter(blocked_words=["password"], case_sensitive=False)
        matches = filter_._check_text("PASSWORD is here")
        assert len(matches) == 1
    
    def test_case_sensitive(self):
        filter_ = ContentFilter(blocked_words=["password"], case_sensitive=True)
        matches_upper = filter_._check_text("PASSWORD is here")
        matches_lower = filter_._check_text("password is here")
        assert len(matches_upper) == 0
        assert len(matches_lower) == 1
    
    @pytest.mark.asyncio
    async def test_block_action(self):
        filter_ = ContentFilter(blocked_words=["forbidden"], action="block")
        
        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "The forbidden word"}]
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_THINK,
            task="test",
            messages=messages,
        )
        
        with pytest.raises(StopExecution):
            await filter_.pre_think(ctx)
    
    @pytest.mark.asyncio
    async def test_no_match_passes(self):
        filter_ = ContentFilter(blocked_words=["forbidden"])
        
        mock_agent = MagicMock()
        messages = [{"role": "user", "content": "A normal message"}]
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_THINK,
            task="test",
            messages=messages,
        )
        
        result = await filter_.pre_think(ctx)
        
        assert result.proceed is True
    
    def test_regex_pattern(self):
        filter_ = ContentFilter(blocked_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"])
        matches = filter_._check_text("SSN format: 123-45-6789")
        assert len(matches) == 1


# =============================================================================
# Test RateLimiter
# =============================================================================

class TestRateLimiter:
    """Test RateLimiter interceptor."""
    
    def test_default_values(self):
        limiter = RateLimiter()
        assert limiter.calls_per_window == 10
        assert limiter.window_seconds == 60.0
        assert limiter.action == "wait"
        assert limiter.per_tool is False
    
    def test_invalid_calls(self):
        with pytest.raises(ValueError, match="calls_per_window must be at least 1"):
            RateLimiter(calls_per_window=0)
    
    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            RateLimiter(window_seconds=0)
    
    def test_invalid_action(self):
        with pytest.raises(ValueError, match="action must be"):
            RateLimiter(action="invalid")
    
    @pytest.mark.asyncio
    async def test_under_limit_passes(self):
        limiter = RateLimiter(calls_per_window=5, window_seconds=60)
        
        mock_agent = MagicMock()
        mock_agent.name = "test"
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_ACT,
            task="test",
            messages=[],
            tool_name="search",
        )
        
        result = await limiter.pre_act(ctx)
        assert result.proceed is True
    
    @pytest.mark.asyncio
    async def test_record_call(self):
        limiter = RateLimiter(calls_per_window=5)
        
        mock_agent = MagicMock()
        mock_agent.name = "test"
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.POST_ACT,
            task="test",
            messages=[],
            tool_name="search",
        )
        
        await limiter.post_act(ctx)
        
        usage = limiter.current_usage
        assert usage["global_calls"] == 1
    
    def test_reset(self):
        limiter = RateLimiter()
        limiter._record_call("test")
        assert limiter.current_usage["global_calls"] == 1
        
        limiter.reset()
        assert limiter.current_usage["global_calls"] == 0


class TestThrottleInterceptor:
    """Test ThrottleInterceptor."""
    
    def test_default_values(self):
        throttle = ThrottleInterceptor()
        assert throttle.min_delay == 0.5
        assert throttle.per_tool is False
    
    def test_invalid_delay(self):
        with pytest.raises(ValueError, match="min_delay must be non-negative"):
            ThrottleInterceptor(min_delay=-1)
    
    @pytest.mark.asyncio
    async def test_first_call_no_delay(self):
        throttle = ThrottleInterceptor(min_delay=1.0)
        
        mock_agent = MagicMock()
        mock_agent.name = "test"
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_ACT,
            task="test",
            messages=[],
            tool_name="search",
        )
        
        import time
        start = time.monotonic()
        result = await throttle.pre_act(ctx)
        elapsed = time.monotonic() - start
        
        assert result.proceed is True
        assert elapsed < 0.1  # First call should be immediate


# =============================================================================
# Test Auditor
# =============================================================================

class TestAuditor:
    """Test Auditor interceptor."""
    
    def test_default_values(self):
        auditor = Auditor()
        assert auditor.log_to_file is None
        assert auditor.callback is None
        assert auditor.include_content is True
        assert auditor.max_content_length == 500
    
    @pytest.mark.asyncio
    async def test_logs_run_start(self):
        auditor = Auditor()
        
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_RUN,
            task="test task",
            messages=[],
        )
        
        await auditor.pre_run(ctx)
        
        assert len(auditor.events) == 1
        assert auditor.events[0].event_type == AuditTraceType.RUN_START
        assert auditor.events[0].agent_name == "TestAgent"
    
    @pytest.mark.asyncio
    async def test_logs_model_request(self):
        auditor = Auditor()
        
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_THINK,
            task="test",
            messages=[{"role": "user", "content": "Hello"}],
            model_calls=0,
        )
        
        await auditor.pre_think(ctx)
        
        assert len(auditor.events) == 1
        assert auditor.events[0].event_type == AuditTraceType.MODEL_REQUEST
        assert auditor.events[0].data["model_call_number"] == 1
    
    @pytest.mark.asyncio
    async def test_logs_tool_request(self):
        auditor = Auditor()
        
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_ACT,
            task="test",
            messages=[],
            tool_name="search",
            tool_args={"query": "hello"},
            tool_calls=0,
        )
        
        await auditor.pre_act(ctx)
        
        assert len(auditor.events) == 1
        assert auditor.events[0].event_type == AuditTraceType.TOOL_REQUEST
        assert auditor.events[0].data["tool_name"] == "search"
    
    @pytest.mark.asyncio
    async def test_callback_called(self):
        events_received = []
        
        def callback(event):
            events_received.append(event)
        
        auditor = Auditor(callback=callback)
        
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        ctx = InterceptContext(
            agent=mock_agent,
            phase=Phase.PRE_RUN,
            task="test",
            messages=[],
        )
        
        await auditor.pre_run(ctx)
        
        assert len(events_received) == 1
        assert events_received[0].event_type == AuditTraceType.RUN_START
    
    def test_summary(self):
        auditor = Auditor()
        auditor._events = [
            AuditEvent(
                timestamp="2024-01-01T00:00:00Z",
                event_type=AuditTraceType.RUN_START,
                agent_name="Test",
                task="test",
                phase=Phase.PRE_RUN,
            ),
            AuditEvent(
                timestamp="2024-01-01T00:00:01Z",
                event_type=AuditTraceType.MODEL_REQUEST,
                agent_name="Test",
                task="test",
                phase=Phase.PRE_THINK,
                duration_ms=100,
            ),
        ]
        
        summary = auditor.summary()
        assert summary["total_events"] == 2
        assert summary["by_type"]["run_start"] == 1
        assert summary["by_type"]["model_request"] == 1
    
    def test_clear(self):
        auditor = Auditor()
        auditor._events = [
            AuditEvent(
                timestamp="2024-01-01T00:00:00Z",
                event_type=AuditTraceType.RUN_START,
                agent_name="Test",
                task="test",
                phase=Phase.PRE_RUN,
            ),
        ]
        
        auditor.clear()
        assert len(auditor.events) == 0
    
    def test_redaction(self):
        auditor = Auditor(redact_patterns=["password=\\w+"])
        result = auditor._redact("Config: password=secret123")
        assert "secret123" not in result
        assert "[REDACTED]" in result
    
    def test_truncation(self):
        auditor = Auditor(max_content_length=10)
        result = auditor._truncate("This is a very long message")
        assert len(result) <= 13  # 10 + "..."
        assert result.endswith("...")


class TestAuditEvent:
    """Test AuditEvent dataclass."""
    
    def test_to_dict(self):
        event = AuditEvent(
            timestamp="2024-01-01T00:00:00Z",
            event_type=AuditTraceType.RUN_START,
            agent_name="TestAgent",
            task="test task",
            phase=Phase.PRE_RUN,
            data={"key": "value"},
            duration_ms=100.5,
        )
        
        d = event.to_dict()
        assert d["timestamp"] == "2024-01-01T00:00:00Z"
        assert d["event_type"] == "run_start"
        assert d["agent_name"] == "TestAgent"
        assert d["phase"] == "pre_run"
        assert d["duration_ms"] == 100.5
    
    def test_to_json(self):
        event = AuditEvent(
            timestamp="2024-01-01T00:00:00Z",
            event_type=AuditTraceType.RUN_START,
            agent_name="Test",
            task="test",
            phase=Phase.PRE_RUN,
        )
        
        json_str = event.to_json()
        assert '"event_type": "run_start"' in json_str


# =============================================================================
# Test Context Layer Interceptors
# =============================================================================

from agenticflow.interceptors.gates import ToolGate, PermissionGate, ConversationGate
from agenticflow.interceptors.failover import Failover, FailoverTrigger
from agenticflow.interceptors.guards import ToolGuard, CircuitBreaker
from agenticflow.interceptors.prompt import (
    PromptAdapter,
    ContextPrompt,
    ConversationPrompt,
    LambdaPrompt,
)
from agenticflow.core.context import RunContext, EMPTY_CONTEXT


class TestToolGate:
    """Test ToolGate base class."""

    @pytest.mark.asyncio
    async def test_pre_run_filters_tools(self):
        """ToolGate subclass filters tools via filter method."""
        class TestGate(ToolGate):
            async def filter(self, tools, ctx):
                return [t for t in tools if t.name != "banned"]

        gate = TestGate()
        
        # Create mock tools
        tool1 = MagicMock()
        tool1.name = "allowed"
        tool2 = MagicMock()
        tool2.name = "banned"
        
        mock_agent = MagicMock()
        context = InterceptContext(
            agent=mock_agent,
            task="task",
            phase=Phase.PRE_THINK,
            messages=[],
            tools=[tool1, tool2],
        )
        
        result = await gate.pre_think(context)
        assert result.proceed is True
        assert result.modified_tools == [tool1]


class TestPermissionGate:
    """Test PermissionGate interceptor."""

    def test_default_values(self):
        gate = PermissionGate()
        assert gate.default_tools == []

    def test_custom_default_tools(self):
        gate = PermissionGate(default_tools=["search", "read"])
        assert gate.default_tools == ["search", "read"]

    @pytest.mark.asyncio
    async def test_filter_with_default_tools(self):
        gate = PermissionGate(default_tools=["search"])
        
        tool1 = MagicMock()
        tool1.name = "search"
        tool2 = MagicMock()
        tool2.name = "delete"
        
        mock_agent = MagicMock()
        ctx = InterceptContext(
            agent=mock_agent,
            task="test",
            phase=Phase.PRE_THINK,
            messages=[],
        )
        
        filtered = await gate.filter([tool1, tool2], ctx)
        assert len(filtered) == 1
        assert filtered[0].name == "search"

    @pytest.mark.asyncio
    async def test_filter_with_wildcard(self):
        gate = PermissionGate(default_tools=["*"])
        
        tool1 = MagicMock()
        tool1.name = "search"
        tool2 = MagicMock()
        tool2.name = "delete"
        
        mock_agent = MagicMock()
        ctx = InterceptContext(
            agent=mock_agent,
            task="test",
            phase=Phase.PRE_THINK,
            messages=[],
        )
        
        filtered = await gate.filter([tool1, tool2], ctx)
        assert len(filtered) == 2


class TestConversationGate:
    """Test ConversationGate interceptor."""

    def test_creation_with_stages(self):
        stages = {0: ["greeting"], 5: ["search"]}
        gate = ConversationGate(stages=stages)
        assert gate.stages == stages

    @pytest.mark.asyncio
    async def test_filter_tools_by_message_count(self):
        stages = {
            0: ["greeting"],
            5: ["search"],
            10: ["write"],
        }
        gate = ConversationGate(stages=stages)
        
        tool1 = MagicMock()
        tool1.name = "greeting"
        tool2 = MagicMock()
        tool2.name = "search"
        tool3 = MagicMock()
        tool3.name = "write"
        
        mock_agent = MagicMock()
        
        # At 3 messages - only greeting available
        ctx3 = InterceptContext(
            agent=mock_agent,
            task="test",
            phase=Phase.PRE_THINK,
            messages=[{"role": "user", "content": "hi"}] * 3,
        )
        filtered = await gate.filter([tool1, tool2, tool3], ctx3)
        names = [t.name for t in filtered]
        assert "greeting" in names
        assert "search" not in names
        
        # At 7 messages - greeting + search
        ctx7 = InterceptContext(
            agent=mock_agent,
            task="test",
            phase=Phase.PRE_THINK,
            messages=[{"role": "user", "content": "hi"}] * 7,
        )
        filtered = await gate.filter([tool1, tool2, tool3], ctx7)
        names = [t.name for t in filtered]
        assert "greeting" in names
        assert "search" in names
        assert "write" not in names


class TestFailover:
    """Test Failover interceptor."""

    def test_creation_with_fallbacks(self):
        failover = Failover(fallbacks=["gpt-4", "gpt-3.5-turbo"])
        assert failover.fallbacks == ["gpt-4", "gpt-3.5-turbo"]

    def test_default_triggers(self):
        failover = Failover(fallbacks=["gpt-4"])
        # Default triggers are rate_limit, timeout, error
        assert FailoverTrigger.RATE_LIMIT in failover.triggers
        assert FailoverTrigger.TIMEOUT in failover.triggers
        assert FailoverTrigger.ERROR in failover.triggers

    def test_custom_triggers(self):
        failover = Failover(
            fallbacks=["gpt-4"],
            on=["rate_limit", "context_length"],
        )
        assert FailoverTrigger.RATE_LIMIT in failover.triggers
        assert FailoverTrigger.CONTEXT_LENGTH in failover.triggers
        assert FailoverTrigger.TIMEOUT not in failover.triggers

    def test_should_trigger_rate_limit(self):
        failover = Failover(fallbacks=["gpt-4"], on=["rate_limit"])
        
        class RateLimitError(Exception):
            pass
        
        assert failover._should_trigger(RateLimitError("rate limit exceeded")) is True
        assert failover._should_trigger(Exception("429 error")) is True
        assert failover._should_trigger(Exception("unknown error")) is False

    def test_should_trigger_timeout(self):
        failover = Failover(fallbacks=["gpt-4"], on=["timeout"])
        
        assert failover._should_trigger(TimeoutError("request timed out")) is True
        assert failover._should_trigger(Exception("connection timeout")) is True

    def test_should_trigger_error(self):
        failover = Failover(fallbacks=["gpt-4"], on=["error"])
        # ERROR trigger matches anything
        assert failover._should_trigger(Exception("any error")) is True


class TestToolGuard:
    """Test ToolGuard interceptor."""

    def test_default_values(self):
        guard = ToolGuard()
        assert guard.max_retries == 3
        assert guard.backoff == 2.0
        assert guard.initial_delay == 0.5
        assert guard.max_delay == 30.0

    def test_custom_values(self):
        guard = ToolGuard(
            max_retries=5,
            backoff=3.0,
            initial_delay=1.0,
            max_delay=60.0,
            retry_on=[TimeoutError, ConnectionError],
        )
        assert guard.max_retries == 5
        assert guard.backoff == 3.0
        assert guard.initial_delay == 1.0
        assert guard.max_delay == 60.0
        assert guard.retry_on == (TimeoutError, ConnectionError)

    def test_should_retry_default(self):
        guard = ToolGuard()
        # Default: retry on common transient errors
        assert guard._should_retry(TimeoutError()) is True
        assert guard._should_retry(ConnectionError()) is True
        # But not arbitrary exceptions
        assert guard._should_retry(ValueError()) is False

    def test_should_retry_specific_errors(self):
        guard = ToolGuard(retry_on=[ValueError, TypeError])
        assert guard._should_retry(ValueError()) is True
        assert guard._should_retry(TypeError()) is True
        assert guard._should_retry(RuntimeError()) is False

    def test_should_retry_skip_on(self):
        guard = ToolGuard(skip_on=[KeyboardInterrupt])
        assert guard._should_retry(KeyboardInterrupt()) is False

    def test_calculate_delay(self):
        guard = ToolGuard(initial_delay=1.0, backoff=2.0, max_delay=10.0)
        assert guard._calculate_delay(0) == 1.0
        assert guard._calculate_delay(1) == 2.0
        assert guard._calculate_delay(2) == 4.0
        assert guard._calculate_delay(3) == 8.0
        # Capped at max_delay
        assert guard._calculate_delay(4) == 10.0


class TestCircuitBreaker:
    """Test CircuitBreaker interceptor."""

    def test_default_values(self):
        cb = CircuitBreaker()
        assert cb.failure_threshold == 5
        assert cb.reset_timeout == 30.0
        assert cb.protected_tools is None  # All tools

    def test_custom_values(self):
        cb = CircuitBreaker(
            failure_threshold=3,
            reset_timeout=60.0,
            tools=["api_call", "database"],
        )
        assert cb.failure_threshold == 3
        assert cb.reset_timeout == 60.0
        assert cb.protected_tools == {"api_call", "database"}

    def test_is_protected(self):
        cb = CircuitBreaker(tools=["api_call"])
        assert cb._is_protected("api_call") is True
        assert cb._is_protected("other_tool") is False

    def test_is_protected_all(self):
        cb = CircuitBreaker()  # No specific tools = all protected
        assert cb._is_protected("any_tool") is True

    @pytest.mark.asyncio
    async def test_circuit_state_tracking(self):
        cb = CircuitBreaker(failure_threshold=2)
        mock_agent = MagicMock()
        
        # Simulate failures
        ctx = InterceptContext(
            agent=mock_agent,
            task="test",
            phase=Phase.POST_ACT,
            messages=[],
            state={},
            tool_name="api_call",
            tool_result="Error: connection failed",
        )
        
        # First failure
        await cb.post_act(ctx)
        circuit = ctx.state[cb.STATE_KEY].get("api_call", {})
        assert circuit.get("failures") == 1
        assert circuit.get("state") == "closed"
        
        # Second failure - should open circuit
        await cb.post_act(ctx)
        circuit = ctx.state[cb.STATE_KEY].get("api_call", {})
        assert circuit.get("failures") == 2
        assert circuit.get("state") == "open"


class TestPromptAdapter:
    """Test PromptAdapter base class."""

    @pytest.mark.asyncio
    async def test_pre_run_adapts_prompt(self):
        class TestAdapter(PromptAdapter):
            async def adapt(self, prompt, ctx):
                return f"{prompt}\n\nExtra context!"

        adapter = TestAdapter()
        
        mock_agent = MagicMock()
        mock_agent.config.system_prompt = "You are helpful."
        
        context = InterceptContext(
            agent=mock_agent,
            task="task",
            phase=Phase.PRE_RUN,
            messages=[],
        )
        
        result = await adapter.pre_run(context)
        assert result.proceed is True
        assert result.modified_prompt == "You are helpful.\n\nExtra context!"


class TestContextPrompt:
    """Test ContextPrompt interceptor."""

    def test_creation(self):
        adapter = ContextPrompt(template="User: {user}")
        assert adapter.template == "User: {user}"
        assert adapter.separator == "\n\n"

    @pytest.mark.asyncio
    async def test_adapt_with_context(self):
        adapter = ContextPrompt(template="User: {user_name}")
        
        mock_agent = MagicMock()
        run_ctx = RunContext(metadata={"user_name": "Alice"})
        # Simulate accessing user_name from metadata via attribute
        run_ctx.user_name = "Alice"
        
        ctx = InterceptContext(
            agent=mock_agent,
            task="test",
            phase=Phase.PRE_RUN,
            messages=[],
            run_context=run_ctx,
        )
        
        result = await adapter.adapt("Base prompt", ctx)
        assert "Base prompt" in result
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_adapt_no_context(self):
        adapter = ContextPrompt(template="Extra: {data}")
        
        mock_agent = MagicMock()
        ctx = InterceptContext(
            agent=mock_agent,
            task="test",
            phase=Phase.PRE_RUN,
            messages=[],
            run_context=None,
        )
        
        result = await adapter.adapt("Base prompt", ctx)
        assert result == "Base prompt"


class TestConversationPrompt:
    """Test ConversationPrompt interceptor."""

    def test_creation(self):
        stages = {0: "Be friendly", 10: "Be concise"}
        adapter = ConversationPrompt(stages=stages)
        assert adapter.stages == stages

    @pytest.mark.asyncio
    async def test_adapt_short_conversation(self):
        adapter = ConversationPrompt(stages={0: "Be verbose"})
        
        mock_agent = MagicMock()
        ctx = InterceptContext(
            agent=mock_agent,
            task="test",
            phase=Phase.PRE_RUN,
            messages=[{"role": "user", "content": "Hi"}] * 3,
        )
        
        result = await adapter.adapt("Base", ctx)
        assert "Be verbose" in result

    @pytest.mark.asyncio
    async def test_adapt_long_conversation(self):
        adapter = ConversationPrompt(stages={0: "Start", 10: "Be brief"})
        
        mock_agent = MagicMock()
        ctx = InterceptContext(
            agent=mock_agent,
            task="test",
            phase=Phase.PRE_RUN,
            messages=[{"role": "user", "content": "Hi"}] * 15,
        )
        
        result = await adapter.adapt("Base", ctx)
        assert "Start" in result
        assert "Be brief" in result


class TestLambdaPrompt:
    """Test LambdaPrompt interceptor."""

    @pytest.mark.asyncio
    async def test_adapter_function(self):
        def my_adapter(prompt, ctx):
            return f"{prompt} [Modified]"
        
        adapter = LambdaPrompt(adapter_fn=my_adapter)
        
        mock_agent = MagicMock()
        ctx = InterceptContext(
            agent=mock_agent,
            task="test",
            phase=Phase.PRE_RUN,
            messages=[],
        )
        
        result = await adapter.adapt("Test", ctx)
        assert result == "Test [Modified]"

    @pytest.mark.asyncio
    async def test_adapter_with_context(self):
        def my_adapter(prompt, ctx):
            user = "Guest"
            if ctx.run_context:
                user = getattr(ctx.run_context, "user", "Guest")
            return f"{prompt}\n\nHello, {user}!"
        
        adapter = LambdaPrompt(adapter_fn=my_adapter)
        
        run_ctx = RunContext()
        run_ctx.user = "Alice"
        
        mock_agent = MagicMock()
        ctx = InterceptContext(
            agent=mock_agent,
            task="test",
            phase=Phase.PRE_RUN,
            messages=[],
            run_context=run_ctx,
        )
        
        result = await adapter.adapt("Welcome", ctx)
        assert result == "Welcome\n\nHello, Alice!"


class TestRunContext:
    """Test RunContext dataclass."""

    def test_default_empty(self):
        ctx = RunContext()
        assert ctx.metadata == {}

    def test_with_metadata(self):
        ctx = RunContext(metadata={"key": "value"})
        assert ctx.metadata == {"key": "value"}

    def test_get_existing_key(self):
        ctx = RunContext(metadata={"name": "Alice"})
        assert ctx.get("name") == "Alice"

    def test_get_missing_key(self):
        ctx = RunContext()
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"

    def test_with_metadata_method(self):
        ctx = RunContext(metadata={"a": 1})
        new_ctx = ctx.with_metadata(b=2)
        
        # Original unchanged
        assert ctx.metadata == {"a": 1}
        # New has merged
        assert new_ctx.metadata == {"a": 1, "b": 2}

    def test_empty_context_constant(self):
        assert EMPTY_CONTEXT.metadata == {}


class TestInterceptResultModifiers:
    """Test new InterceptResult modifier fields."""

    def test_modified_tools(self):
        tool = MagicMock()
        result = InterceptResult(proceed=True, modified_tools=[tool])
        assert result.modified_tools == [tool]

    def test_modified_model(self):
        result = InterceptResult(proceed=True, modified_model="gpt-4")
        assert result.modified_model == "gpt-4"

    def test_modified_prompt(self):
        result = InterceptResult(proceed=True, modified_prompt="New prompt")
        assert result.modified_prompt == "New prompt"

    def test_modify_tools_factory(self):
        tool = MagicMock()
        result = InterceptResult.modify_tools([tool])
        assert result.proceed is True
        assert result.modified_tools == [tool]

    def test_use_model_factory(self):
        result = InterceptResult.use_model("gpt-4")
        assert result.proceed is True
        assert result.modified_model == "gpt-4"

    def test_modify_prompt_factory(self):
        result = InterceptResult.modify_prompt("New prompt")
        assert result.proceed is True
        assert result.modified_prompt == "New prompt"