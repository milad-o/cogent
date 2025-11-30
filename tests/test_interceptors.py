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
