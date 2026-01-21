"""Tests for coordination patterns (all_sources and StatefulSourceFilter)."""

import pytest
import threading

from agenticflow import Flow
from agenticflow.events import Event
from agenticflow.events.patterns import all_sources, StatefulSourceFilter
from agenticflow.flow.state import CoordinationManager, CoordinationState


# -------------------------------------------------------------------------
# Basic Functionality Tests
# -------------------------------------------------------------------------


def test_all_sources_waits_for_all_sources():
    """Coordination requires all sources before triggering."""
    manager = CoordinationManager()
    filter_obj = all_sources(["a", "b", "c"], reset_after=True)
    filter_obj.initialize(manager, "binding_1")
    
    # Emit from only 2 sources - should not trigger
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is False
    
    # Emit from 3rd source - should trigger
    assert filter_obj(Event(name="task.done", source="c")) is True


def test_all_sources_triggers_only_once():
    """Coordination doesn't retrigger within same cycle."""
    manager = CoordinationManager()
    filter_obj = all_sources(["a", "b"], reset_after=True)
    filter_obj.initialize(manager, "binding_1")
    
    # Emit from both sources
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True
    
    # After auto-reset, new cycle starts
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True


def test_all_sources_ignores_duplicate_events():
    """Same source emitting twice counts only once."""
    manager = CoordinationManager()
    filter_obj = all_sources(["a", "b"], reset_after=True)
    filter_obj.initialize(manager, "binding_1")
    
    # Emit from 'a' twice
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="a")) is False
    
    # Emit from 'b' - should trigger
    assert filter_obj(Event(name="task.done", source="b")) is True


def test_all_sources_ignores_wrong_event_type():
    """Base SourceFilter handles event pattern matching."""
    # This is tested at the StatefulSourceFilter level
    # The coordination logic only checks sources, not event names
    manager = CoordinationManager()
    filter_obj = all_sources(["a", "b"], reset_after=True)
    filter_obj.initialize(manager, "binding_1")
    
    # All events contribute to coordination regardless of name
    # (pattern matching happens at higher level in Flow)
    assert filter_obj(Event(name="task.created", source="a")) is False
    assert filter_obj(Event(name="other.event", source="b")) is True


def test_multiple_independent_coordinations():
    """Multiple coordinations work independently."""
    manager = CoordinationManager()
    
    filter1 = all_sources(["a", "b"], reset_after=True)
    filter1.initialize(manager, "binding_1")
    
    filter2 = all_sources(["x", "y", "z"], reset_after=True)
    filter2.initialize(manager, "binding_2")
    
    # Complete first coordination
    assert filter1(Event(name="task.done", source="a")) is False
    assert filter1(Event(name="task.done", source="b")) is True
    
    # Second coordination independent
    assert filter2(Event(name="task.done", source="x")) is False
    assert filter2(Event(name="task.done", source="y")) is False
    assert filter2(Event(name="task.done", source="z")) is True


def test_all_sources_ignores_unknown_sources():
    """Sources not in coordination list are ignored."""
    manager = CoordinationManager()
    filter_obj = all_sources(["a", "b"], reset_after=True)
    filter_obj.initialize(manager, "binding_1")
    
    # Emit from unknown sources
    assert filter_obj(Event(name="task.done", source="x")) is False
    assert filter_obj(Event(name="task.done", source="y")) is False
    
    # Emit from correct sources
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True


def test_coordination_with_single_source():
    """Coordination with single source triggers immediately."""
    manager = CoordinationManager()
    filter_obj = all_sources(["a"], reset_after=True)
    filter_obj.initialize(manager, "binding_1")
    
    # Should trigger immediately when 'a' emits
    assert filter_obj(Event(name="task.done", source="a")) is True


def test_coordination_with_empty_sources():
    """Empty sources list raises ValueError."""
    with pytest.raises(ValueError, match="at least one source"):
        all_sources([])


# -------------------------------------------------------------------------
# Auto-Reset Semantics Tests
# -------------------------------------------------------------------------


def test_coordination_auto_resets_by_default():
    """Default behavior: coordination resets after triggering."""
    manager = CoordinationManager()
    filter_obj = all_sources(["a", "b"], reset_after=True)
    filter_obj.initialize(manager, "binding_1")
    
    # First cycle
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True
    
    # Second cycle - should work again
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True


def test_coordination_multiple_cycles_with_auto_reset():
    """Multiple cycles work with auto-reset."""
    manager = CoordinationManager()
    filter_obj = all_sources(["a", "b", "c"], reset_after=True)
    filter_obj.initialize(manager, "binding_1")
    
    # Cycle 1
    filter_obj(Event(name="task.done", source="a"))
    filter_obj(Event(name="task.done", source="b"))
    assert filter_obj(Event(name="task.done", source="c")) is True
    
    # Cycle 2
    filter_obj(Event(name="task.done", source="a"))
    filter_obj(Event(name="task.done", source="b"))
    assert filter_obj(Event(name="task.done", source="c")) is True
    
    # Cycle 3
    filter_obj(Event(name="task.done", source="a"))
    filter_obj(Event(name="task.done", source="b"))
    assert filter_obj(Event(name="task.done", source="c")) is True


def test_coordination_no_auto_reset_when_disabled():
    """reset_after=False prevents automatic reset."""
    manager = CoordinationManager()
    filter_obj = all_sources(["a", "b"], reset_after=False)
    filter_obj.initialize(manager, "binding_1")
    
    # First completion
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True
    
    # Should NOT trigger again
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is False


def test_manual_reset_via_flow_method():
    """Manual reset via flow.reset_coordination()."""
    flow = Flow()
    filter_obj = all_sources(["a", "b"], reset_after=False)
    
    # Register returns binding_id
    binding_id = flow.register(
        lambda e: None,
        on="task.done",
        when=filter_obj,
    )
    
    # Initialize happens during register
    manager = flow._coordination
    
    # Complete coordination
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True
    
    # Should NOT trigger again
    assert filter_obj(Event(name="task.done", source="a")) is False
    
    # Manual reset
    flow.reset_coordination(binding_id)
    
    # Should work again
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True


def test_reset_clears_seen_sources():
    """Reset clears seen sources for new cycle."""
    state = CoordinationState(required_sources=frozenset(["a", "b"]))
    
    # Add sources
    state.add_source("a")
    assert "a" in state.seen_sources
    
    # Reset
    state.reset()
    assert len(state.seen_sources) == 0
    assert state.completed is False


def test_reset_allows_retrigger():
    """After reset, coordination can trigger again."""
    manager = CoordinationManager()
    filter_obj = all_sources(["a", "b"], reset_after=False)
    filter_obj.initialize(manager, "binding_1")
    
    # First completion
    filter_obj(Event(name="task.done", source="a"))
    assert filter_obj(Event(name="task.done", source="b")) is True
    
    # Should not trigger again
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is False
    
    # Manual reset
    manager.reset_coordination("binding_1")
    
    # Should trigger again
    filter_obj(Event(name="task.done", source="a"))
    assert filter_obj(Event(name="task.done", source="b")) is True


# -------------------------------------------------------------------------
# Composability Tests
# -------------------------------------------------------------------------


def test_all_sources_and_lambda_filter():
    """all_sources can be combined with & operator."""
    manager = CoordinationManager()
    sf = all_sources(["a", "b"], reset_after=True)
    sf.initialize(manager, "binding_1")
    filter_obj = sf & (lambda e: e.data.get("priority") == "high")
    
    # Both conditions must be true
    assert filter_obj(Event(name="task.done", source="a", data={"priority": "high"})) is False
    assert filter_obj(Event(name="task.done", source="b", data={"priority": "low"})) is False
    
    # Reset and try with both matching
    manager.reset_coordination("binding_1")
    assert filter_obj(Event(name="task.done", source="a", data={"priority": "high"})) is False
    assert filter_obj(Event(name="task.done", source="b", data={"priority": "high"})) is True


def test_all_sources_or_other_condition():
    """all_sources can be combined with | operator."""
    manager = CoordinationManager()
    sf = all_sources(["a", "b"], reset_after=True)
    sf.initialize(manager, "binding_1")
    filter_obj = sf | (lambda e: e.data.get("urgent") is True)
    
    # Urgent event triggers immediately
    assert filter_obj(Event(name="task.done", source="x", data={"urgent": True})) is True
    
    # Or all sources
    manager.reset_coordination("binding_1")
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True


def test_all_sources_with_not():
    """all_sources can be negated with ~ operator."""
    manager = CoordinationManager()
    sf = all_sources(["a", "b"], reset_after=True)
    sf.initialize(manager, "binding_1")
    filter_obj = ~sf
    
    # Returns opposite
    assert filter_obj(Event(name="task.done", source="a")) is True  # NOT complete
    assert filter_obj(Event(name="task.done", source="b")) is False  # Now complete, so negation is False


# -------------------------------------------------------------------------
# Edge Cases Tests
# -------------------------------------------------------------------------


def test_uninitialized_filter_raises_error():
    """Using filter before initialization raises RuntimeError."""
    filter_obj = all_sources(["a", "b"])
    event = Event(name="test", source="a")
    
    with pytest.raises(RuntimeError, match="must be initialized"):
        filter_obj(event)


def test_coordination_thread_safety():
    """CoordinationManager is thread-safe."""
    manager = CoordinationManager()
    manager.register_coordination("binding_1", frozenset(["a"]))
    
    results = []
    
    def worker():
        for _ in range(100):
            is_complete = manager.check_coordination("binding_1", "a")
            if is_complete:
                results.append(True)
    
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Should have exactly one completion (thread-safe prevents race conditions)
    assert len([r for r in results if r]) == 1


def test_coordination_state_isolation():
    """Each binding has isolated coordination state."""
    manager = CoordinationManager()
    
    filter1 = all_sources(["a", "b"], reset_after=True)
    filter1.initialize(manager, "binding_1")
    
    filter2 = all_sources(["a", "b"], reset_after=True)
    filter2.initialize(manager, "binding_2")
    
    # Progress filter1
    assert filter1(Event(name="task.done", source="a")) is False
    
    # filter2 should be independent
    assert filter2(Event(name="task.done", source="a")) is False
    assert filter2(Event(name="task.done", source="b")) is True
    
    # filter1 should still need b
    assert filter1(Event(name="task.done", source="b")) is True


def test_coordination_with_wildcard_event_pattern():
    """Coordination works with wildcard patterns in Flow."""
    flow = Flow()
    manager = flow._coordination
    
    filter_obj = all_sources(["a", "b"], reset_after=True)
    flow.register(lambda e: None, on="task.*", when=filter_obj)
    
    # Filter initialized during register
    # Both task.created and task.done contribute
    assert filter_obj(Event(name="task.created", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True


# -------------------------------------------------------------------------
# CoordinationState Unit Tests
# -------------------------------------------------------------------------


def test_coordination_state_basic():
    """CoordinationState tracks sources correctly."""
    state = CoordinationState(required_sources=frozenset(["a", "b", "c"]))
    
    assert state.add_source("a") is False
    assert state.add_source("b") is False
    assert state.add_source("c") is True
    assert state.completed is True


def test_coordination_state_ignores_unknown_sources():
    """Unknown sources don't affect coordination."""
    state = CoordinationState(required_sources=frozenset(["a", "b"]))
    
    assert state.add_source("x") is False
    assert state.add_source("y") is False
    assert state.completed is False
    
    # Correct sources still work
    assert state.add_source("a") is False
    assert state.add_source("b") is True


def test_coordination_state_reset():
    """Reset clears seen sources."""
    state = CoordinationState(required_sources=frozenset(["a", "b"]))
    
    state.add_source("a")
    state.add_source("b")
    assert state.completed is True
    
    state.reset()
    assert len(state.seen_sources) == 0
    assert state.completed is False


# -------------------------------------------------------------------------
# CoordinationManager Unit Tests
# -------------------------------------------------------------------------


def test_coordination_manager_basic():
    """CoordinationManager tracks multiple coordinations."""
    manager = CoordinationManager()
    
    manager.register_coordination("b1", frozenset(["a", "b"]))
    manager.register_coordination("b2", frozenset(["x", "y", "z"]))
    
    assert manager.check_coordination("b1", "a") is False
    assert manager.check_coordination("b1", "b") is True
    
    assert manager.check_coordination("b2", "x") is False
    assert manager.check_coordination("b2", "y") is False
    assert manager.check_coordination("b2", "z") is True


def test_coordination_manager_prevents_retrigger():
    """Completed coordination doesn't retrigger."""
    manager = CoordinationManager()
    manager.register_coordination("b1", frozenset(["a", "b"]))
    
    assert manager.check_coordination("b1", "a") is False
    assert manager.check_coordination("b1", "b") is True
    
    # Try again
    assert manager.check_coordination("b1", "a") is False
    assert manager.check_coordination("b1", "b") is False


def test_coordination_manager_unknown_binding():
    """Unknown binding returns False."""
    manager = CoordinationManager()
    assert manager.check_coordination("unknown", "a") is False


def test_coordination_manager_empty_sources_raises():
    """Empty sources raises ValueError."""
    manager = CoordinationManager()
    with pytest.raises(ValueError, match="at least one source"):
        manager.register_coordination("b1", frozenset())


def test_coordination_manager_reset():
    """Reset allows coordination to trigger again."""
    manager = CoordinationManager()
    manager.register_coordination("b1", frozenset(["a", "b"]))
    
    manager.check_coordination("b1", "a")
    manager.check_coordination("b1", "b")
    
    # Reset
    manager.reset_coordination("b1")
    
    # Should work again
    assert manager.check_coordination("b1", "a") is False
    assert manager.check_coordination("b1", "b") is True


def test_coordination_manager_remove():
    """Remove coordination cleans up state."""
    manager = CoordinationManager()
    manager.register_coordination("b1", frozenset(["a", "b"]))
    
    manager.remove_coordination("b1")
    
    # Should return False (no longer exists)
    assert manager.check_coordination("b1", "a") is False


def test_coordination_manager_get_state():
    """Get current coordination state."""
    manager = CoordinationManager()
    manager.register_coordination("b1", frozenset(["a", "b", "c"]))
    
    manager.check_coordination("b1", "a")
    manager.check_coordination("b1", "b")
    
    state = manager.get_coordination_state("b1")
    assert state is not None
    assert "a" in state.seen_sources
    assert "b" in state.seen_sources
    assert "c" not in state.seen_sources
    assert state.completed is False
