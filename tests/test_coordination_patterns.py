"""Tests for coordination patterns (all_sources and StatefulSourceFilter)."""

import threading

import pytest

from agenticflow import Flow
from agenticflow.events import Event
from agenticflow.events.patterns import all_sources
from agenticflow.flow.state import CoordinationManager, CoordinationState

# -------------------------------------------------------------------------
# Basic Functionality Tests
# -------------------------------------------------------------------------


def test_all_sources_waits_for_all_sources():
    """Coordination requires all sources before triggering."""
    filter_obj = all_sources(["a", "b", "c"])

    # Emit from only 2 sources - should not trigger
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is False

    # Emit from 3rd source - should trigger
    assert filter_obj(Event(name="task.done", source="c")) is True


def test_all_sources_triggers_only_once():
    """Coordination triggers then auto-resets for next cycle."""
    filter_obj = all_sources(["a", "b"])

    # Emit from both sources
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True

    # After auto-reset, new cycle starts
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True


def test_all_sources_ignores_duplicate_events():
    """Same source emitting twice counts only once."""
    filter_obj = all_sources(["a", "b"])

    # Emit from 'a' twice
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="a")) is False

    # Emit from 'b' - should trigger
    assert filter_obj(Event(name="task.done", source="b")) is True


def test_all_sources_accepts_any_event_type():
    """Coordination accepts any event type (pattern matching at Flow level)."""
    filter_obj = all_sources(["a", "b"])

    # Different event names all count
    assert filter_obj(Event(name="task.created", source="a")) is False
    assert filter_obj(Event(name="workflow.complete", source="b")) is True


def test_multiple_independent_coordinations():
    """Multiple coordinations work independently."""
    filter1 = all_sources(["a", "b"])
    filter2 = all_sources(["x", "y", "z"])

    # Complete first coordination
    assert filter1(Event(name="task.done", source="a")) is False
    assert filter1(Event(name="task.done", source="b")) is True

    # Second coordination independent
    assert filter2(Event(name="task.done", source="x")) is False
    assert filter2(Event(name="task.done", source="y")) is False
    assert filter2(Event(name="task.done", source="z")) is True


def test_all_sources_ignores_unknown_sources():
    """Sources not in coordination list are ignored."""
    filter_obj = all_sources(["a", "b"])

    # Emit from unknown sources
    assert filter_obj(Event(name="task.done", source="x")) is False
    assert filter_obj(Event(name="task.done", source="y")) is False

    # Emit from correct sources
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True


def test_coordination_with_single_source():
    """Coordination with single source triggers immediately."""
    filter_obj = all_sources(["a"])

    # Should trigger immediately when 'a' emits
    assert filter_obj(Event(name="task.done", source="a")) is True


def test_empty_sources_raises_error():
    """Empty source list raises ValueError."""
    with pytest.raises(ValueError, match="at least one source"):
        all_sources([])


# -------------------------------------------------------------------------
# Auto-Reset Semantics Tests
# -------------------------------------------------------------------------


def test_coordination_auto_resets_by_default():
    """Default behavior: coordination resets after triggering."""
    filter_obj = all_sources(["a", "b"])

    # First cycle
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True

    # Second cycle - should work again (auto-reset)
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True


def test_coordination_multiple_cycles_with_auto_reset():
    """Multiple cycles work with auto-reset."""
    filter_obj = all_sources(["a", "b", "c"])

    # Cycle 1
    filter_obj(Event(name="task.done", source="a"))
    filter_obj(Event(name="task.done", source="b"))
    assert filter_obj(Event(name="task.done", source="c")) is True

    # Cycle 2 (auto-reset after trigger)
    filter_obj(Event(name="task.done", source="a"))
    filter_obj(Event(name="task.done", source="b"))
    assert filter_obj(Event(name="task.done", source="c")) is True

    # Cycle 3
    filter_obj(Event(name="task.done", source="a"))
    filter_obj(Event(name="task.done", source="b"))
    assert filter_obj(Event(name="task.done", source="c")) is True


def test_one_time_gate_with_once():
    """.once() creates one-time gate that never resets."""
    filter_obj = all_sources(["a", "b"]).once()

    # First completion
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True

    # Should NOT trigger again (one-time gate)
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is False


def test_manual_reset_method():
    """Manual reset via filter.reset()."""
    filter_obj = all_sources(["a", "b"]).once()  # One-time gate

    # Complete coordination
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True

    # Should NOT trigger again (one-time)
    assert filter_obj(Event(name="task.done", source="a")) is False

    # Manual reset
    filter_obj.reset()

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
    filter_obj = all_sources(["a", "b"]).once()

    # First completion
    filter_obj(Event(name="task.done", source="a"))
    assert filter_obj(Event(name="task.done", source="b")) is True

    # Should not trigger again (one-time)
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is False

    # Manual reset
    filter_obj.reset()

    # Should work again
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True


# -------------------------------------------------------------------------
# Composition Tests
# -------------------------------------------------------------------------


def test_all_sources_with_and():
    """all_sources can be combined with & operator."""
    sf = all_sources(["a", "b"])
    filter_obj = sf & (lambda e: e.data.get("priority") == "high")

    # Both conditions must be true
    assert filter_obj(Event(name="task.done", source="a", data={"priority": "high"})) is False
    assert filter_obj(Event(name="task.done", source="b", data={"priority": "low"})) is False

    # Reset and try with both matching (auto-reset happened)
    assert filter_obj(Event(name="task.done", source="a", data={"priority": "high"})) is False
    assert filter_obj(Event(name="task.done", source="b", data={"priority": "high"})) is True


def test_all_sources_with_or():
    """all_sources can be combined with | operator."""
    sf = all_sources(["a", "b"])
    filter_obj = sf | (lambda e: e.data.get("urgent") is True)

    # Urgent event triggers immediately
    assert filter_obj(Event(name="task.done", source="x", data={"urgent": True})) is True

    # Or all sources (after auto-reset)
    assert filter_obj(Event(name="task.done", source="a")) is False
    assert filter_obj(Event(name="task.done", source="b")) is True


def test_all_sources_with_not():
    """all_sources can be negated with ~ operator."""
    sf = all_sources(["a", "b"])
    filter_obj = ~sf

    # Returns opposite
    assert filter_obj(Event(name="task.done", source="a")) is True  # NOT complete
    assert filter_obj(Event(name="task.done", source="b")) is False  # Now complete, so negation is False


# -------------------------------------------------------------------------
# Edge Cases Tests
# -------------------------------------------------------------------------


def test_thread_safe_coordination():
    """Coordination is thread-safe."""
    filter_obj = all_sources(["a", "b"])
    results = []

    def emit_event(source):
        result = filter_obj(Event(name="task.done", source=source))
        results.append(result)

    # Emit concurrently
    threads = [
        threading.Thread(target=emit_event, args=("a",)),
        threading.Thread(target=emit_event, args=("b",)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have exactly one completion (thread-safe prevents race conditions)
    assert len([r for r in results if r]) == 1


def test_coordination_filter_instance_isolation():
    """Each filter instance has isolated coordination state."""
    filter1 = all_sources(["a", "b"])
    filter2 = all_sources(["a", "b"])

    # Progress filter1
    assert filter1(Event(name="task.done", source="a")) is False

    # filter2 should be independent
    assert filter2(Event(name="task.done", source="a")) is False
    assert filter2(Event(name="task.done", source="b")) is True

    # filter1 should still need b
    assert filter1(Event(name="task.done", source="b")) is True


def test_flow_register_accepts_all_sources():
    """Flow.register() accepts all_sources filter as when parameter."""
    flow = Flow()
    filter_obj = all_sources(["a", "b"])

    def test_reactor(event):
        pass

    # Should not raise - register accepts coordination filter
    binding_id = flow.register(test_reactor, on="*", when=filter_obj)
    assert binding_id is not None

    # Verify the filter object still works independently
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
