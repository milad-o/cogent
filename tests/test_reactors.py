"""Tests for reactor implementations."""

import pytest

from agenticflow.events import Event
from agenticflow.reactors import (
    FunctionReactor,
    function_reactor,
    Aggregator,
    FirstWins,
    WaitAll,
    Router,
    ConditionalRouter,
    Transform,
    MapTransform,
    FanInMode,
)


class TestFunctionReactor:
    """Tests for FunctionReactor."""

    def test_creation(self) -> None:
        """FunctionReactor can be created."""

        def fn(event: Event) -> Event:
            return Event(name="out", source="fn", data={})

        reactor = FunctionReactor(fn=fn, name="test")
        assert reactor.name == "test"

    def test_creation_with_emit_name(self) -> None:
        """FunctionReactor can specify emit_name."""

        def fn(event: Event) -> dict:
            return {"result": "ok"}

        reactor = FunctionReactor(fn=fn, emit_name="processed")
        assert reactor is not None

    def test_decorator(self) -> None:
        """function_reactor decorator creates FunctionReactor."""

        @function_reactor(name="decorated")
        def my_handler(event: Event) -> Event | None:
            return None

        assert isinstance(my_handler, FunctionReactor)
        assert my_handler.name == "decorated"


class TestAggregator:
    """Tests for Aggregator reactor."""

    def test_creation(self) -> None:
        """Aggregator can be created with required params."""
        aggregator = Aggregator(collect=3, emit="all.done")
        assert aggregator is not None

    def test_creation_with_mode(self) -> None:
        """Aggregator accepts mode parameter."""
        aggregator = Aggregator(
            collect=3,
            emit="all.done",
            mode=FanInMode.WAIT_ALL,
        )
        assert aggregator is not None

    def test_creation_with_timeout(self) -> None:
        """Aggregator accepts timeout parameter."""
        aggregator = Aggregator(
            collect=3,
            emit="all.done",
            timeout=30.0,
        )
        assert aggregator is not None

    def test_reset(self) -> None:
        """Aggregator can be reset."""
        aggregator = Aggregator(collect=3, emit="all.done")
        aggregator.reset()
        # Should not raise


class TestFirstWins:
    """Tests for FirstWins aggregator."""

    def test_creation(self) -> None:
        """FirstWins can be created."""
        first_wins = FirstWins(emit="winner")
        assert first_wins is not None


class TestWaitAll:
    """Tests for WaitAll aggregator."""

    def test_creation(self) -> None:
        """WaitAll can be created."""
        wait_all = WaitAll(collect=5, emit="complete")
        assert wait_all is not None


class TestRouter:
    """Tests for Router reactor."""

    def test_creation(self) -> None:
        """Router can be created with routes dict."""
        router = Router(
            routes={"high": "priority.high", "low": "priority.low"},
            key="level",
        )
        assert router is not None

    def test_creation_with_default(self) -> None:
        """Router accepts default route."""
        router = Router(
            routes={"a": "route.a", "b": "route.b"},
            key="type",
            default="route.unknown",
        )
        assert router is not None


class TestConditionalRouter:
    """Tests for ConditionalRouter reactor."""

    def test_creation(self) -> None:
        """ConditionalRouter can be created."""
        router = ConditionalRouter(
            conditions=[
                (lambda e: e.data.get("v", 0) > 100, "high"),
                (lambda e: e.data.get("v", 0) > 50, "medium"),
                (lambda e: True, "low"),  # Default
            ],
        )
        assert router is not None


class TestTransform:
    """Tests for Transform reactor."""

    def test_creation(self) -> None:
        """Transform can be created."""
        transform = Transform(
            transform=lambda d: {"doubled": d.get("v", 0) * 2},
            emit="transformed",
        )
        assert transform is not None


class TestMapTransform:
    """Tests for MapTransform reactor."""

    def test_creation(self) -> None:
        """MapTransform can be created."""
        transform = MapTransform(
            transforms=[
                (lambda d: {"text": d.get("output", "")}, "text.ready"),
                (lambda d: {"length": len(d.get("output", ""))}, "metrics.ready"),
            ],
        )
        assert transform is not None
