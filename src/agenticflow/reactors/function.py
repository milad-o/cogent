"""Function reactor - wraps plain functions as reactors."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agenticflow.events import Event
from agenticflow.reactors.base import BaseReactor

if TYPE_CHECKING:
    from agenticflow.flow.context import Context


# Type for functions that can be wrapped
ReactorFunction = (
    Callable[[Event], Any]
    | Callable[[Event, "Context"], Any]
)


class FunctionReactor(BaseReactor):
    """Wraps a plain function as a reactor.

    The function can be sync or async, and can optionally receive
    the execution context.

    Example:
        ```python
        def process_data(event: Event) -> str:
            return f"Processed: {event.data}"

        async def async_process(event: Event, ctx: Context) -> dict:
            await some_async_operation()
            return {"result": "done"}

        # Register as reactor
        flow.register(
            FunctionReactor(process_data),
            on="data.received",
        )

        # Or use shorthand (flow auto-wraps functions)
        flow.register(process_data, on="data.received")
        ```
    """

    def __init__(
        self,
        fn: ReactorFunction,
        name: str | None = None,
        *,
        emit_name: str | None = None,
    ) -> None:
        """Initialize function reactor.

        Args:
            fn: The function to wrap
            name: Reactor name (defaults to function name)
            emit_name: Event name to emit (defaults to "{name}.done")
        """
        self._fn = fn
        self._emit_name = emit_name

        # Determine name
        fn_name = getattr(fn, "__name__", "function")
        super().__init__(name or fn_name)

        # Check function signature
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        self._takes_context = len(params) >= 2
        self._is_async = asyncio.iscoroutinefunction(fn)

    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> Event | None:
        """Execute the wrapped function."""
        # Call function with appropriate arguments
        result = self._fn(event, ctx) if self._takes_context else self._fn(event)

        # Await if async
        if self._is_async:
            result = await result

        # If function returned None, don't emit
        if result is None:
            return None

        # Convert result to Event
        if isinstance(result, Event):
            return result

        # Wrap other results in Event.done
        emit_name = self._emit_name or "agent.done"

        if isinstance(result, str):
            return Event(
                name=emit_name,
                source=self.name,
                data={"output": result},
                correlation_id=event.correlation_id,
            )

        if isinstance(result, dict):
            return Event(
                name=emit_name,
                source=self.name,
                data=result,
                correlation_id=event.correlation_id,
            )

        # Fallback: convert to string
        return Event(
            name=emit_name,
            source=self.name,
            data={"output": str(result)},
            correlation_id=event.correlation_id,
        )


def function_reactor(
    fn: ReactorFunction | None = None,
    *,
    name: str | None = None,
    emit_name: str | None = None,
) -> FunctionReactor | Callable[[ReactorFunction], FunctionReactor]:
    """Decorator to create a FunctionReactor.

    Can be used with or without arguments.

    Example:
        ```python
        @function_reactor
        def process(event: Event) -> str:
            return f"Result: {event.data}"

        @function_reactor(name="custom_name", emit_name="custom.done")
        def another(event: Event) -> str:
            return "Done"
        ```
    """
    def decorator(f: ReactorFunction) -> FunctionReactor:
        return FunctionReactor(f, name=name, emit_name=emit_name)

    if fn is not None:
        return decorator(fn)
    return decorator
