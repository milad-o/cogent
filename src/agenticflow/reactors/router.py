"""Router reactor - routes events based on conditions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agenticflow.events import Event
from agenticflow.reactors.base import BaseReactor

if TYPE_CHECKING:
    from agenticflow.flow.context import Context


# Route selector types
RouteSelector = Callable[[Event], str | None]


class Router(BaseReactor):
    """Routes events to different paths based on conditions.

    The router examines incoming events and re-emits them with
    a modified name based on routing rules.

    Example:
        ```python
        # Route by event data
        flow.register(
            Router({
                "billing": "route.billing",
                "technical": "route.technical",
                "general": "route.general",
            }, key="category"),
            on="ticket.classified",
        )

        # Route with custom selector
        flow.register(
            Router.from_selector(
                lambda e: "high" if e.data.get("priority") > 5 else "low",
                routes={"high": "priority.high", "low": "priority.low"},
            ),
            on="task.created",
        )
        ```
    """

    def __init__(
        self,
        routes: dict[str, str],
        *,
        name: str | None = None,
        key: str | None = None,
        selector: RouteSelector | None = None,
        default: str | None = None,
    ) -> None:
        """Initialize router.

        Args:
            routes: Mapping from route keys to event names
            name: Reactor name
            key: Event data key to use for routing (if selector not provided)
            selector: Custom function to select route
            default: Default event name if no route matches
        """
        super().__init__(name or "router")
        self._routes = routes
        self._key = key
        self._selector = selector
        self._default = default

    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> Event | None:
        """Route the event based on rules."""
        # Determine route key
        route_key = self._get_route_key(event)

        if route_key is None:
            if self._default:
                emit_name = self._default
            else:
                return None
        else:
            emit_name = self._routes.get(route_key, self._default)
            if emit_name is None:
                return None

        # Re-emit event with new name
        return Event(
            name=emit_name,
            source=self.name,
            data={
                **event.data,
                "_routed_from": event.name,
                "_route_key": route_key,
            },
            correlation_id=event.correlation_id,
        )

    def _get_route_key(self, event: Event) -> str | None:
        """Determine the route key for an event."""
        if self._selector:
            return self._selector(event)

        if self._key:
            return event.data.get(self._key)

        # Try common keys
        for key in ["type", "category", "route", "action"]:
            if key in event.data:
                return str(event.data[key])

        return None

    @classmethod
    def from_selector(
        cls,
        selector: RouteSelector,
        routes: dict[str, str],
        **kwargs: Any,
    ) -> Router:
        """Create a router with a custom selector function.

        Args:
            selector: Function that takes an event and returns a route key
            routes: Mapping from route keys to event names
            **kwargs: Additional router options
        """
        return cls(routes, selector=selector, **kwargs)

    @classmethod
    def by_source(
        cls,
        routes: dict[str, str],
        **kwargs: Any,
    ) -> Router:
        """Create a router that routes by event source.

        Args:
            routes: Mapping from source names to event names
            **kwargs: Additional router options
        """
        return cls(routes, selector=lambda e: e.source, **kwargs)


class ConditionalRouter(BaseReactor):
    """Routes events using condition functions.

    Evaluates conditions in order, routes to first match.

    Example:
        ```python
        flow.register(
            ConditionalRouter([
                (lambda e: e.data.get("urgent"), "priority.urgent"),
                (lambda e: e.data.get("value", 0) > 100, "priority.high"),
                (lambda e: True, "priority.normal"),  # Default
            ]),
            on="order.created",
        )
        ```
    """

    def __init__(
        self,
        conditions: list[tuple[Callable[[Event], bool], str]],
        *,
        name: str | None = None,
    ) -> None:
        """Initialize conditional router.

        Args:
            conditions: List of (condition_fn, event_name) tuples
            name: Reactor name
        """
        super().__init__(name or "conditional_router")
        self._conditions = conditions

    async def handle(
        self,
        event: Event,
        ctx: Context,
    ) -> Event | None:
        """Evaluate conditions and route."""
        for condition, emit_name in self._conditions:
            try:
                if condition(event):
                    return Event(
                        name=emit_name,
                        source=self.name,
                        data=event.data,
                        correlation_id=event.correlation_id,
                    )
            except Exception:
                continue

        return None
