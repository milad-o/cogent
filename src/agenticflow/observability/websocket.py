"""
WebSocket server for real-time event streaming.

Part of the events module - provides real-time streaming of EventBus
events to connected WebSocket clients.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agenticflow.observability.event import EventType
from agenticflow.core.utils import generate_id, now_utc
from agenticflow.observability.event import Event

if TYPE_CHECKING:
    from agenticflow.observability.bus import EventBus

# Check for websockets availability
try:
    import websockets
    from websockets.asyncio.server import serve

    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


class WebSocketServer:
    """
    WebSocket server for real-time event streaming.
    
    Broadcasts all events to connected clients and handles
    client commands like history queries.
    
    Attributes:
        event_bus: EventBus to stream from
        host: Server host
        port: Server port
        
    Example:
        ```python
        server = WebSocketServer(event_bus, host="localhost", port=8765)
        await server.start()
        
        # Later...
        await server.stop()
        ```
    """

    def __init__(
        self,
        event_bus: EventBus,
        host: str = "localhost",
        port: int = 8765,
    ) -> None:
        """
        Initialize the WebSocket server.
        
        Args:
            event_bus: EventBus to stream events from
            host: Server host address
            port: Server port number
        """
        if not WEBSOCKET_AVAILABLE:
            raise ImportError(
                "websockets package not installed. Run: uv add websockets"
            )

        self.event_bus = event_bus
        self.host = host
        self.port = port
        self._server = None
        self._running = False

    async def _handle_client(self, websocket) -> None:
        """
        Handle a WebSocket client connection.
        
        Args:
            websocket: The WebSocket connection
        """
        client_id = generate_id()

        # Emit connection event
        await self.event_bus.publish(
            Event(
                type=EventType.CLIENT_CONNECTED,
                data={"client_id": client_id},
                source="websocket_server",
            )
        )

        # Register for event streaming
        self.event_bus.add_websocket(websocket)

        try:
            # Send welcome message
            await websocket.send(
                json.dumps(
                    {
                        "type": "welcome",
                        "client_id": client_id,
                        "message": "Connected to AgenticFlow Event Stream",
                    }
                )
            )

            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_command(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps({"type": "error", "message": "Invalid JSON"})
                    )

        except Exception as e:
            print(f"WebSocket error: {e}")

        finally:
            # Cleanup
            self.event_bus.remove_websocket(websocket)
            await self.event_bus.publish(
                Event(
                    type=EventType.CLIENT_DISCONNECTED,
                    data={"client_id": client_id},
                    source="websocket_server",
                )
            )

    async def _handle_command(self, websocket, data: dict) -> None:
        """
        Handle a command from a client.
        
        Args:
            websocket: The WebSocket connection
            data: The command data
        """
        command = data.get("command")

        if command == "history":
            # Return event history
            event_type = None
            if data.get("event_type"):
                try:
                    event_type = EventType(data["event_type"])
                except ValueError:
                    pass

            history = self.event_bus.get_history(
                event_type=event_type,
                correlation_id=data.get("correlation_id"),
                limit=data.get("limit", 50),
            )
            await websocket.send(
                json.dumps(
                    {
                        "type": "history",
                        "events": [e.to_dict() for e in history],
                    }
                )
            )

        elif command == "stats":
            # Return event bus stats
            await websocket.send(
                json.dumps(
                    {
                        "type": "stats",
                        "stats": self.event_bus.get_stats(),
                    }
                )
            )

        elif command == "ping":
            await websocket.send(json.dumps({"type": "pong"}))

        else:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Unknown command: {command}",
                    }
                )
            )

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            return

        self._server = await serve(
            self._handle_client,
            self.host,
            self.port,
        )
        self._running = True
        print(f"🌐 WebSocket server started at ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._running = False
            print("🌐 WebSocket server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    async def __aenter__(self) -> WebSocketServer:
        """Context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.stop()


async def websocket_handler(
    websocket,
    path: str,
    event_bus: EventBus,
) -> None:
    """
    Standalone WebSocket handler function.
    
    Can be used with websockets.serve() directly.
    
    Args:
        websocket: The WebSocket connection
        path: The request path
        event_bus: EventBus to stream from
    """
    client_id = generate_id()

    await event_bus.publish(
        Event(
            type=EventType.CLIENT_CONNECTED,
            data={"client_id": client_id, "path": path},
            source="websocket_handler",
        )
    )

    event_bus.add_websocket(websocket)

    try:
        await websocket.send(
            json.dumps(
                {
                    "type": "welcome",
                    "client_id": client_id,
                    "message": "Connected to AgenticFlow Event Stream",
                }
            )
        )

        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("command") == "history":
                    history = event_bus.get_history(limit=data.get("limit", 50))
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "history",
                                "events": [e.to_dict() for e in history],
                            }
                        )
                    )
            except json.JSONDecodeError:
                pass

    finally:
        event_bus.remove_websocket(websocket)
        await event_bus.publish(
            Event(
                type=EventType.CLIENT_DISCONNECTED,
                data={"client_id": client_id},
                source="websocket_handler",
            )
        )


async def start_websocket_server(
    event_bus: EventBus,
    host: str = "localhost",
    port: int = 8765,
) -> WebSocketServer | None:
    """
    Start a WebSocket server.
    
    Convenience function that creates and starts a WebSocketServer.
    
    Args:
        event_bus: EventBus to stream from
        host: Server host
        port: Server port
        
    Returns:
        The started WebSocketServer, or None if websockets not available
    """
    if not WEBSOCKET_AVAILABLE:
        print("⚠️ WebSocket not available. Install with: uv add websockets")
        return None

    server = WebSocketServer(event_bus, host, port)
    await server.start()
    return server
