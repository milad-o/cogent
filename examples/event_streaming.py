"""
Event Streaming Example - demonstrates real-time event streaming via WebSocket.

This example shows how to:
1. Start a WebSocket server
2. Stream events to connected clients
3. Handle client commands

Usage:
    # Start the server
    uv run python examples/event_streaming.py

    # Connect with websocat or similar tool:
    websocat ws://localhost:8765

Requirements:
    uv add websockets
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.tools import tool

from agenticflow import (
    Agent,
    AgentConfig,
    AgentRole,
    ConsoleEventHandler,
    EventBus,
    Orchestrator,
    TaskManager,
    ToolRegistry,
)

load_dotenv()

# Check for websockets
try:
    from agenticflow.server import WebSocketServer, start_websocket_server

    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


@tool
def process_data(input_data: str) -> str:
    """Process input data and return results."""
    import time

    time.sleep(1)  # Simulate work
    return f"Processed: {input_data}"


@tool
def transform_data(data: str, format: str = "json") -> str:
    """Transform data into a specific format."""
    import time

    time.sleep(0.5)
    return f"Transformed to {format}: {data}"


async def run_demo_tasks(orchestrator: Orchestrator) -> None:
    """Run some demo tasks to generate events."""
    await asyncio.sleep(3)  # Give time for clients to connect

    print("\n🎬 Running demo tasks...")

    # Run a few tasks
    for i in range(3):
        await asyncio.sleep(2)
        await orchestrator.run_task(
            name=f"Demo task {i + 1}",
            tool="process_data",
            args={"input_data": f"Sample data #{i + 1}"},
        )


async def main() -> None:
    """Run the event streaming example."""
    if not WEBSOCKET_AVAILABLE:
        print("❌ websockets not installed. Run: uv add websockets")
        return

    print("\n" + "=" * 60)
    print("AgenticFlow Event Streaming Example")
    print("=" * 60)

    # Create infrastructure
    event_bus = EventBus()
    task_manager = TaskManager(event_bus)

    # Add console handler
    console = ConsoleEventHandler()
    event_bus.subscribe_all(console)

    # Create tools and agents
    tool_registry = ToolRegistry()
    tool_registry.register_many([process_data, transform_data])

    worker = Agent(
        config=AgentConfig(
            name="Worker",
            role=AgentRole.WORKER,
            tools=["process_data", "transform_data"],
        ),
        event_bus=event_bus,
        tool_registry=tool_registry,
    )

    orchestrator = Orchestrator(
        event_bus=event_bus,
        task_manager=task_manager,
        tool_registry=tool_registry,
    )
    orchestrator.register_agent(worker)

    # Start WebSocket server
    server = await start_websocket_server(event_bus, port=8765)

    if server:
        print("\n📡 WebSocket server running at ws://localhost:8765")
        print("   Connect with: websocat ws://localhost:8765")
        print("\n   Available commands:")
        print("     {\"command\": \"history\", \"limit\": 10}")
        print("     {\"command\": \"stats\"}")
        print("     {\"command\": \"ping\"}")
        print("\n   Press Ctrl+C to stop\n")

        # Run demo tasks in background
        demo_task = asyncio.create_task(run_demo_tasks(orchestrator))

        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            demo_task.cancel()
            await server.stop()
            print("\n👋 Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
