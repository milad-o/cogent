"""
AgenticFlow CLI - Command-line interface for AgenticFlow.
"""

import argparse
import asyncio
import sys


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="agenticflow",
        description="AgenticFlow - Event-Driven Multi-Agent System Framework",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Version command
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version",
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start servers")
    serve_parser.add_argument(
        "--websocket", "-ws",
        action="store_true",
        help="Start WebSocket server",
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8765,
        help="Server port (default: 8765)",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)",
    )

    # Info command
    subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if args.version:
        from agenticflow import __version__
        print(f"AgenticFlow v{__version__}")
        return

    if args.command == "serve":
        asyncio.run(run_server(args))
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()


async def run_server(args: argparse.Namespace) -> None:
    """Run the server based on arguments."""
    from agenticflow import EventBus, ConsoleEventHandler

    event_bus = EventBus()
    event_bus.subscribe_all(ConsoleEventHandler())

    if args.websocket:
        try:
            from agenticflow.server import start_websocket_server
        except ImportError:
            print("❌ WebSocket support not available.")
            print("   Install with: uv add websockets")
            sys.exit(1)

        server = await start_websocket_server(
            event_bus,
            host=args.host,
            port=args.port,
        )

        if server:
            print(f"📡 WebSocket server running at ws://{args.host}:{args.port}")
            print("Press Ctrl+C to stop...")

            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                await server.stop()
    else:
        print("No server specified. Use --websocket to start WebSocket server.")


def show_info() -> None:
    """Show system information."""
    from agenticflow import __version__

    print(f"""
AgenticFlow v{__version__}
========================

A production-grade event-driven multi-agent system framework.

Components:
  • Agent       - Autonomous entity that thinks and acts
  • Task        - Unit of work with lifecycle tracking
  • Event       - Immutable record of system activity
  • EventBus    - Central pub/sub for event distribution
  • Orchestrator - Coordinates multiple agents

Optional Dependencies:
  • websockets  - Real-time event streaming
  • fastapi     - REST API server

Documentation: https://github.com/milad-o/agenticflow
""")


if __name__ == "__main__":
    main()
