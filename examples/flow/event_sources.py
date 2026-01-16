"""
Event Sources & Sinks Demo
===========================

Demonstrates the new event-driven capabilities with real LLM agents:
- FileWatcherSource: Watch directories for incoming files
- WebhookSink: Send completion notifications to external systems

Scenario: Document processing pipeline
1. Watch a folder for new documents (JSON, CSV, TXT)
2. When a file arrives, trigger agents to process it
3. Send notifications when processing completes

Run with:
    uv run python examples/reactive/event_sources_demo.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add examples to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_model

from agenticflow import Agent, tool
from agenticflow.reactive.flow import EventFlow  # Has source/sink methods
from agenticflow.reactive import react_to, Observer
from agenticflow.events import EventSink


# =============================================================================
# Define tools for file processing
# =============================================================================


@tool
def parse_json_file(path: str) -> str:
    """Parse and validate a JSON file."""
    import json
    try:
        data = json.loads(Path(path).read_text())
        return f"✓ Valid JSON with {len(data) if isinstance(data, (list, dict)) else 1} items: {str(data)[:200]}"
    except Exception as e:
        return f"✗ JSON error: {e}"


@tool
def analyze_csv_file(path: str) -> str:
    """Analyze a CSV file and return statistics."""
    content = Path(path).read_text()
    lines = content.strip().split("\n")
    if lines:
        headers = lines[0].split(",")
        return f"✓ CSV with {len(lines)-1} data rows, columns: {headers}"
    return "✗ Empty CSV file"


@tool
def extract_text(path: str) -> str:
    """Extract and summarize text from a file."""
    content = Path(path).read_text()
    word_count = len(content.split())
    return f"✓ Text file: {word_count} words, preview: {content[:100]}..."


# =============================================================================
# Demo: FileWatcher with Real Agents
# =============================================================================


async def demo_file_processing():
    """
    Demonstrate file-triggered agent processing.
    
    Agents react to different file types and process them accordingly.
    """
    print("\n" + "=" * 60)
    print("Event-Driven File Processing Demo")
    print("=" * 60)

    # Get LLM model
    model = get_model()

    # Create specialized agents for different file types
    json_agent = Agent(
        name="json_processor",
        model=model,
        system_prompt="""You are a JSON data analyst. When you receive a file:
1. Use the parse_json_file tool to read and validate it
2. Summarize what data the file contains
3. Suggest what actions could be taken with this data""",
        tools=[parse_json_file],
    )

    csv_agent = Agent(
        name="csv_processor",
        model=model,
        system_prompt="""You are a CSV data analyst. When you receive a file:
1. Use the analyze_csv_file tool to get statistics
2. Describe the data structure
3. Recommend analysis approaches""",
        tools=[analyze_csv_file],
    )

    text_agent = Agent(
        name="text_processor",
        model=model,
        system_prompt="""You are a text analyst. When you receive a file:
1. Use the extract_text tool to read the content
2. Summarize the main points
3. Suggest relevant follow-up actions""",
        tools=[extract_text],
    )

    # Create observer for progress output
    observer = Observer.progress()

    # Create event-driven flow
    flow = EventFlow(observer=observer)

    # Register agents with file-extension-specific triggers
    flow.register(
        json_agent,
        [react_to("file.created").when(lambda e: e.data.get("extension") == ".json")],
    )
    flow.register(
        csv_agent,
        [react_to("file.created").when(lambda e: e.data.get("extension") == ".csv")],
    )
    flow.register(
        text_agent,
        [react_to("file.created").when(lambda e: e.data.get("extension") == ".txt")],
    )

    print(f"\n✓ Registered agents: {flow.agents}")

    # Create temp directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        watch_dir = Path(tmpdir) / "incoming"
        watch_dir.mkdir()

        # Pre-create test files (in real usage, these would appear dynamically)
        files = {
            "orders.json": '{"orders": [{"id": 1, "product": "Widget", "qty": 5}, {"id": 2, "product": "Gadget", "qty": 3}]}',
            "sales.csv": "date,product,amount\n2024-01-01,Widget,100\n2024-01-02,Gadget,200\n2024-01-03,Widget,150",
            "notes.txt": "Meeting notes from Q1 planning session. Topics: product roadmap, hiring plan, budget allocation.",
        }

        print(f"\n📁 Processing files from: {watch_dir}")

        for filename, content in files.items():
            filepath = watch_dir / filename
            filepath.write_text(content)
            print(f"   📄 Created: {filename}")

        # Note: FileWatcherSource is for production use with real-time file watching.
        # For this demo, we'll emit file events directly to show the processing flow.
        
        print("\n🔄 Processing files...")

        # Process each file by emitting an event
        for filename in files.keys():
            filepath = watch_dir / filename
            
            result = await flow.run(
                f"Analyze the file: {filename}",
                initial_event="file.created",
                initial_data={
                    "path": str(filepath),
                    "filename": filename,
                    "extension": filepath.suffix,
                },
            )
            
            print(f"\n--- {filename} ---")
            if result.reactions:
                for reaction in result.reactions:
                    output = reaction.output or "(no output)"
                    # Show first 300 chars
                    print(output[:300] + ("..." if len(output) > 300 else ""))
            else:
                print("(no matching agent)")

        print(f"\n✅ Processed {len(files)} files")


# =============================================================================
# Demo: Event Sinks for Notifications  
# =============================================================================


async def demo_notification_sink():
    """
    Demonstrate outbound event sinks for notifications.
    
    Uses a mock sink to show how events can be sent to external systems.
    """
    print("\n" + "=" * 60)
    print("Event Sink Notification Demo")
    print("=" * 60)

    # Create a logging sink (in production, use WebhookSink)
    received_events = []
    
    class NotificationSink(EventSink):
        """Mock sink that logs notifications (simulates Slack/email/webhook)."""
        
        async def send(self, event) -> None:
            received_events.append(event)
            print(f"   � Notification: {event.name}")
            print(f"      Agent: {event.data.get('agent', 'unknown')}")
        
        @property
        def name(self) -> str:
            return "NotificationSink"

    # Get LLM model
    model = get_model()

    # Create a simple processing agent
    processor = Agent(
        name="order_processor",
        model=model,
        system_prompt="You process orders. Acknowledge the order and confirm it's been processed.",
    )

    # Create flow with observer and notification sink
    observer = Observer.minimal()
    flow = EventFlow(observer=observer)

    # Register agent
    flow.register(processor, [react_to("order.created")])

    # Add sink for completion events
    flow.sink(NotificationSink(), pattern="*.completed")

    print("\n🚀 Processing an order...")

    result = await flow.run(
        "Process order #12345 for customer Alice",
        initial_event="order.created",
        initial_data={"order_id": "12345", "customer": "Alice", "total": 99.99},
    )

    print(f"\n✅ Flow completed in {result.execution_time_ms:.0f}ms")
    print(f"   Notifications sent: {len(received_events)}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run demos."""
    print("\n🎯 Event Sources & Sinks Demo (with LLM)")
    print("=" * 60)

    # Demo 1: File-triggered processing
    await demo_file_processing()

    # Demo 2: Notification sinks
    await demo_notification_sink()

    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
