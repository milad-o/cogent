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
import tempfile
from pathlib import Path

from agenticflow import Agent, tool
from agenticflow.flow import Flow

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
    model = "gpt4"

    # Create specialized agents for different file types
    json_agent = Agent(
        name="json_processor",
        model="gpt4",
        system_prompt="""You are a JSON data analyst. When you receive a file:
1. Use the parse_json_file tool to read and validate it
2. Summarize what data the file contains
3. Suggest what actions could be taken with this data""",
        tools=[parse_json_file],
    )

    csv_agent = Agent(
        name="csv_processor",
        model="gpt4",
        system_prompt="""You are a CSV data analyst. When you receive a file:
1. Use the analyze_csv_file tool to get statistics
2. Describe the data structure
3. Recommend analysis approaches""",
        tools=[analyze_csv_file],
    )

    text_agent = Agent(
        name="text_processor",
        model="gpt4",
        system_prompt="""You are a text analyst. When you receive a file:
1. Use the extract_text tool to read the content
2. Summarize the main points
3. Suggest relevant follow-up actions""",
        tools=[extract_text],
    )

    # Create event-driven flow
    flow = Flow()

    # Register agents with file-extension-specific triggers
    flow.register(
        json_agent,
        on="file.created",
        when=lambda e: e.data.get("extension") == ".json",
    )
    flow.register(
        csv_agent,
        on="file.created",
        when=lambda e: e.data.get("extension") == ".csv",
    )
    flow.register(
        text_agent,
        on="file.created",
        when=lambda e: e.data.get("extension") == ".txt",
    )

    print(f"\n✓ Registered event handlers successfully")

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
        for filename in files:
            filepath = watch_dir / filename

            result = await flow.run(
                f"Analyze the file: {filename}",
                initial_event="file.created",
                data={
                    "path": str(filepath),
                    "filename": filename,
                    "extension": filepath.suffix,
                },
            )

            print(f"\n--- {filename} ---")
            if result.success:
                output = str(result.output) if result.output else "(no output)"
                # Show first 300 chars
                print(output[:300] + ("..." if len(output) > 300 else ""))
            else:
                print(f"Error: {result.error or '(no output)'}")

        print(f"\n✅ Processed {len(files)} files")


# =============================================================================
# Demo: Event Sinks for Notifications
# =============================================================================


async def demo_notification_sink():
    """
    Demonstrate event-based order processing.
    
    Shows how to trigger agents on events and handle results.
    """
    print("\n" + "=" * 60)
    print("Order Processing Demo")
    print("=" * 60)

    processor = Agent(
        name="order_processor",
        model="gpt4",
        system_prompt="You process orders. Acknowledge the order and confirm it's been processed.",
    )

    flow = Flow()
    flow.register(processor, on="order.created")

    print("\n🚀 Processing an order...")

    result = await flow.run(
        "Process order #12345 for customer Alice",
        initial_event="order.created",
        data={"order_id": "12345", "customer": "Alice", "total": 99.99},
    )

    if result.success:
        print(f"✅ Order processed successfully")
        response_text = str(result.output)[:300]
        print(f"   Response: {response_text}")
    else:
        print(f"❌ Error: {result.error}")


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all demos."""
    print("\n🎯 Event Sources & Sinks Demo (with LLM)")
    print("=" * 60)

    await demo_file_processing()
    await demo_notification_sink()

    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
