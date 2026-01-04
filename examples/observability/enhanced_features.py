"""
Demonstrates enhanced observability features:
- Token usage tracking
- Structured export (JSONL, JSON, CSV)
- Progress step indicators
- Error context with suggestions
- State change diff visualization
"""

import asyncio
import sys
from pathlib import Path

# Add examples directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_model

from agenticflow import Agent, Observer, ObservabilityLevel


async def demo_token_tracking():
    """Demonstrate token usage tracking in LLM calls."""
    print("\n" + "═" * 60)
    print("DEMO 1: Token Usage Tracking")
    print("═" * 60 + "\n")
    
    # Observer with token tracking enabled (default)
    observer = Observer(
        level=ObservabilityLevel.DEBUG,
        show_token_usage=True,  # Show tokens in LLM events
        track_tokens=True,       # Track cumulative usage
    )
    
    model = get_model()
    agent = Agent(
        name="Assistant",
        model=model,
        instructions="You are a helpful assistant.",
        observer=observer,
    )
    
    # Run a simple task
    await agent.run("Explain quantum computing in 2 sentences.")
    
    # Show token summary
    print("\n" + observer.summary())


async def demo_structured_export():
    """Demonstrate exporting events to structured formats."""
    print("\n" + "═" * 60)
    print("DEMO 2: Structured Export")
    print("═" * 60 + "\n")
    
    observer = Observer.trace()
    
    model = get_model()
    agent = Agent(
        name="Researcher",
        model=model,
        instructions="Research topics thoroughly.",
        observer=observer,
    )
    
    await agent.run("What are the main benefits of Rust programming language?")
    
    # Export to different formats
    output_dir = Path("observability_exports")
    output_dir.mkdir(exist_ok=True)
    
    # JSON Lines format (one event per line - great for streaming logs)
    observer.export(output_dir / "events.jsonl", format="jsonl")
    print(f"✓ Exported to {output_dir / 'events.jsonl'} (JSONL format)")
    
    # JSON array format (full structured data)
    observer.export(output_dir / "events.json", format="json")
    print(f"✓ Exported to {output_dir / 'events.json'} (JSON format)")
    
    # CSV format (tabular data for spreadsheets)
    observer.export(output_dir / "events.csv", format="csv")
    print(f"✓ Exported to {output_dir / 'events.csv'} (CSV format)")
    
    print(f"\n📁 All exports saved to: {output_dir.absolute()}")


async def demo_progress_indicators():
    """Demonstrate progress step indicators for multi-step operations."""
    print("\n" + "═" * 60)
    print("DEMO 3: Progress Step Indicators")
    print("═" * 60 + "\n")
    
    observer = Observer(
        level=ObservabilityLevel.PROGRESS,
        show_progress_steps=True,  # Enable step progress
    )
    
    model = get_model()
    agent = Agent(
        name="Analyzer",
        model=model,
        instructions="Break down problems into clear steps.",
        observer=observer,
    )
    
    # Note: Progress steps would typically be emitted by the agent/tool implementation
    # This example shows what it looks like when enabled
    await agent.run("Analyze the pros and cons of electric vehicles.")


async def demo_error_context():
    """Demonstrate enhanced error messages with suggestions."""
    print("\n" + "═" * 60)
    print("DEMO 4: Error Context & Suggestions")
    print("═" * 60 + "\n")
    
    # Use DEBUG level to see suggestions
    observer = Observer.debug()
    
    print("Error context enhancement is automatic!")
    print("When errors occur at DEBUG level, you'll see:")
    print("  - File/line/tool context")
    print("  - Actionable suggestions based on error patterns")
    print("")
    print("Example patterns matched:")
    print("  • 'permission denied' → Check permissions, verify access")
    print("  • 'connection refused' → Verify service running, check network")
    print("  • 'timeout' → Check connectivity, increase timeout")
    print("  • 'not found' → Verify resource exists, check path")
    print("  • 'invalid credentials' → Verify API keys, check expiration")
    print("")
    print("(See real errors in DEBUG mode when agents encounter issues)")


async def demo_state_diff():
    """Demonstrate state change diff visualization."""
    print("\n" + "═" * 60)
    print("DEMO 5: State Change Diff Visualization")
    print("═" * 60 + "\n")
    
    observer = Observer.detailed()
    
    print("State diff visualization is available for reactive agents!")
    print("When AGENT_STATUS_CHANGED events are emitted, you'll see:")
    print("")
    print("Example output:")
    print("  [TaskManager] [state-change]")
    print("             status: pending → in_progress")
    print("             assignee: null → alice@example.com")
    print("             priority: medium → high")
    print("             progress: 0 → 25")
    print("")
    print("(Enable at DETAILED level or higher)")
    print("(Useful for tracking task status, agent state, entity changes)")


async def main():
    """Run all demos."""
    print("\n" + "┏" + "━" * 58 + "┓")
    print("┃" + " " * 10 + "Enhanced Observability Features Demo" + " " * 11 + "┃")
    print("┗" + "━" * 58 + "┛")
    
    try:
        # Demo 1: Token tracking
        await demo_token_tracking()
        
        # Demo 2: Structured export
        await demo_structured_export()
        
        # Demo 3: Progress indicators
        await demo_progress_indicators()
        
        # Demo 4: Error context
        await demo_error_context()
        
        # Demo 5: State diff
        await demo_state_diff()
        
        print("\n" + "═" * 60)
        print("✓ All demos completed successfully!")
        print("═" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
