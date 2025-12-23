"""
Agent + TaskBoard Observability - Complete Event Tracking

This example demonstrates comprehensive observability of both:
1. **Agent lifecycle events** (invoked, thinking, LLM calls, tools, responses)
2. **TaskBoard events** (tasks added, started, completed, notes added)

Shows how to track an agent's entire execution including its internal
task planning and progress.

## How TaskBoard Events Work

TaskBoard emits events from sync tool functions (plan_tasks, update_task, etc.)
which run in thread pools. To make this work with async EventBus:

1. EventBus captures the event loop reference on first publish()
2. TaskBoard calls EventBus.publish_sync() from tool functions
3. publish_sync() uses run_coroutine_threadsafe() to emit events from threads
4. Events are delivered to observers in the main async context

This demonstrates **native async** design - events flow seamlessly from sync
tool functions to async observers via thread-safe event bus.

## Run

    uv run python observability/agent_and_taskboard_events.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent
from agenticflow.observability import Observer, EventBus, EventType, ObservabilityLevel, Channel
from agenticflow.observability.event import Event


class ComprehensiveObserver(Observer):
    """Observer that tracks both agent and taskboard events."""
    
    def __init__(self):
        super().__init__()
        self.events: list[Event] = []
        self.agent_events: list[Event] = []
        self.taskboard_events: list[Event] = []
        
    def _handle_event(self, event: Event) -> None:
        """Override to track all events."""
        # Call parent handler
        super()._handle_event(event)
        
        # Track event
        self.events.append(event)
        
        # Categorize by type
        if event.type.value.startswith("agent.") or event.type.value.startswith("llm.") or event.type.value.startswith("tool.") or event.type.value.startswith("output."):
            self.agent_events.append(event)
        elif event.type.value == "custom":
            # Check if it's a taskboard event
            if "event_name" in event.data and event.data["event_name"] and event.data["event_name"].startswith("taskboard."):
                self.taskboard_events.append(event)
        
        # Print events in real-time
        self._print_event(event)
    
    def _print_event(self, event: Event) -> None:
        """Print event in real-time."""
        # Determine event category for coloring
        event_name = event.type.value
        if event_name == "custom" and "event_name" in event.data:
            event_name = event.data.get("event_name", event_name)
        
        if event_name.startswith("agent."):
            prefix = "🤖 AGENT"
        elif event_name.startswith("llm."):
            prefix = "🧠 LLM"
        elif event_name.startswith("tool."):
            prefix = "🔧 TOOL"
        elif event_name.startswith("taskboard."):
            prefix = "📋 TASKBOARD"
        elif event_name.startswith("output."):
            prefix = "📤 OUTPUT"
        else:
            prefix = "📡 EVENT"
        
        # Extract key info based on event type
        if event_name == "taskboard.task_added":
            desc = event.data.get("description", "")
            print(f"{prefix}: Task added - {desc}")
        elif event_name == "taskboard.task_started":
            desc = event.data.get("description", "")
            print(f"{prefix}: Task started - {desc}")
        elif event_name == "taskboard.task_completed":
            desc = event.data.get("description", "")
            result = event.data.get("result", "")
            if result:
                print(f"{prefix}: Task completed - {desc} → {result[:50]}...")
            else:
                print(f"{prefix}: Task completed - {desc}")
        elif event_name == "taskboard.note_added":
            content = event.data.get("content", "")[:60]
            category = event.data.get("category", "")
            print(f"{prefix}: Note added [{category}] - {content}...")
        elif event_name == "agent.thinking":
            phase = event.data.get("phase", "")
            print(f"{prefix}: Thinking ({phase})...")
        elif event_name == "llm.tool_decision":
            count = len(event.data.get("tool_calls", []))
            names = [tc.get("name") for tc in event.data.get("tool_calls", [])]
            print(f"{prefix}: Calling {count} tool(s) - {', '.join(names)}")
        elif event_name == "tool.called":
            name = event.data.get("tool_name", "")
            print(f"{prefix}: Executing {name}...")
        elif event_name == "tool.result":
            name = event.data.get("tool_name", "")
            success = not event.data.get("error")
            status = "✓" if success else "✗"
            print(f"{prefix}: {name} {status}")
        else:
            print(f"{prefix}: {event_name}")
    
    def summary(self) -> None:
        """Print summary of all captured events."""
        print("\n" + "="*80)
        print("EVENT SUMMARY")
        print("="*80)
        
        print(f"\n📊 Total Events: {len(self.events)}")
        print(f"   🤖 Agent events: {len(self.agent_events)}")
        print(f"   📋 TaskBoard events: {len(self.taskboard_events)}")
        
        # Agent event breakdown
        if self.agent_events:
            print("\n🤖 Agent Events:")
            agent_types = {}
            for e in self.agent_events:
                agent_types[e.type] = agent_types.get(e.type, 0) + 1
            for etype, count in sorted(agent_types.items(), key=lambda x: x[0].value):
                print(f"   • {etype.value}: {count}")
        
        # TaskBoard event breakdown
        if self.taskboard_events:
            print("\n📋 TaskBoard Events:")
            tb_types = {}
            for e in self.taskboard_events:
                event_name = e.data.get("event_name", "unknown")
                tb_types[event_name] = tb_types.get(event_name, 0) + 1
            for event_name, count in sorted(tb_types.items()):
                print(f"   • {event_name}: {count}")
        
        print("\n" + "="*80)


async def main():
    """Run comprehensive observability demo."""
    print("="*80)
    print("COMPREHENSIVE AGENT + TASKBOARD OBSERVABILITY")
    print("="*80)
    
    # Create model
    model = get_model()
    
    # Create event bus and observer
    event_bus = EventBus()
    observer = ComprehensiveObserver()
    observer.attach(event_bus)
    
    # Create agent with taskboard and observability
    print("\n✓ Creating agent with taskboard and event tracking...")
    agent = Agent(
        name="ResearchAssistant",
        model=model,
        taskboard=True,  # Enable taskboard with tools
        event_bus=event_bus,
    )
    
    # Define a multi-step task
    task = """
    Research Python async programming and create a summary:
    1. First, understand what async/await is
    2. Then, identify 3 key benefits
    3. Finally, provide a simple example
    
    Track your progress using tasks and notes.
    """
    
    print(f"\n📝 Task:\n{task}\n")
    print("="*80)
    print("EXECUTION LOG (Real-time Events)")
    print("="*80 + "\n")
    
    # Run the agent
    result = await agent.run(task)
    
    # Print result
    print("\n" + "="*80)
    print("AGENT RESULT")
    print("="*80)
    print(result)
    
    # Print event summary
    observer.summary()
    
    # Show taskboard state
    print("\n" + "="*80)
    print("TASKBOARD STATE")
    print("="*80)
    print(agent.taskboard.summary())


if __name__ == "__main__":
    asyncio.run(main())
