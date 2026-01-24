"""
Complete Event System

AgenticFlow event types for observability and monitoring:
- User/Output: USER_INPUT, OUTPUT_GENERATED
- Agent Lifecycle: AGENT_INVOKED, AGENT_THINKING, AGENT_RESPONDED
- LLM Tracing: LLM_REQUEST, LLM_RESPONSE, LLM_TOOL_DECISION
- Tool Execution: TOOL_CALLED, TOOL_RESULT, TOOL_ERROR
- Message Passing: MESSAGE_SENT, MESSAGE_RECEIVED

Usage: uv run python examples/observability/events.py
"""

import asyncio
from collections import defaultdict

from agenticflow import Agent, tool
from agenticflow.events import EventBus
from agenticflow.observability import ObservabilityLevel, Observer, TraceType


def create_agent_with_observer(observer: Observer, **agent_kwargs) -> Agent:
    """Create an agent with observer attached via event bus."""
    event_bus = EventBus()
    observer.attach(event_bus)
    agent = Agent(**agent_kwargs)
    agent.event_bus = event_bus
    return agent


@tool
def calculate(expression: str) -> float:
    """Calculate a math expression."""
    return eval(expression)


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"


async def demo_basic_events():
    """Collect and categorize all events from a simple agent run."""
    print("\n" + "=" * 60)
    print("1. Basic Event Collection")
    print("=" * 60)

    events_by_category: dict[str, list[str]] = defaultdict(list)

    def collect_event(event):
        category = event.type.value.split(".")[0]
        events_by_category[category].append(event.type.value)

    observer = Observer(level=ObservabilityLevel.OFF, on_event=collect_event)

    agent = create_agent_with_observer(
        observer,
        name="Assistant",
        model="gpt4",
        instructions="Be concise.",
    )

    print("\nRunning: 'What is 2+2?'")
    result = await agent.run("What is 2+2?")

    print(f"\nResult: {result.unwrap()[:100]}...")
    print("\nEvents by Category:")
    for category, events in sorted(events_by_category.items()):
        unique = list(dict.fromkeys(events))
        print(f"  {category}: {unique}")

    print(f"\nTotal events: {sum(len(e) for e in events_by_category.values())}")


async def demo_user_output_events():
    """Track user input and final output events."""
    print("\n" + "=" * 60)
    print("2. User Input & Output Events")
    print("=" * 60)

    user_events = []
    output_events = []

    def track_io(event):
        if event.type == TraceType.USER_INPUT:
            user_events.append(event.data)
        elif event.type == TraceType.OUTPUT_GENERATED:
            output_events.append(event.data)

    observer = Observer(level=ObservabilityLevel.OFF, on_event=track_io)

    agent = create_agent_with_observer(
        observer,
        name="Helper",
        model="gpt4",
        instructions="Be helpful and concise.",
    )

    print("\nRunning conversation...")
    await agent.run("Hello!")
    await agent.run("What is 5+5?")

    print(f"\nUser inputs: {len(user_events)}")
    for i, data in enumerate(user_events, 1):
        msg = data.get("message", "")
        print(f"  [{i}] {msg[:60]}...")

    print(f"\nOutputs generated: {len(output_events)}")
    for i, data in enumerate(output_events, 1):
        content = data.get("content", "")
        print(f"  [{i}] {content[:60]}...")


async def demo_agent_lifecycle():
    """Track agent lifecycle events."""
    print("\n" + "=" * 60)
    print("3. Agent Lifecycle Events")
    print("=" * 60)

    lifecycle_events = []

    lifecycle_types = {
        TraceType.AGENT_INVOKED,
        TraceType.AGENT_THINKING,
        TraceType.AGENT_RESPONDED,
    }

    def track_lifecycle(event):
        if event.type in lifecycle_types:
            lifecycle_events.append((event.type.value, event.timestamp))

    observer = Observer(level=ObservabilityLevel.OFF, on_event=track_lifecycle)

    agent = create_agent_with_observer(
        observer,
        name="Thinker",
        model="gpt4",
        instructions="Think deeply. Be concise.",
    )

    print("\nRunning agent...")
    await agent.run("What is the meaning of life?")

    print("\nLifecycle Events:")
    for event_type, timestamp in lifecycle_events:
        print(f"  {event_type} at {timestamp}")

    print(f"\nTotal: {len(lifecycle_events)} lifecycle events")


async def demo_tool_events():
    """Track tool execution events."""
    print("\n" + "=" * 60)
    print("4. Tool Execution Events")
    print("=" * 60)

    tool_events = []

    tool_event_types = {
        TraceType.TOOL_CALLED,
        TraceType.TOOL_RESULT,
        TraceType.LLM_TOOL_DECISION,
    }

    def track_tools(event):
        if event.type in tool_event_types:
            tool_events.append((event.type.value, event.data))

    observer = Observer(level=ObservabilityLevel.OFF, on_event=track_tools)

    agent = create_agent_with_observer(
        observer,
        name="Calculator",
        model="gpt4",
        tools=[calculate, get_weather],
        instructions="Use tools to answer. Be concise.",
    )

    print("\nRunning agent with tools...")
    await agent.run("What is 15 * 7? Also, weather in Paris?")

    print("\nTool Events:")
    for event_type, data in tool_events:
        if event_type == "tool.called":
            print(f"  📤 TOOL_CALLED: {data.get('tool_name')} with {data.get('args')}")
        elif event_type == "tool.result":
            result_preview = str(data.get('result', ''))[:50]
            print(f"  📥 TOOL_RESULT: {data.get('tool_name')} → {result_preview}")
        elif event_type == "llm.tool_decision":
            print(f"  🎯 LLM_TOOL_DECISION: {data.get('tools_selected')}")

    print(f"\nTotal: {len(tool_events)} tool events")


async def demo_llm_tracing():
    """Track LLM request/response for deep observability."""
    print("\n" + "=" * 60)
    print("5. LLM Deep Tracing")
    print("=" * 60)

    llm_events = []

    llm_event_types = {TraceType.LLM_REQUEST, TraceType.LLM_RESPONSE}

    def track_llm(event):
        if event.type in llm_event_types:
            llm_events.append((event.type.value, event.data))

    observer = Observer(level=ObservabilityLevel.OFF, on_event=track_llm)

    agent = create_agent_with_observer(
        observer,
        name="Analyst",
        model="gpt4",
        instructions="Analyze data. Be concise.",
    )

    print("\nRunning agent (tracking LLM calls)...")
    await agent.run("Analyze the number 42")

    print("\nLLM Request/Response Events:")
    for event_type, data in llm_events:
        if event_type == "llm.request":
            msg_count = data.get("message_count", 0)
            model_name = data.get("model", "unknown")
            print(f"  📤 LLM_REQUEST: {msg_count} messages to {model_name}")
        elif event_type == "llm.response":
            duration = data.get("duration_ms", 0)
            has_tools = data.get("has_tool_calls", False)
            content = data.get("content", "")[:50]
            print(f"  📥 LLM_RESPONSE: {duration:.0f}ms, tools={has_tools}")
            print(f"     Content: \"{content}...\"")


async def demo_custom_observer():
    """Custom observer for specific event types."""
    print("\n" + "=" * 60)
    print("6. Custom Observer (Tool Focus)")
    print("=" * 60)

    class ToolObserver:
        """Custom observer that only tracks tool events."""
        def __init__(self):
            self.tool_calls = []

        def __call__(self, event):
            if event.type == TraceType.TOOL_CALLED:
                self.tool_calls.append(event.data.get("tool_name"))
                print(f"  🔧 Tool called: {event.data.get('tool_name')}")

    tool_observer = ToolObserver()
    observer = Observer(level=ObservabilityLevel.OFF, on_event=tool_observer)

    agent = create_agent_with_observer(
        observer,
        name="Worker",
        model="gpt4",
        tools=[calculate, get_weather],
        instructions="Use tools to answer.",
    )

    print("\nRunning agent...")
    await agent.run("What is 8+9? And weather in London?")

    print(f"\nTools used: {tool_observer.tool_calls}")


async def main():
    """Run all event system examples."""
    print("\n" + "=" * 60)
    print("AGENTICFLOW EVENT SYSTEM EXAMPLES")
    print("=" * 60)

    await demo_basic_events()
    await demo_user_output_events()
    await demo_agent_lifecycle()
    await demo_tool_events()
    await demo_llm_tracing()
    await demo_custom_observer()

    print("\n" + "=" * 60)
    print("✓ All demos completed")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
