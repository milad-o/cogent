"""
Demo: Complete Event System

Showcases ALL event types in AgenticFlow with practical examples.
Events are organized by category for easy understanding.

Event Categories:
  - User/Output: USER_INPUT, OUTPUT_GENERATED, OUTPUT_STREAMED
  - Agent Lifecycle: AGENT_INVOKED, AGENT_THINKING, AGENT_RESPONDED, etc.
  - LLM Deep Tracing: LLM_REQUEST, LLM_RESPONSE, LLM_TOOL_DECISION
  - Tool Execution: TOOL_CALLED, TOOL_RESULT, TOOL_ERROR
  - Task Management: TASK_STARTED, TASK_COMPLETED, TASK_FAILED
  - Message Passing: MESSAGE_SENT, MESSAGE_RECEIVED
  - Memory: MEMORY_READ, MEMORY_WRITE, MEMORY_SEARCH
  - Retrieval (RAG): RETRIEVAL_START, RETRIEVAL_COMPLETE
  - Streaming: STREAM_START, TOKEN_STREAMED, STREAM_END

Usage:
    uv run python examples/observability/01_events.py
"""

import asyncio
from collections import defaultdict

from config import get_model

from agenticflow import tool
from agenticflow import Agent, Flow, Observer, ObservabilityLevel, EventBus
from agenticflow.observability.event import EventType


# ============================================================
# Helper: Create agent with event bus attached
# ============================================================
def create_agent_with_observer(
    observer: Observer,
    **agent_kwargs,
) -> Agent:
    """Create an agent with observer attached via event bus."""
    event_bus = EventBus()
    observer.attach(event_bus)
    agent = Agent(**agent_kwargs)
    agent.event_bus = event_bus
    return agent


# ============================================================
# Example 1: Basic Event Collection
# ============================================================
async def example_basic_events():
    """Collect and categorize all events from a simple agent run."""
    print("\n" + "=" * 60)
    print("  Example 1: Basic Event Collection")
    print("=" * 60)
    
    model = get_model()
    events_by_category: dict[str, list[str]] = defaultdict(list)
    
    def collect_event(event):
        category = event.type.value.split(".")[0]
        events_by_category[category].append(event.type.value)
    
    observer = Observer(
        level=ObservabilityLevel.OFF,  # Silent - we handle output
        on_event=collect_event,
    )
    
    agent = create_agent_with_observer(
        observer,
        name="Assistant",
        model=model,
        instructions="You are a helpful assistant. Be concise.",
    )
    
    # Run a simple task
    print("\nRunning: 'What is 2+2?'")
    result = await agent.run("What is 2+2?")
    
    print(f"\nResult: {result[:100]}...")
    print("\n📊 Events by Category:")
    for category, events in sorted(events_by_category.items()):
        unique = list(dict.fromkeys(events))  # Preserve order, remove dupes
        print(f"  {category}: {unique}")
    
    print(f"\n  Total events: {sum(len(e) for e in events_by_category.values())}")


# ============================================================
# Example 2: User Input & Output Events
# ============================================================
async def example_user_output_events():
    """Track user input and final output events."""
    print("\n" + "=" * 60)
    print("  Example 2: User Input & Output Events")
    print("=" * 60)
    
    model = get_model()
    user_events = []
    output_events = []
    
    def track_io(event):
        if event.type == EventType.USER_INPUT:
            user_events.append(event.data)
        elif event.type == EventType.OUTPUT_GENERATED:
            output_events.append(event.data)
    
    observer = Observer(level=ObservabilityLevel.OFF, on_event=track_io)
    
    planner = Agent(name="Planner", model=model, instructions="Create brief plans.")
    executor = Agent(name="Executor", model=model, instructions="Execute plans briefly.")
    
    flow = Flow(
        name="planning",
        agents=[planner, executor],
        topology="pipeline",
        observer=observer,
    )
    
    print("\nRunning pipeline flow...")
    result = await flow.run("Plan a 10-minute workout")
    
    print("\n👤 USER_INPUT Events:")
    for data in user_events:
        input_text = data.get("input", "")[:80]
        print(f"  → {input_text}...")
    
    print("\n📝 OUTPUT_GENERATED Events:")
    for data in output_events:
        output_text = data.get("output", "")[:80]
        print(f"  → {output_text}...")


# ============================================================
# Example 3: Agent Lifecycle Events
# ============================================================
async def example_agent_lifecycle():
    """Track the complete agent lifecycle."""
    print("\n" + "=" * 60)
    print("  Example 3: Agent Lifecycle Events")
    print("=" * 60)
    
    model = get_model()
    lifecycle_events = []
    
    # Agent lifecycle event types
    agent_events = {
        EventType.AGENT_INVOKED,
        EventType.AGENT_THINKING,
        EventType.AGENT_REASONING,
        EventType.AGENT_RESPONDED,
        EventType.AGENT_ERROR,
    }
    
    def track_lifecycle(event):
        if event.type in agent_events:
            agent = event.data.get("agent_name", "unknown")
            lifecycle_events.append((event.type.value, agent))
    
    observer = Observer(level=ObservabilityLevel.OFF, on_event=track_lifecycle)
    
    researcher = Agent(name="Researcher", model=model, instructions="Research topics briefly.")
    writer = Agent(name="Writer", model=model, instructions="Write content briefly.")
    
    flow = Flow(
        name="content",
        agents=[researcher, writer],
        topology="pipeline",
        observer=observer,
    )
    
    print("\nRunning 2-agent pipeline...")
    await flow.run("Write about Python")
    
    print("\n🔄 Agent Lifecycle Timeline:")
    for event_type, agent in lifecycle_events:
        symbol = {
            "agent.invoked": "▶",
            "agent.thinking": "🧠",
            "agent.reasoning": "💭",
            "agent.responded": "✓",
            "agent.error": "❌",
        }.get(event_type, "·")
        print(f"  {symbol} [{agent}] {event_type}")


# ============================================================
# Example 4: Tool Events
# ============================================================
async def example_tool_events():
    """Track tool calls and results."""
    print("\n" + "=" * 60)
    print("  Example 4: Tool Execution Events")
    print("=" * 60)
    
    @tool
    def calculate(expression: str) -> str:
        """Calculate a math expression."""
        try:
            result = eval(expression, {"__builtins__": {}})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    
    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}: Sunny, 72°F"
    
    model = get_model()
    tool_events = []
    
    tool_event_types = {
        EventType.TOOL_CALLED,
        EventType.TOOL_RESULT,
        EventType.TOOL_ERROR,
        EventType.LLM_TOOL_DECISION,
    }
    
    def track_tools(event):
        if event.type in tool_event_types:
            tool_events.append((event.type.value, event.data))
    
    observer = Observer(level=ObservabilityLevel.OFF, on_event=track_tools)
    
    agent = create_agent_with_observer(
        observer,
        name="Calculator",
        model=model,
        tools=[calculate, get_weather],
        instructions="Use tools to answer questions. Be concise.",
    )
    
    print("\nRunning agent with tools...")
    result = await agent.run("What is 15 * 7? Also, what's the weather in Paris?")
    
    print("\n🔧 Tool Events:")
    for event_type, data in tool_events:
        if event_type == "tool.called":
            print(f"  📤 TOOL_CALLED: {data.get('tool_name')} with {data.get('args')}")
        elif event_type == "tool.result":
            result_preview = str(data.get('result', ''))[:50]
            print(f"  📥 TOOL_RESULT: {data.get('tool_name')} → {result_preview}")
        elif event_type == "llm.tool_decision":
            print(f"  🎯 LLM_TOOL_DECISION: {data.get('tools_selected')}")
    
    print(f"\n  Total tool events: {len(tool_events)}")


# ============================================================
# Example 5: LLM Deep Tracing Events
# ============================================================
async def example_llm_tracing():
    """Track LLM request/response for deep observability."""
    print("\n" + "=" * 60)
    print("  Example 5: LLM Deep Tracing")
    print("=" * 60)
    
    model = get_model()
    llm_events = []
    
    llm_event_types = {
        EventType.LLM_REQUEST,
        EventType.LLM_RESPONSE,
    }
    
    def track_llm(event):
        if event.type in llm_event_types:
            llm_events.append((event.type.value, event.data))
    
    observer = Observer(level=ObservabilityLevel.OFF, on_event=track_llm)
    
    agent = create_agent_with_observer(
        observer,
        name="Analyst",
        model=model,
        instructions="You analyze data. Be concise.",
    )
    
    print("\nRunning agent (tracking LLM calls)...")
    await agent.run("Analyze the number 42")
    
    print("\n📡 LLM Request/Response Events:")
    for event_type, data in llm_events:
        if event_type == "llm.request":
            msg_count = data.get("message_count", 0)
            model_name = data.get("model", "unknown")
            prompt = data.get("prompt", "")[:50]
            print(f"  📤 LLM_REQUEST: {msg_count} messages to {model_name}")
            print(f"     Prompt: \"{prompt}...\"")
        elif event_type == "llm.response":
            duration = data.get("duration_ms", 0)
            has_tools = data.get("has_tool_calls", False)
            content = data.get("content", "")[:50]
            print(f"  📥 LLM_RESPONSE: {duration:.0f}ms, tools={has_tools}")
            print(f"     Content: \"{content}...\"")


# ============================================================
# Example 6: Message Passing Events
# ============================================================
async def example_message_events():
    """Track inter-agent message passing."""
    print("\n" + "=" * 60)
    print("  Example 6: Message Passing Events")
    print("=" * 60)
    
    model = get_model()
    message_events = []
    
    message_event_types = {
        EventType.MESSAGE_SENT,
        EventType.MESSAGE_RECEIVED,
    }
    
    def track_messages(event):
        if event.type in message_event_types:
            message_events.append((event.type.value, event.data))
    
    observer = Observer(level=ObservabilityLevel.OFF, on_event=track_messages)
    
    # Mesh topology has agents communicating with each other
    agent1 = Agent(name="Alice", model=model, instructions="You are Alice. Collaborate briefly.")
    agent2 = Agent(name="Bob", model=model, instructions="You are Bob. Collaborate briefly.")
    agent3 = Agent(name="Charlie", model=model, instructions="You are Charlie. Synthesize briefly.")
    
    flow = Flow(
        name="collaboration",
        agents=[agent1, agent2, agent3],
        topology="mesh",
        mesh_rounds=1,
        observer=observer,
    )
    
    print("\nRunning mesh flow (agents exchange messages)...")
    await flow.run("What is the best programming language?")
    
    print("\n💬 Message Events:")
    for event_type, data in message_events[:10]:  # Show first 10
        if event_type == "message.sent":
            sender = data.get("from", "?")
            receiver = data.get("to", "?")
            content = data.get("content", "")[:40]
            print(f"  📤 {sender} → {receiver}: \"{content}...\"")
        elif event_type == "message.received":
            receiver = data.get("to", "?")
            sender = data.get("from", "?")
            print(f"  📥 {receiver} ← {sender}")
    
    if len(message_events) > 10:
        print(f"  ... and {len(message_events) - 10} more")


# ============================================================
# Example 7: Task Events (Pipeline)
# ============================================================
async def example_task_events():
    """Track task lifecycle in a pipeline."""
    print("\n" + "=" * 60)
    print("  Example 7: Task Lifecycle Events")
    print("=" * 60)
    
    model = get_model()
    task_events = []
    
    task_event_types = {
        EventType.TASK_CREATED,
        EventType.TASK_STARTED,
        EventType.TASK_COMPLETED,
        EventType.TASK_FAILED,
    }
    
    def track_tasks(event):
        if event.type in task_event_types:
            task_events.append((event.type.value, event.data))
    
    observer = Observer(level=ObservabilityLevel.OFF, on_event=track_tasks)
    
    step1 = Agent(name="Step1", model=model, instructions="First step. Be brief.")
    step2 = Agent(name="Step2", model=model, instructions="Second step. Be brief.")
    step3 = Agent(name="Step3", model=model, instructions="Third step. Be brief.")
    
    flow = Flow(
        name="pipeline",
        agents=[step1, step2, step3],
        topology="pipeline",
        observer=observer,
    )
    
    print("\nRunning 3-step pipeline...")
    await flow.run("Process this data")
    
    print("\n📋 Task Events:")
    for event_type, data in task_events:
        agent = data.get("agent", data.get("agent_name", "?"))
        symbol = {
            "task.created": "📝",
            "task.started": "▶",
            "task.completed": "✅",
            "task.failed": "❌",
        }.get(event_type, "·")
        print(f"  {symbol} {event_type}: {agent}")


# ============================================================
# Example 8: All Events Summary
# ============================================================
async def example_all_events_summary():
    """Show complete event flow with visual timeline."""
    print("\n" + "=" * 60)
    print("  Example 8: Complete Event Timeline")
    print("=" * 60)
    
    @tool
    def lookup(topic: str) -> str:
        """Look up information about a topic."""
        return f"Information about {topic}: It's interesting!"
    
    model = get_model()
    all_events = []
    
    def collect_all(event):
        all_events.append((event.type.value, event.timestamp, event.data))
    
    observer = Observer(level=ObservabilityLevel.OFF, on_event=collect_all)
    
    agent = create_agent_with_observer(
        observer,
        name="ResearchBot",
        model=model,
        tools=[lookup],
        instructions="Use the lookup tool to research topics. Be concise.",
    )
    
    print("\nRunning agent with tool (capturing all events)...")
    await agent.run("Research quantum computing")
    
    print("\n📊 Complete Event Timeline:")
    print("-" * 50)
    
    # Group by category for cleaner display
    categories_seen = set()
    for event_type, timestamp, data in all_events:
        category = event_type.split(".")[0]
        
        # Visual indicators
        symbols = {
            "user": "👤",
            "output": "📝",
            "agent": "🤖",
            "llm": "🧠",
            "tool": "🔧",
            "task": "📋",
            "message": "💬",
            "stream": "📺",
            "memory": "💾",
            "retrieval": "🔍",
        }
        symbol = symbols.get(category, "·")
        
        # Show each unique category once with all its events
        if category not in categories_seen:
            categories_seen.add(category)
        
        # Format event display
        time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]
        print(f"  {time_str} {symbol} {event_type}")
    
    print("-" * 50)
    print(f"  Total: {len(all_events)} events")
    
    # Summary by category
    category_counts = defaultdict(int)
    for event_type, _, _ in all_events:
        category = event_type.split(".")[0]
        category_counts[category] += 1
    
    print("\n  By Category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")


# ============================================================
# Main
# ============================================================
async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("  AgenticFlow Event System Demo")
    print("=" * 60)
    print("\nThis demo showcases all event types with practical examples.")
    print("Events enable observability, debugging, and monitoring.")
    
    # Run examples
    await example_basic_events()
    await example_user_output_events()
    await example_agent_lifecycle()
    await example_tool_events()
    await example_llm_tracing()
    await example_message_events()
    await example_task_events()
    await example_all_events_summary()
    
    print("\n" + "=" * 60)
    print("  Event Categories Reference")
    print("=" * 60)
    print("""
  👤 USER EVENTS
     USER_INPUT          - User provided input/prompt
     USER_FEEDBACK       - User provided feedback (HITL)
  
  📝 OUTPUT EVENTS  
     OUTPUT_GENERATED    - Final output ready for user
     OUTPUT_STREAMED     - Output token streamed to user
  
  🤖 AGENT EVENTS
     AGENT_INVOKED       - Agent started working on task
     AGENT_THINKING      - Agent is processing (LLM call)
     AGENT_REASONING     - Extended thinking/chain-of-thought
     AGENT_RESPONDED     - Agent produced a response
     AGENT_ERROR         - Agent encountered an error
     AGENT_SPAWNED       - Dynamic agent created
  
  🧠 LLM EVENTS (Deep Tracing)
     LLM_REQUEST         - Request sent to LLM
     LLM_RESPONSE        - Response received from LLM
     LLM_TOOL_DECISION   - LLM decided to call tool(s)
  
  🔧 TOOL EVENTS
     TOOL_CALLED         - Tool invocation started
     TOOL_RESULT         - Tool returned result
     TOOL_ERROR          - Tool execution failed
     TOOL_DEFERRED       - Tool returned async result
  
  📋 TASK EVENTS
     TASK_CREATED        - New task created
     TASK_STARTED        - Task execution began
     TASK_COMPLETED      - Task finished successfully
     TASK_FAILED         - Task failed with error
  
  💬 MESSAGE EVENTS
     MESSAGE_SENT        - Inter-agent message sent
     MESSAGE_RECEIVED    - Inter-agent message received
  
  💾 MEMORY EVENTS
     MEMORY_READ         - Memory retrieved
     MEMORY_WRITE        - Memory stored
     MEMORY_SEARCH       - Semantic search performed
  
  🔍 RETRIEVAL EVENTS (RAG)
     RETRIEVAL_START     - Retrieval query started
     RETRIEVAL_COMPLETE  - Retrieval finished
     RERANK_START        - Reranking started
     RERANK_COMPLETE     - Reranking finished
  
  📺 STREAMING EVENTS
     STREAM_START        - Streaming began
     TOKEN_STREAMED      - Token received
     STREAM_END          - Streaming completed
""")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
