"""
Agent Lifecycle Events - Track agent execution from start to finish.

Demonstrates observability of agent lifecycle:
- Agent invoked (started)
- Agent thinking
- LLM requests/responses
- Tool calls
- Agent responded (finished)
- Errors

Run:
    uv run python observability/agent_lifecycle.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_model

from agenticflow import Agent
from agenticflow.observability import Observer, EventBus, EventType
from agenticflow.observability.observer import ObservabilityLevel, Channel
from agenticflow.tools.base import tool


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: Math expression like "2 + 2"
        
    Returns:
        Result of calculation
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_info(topic: str) -> str:
    """Get information about a topic (mock).
    
    Args:
        topic: Topic to get info about
        
    Returns:
        Information about the topic
    """
    info = {
        "python": "Python is a high-level programming language",
        "async": "Async programming enables concurrent execution",
        "agents": "AI agents are autonomous programs that can reason and act",
    }
    return info.get(topic.lower(), f"No information available about {topic}")


class AgentLifecycleObserver(Observer):
    """Observer that tracks agent lifecycle events."""
    
    def __init__(self):
        super().__init__(
            level=ObservabilityLevel.DEBUG,
            channels=[Channel.ALL],
        )
        self.events = []
        self.agent_invocations = 0
        self.tool_calls = 0
        self.llm_requests = 0
        
    def _handle_event(self, event):
        """Override to handle events."""
        # Call parent to maintain Observer functionality
        super()._handle_event(event)
        
        # Custom tracking
        self.events.append(event)
        
        # Agent lifecycle events - print our own summaries
        if event.type == EventType.AGENT_INVOKED.value:
            self.agent_invocations += 1
            agent_name = event.data.get('agent_name', 'Unknown')
            task = event.data.get('task', 'No task specified')
            print(f"🚀 [AGENT INVOKED] {agent_name}")
            print(f"   Task: {task}")
            print()
        
        elif event.type == EventType.AGENT_THINKING.value:
            agent_name = event.data.get('agent_name', 'Unknown')
            print(f"🧠 [AGENT THINKING] {agent_name} is processing...")
        
        elif event.type == EventType.LLM_REQUEST.value:
            self.llm_requests += 1
            model = event.data.get('model', 'Unknown')
            messages = event.data.get('messages', [])
            print(f"💬 [LLM REQUEST #{self.llm_requests}] Model: {model}")
            print(f"   Messages: {len(messages)} in conversation")
        
        elif event.type == EventType.LLM_RESPONSE.value:
            content = event.data.get('content', '')
            truncated = content[:100] + "..." if len(content) > 100 else content
            print(f"✨ [LLM RESPONSE] {truncated}")
        
        elif event.type == EventType.LLM_TOOL_DECISION.value:
            tools = event.data.get('tools', [])
            if tools:
                print(f"🔧 [TOOL DECISION] Agent decided to call {len(tools)} tool(s):")
                for tool_call in tools:
                    print(f"   - {tool_call.get('name', 'unknown')}")
            else:
                print(f"💭 [TOOL DECISION] Agent decided NOT to use tools")
        
        elif event.type == EventType.TOOL_CALLED.value:
            self.tool_calls += 1
            tool_name = event.data.get('tool_name', 'Unknown')
            args = event.data.get('args', {})
            print(f"⚙️  [TOOL CALL #{self.tool_calls}] {tool_name}")
            print(f"   Args: {args}")
        
        elif event.type == EventType.TOOL_RESULT.value:
            tool_name = event.data.get('tool_name', 'Unknown')
            result = str(event.data.get('result', ''))[:100]
            print(f"✓ [TOOL RESULT] {tool_name}: {result}")
        
        elif event.type == EventType.AGENT_RESPONDED.value:
            agent_name = event.data.get('agent_name', 'Unknown')
            response = event.data.get('response', '')
            truncated = response[:100] + "..." if len(response) > 100 else response
            print(f"✅ [AGENT RESPONDED] {agent_name}")
            print(f"   Response: {truncated}")
            print()
        
        elif event.type == EventType.AGENT_ERROR.value:
            agent_name = event.data.get('agent_name', 'Unknown')
            error = event.data.get('error', 'Unknown error')
            print(f"❌ [AGENT ERROR] {agent_name}")
            print(f"   Error: {error}")
            print()
    
    def summary(self):
        """Print summary of observed events."""
        print()
        print("=" * 70)
        print("Observer Summary")
        print("=" * 70)
        print(f"Total Events: {len(self.events)}")
        print(f"Agent Invocations: {self.agent_invocations}")
        print(f"LLM Requests: {self.llm_requests}")
        print(f"Tool Calls: {self.tool_calls}")
        print()
        
        # Count by event type
        type_counts = {}
        for event in self.events:
            type_counts[event.type] = type_counts.get(event.type, 0) + 1
        
        print("Events by Type:")
        for event_type, count in sorted(type_counts.items(), key=lambda x: x[0].value if hasattr(x[0], 'value') else str(x[0])):
            type_name = event_type.value if hasattr(event_type, 'value') else str(event_type)
            print(f"  {type_name}: {count}")


async def main():
    """Demo agent lifecycle observability."""
    print("=" * 70)
    print("Agent Lifecycle Observability Demo")
    print("=" * 70)
    print()
    
    model = get_model()
    
    # Create event bus and observer
    event_bus = EventBus()
    observer = AgentLifecycleObserver()
    observer.attach(event_bus)
    
    # Create agent with event bus
    agent = Agent(
        name="MathAssistant",
        model=model,
        tools=[calculate, get_info],
        event_bus=event_bus,
    )
    
    print("✅ Agent created with lifecycle observability enabled")
    print("✅ Observer subscribed to all agent events")
    print(f"   Agent event_bus: {agent.event_bus}")
    print(f"   Same as our bus: {agent.event_bus is event_bus}")
    print()
    
    # Task that will use tools
    task = "What is 15 multiplied by 23, and then explain what Python async is"
    
    print(f"Task: {task}")
    print()
    print("=" * 70)
    print("Agent Lifecycle Events (Real-time):")
    print("=" * 70)
    print()
    
    result = await agent.run(task)
    
    print()
    print("=" * 70)
    print("Final Result:")
    print("=" * 70)
    print(result)
    print()
    
    # Print observer summary
    observer.summary()
    
    print()
    print("=" * 70)
    print("✅ Agent Lifecycle Observability Demo Complete!")
    print("=" * 70)
    print()
    print("Key Events Demonstrated:")
    print("  1. AGENT_INVOKED - When agent starts execution")
    print("  2. AGENT_THINKING - Agent is processing")
    print("  3. LLM_REQUEST - Sent request to LLM")
    print("  4. LLM_RESPONSE - Received LLM response")
    print("  5. LLM_TOOL_DECISION - Agent decided whether to use tools")
    print("  6. TOOL_CALLED - Agent called a tool")
    print("  7. TOOL_RESULT - Tool execution completed")
    print("  8. AGENT_RESPONDED - Agent finished with final answer")
    print()


if __name__ == "__main__":
    asyncio.run(main())
