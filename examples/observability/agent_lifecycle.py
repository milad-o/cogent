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
    uv run ./examples/observability/agent_lifecycle.py
"""

import asyncio
from typing import Literal

from cogent import Agent
from cogent.observability import Channel, ObservabilityLevel, Observer, TraceType
from cogent.tools.base import tool


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
def get_info(topic: Literal["python", "async", "agents"]) -> str:
    """Get information about a specific topic.

    Args:
        topic: Must be one of: 'python', 'async', or 'agents'

    Returns:
        Information about the topic
    """
    info = {
        "python": "Python is a high-level programming language known for readability and versatility",
        "async": "Async programming in Python enables concurrent execution using asyncio, allowing non-blocking I/O operations",
        "agents": "AI agents are autonomous programs that can reason, make decisions, and act to achieve goals",
    }
    return info.get(
        topic,
        "Topic not recognized. Please choose from 'python', 'async', or 'agents'.",
    )


class AgentLifecycleObserver(Observer):
    """Observer that tracks agent lifecycle events."""

    def __init__(self):
        super().__init__(
            level=ObservabilityLevel.DEBUG,
            channels=[Channel.ALL],
            truncate=200,
        )
        self.captured_events = []
        self.agent_invocations = 0
        self.tool_calls = 0
        self.llm_requests = 0
        self._seen_thinking: set[str] = set()

    def _truncate(self, text: str, *, default: int = 100) -> str:
        limit = self.config.truncate if self.config.truncate else default
        if not limit:
            return text
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _handle_event(self, event: object) -> None:
        """Override to handle events."""
        # Call parent to maintain Observer functionality
        super()._handle_event(event)

        # Observer._handle_event expects an Event; if something else comes through,
        # just skip custom logic.
        if not hasattr(event, "type") or not hasattr(event, "data"):
            return

        # Custom tracking
        self.captured_events.append(event)

        # Agent lifecycle events - print our own summaries
        if event.type == TraceType.AGENT_INVOKED:
            self.agent_invocations += 1
            agent_name = event.data.get("agent_name", "Unknown")
            task = event.data.get("task", "No task specified")
            print(f"🚀 [AGENT INVOKED] {agent_name}")
            print(f"   Task: {task}")
            print()

        elif event.type == TraceType.AGENT_THINKING:
            agent_name = event.data.get("agent_name", "Unknown")
            if agent_name in self._seen_thinking:
                return
            self._seen_thinking.add(agent_name)
            print(f"🧠 [AGENT THINKING] {agent_name} is processing...")

        elif event.type == TraceType.LLM_REQUEST:
            self.llm_requests += 1
            model = event.data.get("model", "Unknown")
            messages = event.data.get("messages", [])
            print(f"💬 [LLM REQUEST #{self.llm_requests}] Model: {model}")
            print(f"   Messages: {len(messages)} in conversation")

        elif event.type == TraceType.LLM_RESPONSE:
            content = str(event.data.get("content", ""))
            content_clean = content.replace("\n", " ").strip()
            if not content_clean and event.data.get("has_tool_calls"):
                tool_calls = event.data.get("tool_calls", [])
                tools = (
                    ", ".join(tc.get("name", "?") for tc in tool_calls)
                    if tool_calls
                    else "(unknown)"
                )
                print(f"✨ [LLM RESPONSE] (tool_calls: {tools})")
            else:
                print(f"✨ [LLM RESPONSE] {self._truncate(content_clean)}")

        elif event.type == TraceType.LLM_TOOL_DECISION:
            tools_selected = event.data.get("tools_selected", [])
            if tools_selected:
                print(
                    f"🔧 [TOOL DECISION] Agent decided to call {len(tools_selected)} tool(s):"
                )
                for tool_name in tools_selected:
                    print(f"   - {tool_name}")
            else:
                print("💭 [TOOL DECISION] Agent decided NOT to use tools")

        elif event.type == TraceType.TOOL_CALLED:
            self.tool_calls += 1
            tool_name = event.data.get("tool_name") or event.data.get("tool", "Unknown")
            args = event.data.get("args", {})
            print(f"⚙️  [TOOL CALL #{self.tool_calls}] {tool_name}")
            print(f"   Args: {args}")

        elif event.type == TraceType.TOOL_RESULT:
            tool_name = event.data.get("tool_name") or event.data.get("tool", "Unknown")
            result = str(event.data.get("result", ""))
            print(
                f"✓ [TOOL RESULT] {tool_name}: {self._truncate(result.replace('\n', ' ').strip())}"
            )

        elif event.type == TraceType.AGENT_RESPONDED:
            agent_name = event.data.get("agent_name", "Unknown")
            response = str(event.data.get("response", ""))
            self._seen_thinking.discard(agent_name)
            print(f"✅ [AGENT RESPONDED] {agent_name}")
            print(f"   Response: {self._truncate(response.replace('\n', ' ').strip())}")
            print()

        elif event.type == TraceType.AGENT_ERROR:
            agent_name = event.data.get("agent_name", "Unknown")
            error = event.data.get("error", "Unknown error")
            print(f"❌ [AGENT ERROR] {agent_name}")
            print(f"   Error: {error}")
            print()

    def summary(self):
        """Print summary of observed events."""
        print()
        print("=" * 70)
        print("Observer Summary")
        print("=" * 70)
        print(f"Total Events: {len(self.captured_events)}")
        print(f"Agent Invocations: {self.agent_invocations}")
        print(f"LLM Requests: {self.llm_requests}")
        print(f"Tool Calls: {self.tool_calls}")
        print()

        # Count by event type
        type_counts = {}
        for event in self.captured_events:
            type_counts[event.type] = type_counts.get(event.type, 0) + 1

        print("Events by Type:")
        for event_type, count in sorted(type_counts.items(), key=lambda x: x[0].value):
            type_name = event_type.value
            print(f"  {type_name}: {count}")


async def main():
    """Demo agent lifecycle observability."""
    print("=" * 70)
    print("Agent Lifecycle Observability Demo")
    print("=" * 70)
    print()

    # Create observer
    observer = AgentLifecycleObserver()

    # Create agent with observer
    agent = Agent(
        name="MathAssistant",
        model="gpt4",
        tools=[calculate, get_info],
        observer=observer,
    )

    print("✅ Agent created with lifecycle observability enabled")
    print()

    # Task that will use tools
    task = "What is 15 multiplied by 23, and then explain what 'async' is"

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


if __name__ == "__main__":
    asyncio.run(main())
