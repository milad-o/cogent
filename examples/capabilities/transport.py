"""
Event Transport - LocalTransport & RedisTransport

Demonstrates both LocalTransport (no dependencies) and RedisTransport (distributed).

Features:
    ✓ LocalTransport: Pattern matching, event delivery, agent coordination
    ✓ RedisTransport: Cross-process communication (mocked if Redis unavailable)
    ✓ Real LLM integration with reactive flows
    ✓ Multiple agents coordinating via events

Run:
    uv run python examples/reactive/transport.py

Optional: Real Redis for production use
    docker run -d -p 6379:6379 redis:7-alpine
    # or: brew install redis && redis-server
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Add examples to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agenticflow import Agent
from agenticflow.events import Event, EventBus
from agenticflow.reactive import EventFlow, EventFlowConfig, react_to
from agenticflow.reactive.transport import LocalTransport, RedisTransport
from config import get_model


# Mock Redis classes for demo when Redis is unavailable
class MockRedisPubSub:
    """Mock Redis Pub/Sub - simulates Redis without server."""
    
    def __init__(self):
        self.subscriptions = {}
        self.messages = []
        self._closed = False
    
    async def psubscribe(self, pattern: str):
        self.subscriptions[pattern] = True
    
    async def punsubscribe(self, pattern: str):
        self.subscriptions.pop(pattern, None)
    
    async def get_message(self, ignore_subscribe_messages=False, timeout=None):
        if self.messages:
            return self.messages.pop(0)
        await asyncio.sleep(0.01)
        raise asyncio.TimeoutError()
    
    async def close(self):
        self._closed = True


class MockRedisClient:
    """Mock Redis client - simulates Redis without server."""
    
    def __init__(self):
        self.published_events = []
        self._pubsub = MockRedisPubSub()
        self._closed = False
    
    async def ping(self):
        return True
    
    async def publish(self, channel: str, message: str):
        event_data = json.loads(message)
        self.published_events.append({"channel": channel, "event": event_data})
        
        # Simulate pub/sub delivery
        self._pubsub.messages.append({
            "type": "pmessage",
            "pattern": channel,
            "channel": channel,
            "data": message
        })
        return 1
    
    def pubsub(self):
        return self._pubsub
    
    async def close(self):
        self._closed = True


async def basic_transport():
    """Basic LocalTransport - Pattern matching and event delivery."""
    print("\n" + "="*70)
    print("1. Basic LocalTransport - Pattern Matching")
    print("="*70 + "\n")
    
    transport = LocalTransport()
    await transport.connect()
    
    received = []
    
    async def handler(event: Event):
        received.append(event)
        print(f"  ✓ Received: {event.name} → {event.data}")
    
    # Subscribe to pattern
    print("Subscribing to pattern 'task.*'...\n")
    await transport.subscribe("task.*", handler)
    
    # Publish events
    print("Publishing events:")
    events = [
        Event(name="task.created", data={"id": "1", "title": "Design API"}),
        Event(name="task.updated", data={"id": "1", "status": "in_progress"}),
        Event(name="task.completed", data={"id": "1", "result": "success"}),
        Event(name="user.login", data={"user": "alice"}),  # Won't match
    ]
    
    for event in events:
        await transport.publish(event)
        print(f"  • Published: {event.name}")
    
    await asyncio.sleep(0.2)
    
    print(f"\n✅ Result: {len(received)}/3 matching events received")
    print("   (user.login was filtered by pattern 'task.*')")
    
    await transport.disconnect()


async def pattern_matching():
    """Advanced pattern matching with wildcards."""
    print("\n" + "="*70)
    print("2. Advanced Pattern Matching")
    print("="*70 + "\n")
    
    transport = LocalTransport()
    await transport.connect()
    
    patterns = {
        "task.*": [],
        "agent.**": [],
        "workflow.step.complete": []
    }
    
    print("Subscribing to multiple patterns:")
    for pattern in patterns:
        async def make_handler(p):
            async def handler(event):
                patterns[p].append(event)
            return handler
        
        handler = await make_handler(pattern)
        await transport.subscribe(pattern, handler)
        print(f"  ✓ {pattern}")
    
    print("\nPublishing events:")
    events = [
        Event(name="task.created", data={"id": "1"}),
        Event(name="task.updated", data={"id": "2"}),
        Event(name="agent.task.started", data={"id": "3"}),
        Event(name="agent.task.step.done", data={"id": "4"}),
        Event(name="workflow.step.complete", data={"id": "5"}),
        Event(name="user.login", data={"id": "6"}),
    ]
    
    for event in events:
        await transport.publish(event)
    
    await asyncio.sleep(0.3)
    
    print("\n✅ Pattern matching results:")
    for pattern, received in patterns.items():
        print(f"   {pattern:30} → {len(received)} matches")
        for evt in received:
            print(f"      - {evt.name}")
    
    await transport.disconnect()


async def agent_coordination():
    """Multiple agents coordinating via transport."""
    print("\n" + "="*70)
    print("3. Multi-Agent Coordination with LocalTransport")
    print("="*70 + "\n")
    
    transport = LocalTransport()
    await transport.connect()
    
    bus = EventBus(transport=transport)
    model = get_model()
    
    # Create specialized agents
    analyzer = Agent(
        name="analyzer",
        model=model,
        instructions="Analyze problems and identify root causes. Be concise (2-3 sentences).",
    )
    
    planner = Agent(
        name="planner",
        model=model,
        instructions="Create action plans. List exactly 3 steps.",
    )
    
    # Track events for demo
    events_seen = []
    async def track_events(event: Event):
        events_seen.append(event.name)
        print(f"  📡 Event: {event.name}")
    
    await transport.subscribe("**", track_events)
    
    # Create flow
    flow = EventFlow(
        config=EventFlowConfig(max_rounds=3),
        event_bus=bus,
    )
    
    flow.register(analyzer, [react_to("request.analyze")])
    flow.register(planner, [react_to("request.plan")])
    
    print("Registered agents:")
    print("  • analyzer → reacts to 'request.analyze'")
    print("  • planner  → reacts to 'request.plan'\n")
    
    # Run flow with event publishing
    async def publish_requests():
        await asyncio.sleep(0.1)
        print("Publishing: request.analyze")
        await bus.publish(Event(
            name="request.analyze",
            data={"problem": "API response time increased 50%"}
        ))
        await asyncio.sleep(0.2)
        print("Publishing: request.plan")
        await bus.publish(Event(
            name="request.plan",
            data={"objective": "Reduce API latency"}
        ))
    
    print("Starting coordinated flow...\n")
    publish_task = asyncio.create_task(publish_requests())
    result = await flow.run("Coordinate analysis and planning")
    await publish_task
    
    print(f"\n✅ Flow completed:")
    print(f"   Events processed: {result.events_processed}")
    print(f"   Total events seen: {len(events_seen)}")
    
    await transport.disconnect()


async def redis_transport():
    """Redis Transport for distributed systems."""
    print("\n" + "="*70)
    print("4. RedisTransport - Distributed Communication")
    print("="*70 + "\n")
    
    # Try real Redis first, fall back to mock
    use_mock = False
    mock_client = None
    
    try:
        # Test real Redis connection
        import redis.asyncio as aioredis
        test_client = aioredis.from_url("redis://localhost:6379")
        await test_client.ping()
        await test_client.close()
        print("✅ Connected to real Redis server\n")
    except Exception:
        print("📝 Redis server not found - using mock (demo mode)")
        print("   Install Redis for production: docker run -d -p 6379:6379 redis:7-alpine\n")
        use_mock = True
        mock_client = MockRedisClient()
    
    # Create Redis transport
    if use_mock:
        with patch('redis.asyncio.from_url', return_value=mock_client):
            await _run_redis(use_mock)
    else:
        await _run_redis(use_mock)


async def _run_redis(is_mock: bool):
    """Run Redis transport."""
    transport = RedisTransport(
        url="redis://localhost:6379",
        channel_prefix="agenticflow:demo",
    )
    await transport.connect()
    
    bus = EventBus(transport=transport)
    
    # Subscribe to events
    events_received = []
    
    async def handler(event: Event):
        events_received.append(event)
        print(f"  ✓ Received: {event.name} → {event.data}")
    
    await transport.subscribe("task.*", handler)
    print("Subscribed to 'task.*' pattern\n")
    
    # Publish events
    print("Publishing events:")
    events = [
        Event(name="task.created", data={"id": "456", "title": "Deploy"}),
        Event(name="task.updated", data={"id": "456", "status": "running"}),
        Event(name="task.completed", data={"id": "456", "result": "success"}),
    ]
    
    for event in events:
        await bus.publish(event)
        print(f"  • Published: {event.name}")
    
    # Give time for pub/sub processing
    await asyncio.sleep(0.5)
    
    print(f"\n✅ RedisTransport verified:")
    print(f"   Events published: {len(events)}")
    print(f"   Events received: {len(events_received)}")
    if is_mock:
        print(f"   Mode: Mock - install Redis for production")
    else:
        print(f"   Mode: Real Redis - events visible to ALL processes!")
    
    await transport.disconnect()


async def redis_multi_process():
    """Multi-process coordination with Redis and real LLM."""
    print("\n" + "="*70)
    print("5. Multi-Process Agent Coordination with Real LLM")
    print("="*70 + "\n")
    
    # Try real Redis first, fall back to mock
    use_mock = False
    mock_client = None
    
    try:
        import redis.asyncio as aioredis
        test_client = aioredis.from_url("redis://localhost:6379")
        await test_client.ping()
        await test_client.close()
    except Exception:
        use_mock = True
        mock_client = MockRedisClient()
        print("📝 Using mock Redis for demo (real Redis not available)")
        print()
    
    # Run with or without mock
    if use_mock:
        with patch('redis.asyncio.from_url', return_value=mock_client):
            await _run_multi_process(use_mock)
    else:
        await _run_multi_process(use_mock)


async def _run_multi_process(is_mock: bool):
    """Run multi-process coordination with real LLM."""
    transport = RedisTransport(
        url="redis://localhost:6379",
        channel_prefix="agenticflow:agents",
    )
    await transport.connect()
    
    bus = EventBus(transport=transport)
    model = get_model()
    
    # Create real agent with LLM
    agent = Agent(
        name="task_processor",
        model=model,
        instructions="You are a task processor. Analyze tasks and provide brief, actionable insights (2-3 sentences max).",
    )
    
    flow = EventFlow(
        config=EventFlowConfig(max_rounds=2),
        event_bus=bus,
    )
    
    flow.register(agent, [react_to("task.process")])
    
    print("✅ Agent registered to process 'task.process' events")
    if is_mock:
        print(f"   Mode: Mock Redis")
    else:
        print("   Mode: Real Redis - agent accessible from ANY process!")
    print()
    
    # Simulate external task
    async def publish_task():
        await asyncio.sleep(0.1)
        print("Publishing task event...")
        await bus.publish(Event(
            name="task.process",
            data={"task": "Analyze customer churn: 15% increase last quarter"}
        ))
    
    print("Starting agent flow with real LLM...\n")
    publish_task_coro = asyncio.create_task(publish_task())
    result = await flow.run("Process distributed tasks with AI")
    await publish_task_coro
    
    print(f"\n✅ Agent completed processing:")
    print(f"   Events processed: {result.events_processed}")
    if result.output:
        print(f"   Agent response: {result.output[:200]}...")
    
    await transport.disconnect()


async def main():
    """Run comprehensive transport examples."""
    print("\n" + "="*70)
    print("    EVENT TRANSPORT - LocalTransport & RedisTransport")
    print("="*70)
    
    # Core LocalTransport (always work)
    await basic_transport()
    await pattern_matching()
    await agent_coordination()
    
    # Optional Redis (require Redis server)
    await redis_transport()
    await redis_multi_process()
    
    print("\n" + "="*70)
    print("✅ Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • LocalTransport: Zero dependencies, perfect for single-process apps")
    print("  • Pattern matching: Wildcards (* and **) for flexible subscriptions")
    print("  • Agent coordination: Event-driven multi-agent workflows")
    print("  • RedisTransport: Optional for distributed multi-process systems")
    print("\nNext Steps:")
    print("  • Explore examples/reactive/ for more patterns")
    print("  • See docs/transport.md for full documentation")
    print("  • Check docs/reactive.md for event-driven flows")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
