"""
Example 21: Deferred/Event-Driven Tools

Demonstrates how to create tools that don't complete immediately,
allowing agents to wait for external events (webhooks, callbacks,
long-running jobs) before continuing.

Use cases:
- Webhook callbacks from external APIs
- Long-running job processing
- Human-in-the-loop approvals
- External system integrations
- Polling-based job completion
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from agenticflow import Agent
from agenticflow.observability.bus import EventBus
from agenticflow.observability.event import EventType
from agenticflow.tools import tool, DeferredResult

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from config import get_model  # type: ignore  # noqa: E402


# ==============================================================================
# Example 1: Webhook Callback Pattern
# ==============================================================================

@tool
async def process_video(video_url: str) -> DeferredResult:
    """
    Submit video for processing and wait for webhook callback.
    
    The video processing service will call our webhook endpoint
    when processing is complete. This tool returns immediately
    but the agent will wait for the completion event.
    """
    job_id = f"video-{uuid4().hex[:8]}"
    
    # In real code: submit to video API
    # await video_api.submit(video_url, callback_url="/webhooks/video")
    print(f"📹 Submitted video job: {job_id}")
    
    return DeferredResult(
        job_id=job_id,
        wait_for="webhook.video_complete",
        match={"job_id": job_id},
        timeout=300,  # 5 minutes
        metadata={"video_url": video_url},
    )


@tool
async def transcribe_audio(audio_url: str) -> DeferredResult:
    """
    Submit audio for transcription with webhook completion.
    """
    job_id = f"transcribe-{uuid4().hex[:8]}"
    
    print(f"🎤 Submitted transcription job: {job_id}")
    
    return DeferredResult(
        job_id=job_id,
        wait_for="webhook.transcription_complete",
        match={"job_id": job_id},
        timeout=600,  # 10 minutes
        on_timeout={"transcript": "[Transcription timed out]"},  # Default value on timeout
    )


# ==============================================================================
# Example 2: Human Approval Pattern
# ==============================================================================

@tool
async def request_payment_approval(
    amount: float,
    recipient: str,
    reason: str,
) -> DeferredResult:
    """
    Request human approval for a payment.
    
    Sends notification to approver and waits for their response.
    The agent will pause until approval or rejection is received.
    """
    approval_id = f"approval-{uuid4().hex[:8]}"
    
    # In real code: send Slack/email notification
    print(f"💳 Requesting approval for ${amount:.2f} to {recipient}")
    print(f"   Reason: {reason}")
    print(f"   Approval ID: {approval_id}")
    
    return DeferredResult(
        job_id=approval_id,
        wait_for="human.approval_response",
        match={"approval_id": approval_id},
        timeout=3600,  # 1 hour to respond
        on_timeout={"approved": False, "reason": "No response received"},
        metadata={
            "amount": amount,
            "recipient": recipient,
            "reason": reason,
            "requested_at": datetime.now().isoformat(),
        },
    )


# ==============================================================================
# Example 3: Polling Pattern (for APIs without webhooks)
# ==============================================================================

@tool
async def run_ml_training(
    dataset_id: str,
    model_type: str,
) -> DeferredResult:
    """
    Start ML model training with polling for completion.
    
    For APIs that don't support webhooks, we can poll a status
    endpoint until the job completes.
    """
    job_id = f"training-{uuid4().hex[:8]}"
    
    print(f"🤖 Started ML training job: {job_id}")
    print(f"   Dataset: {dataset_id}")
    print(f"   Model: {model_type}")
    
    return DeferredResult(
        job_id=job_id,
        poll_url=f"https://ml-api.example.com/jobs/{job_id}/status",
        poll_interval=10,  # Check every 10 seconds
        complete_when=lambda status: status.get("state") in ("complete", "failed"),
        timeout=1800,  # 30 minutes max
        metadata={"dataset_id": dataset_id, "model_type": model_type},
    )


# ==============================================================================
# Simulated Webhook Server
# ==============================================================================

async def simulate_webhook_server(event_bus: EventBus):
    """
    Simulate external services sending webhook callbacks.
    
    In production, this would be an HTTP endpoint receiving
    webhooks from external services.
    """
    await asyncio.sleep(2)  # Simulate processing delay
    
    # Simulate video processing complete
    print("\n🔔 [Webhook Received] Video processing complete!")
    # Publish to event bus (agent will resume)
    # Note: we include event_name via the string publish API.
    await event_bus.publish(
        "webhook.video_complete",
        {
            "job_id": "video-test",
            "result": {
                "status": "complete",
                "output_url": "https://cdn.example.com/processed-video.mp4",
                "duration": 120,
                "resolution": "1080p",
            },
        },
    )


async def simulate_human_approval(event_bus: EventBus, approval_id: str):
    """Simulate a human approving a payment request."""
    await asyncio.sleep(1)
    
    print(f"\n✅ [Human Response] Payment approved: {approval_id}")
    await event_bus.publish(
        "human.approval_response",
        {
            "approval_id": approval_id,
            "result": {
                "approved": True,
                "approver": "John Manager",
                "approved_at": datetime.now().isoformat(),
                "notes": "Looks good, proceed with payment.",
            },
        },
    )


async def demo_end_to_end_agent_deferred() -> None:
    """End-to-end: agent.act() executes a deferred tool and resumes on event."""
    print("\n" + "=" * 40)
    print("Example 5: End-to-End Agent.act() + DeferredResult")
    print("=" * 40)

    event_bus = EventBus()
    model = get_model()
    agent = Agent(
        name="DeferredE2EAgent",
        system_prompt="You can run tools. Some tools complete later.",
        model=model,
        event_bus=event_bus,
        tools=[process_video, request_payment_approval],
    )

    completed_once = {"video": False}

    def auto_complete_when_waiting(event) -> None:
        # Important: don't publish completion *inside* TOOL_DEFERRED handler.
        # The agent only starts subscribing for the completion event after it
        # emits TOOL_DEFERRED_WAITING; otherwise we'd race and miss the event.
        if event.type != EventType.TOOL_DEFERRED_WAITING:
            return

        tool_name = event.data.get("tool_name")
        job_id = event.data.get("job_id")

        if tool_name != "process_video" or not job_id or completed_once["video"]:
            return

        completed_once["video"] = True

        async def publish_completion() -> None:
            # Yield control so the agent enters DeferredWaiter.wait() and subscribes.
            await asyncio.sleep(0)
            await asyncio.sleep(0.25)
            await event_bus.publish(
                "webhook.video_complete",
                {
                    "job_id": job_id,
                    "result": {
                        "status": "complete",
                        "output_url": "https://cdn.example.com/processed-video.mp4",
                    },
                },
            )

        asyncio.create_task(publish_completion())

    event_bus.subscribe(EventType.TOOL_DEFERRED_WAITING, auto_complete_when_waiting)

    print("Calling tool via agent.act('process_video', ...) — it should wait then resume...")
    result = await agent.act(
        "process_video",
        {"video_url": "https://example.com/video.mp4"},
        use_resilience=False,
    )
    print(f"\n✅ agent.act returned: {result}")

    print("\n" + "=" * 40)
    print("Example 6: LLM picks the tool")
    print("=" * 40)
    print("Asking the agent to process a video; it should decide to call process_video.")
    output = await agent.run(
        "Please process this video URL: https://example.com/video.mp4. "
        "Use tools if needed, and wait for completion.",
    )
    output_text = output.output if hasattr(output, "output") else str(output)
    print(f"\n✅ agent.run output: {output_text}")


# ==============================================================================
# Demo Runner
# ==============================================================================

async def demo_deferred_tools():
    """Demonstrate deferred/event-driven tools."""
    print("=" * 60)
    print("🔄 Deferred/Event-Driven Tools Demo")
    print("=" * 60)
    
    # Create event bus
    event_bus = EventBus()

    model = get_model()
    
    # Create agent with deferred tool capabilities
    agent = Agent(
        name="DeferredTaskAgent",
        system_prompt="""You are an assistant that handles asynchronous tasks.
When using deferred tools, wait for them to complete before proceeding.""",
        model=model,
        event_bus=event_bus,
        tools=[process_video, transcribe_audio, request_payment_approval],
    )
    
    print("\n📋 Available deferred tools:")
    for tool_obj in agent.all_tools:
        print(f"   - {tool_obj.name}")
    
    # Example 1: Show how DeferredResult works
    print("\n" + "=" * 40)
    print("Example 1: Creating a Deferred Result")
    print("=" * 40)
    
    # Call the tool's underlying function
    result = await process_video.func("https://example.com/video.mp4")
    print(f"\n📦 Tool returned: {type(result).__name__}")
    print(f"   Job ID: {result.job_id}")
    print(f"   Wait for: {result.wait_for}")
    print(f"   Match: {result.match}")
    print(f"   Timeout: {result.timeout}s")
    print(f"   Status: {result.status.value}")
    
    # Example 2: Show DeferredManager
    print("\n" + "=" * 40)
    print("Example 2: DeferredManager Tracking")
    print("=" * 40)
    
    manager = agent.deferred_manager
    manager.register(result)
    
    print(f"\n📊 Manager status:")
    print(f"   Pending: {manager.pending_count}")
    print(f"   Jobs: {manager.pending_jobs}")
    
    summary = manager.get_summary()
    print(f"   Summary: {summary}")
    
    # Cancel for demo purposes
    manager.cancel(result.job_id)
    print(f"\n   After cancel: {manager.pending_count} pending")
    
    # Example 3: Show approval pattern
    print("\n" + "=" * 40)
    print("Example 3: Human Approval Pattern")
    print("=" * 40)
    
    approval_result = await request_payment_approval.func(
        amount=1500.00,
        recipient="Vendor Inc.",
        reason="Monthly software subscription",
    )
    
    print(f"\n📦 Approval deferred result:")
    print(f"   Job ID: {approval_result.job_id}")
    print(f"   Timeout: {approval_result.timeout}s (1 hour)")
    print(f"   On timeout: {approval_result.on_timeout}")
    
    # Example 4: Timeout with default value
    print("\n" + "=" * 40)
    print("Example 4: Timeout with Default Value")
    print("=" * 40)
    
    quick_result = DeferredResult(
        job_id="quick-test",
        wait_for="test.event",
        timeout=0.5,  # Very short timeout
        on_timeout={"status": "timed_out", "fallback": True},
    )
    
    print(f"Waiting for event with 0.5s timeout...")
    
    from agenticflow.tools.deferred import DeferredWaiter
    from unittest.mock import MagicMock
    
    mock_bus = MagicMock()
    mock_bus.subscribe = MagicMock()
    mock_bus.unsubscribe = MagicMock()
    
    waiter = DeferredWaiter(deferred=quick_result, event_bus=mock_bus)
    timeout_result = await waiter.wait()
    
    print(f"Result after timeout: {timeout_result}")
    print(f"Status: {quick_result.status.value}")

    # Example 5: Full end-to-end agent execution that waits and resumes
    await demo_end_to_end_agent_deferred()
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_deferred_tools())
