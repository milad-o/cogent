#!/usr/bin/env python3
"""
Deferred Tools - Event-Driven Async Completions
================================================

Demonstrates tools that don't complete immediately but return a DeferredResult
that the agent waits for. Perfect for:
- Webhook callbacks from external APIs
- Long-running job processing
- Human-in-the-loop approvals
- External system integrations

Key Concepts:
1. Tool returns DeferredResult instead of direct value
2. Agent automatically waits for completion event via trace_bus
3. External system publishes completion event to agent's trace_bus
4. Agent resumes with the result

Prerequisites:
    - Set API key: export OPENAI_API_KEY=your-key

Run:
    uv run python examples/advanced/deferred_tools.py
"""

import asyncio
from datetime import datetime

from cogent import Agent
from cogent.observability.trace_record import Trace, TraceType
from cogent.tools import DeferredResult, tool

# =============================================================================
# Demo 1: Webhook Callback Pattern
# =============================================================================


async def demo_webhook_callback():
    """Demo 1: Tool that waits for external webhook callback."""
    print("\n" + "=" * 70)
    print("Demo 1: Webhook Callback Pattern")
    print("=" * 70)

    # Create agent first with placeholder tool list
    agent = Agent(
        name="VideoProcessor",
        model="gpt-4o-mini",
        instructions="You process video requests. Always use the submit_video_processing tool.",
        verbosity=True,  # Enable verbose logging to see what's happening
    )

    @tool
    async def submit_video_processing(video_url: str) -> DeferredResult:
        """Submit video for processing, wait for webhook callback."""
        job_id = f"video_{datetime.now().timestamp()}"

        print(f"  🎬 Submitting video for processing: {video_url}")
        print(f"  📋 Job ID: {job_id}")
        print("  ⏳ Waiting for webhook callback...")

        # Simulate async job submission
        await asyncio.sleep(0.1)

        # Schedule webhook callback simulation
        # In reality, external service would call your webhook endpoint
        async def simulate_webhook():
            await asyncio.sleep(2)  # Simulate processing time
            print(f"  📨 Webhook callback received! Completing job {job_id}")

            # Ensure trace_bus is initialized (it's lazy-loaded)
            # Access deferred_manager to trigger trace_bus initialization
            _ = agent.deferred_manager

            # Publish completion event to agent's trace_bus
            # This is what your webhook endpoint would do
            trace = Trace(
                type=TraceType.CUSTOM,
                data={
                    "job_id": job_id,
                    "result": {
                        "status": "success",
                        "output_url": f"https://cdn.example.com/{job_id}.mp4",
                        "duration": "2m 34s",
                        "size_mb": 45.2,
                    },
                },
            )
            await agent.trace_bus.publish(trace)

        asyncio.create_task(simulate_webhook())

        return DeferredResult(
            job_id=job_id,
            wait_for=TraceType.CUSTOM,  # Wait for CUSTOM trace type
            match={"job_id": job_id},  # Match on job_id field
            timeout=10,  # 10 seconds for demo
        )

    # Now add the tool to the agent
    # Access private _direct_tools since tools param is only for init
    agent._direct_tools.append(submit_video_processing)

    print("\n📝 Task: Process a promotional video")
    print("-" * 70)

    result = await agent.run("Process this video: https://example.com/promo.mp4")

    print("\n✅ Agent Response:")
    print(f"   {result.content}")
    print()


# =============================================================================
# Demo 2: Human Approval Pattern
# =============================================================================


async def demo_human_approval():
    """Demo 2: Tool requiring human approval before execution."""
    print("\n" + "=" * 70)
    print("Demo 2: Human Approval Pattern")
    print("=" * 70)

    # Create agent first
    agent = Agent(
        name="MarketingBot",
        model="gpt-4o-mini",
        instructions="You help with marketing campaigns. Always get approval for promotional emails.",
    )

    @tool
    async def send_promotional_email(
        recipient_list: str, subject: str, campaign_name: str
    ) -> DeferredResult:
        """Send promotional email campaign (requires approval)."""
        approval_id = f"approval_{datetime.now().timestamp()}"

        print("\n  📧 Email Campaign Submitted:")
        print(f"     Campaign: {campaign_name}")
        print(f"     Subject: {subject}")
        print(f"     Recipients: {recipient_list}")
        print("  🔒 Waiting for human approval...")
        print(f"     Approval ID: {approval_id}")

        # Simulate notification to human (Slack, email, etc.)
        await asyncio.sleep(0.1)

        # Simulate human approval after review
        async def simulate_approval():
            await asyncio.sleep(3)  # Human review time
            print("  ✅ APPROVED by admin@company.com")

            # Ensure trace_bus is initialized
            _ = agent.deferred_manager

            # Publish approval event to agent's trace_bus
            trace = Trace(
                type=TraceType.CUSTOM,
                data={
                    "approval_id": approval_id,
                    "approved_by": "admin@company.com",
                    "result": "Email campaign sent successfully to 1,250 recipients",
                },
            )
            await agent.trace_bus.publish(trace)

        asyncio.create_task(simulate_approval())

        return DeferredResult(
            job_id=approval_id,
            wait_for=TraceType.CUSTOM,
            match={"approval_id": approval_id},
            timeout=10,  # 10 seconds for demo
            on_timeout="reject",
        )

    # Add tool to agent
    agent._direct_tools.append(send_promotional_email)

    print("\n📝 Task: Send Q1 promotion campaign")
    print("-" * 70)

    result = await agent.run(
        "Send a promotional email campaign about our Q1 sale to our customer list. "
        "Use subject: 'Exclusive Q1 Sale - 30% Off!' and campaign name 'Q1-2026-Sale'"
    )

    print("\n✅ Agent Response:")
    print(f"   {result.content}")
    print()


# =============================================================================
# Demo 3: Long-Running Job Pattern
# =============================================================================


async def demo_long_running_job():
    """Demo 3: Long-running job with progress updates."""
    print("\n" + "=" * 70)
    print("Demo 3: Long-Running Job Pattern")
    print("=" * 70)

    # Create agent first
    agent = Agent(
        name="AnalyticsBot",
        model="gpt-4o-mini",
        instructions="You run data analysis jobs. Use the run_data_analysis tool.",
    )

    @tool
    async def run_data_analysis(dataset_url: str, analysis_type: str) -> DeferredResult:
        """Run long-running data analysis job."""
        job_id = f"analysis_{datetime.now().timestamp()}"

        print("\n  📊 Analysis Job Started:")
        print(f"     Dataset: {dataset_url}")
        print(f"     Type: {analysis_type}")
        print(f"     Job ID: {job_id}")
        print("  🔄 Processing data...")

        # Simulate job progression
        async def simulate_job():
            for progress in [25, 50, 75, 100]:
                await asyncio.sleep(0.8)
                print(f"  ⏳ Progress: {progress}%")

                if progress == 100:
                    print("  ✅ Analysis complete!")

                    # Ensure trace_bus is initialized
                    _ = agent.deferred_manager

                    # Publish completion event to agent's trace_bus
                    trace = Trace(
                        type=TraceType.CUSTOM,
                        data={
                            "job_id": job_id,
                            "result": {
                                "insights": [
                                    "Revenue increased 23% YoY",
                                    "Customer retention improved to 94%",
                                    "Top product: Widget Pro (45% of sales)",
                                ],
                                "confidence": 0.95,
                            },
                        },
                    )
                    await agent.trace_bus.publish(trace)

        asyncio.create_task(simulate_job())

        return DeferredResult(
            job_id=job_id,
            wait_for=TraceType.CUSTOM,
            match={"job_id": job_id},
            timeout=10,  # 10 seconds for demo
        )

    # Add tool to agent
    agent._direct_tools.append(run_data_analysis)

    print("\n📝 Task: Analyze Q4 sales data")
    print("-" * 70)

    result = await agent.run(
        "Run a sales trend analysis on the Q4 dataset at https://data.company.com/q4_sales.csv"
    )

    print("\n✅ Agent Response:")
    print(f"   {result.content}")
    print()


# =============================================================================
# Demo 4: Timeout Handling
# =============================================================================


async def demo_timeout_handling():
    """Demo 4: Handling deferred tool timeouts."""
    print("\n" + "=" * 70)
    print("Demo 4: Timeout Handling")
    print("=" * 70)

    # Create agent first
    agent = Agent(
        name="ApprovalBot",
        model="gpt-4o-mini",
        instructions="You handle approval requests. Use the tool and report results.",
    )

    @tool
    async def request_external_approval(request: str) -> DeferredResult:
        """Request approval that will timeout (for demo)."""
        approval_id = f"timeout_{datetime.now().timestamp()}"

        print("\n  🔔 Approval Request Sent:")
        print(f"     Request: {request}")
        print(f"     Approval ID: {approval_id}")
        print("  ⏱️  Timeout: 2 seconds")
        print("  ⏳ Waiting...")

        # Intentionally don't publish completion event - let it timeout
        # This demonstrates timeout handling

        return DeferredResult(
            job_id=approval_id,
            wait_for=TraceType.CUSTOM,
            match={"approval_id": approval_id},
            timeout=2,  # Very short timeout for demo
            on_timeout="auto_reject",  # What to do on timeout
        )

    # Add tool to agent
    agent._direct_tools.append(request_external_approval)

    print("\n📝 Task: Request approval for budget increase")
    print("-" * 70)

    result = await agent.run(
        "Request approval to increase the marketing budget by $50,000"
    )

    print("\n✅ Agent Response (after timeout):")
    print(f"   {result.content}")
    print()


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all deferred tools demos."""
    print("=" * 70)
    print("        COGENT DEFERRED TOOLS EXAMPLES")
    print("=" * 70)

    await demo_webhook_callback()
    await demo_human_approval()
    await demo_long_running_job()
    await demo_timeout_handling()

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
