"""Example: Reactive Flow + Webhook-style async completion (no DeferredResult waiting).

This demonstrates the *reactive* approach to async workflows:
- Agents do NOT block waiting for a single DeferredResult
- External events (webhooks/callbacks) are emitted into the flow
- Multiple jobs can be in-flight concurrently

Run:
    uv run ./examples/advanced/reactive_webhook_jobs.py

Notes:
- Requires a real provider-backed model configured in examples/.env.
- Shows how tools can emit events back into the running EventFlow.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from agenticflow import tool
from agenticflow.reactive import EventFlow, EventFlowConfig, ReactiveAgent

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from config import get_model  # type: ignore  # noqa: E402


async def main() -> None:
    print("=" * 70)
    print("⚡ Reactive Flow + Webhook Jobs (multi in-flight)")
    print("=" * 70)

    flow = (
        EventFlow(
            config=EventFlowConfig(
                max_rounds=200,
                max_concurrent_agents=10,
                event_timeout=5.0,
                stop_on_idle=False,
                stop_events=frozenset({"flow.completed", "flow.failed"}),
            )
        )
        .thread_by_data("job_id")
        .with_memory()
    )

    state: dict[str, object] = {
        "expected": 2,
        "submitted": 0,
        "completed": 0,
        "jobs": {},  # job_id -> {video_url, output_url?, status}
    }

    # Tools close over the running flow so they can emit events back into it.
    @tool
    async def submit_video_job(job_id: str, video_url: str) -> str:
        # Real LLMs can occasionally emit duplicate tool calls.
        # Keep this example stable by enforcing the expected count.
        jobs: dict = state["jobs"]  # type: ignore[assignment]
        # Idempotency: if already submitted for this job_id, return it.
        if job_id in jobs and jobs[job_id].get("status") in {"submitted", "completed"}:
            return job_id

        if int(state["submitted"]) >= int(state["expected"]):
            return "ignored"

        jobs[job_id] = {"video_url": video_url, "status": "submitted"}
        state["submitted"] = int(state["submitted"]) + 1

        print(f"📤 Submitted job {job_id} for {video_url}")

        # Emit an internal event so the flow can track in-flight jobs.
        await flow.emit("video.job_submitted", {"job_id": job_id, "video_url": video_url})
        return job_id

    @tool
    async def record_video_completion(job_id: str, output_url: str) -> str:
        jobs: dict = state["jobs"]  # type: ignore[assignment]
        jobs.setdefault(job_id, {})

        # Idempotency: ignore duplicate webhook handling for already-completed jobs.
        if jobs[job_id].get("status") == "completed":
            return "ok"

        jobs[job_id].update({"output_url": output_url, "status": "completed"})
        state["completed"] = int(state["completed"]) + 1

        print(f"✅ Completed job {job_id} -> {output_url}")

        # Stop condition: when all expected jobs complete, emit flow.completed.
        if int(state["completed"]) >= int(state["expected"]):
            await flow.emit(
                "flow.completed",
                {
                    "summary": {
                        "expected": state["expected"],
                        "submitted": state["submitted"],
                        "completed": state["completed"],
                    }
                },
            )

        return "ok"

    # Flow acts as a container: it injects shared memory on register().
    submitter = ReactiveAgent(name="submitter", model=get_model()).on(
        "video.requested",
        tool=submit_video_job,
    )

    webhook_handler = ReactiveAgent(name="webhook_handler", model=get_model()).on(
        "webhook.video_complete",
        tool=record_video_completion,
    )

    submitter.register(flow)
    webhook_handler.register(flow)

    # External system simulator: when a job is submitted, schedule a webhook event.
    def on_any_event(event) -> None:
        if getattr(event, "name", None) != "video.job_submitted":
            return

        job_id = (event.data or {}).get("job_id")
        video_url = (event.data or {}).get("video_url")

        async def deliver_webhook() -> None:
            # Different delays to show multiple in-flight jobs.
            delay = 0.4 if str(video_url).endswith("1.mp4") else 0.8
            await asyncio.sleep(delay)
            await flow.emit(
                "webhook.video_complete",
                {
                    "job_id": job_id,
                    "result": {
                        "status": "complete",
                        "output_url": f"https://cdn.example.com/processed/{job_id}.mp4",
                    },
                },
            )

        flow.spawn(deliver_webhook())

    # Subscribe on the *core* bus (orchestration), not observability.
    flow.events.subscribe_all(on_any_event)

    async def emit_requests() -> None:
        # Kick off two jobs quickly.
        await asyncio.sleep(0.1)
        await flow.emit(
            "video.requested",
            {"job_id": f"video-{uuid4().hex[:8]}", "video_url": "https://example.com/video1.mp4"},
        )
        await flow.emit(
            "video.requested",
            {"job_id": f"video-{uuid4().hex[:8]}", "video_url": "https://example.com/video2.mp4"},
        )

    # Run the flow and concurrently inject events.
    emitter_task = flow.spawn(emit_requests())
    result = await flow.run(
        task="Process 2 videos via reactive webhooks",
        initial_event="demo.start",
        initial_data={"note": "Reactive demo"},
        context={"state": state},
    )
    await emitter_task

    print("\n" + "=" * 70)
    print("✅ Flow stopped")
    print("=" * 70)
    print(f"Final output: {result.output[:200] if result.output else '(none)'}")
    print(f"State: {state}")


if __name__ == "__main__":
    asyncio.run(main())
