"""Example: Reactive Flow + multi-stage triage with async webhook completions.

This is a more realistic reactive scenario than the basic webhook demo:
- Multiple cases are in-flight concurrently
- An LLM-driven triage agent decides what to do (tool calling)
- Some cases require an async external scan (webhook callback)
- Decision + responder agents handle follow-up events
- Per-case threaded memory is enabled automatically via flow.thread_by_data("case_id")

Run:
    uv run ./examples/advanced/reactive_case_triage_async.py

Notes:
- Requires a real provider-backed model configured in examples/.env.
- Uses the core orchestration bus (flow.events) for internal simulation hooks.
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from pathlib import Path
from uuid import uuid4

from random import random, uniform

from agenticflow import Agent, tool
from agenticflow import Channel, ObservabilityLevel, Observer
from agenticflow.agent.resilience import RecoveryAction, ResilienceConfig, RetryPolicy
from agenticflow.core.messages import HumanMessage
from agenticflow.reactive import EventFlow, EventFlowConfig, IdempotencyGuard, ReactiveAgent, emit_later, react_to

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from config import get_model, settings  # type: ignore  # noqa: E402


def _make_observer() -> Observer:
    """Create an Observer based on examples/.env VERBOSE_LEVEL.

    Note: domain/orchestration events like case.created are mirrored onto
    flow.bus as EventType.CUSTOM, so we include Channel.SYSTEM to show them.
    """

    level = settings.verbose_level
    if level == "trace":
        return Observer.trace()
    if level == "debug":
        return Observer.debug()
    if level == "verbose":
        return Observer(
            level=ObservabilityLevel.DETAILED,
            channels=[
                Channel.REACTIVE,
                Channel.SYSTEM,
                Channel.AGENTS,
                Channel.TOOLS,
                Channel.RESILIENCE,
                Channel.TASKS,
            ],
            show_timestamps=True,
            show_duration=True,
            truncate=500,
        )
    # "minimal" (and any unknown) -> milestone view
    return Observer(
        level=ObservabilityLevel.PROGRESS,
        channels=[Channel.REACTIVE, Channel.SYSTEM],
        show_duration=True,
    )


def _format_exception(e: BaseException) -> str:
    name = type(e).__name__
    text = str(e).strip()
    if text:
        return f"{name}: {text}"
    # Some exceptions (notably TimeoutError/CancelledError) stringify to "".
    return f"{name}: {repr(e)}"


def _configure_model_for_demo(model: object, *, timeout_seconds: float) -> object:
    # Best-effort: not all model backends expose these attributes.
    if hasattr(model, "timeout"):
        try:
            setattr(model, "timeout", float(timeout_seconds))
        except Exception:
            pass
    if hasattr(model, "max_retries"):
        try:
            setattr(model, "max_retries", 0)
        except Exception:
            pass
    return model


async def main() -> None:
    print("=" * 78)
    print("Reactive Case Triage + Async Scans (multi-agent, multi in-flight)")
    print("=" * 78)

    observer = _make_observer()

    # Fast preflight: if the model/provider is unreachable, fail immediately
    # instead of spending ~N * timeout_seconds waiting for each case.
    try:
        model_probe = get_model()
        _configure_model_for_demo(model_probe, timeout_seconds=8.0)
        await asyncio.wait_for(
            model_probe.ainvoke([HumanMessage(content="ping")]),
            timeout=8.0,
        )
    except asyncio.TimeoutError as e:
        print("\nLLM preflight timed out; demo aborting early.")
        print(f"Provider: {settings.llm_provider}")
        print(f"Model: {getattr(settings, 'github_model', None) if settings.llm_provider == 'github' else 'default'}")
        print(f"Error: {_format_exception(e)}")
        return
    except Exception as e:
        print("\nLLM preflight failed; demo aborting early.")
        print(f"Provider: {settings.llm_provider}")
        print(f"Error: {_format_exception(e)}")
        # Helpful when the exception nests the real cause (common with HTTP clients).
        cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
        if cause is not None and cause is not e:
            print(f"Cause: {_format_exception(cause)}")
        if settings.verbose_level in {"debug", "trace"}:
            print("Traceback (debug):")
            print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        return

    flow = (
        EventFlow(
            config=EventFlowConfig(
                max_rounds=400,
                # Real providers often have low concurrency / rate limits.
                # Keeping this small avoids long queued requests and 120s timeouts.
                max_concurrent_agents=2,
                event_timeout=5.0,
                # Demo safety: if we end up with no pending events (e.g. a tool call failed),
                # don't loop forever waiting for more.
                stop_on_idle=True,
                stop_events=frozenset({"flow.completed", "flow.failed"}),
            ),
            observer=observer,
        )
        .thread_by_data("case_id")
        .with_memory()
    )

    # Attach observer to the flow's observability bus so domain events
    # mirrored as EventType.CUSTOM (case.*, scan.*, webhook.*) are visible.
    observer.attach(flow.bus)

    # Process-local idempotency for simulated external callbacks.
    guard = IdempotencyGuard()

    # Shared state for idempotency + stop conditions.
    state: dict[str, object] = {
        "expected": 3,
        "terminal": 0,
        "cases": {},  # case_id -> {status, category, priority, scan_id?, resolution?}
    }

    # If the provider is timing out, end the demo quickly.
    llm_unhealthy = {"value": False}

    # Map event_id -> case_id so failure handlers can identify the affected case.
    event_case_index: dict[str, str] = {}

    def _case(case_id: str) -> dict:
        cases: dict = state["cases"]  # type: ignore[assignment]
        cases.setdefault(case_id, {"status": "new"})
        return cases[case_id]

    def _effective_customer_id(case_id: str, supplied: str | None) -> str:
        case = _case(case_id)
        existing = case.get("customer_id")
        if isinstance(existing, str) and existing:
            if supplied and supplied != existing:
                case["customer_id_mismatch"] = {"expected": existing, "got": supplied}
            return existing
        if supplied:
            case["customer_id"] = supplied
            return supplied
        return ""

    def _mark_terminal(case_id: str, status: str) -> None:
        case = _case(case_id)
        if case.get("terminal"):
            return
        case["terminal"] = True
        case["status"] = status
        state["terminal"] = int(state["terminal"]) + 1

    async def _maybe_complete_flow() -> None:
        if int(state["terminal"]) >= int(state["expected"]):
            await flow.emit(
                "flow.completed",
                {
                    "summary": {
                        "expected": state["expected"],
                        "terminal": state["terminal"],
                    }
                },
            )

    # --- LLM-driven routing tools (agent decides which tool to call) ---

    @tool
    async def classify_case(
        case_id: str,
        customer_id: str,
        category: str,
        priority: str,
        summary: str,
    ) -> str:
        """Persist triage classification and emit case.classified."""
        case = _case(case_id)
        customer_id = _effective_customer_id(case_id, customer_id)
        if case.get("status") not in {"new", "created", "awaiting_customer"}:
            return "ok"

        case.update(
            {
                "status": "classified",
                "customer_id": customer_id,
                "category": category,
                "priority": priority,
                "summary": summary,
            }
        )

        await flow.emit(
            "case.classified",
            {
                "case_id": case_id,
                "customer_id": customer_id,
                "category": category,
                "priority": priority,
                "summary": summary,
            },
        )
        return "ok"

    @tool
    async def request_scan(
        case_id: str,
        customer_id: str,
        scan_kind: str = "risk_scan",
        reason: str | None = None,
    ) -> str:
        """Request an async scan (LLM chooses if/when) by emitting scan.requested."""
        case = _case(case_id)
        customer_id = _effective_customer_id(case_id, customer_id)
        if case.get("terminal"):
            return "ok"
        # Guardrail: don't request scans repeatedly.
        if case.get("scan_id") or case.get("status") in {"scan_submitted", "resolution_ready", "resolved", "escalated"}:
            return "ok"

        scan_requests = int(case.get("scan_requests", 0))
        if scan_requests >= 1:
            return "blocked: scan already requested"
        case["scan_requests"] = scan_requests + 1

        case["requires_scan"] = True
        case["scan_kind"] = scan_kind
        if reason:
            case["scan_reason"] = reason

        # Initialize attempt counter if missing
        case.setdefault("scan_attempt", 0)

        await flow.emit(
            "scan.requested",
            {
                "case_id": case_id,
                "customer_id": customer_id,
                "scan_kind": scan_kind,
                "reason": reason,
            },
        )
        return "ok"

    @tool
    async def ask_customer_question(
        case_id: str,
        customer_id: str,
        question: str,
        why: str | None = None,
    ) -> str:
        """Ask the customer a question (demo will simulate a reply)."""
        case = _case(case_id)
        customer_id = _effective_customer_id(case_id, customer_id)
        if case.get("terminal"):
            return "ok"

        # Guardrail: don't get stuck in back-and-forth loops.
        asked = int(case.get("questions_asked", 0))
        if asked >= 1:
            return "blocked: question already asked"
        case["questions_asked"] = asked + 1

        case.update({"status": "awaiting_customer", "question": question, "question_why": why})
        await flow.emit(
            "customer.question_requested",
            {"case_id": case_id, "customer_id": customer_id, "question": question, "why": why},
        )
        return "ok"

    @tool
    async def escalate_to_human(
        case_id: str,
        customer_id: str,
        reason: str,
        queue: str = "manual_review",
    ) -> str:
        """Escalate case to a human queue and mark terminal (demo ends for this case)."""
        case = _case(case_id)
        customer_id = _effective_customer_id(case_id, customer_id)
        case.update({"customer_id": customer_id, "escalation_reason": reason, "queue": queue})
        await flow.emit(
            "case.escalated",
            {"case_id": case_id, "customer_id": customer_id, "reason": reason, "queue": queue},
        )
        _mark_terminal(case_id, "escalated")
        await _maybe_complete_flow()
        return "ok"

    @tool
    async def emit_auto_resolution(
        case_id: str,
        customer_id: str,
        decision: str,
        customer_message: str,
        internal_notes: str | None = None,
    ) -> str:
        """Emit case.resolution_ready without requiring a scan."""
        case = _case(case_id)
        customer_id = _effective_customer_id(case_id, customer_id)
        # Guardrail: don't auto-resolve if a scan is required/started.
        if case.get("requires_scan") or case.get("scan_id"):
            return "blocked: scan already requested"
        if case.get("terminal"):
            return "ok"
        await flow.emit(
            "case.resolution_ready",
            {
                "case_id": case_id,
                "customer_id": customer_id,
                "decision": decision,
                "customer_message": customer_message,
                "internal_notes": internal_notes,
            },
        )
        return "ok"

    @tool
    async def submit_scan(case_id: str, customer_id: str, scan_kind: str) -> str:
        """Submit an async scan and emit scan.submitted.

        This also triggers the external system simulator via orchestration events.
        """
        case = _case(case_id)
        customer_id = _effective_customer_id(case_id, customer_id)
        if case.get("terminal"):
            return str(case.get("scan_id") or "")
        # Allow limited retries: only submit once per attempt.
        attempt = int(case.get("scan_attempt", 0))
        existing_scan_id = case.get("scan_id")
        if existing_scan_id and attempt == int(case.get("scan_attempt_for_id", attempt)):
            return str(existing_scan_id)

        scan_id = f"scan-{uuid4().hex[:10]}"
        case.update(
            {
                "status": "scan_submitted",
                "scan_id": scan_id,
                "scan_kind": scan_kind,
                "scan_attempt": attempt,
                "scan_attempt_for_id": attempt,
            }
        )

        await flow.emit(
            "scan.submitted",
            {
                "case_id": case_id,
                "customer_id": customer_id,
                "scan_kind": scan_kind,
                "scan_id": scan_id,
                "attempt": attempt,
            },
        )

        return scan_id

    @tool
    async def request_rescan(
        case_id: str,
        customer_id: str,
        scan_kind: str = "risk_scan",
        reason: str | None = None,
    ) -> str:
        """Request a re-scan (limited) by emitting scan.requested with incremented attempt."""
        case = _case(case_id)
        customer_id = _effective_customer_id(case_id, customer_id)
        if case.get("terminal"):
            return "ok"
        # Hard cap: allow at most one rescan to keep the demo terminating.
        if int(case.get("scan_attempt", 0)) >= 1:
            return "blocked: rescan limit reached"
        attempt = int(case.get("scan_attempt", 0)) + 1
        case["scan_attempt"] = attempt

        await flow.emit(
            "scan.requested",
            {
                "case_id": case_id,
                "customer_id": customer_id,
                "scan_kind": scan_kind,
                "reason": reason or "rescan_requested",
                "attempt": attempt,
            },
        )
        return "ok"

    @tool
    async def emit_resolution(
        case_id: str,
        customer_id: str,
        decision: str,
        customer_message: str,
        internal_notes: str | None = None,
    ) -> str:
        """Emit resolution-ready event.

        The LLM supplies the decision + message based on scan results.
        """
        case = _case(case_id)
        if case.get("status") in {"resolved"}:
            return "ok"

        case.update(
            {
                "status": "resolution_ready",
                "decision": decision,
                "customer_message": customer_message,
                "internal_notes": internal_notes,
            }
        )

        await flow.emit(
            "case.resolution_ready",
            {
                "case_id": case_id,
                "customer_id": customer_id,
                "decision": decision,
                "customer_message": customer_message,
                "internal_notes": internal_notes,
            },
        )
        return "ok"

    @tool
    async def note_scan_progress(
        case_id: str,
        customer_id: str,
        scan_id: str,
        progress: float,
        attempt: int | None = None,
    ) -> str:
        """Record scan progress updates (non-terminal)."""
        case = _case(case_id)
        customer_id = _effective_customer_id(case_id, customer_id)
        # Normalize: some models send percentages (e.g. 40 for 40%).
        p = float(progress)
        if p > 1.0:
            p = p / 100.0
        p = max(0.0, min(1.0, p))
        case.update(
            {
                "customer_id": customer_id,
                "scan_id": scan_id,
                "scan_progress": round(p, 2),
                "scan_attempt": int(attempt) if attempt is not None else int(case.get("scan_attempt", 0)),
            }
        )
        return "ok"

    @tool
    async def handle_scan_failure(
        case_id: str,
        customer_id: str,
        scan_id: str,
        error: str,
        attempt: int = 0,
    ) -> str:
        """Record scan failure; let the decision agent choose rescan or escalation."""
        case = _case(case_id)
        customer_id = _effective_customer_id(case_id, customer_id)
        case.update(
            {
                "customer_id": customer_id,
                "scan_id": scan_id,
                "scan_error": error,
                "scan_attempt": int(attempt),
            }
        )
        return "ok"

    @tool
    async def record_resolution(
        case_id: str,
        customer_id: str,
        decision: str,
        customer_message: str,
        internal_notes: str | None = None,
        scan_id: str | None = None,
    ) -> str:
        """Record final resolution and stop the flow when all are done."""
        case = _case(case_id)
        if case.get("status") == "resolved":
            return "ok"

        case.update(
            {
                "status": "resolved",
                "decision": decision,
                "customer_message": customer_message,
                "internal_notes": internal_notes,
                "scan_id": scan_id,
            }
        )

        _mark_terminal(case_id, "resolved")
        await _maybe_complete_flow()

        return "ok"

    # Agents (real model): allow the LLM to choose among multiple tools.
    # This makes the workflow branching genuinely agent-driven.

    triage_router = Agent(
        name="triage",
        model=_configure_model_for_demo(get_model(), timeout_seconds=15.0),
        instructions=(
            "You triage customer cases. IMPORTANT: only act on the triggering event's case_id. "
            "You MUST choose the next step by calling tools. "
            "First call classify_case once. Then call exactly ONE of: request_scan, emit_auto_resolution, "
            "ask_customer_question, or escalate_to_human (do not call multiple next-step tools). "
            "Category must be short (billing, security, access, outage). "
            "Priority must be one of: low, medium, high, critical. "
            "Escalate if you cannot proceed safely."
        ),
        tools=[
            classify_case,
            request_scan,
            emit_auto_resolution,
            ask_customer_question,
            escalate_to_human,
        ],
    )

    scan_agent = ReactiveAgent(
        name="scanner",
        model=_configure_model_for_demo(get_model(), timeout_seconds=15.0),
        instructions=(
            "You handle scan requests. Always submit a scan via the tool. "
            "Use scan_kind from the event, keep it stable."
        ),
    ).on("scan.requested", tool=submit_scan)

    decision_router = Agent(
        name="decision",
        model=_configure_model_for_demo(get_model(), timeout_seconds=15.0),
        instructions=(
            "You decide case outcome based on scan results. IMPORTANT: only act on the triggering event's case_id. "
            "Choose next step by calling tools. "
            "On webhook.scan_progress: call note_scan_progress and do nothing else. "
            "On webhook.scan_failed: call handle_scan_failure, then request_rescan once if attempt < 1; otherwise escalate_to_human. "
            "On webhook.scan_complete: if risk_score >= 0.8 or flags include 'fraud'/'compromised', escalate_to_human. "
            "Otherwise emit_resolution. Keep customer_message short and clear."
        ),
        tools=[emit_resolution, request_rescan, escalate_to_human, handle_scan_failure, note_scan_progress],
    )

    responder_agent = ReactiveAgent(
        name="responder",
        model=_configure_model_for_demo(get_model(), timeout_seconds=15.0),
        instructions=(
            "You finalize cases. Call the tool to record resolution. "
            "Do not add extra steps."
        ),
    ).on("case.resolution_ready", tool=record_resolution)

    # Bounded LLM resilience for real providers: avoid multi-minute hangs,
    # but allow a little time for occasional slow responses.
    demo_resilience = ResilienceConfig(
        # Keep retries off for the demo to avoid repeated 45s stalls.
        retry_policy=RetryPolicy(max_retries=0, base_delay=0.2, max_delay=1.0),
        # Short timeout: if a provider is slow/unavailable, fail fast and let
        # the deterministic failure handler mark the case as terminal.
        timeout_seconds=15.0,
        circuit_breaker_enabled=False,
        fallback_enabled=False,
        learning_enabled=False,
        on_failure=RecoveryAction.SKIP,
    )
    # NativeExecutor will honor this for LLM retries/timeouts.
    for agent in (triage_router, scan_agent, decision_router, responder_agent):
        agent.config.resilience_config = demo_resilience

    # Register triggers
    flow.register(triage_router, [react_to("case.created"), react_to("customer.responded")])
    scan_agent.register(flow)
    flow.register(
        decision_router,
        [
            react_to("webhook.scan_complete"),
            react_to("webhook.scan_failed"),
            react_to("webhook.scan_progress"),
        ],
    )
    responder_agent.register(flow)

    # Index events to their case_id (best-effort), and deterministically
    # terminate cases on agent failures to prevent demo hangs.
    def on_index(event) -> None:
        data = getattr(event, "data", None) or {}
        case_id = data.get("case_id")
        event_id = getattr(event, "id", None)
        if event_id and case_id:
            event_case_index[str(event_id)] = str(case_id)

    async def _handle_agent_error(event) -> None:
        data = getattr(event, "data", None) or {}
        agent = str(data.get("agent") or "")
        error = str(data.get("error") or "")
        trigger_event_id = str(data.get("trigger_event") or "")

        case_id = event_case_index.get(trigger_event_id) or str(data.get("case_id") or "")
        if not case_id:
            return

        # If the LLM provider is timing out, avoid repeated stalls by marking
        # all remaining cases terminal and completing the flow.
        if "timed out" in error.lower():
            llm_unhealthy["value"] = True
            cases: dict = state["cases"]  # type: ignore[assignment]
            for other_case_id in list(cases.keys()):
                if not cases[other_case_id].get("terminal"):
                    _mark_terminal(other_case_id, "failed")
            await _maybe_complete_flow()
            return

        # Mark terminal and emit a domain event that shows up in the observer.
        _mark_terminal(case_id, "failed")
        await flow.emit(
            "case.escalated",
            {
                "case_id": case_id,
                "customer_id": _case(case_id).get("customer_id", ""),
                "reason": f"agent_failed:{agent}",
                "error": error,
                "queue": "manual_review",
            },
        )
        await _maybe_complete_flow()

    def on_failures(event) -> None:
        name = getattr(event, "name", None)
        if not isinstance(name, str) or not name.endswith(".error"):
            return
        flow.spawn(_handle_agent_error(event))

    # External system simulator: when a scan is submitted, later deliver a webhook.
    def on_any_event(event) -> None:
        if getattr(event, "name", None) != "scan.submitted":
            return

        data = getattr(event, "data", None) or {}
        case_id = data.get("case_id")
        customer_id = data.get("customer_id")
        scan_id = data.get("scan_id")
        scan_kind = data.get("scan_kind")

        attempt = data.get("attempt", 0)

        # De-dupe webhook delivery per (scan_id, attempt).
        delivery_key = f"webhook:{scan_id}:{attempt}"

        async def deliver_webhook() -> None:
            if not await guard.claim(delivery_key):
                return

            # Variable delays: show multiple in-flight cases.
            base = 0.20 if str(scan_kind).endswith("risk_scan") else 0.45
            await asyncio.sleep(base + uniform(0.05, 0.55))

            # Occasionally emit a progress event before completion.
            if random() < 0.25:
                await flow.emit(
                    "webhook.scan_progress",
                    {
                        "case_id": case_id,
                        "customer_id": customer_id,
                        "scan_id": scan_id,
                        "scan_kind": scan_kind,
                        "attempt": attempt,
                        "progress": round(uniform(0.35, 0.85), 2),
                    },
                )
                await asyncio.sleep(uniform(0.05, 0.35))

            # Occasionally fail to simulate flaky external systems.
            if random() < 0.18:
                await flow.emit(
                    "webhook.scan_failed",
                    {
                        "case_id": case_id,
                        "customer_id": customer_id,
                        "scan_id": scan_id,
                        "scan_kind": scan_kind,
                        "attempt": attempt,
                        "error": "scan_provider_timeout",
                    },
                )
                return

            # Deterministic-ish mock scan output based on case_id.
            bucket = (sum(ord(c) for c in str(case_id)) % 10) / 10.0
            risk_score = round(0.2 + bucket * 0.9, 2)
            flags = ["compromised"] if risk_score >= 0.85 else ([] if risk_score < 0.6 else ["suspicious"])

            await flow.emit(
                "webhook.scan_complete",
                {
                    "case_id": case_id,
                    "customer_id": customer_id,
                    "scan_id": scan_id,
                    "scan_kind": scan_kind,
                    "attempt": attempt,
                    "result": {"risk_score": risk_score, "flags": flags},
                },
            )

        flow.spawn(deliver_webhook())

    flow.events.subscribe_all(on_index)
    flow.events.subscribe_all(on_failures)
    flow.events.subscribe_all(on_any_event)

    # Simulated customer replies: when a question is requested, respond later.
    def on_question(event) -> None:
        if getattr(event, "name", None) != "customer.question_requested":
            return

        data = getattr(event, "data", None) or {}
        case_id = data.get("case_id")
        customer_id = data.get("customer_id")
        question = data.get("question")

        # De-dupe customer reply per question.
        reply_key = f"customer_reply:{case_id}:{hash(str(question))}"

        async def deliver_reply() -> None:
            if not await guard.claim(reply_key):
                return
            await emit_later(
                flow=flow,
                delay_seconds=0.35,
                event_name="customer.responded",
                data={
                    "case_id": case_id,
                    "customer_id": customer_id,
                    "question": question,
                    "answer": "Here are the details you asked for. Please proceed.",
                },
            )

        flow.spawn(deliver_reply())

    flow.events.subscribe_all(on_question)

    # Seed initial cases before starting the flow loop.
    # With stop_on_idle=True, we must ensure there is work queued immediately.
    cases = [
        {
            "customer_id": "cust-001",
            "text": "I see suspicious charges on my card after logging in from a new device.",
        },
        {
            "customer_id": "cust-002",
            "text": "Can you help me update the billing address on my invoice?",
        },
        {
            "customer_id": "cust-003",
            "text": "My account was locked after too many login attempts. I need access urgently.",
        },
    ]

    for i, payload in enumerate(cases, start=1):
        case_id = f"case-{i}-{uuid4().hex[:6]}"
        _case(case_id).update({"status": "created"})

        await flow.emit(
            "case.created",
            {
                "case_id": case_id,
                "customer_id": payload["customer_id"],
                "text": payload["text"],
            },
        )

    try:
        result = await asyncio.wait_for(
            flow.run(
                task="Triage and resolve multiple cases with async scans when needed",
                initial_event="demo.start",
                initial_data={"note": "reactive case triage demo"},
                # Avoid passing the full shared state into the LLM prompt context.
                # Tools still close over `state`, but the model sees only the triggering event.
                context=None,
            ),
            timeout=180.0,
        )
    except asyncio.TimeoutError:
        await flow.emit("flow.failed", {"reason": "demo_timeout"})
        raise

    print("\n" + "=" * 78)
    print("Flow stopped")
    print("=" * 78)
    print(f"Final output: {result.output[:200] if result.output else '(none)'}")
    print("Cases:")
    print(json.dumps(state["cases"], indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
