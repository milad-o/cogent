"""Tests for ReactiveAgent convenience API."""

from __future__ import annotations

import pytest

from agenticflow.events.event import Event
from agenticflow.models.base import BaseChatModel
from agenticflow.models.mock import MockChatModel
from agenticflow.memory import Memory
from agenticflow.reactive import EventFlow, ReactiveAgent
from agenticflow.reactive.core import react_to
from agenticflow.tools import tool


@tool
async def do_thing(x: int) -> str:
    return str(x)


@pytest.mark.asyncio
async def test_reactive_agent_builds_triggers_and_tools() -> None:
    model = MockChatModel(responses=["ok"])
    agent = ReactiveAgent(name="ra", model=model).on("evt.test", tool=do_thing)

    assert agent.config.tools == [do_thing.name]
    assert len(agent.triggers) == 1
    assert agent.triggers[0].build().on == "evt.test"


@pytest.mark.asyncio
async def test_reactive_agent_registers_into_flow() -> None:
    model = MockChatModel(responses=["ok"])
    agent = ReactiveAgent(name="ra", model=model).on("evt.test", tool=do_thing)

    flow = EventFlow()
    agent.register(flow)

    assert "ra" in flow.agents


@pytest.mark.asyncio
async def test_reactive_agent_react_calls_run() -> None:
    class RecordingChatModel(BaseChatModel):
        model_name: str = "recording"

        def __init__(self) -> None:
            self.seen: list[list[dict[str, object]]] = []

        def _init_client(self) -> None:
            pass

        def invoke(self, messages: list[dict[str, object]], **kwargs: object):  # type: ignore[override]
            self.seen.append(messages)
            from agenticflow.core.messages import AIMessage

            return AIMessage(content="hello")

        async def ainvoke(self, messages: list[dict[str, object]], **kwargs: object):  # type: ignore[override]
            return self.invoke(messages, **kwargs)

        def bind_tools(self, tools: list[object], **kwargs: object):  # type: ignore[override]
            return self

    model = RecordingChatModel()
    agent = ReactiveAgent(name="ra", model=model).on("evt.test", tool=do_thing)

    out = await agent.react(Event(name="evt.test", data={"x": 1}), task="t", context={"k": "v"})
    out_text = out.output if hasattr(out, "output") else str(out)
    assert "hello" in out_text

    assert model.seen, "Expected the model to be invoked at least once"
    last_call = model.seen[-1]
    # Should include a user message with a JSON envelope
    user_msgs = [m for m in last_call if getattr(m, "role", None) == "user"]
    assert user_msgs, "Expected a user message"
    content = getattr(user_msgs[-1], "content", None)
    assert isinstance(content, str)
    assert content.strip().startswith("{")
    assert '"event"' in content
    assert '"name"' in content
    assert '"evt.test"' in content


@pytest.mark.asyncio
async def test_reactive_agent_react_thread_memory_roundtrip() -> None:
    class RecordingChatModel(BaseChatModel):
        model_name: str = "recording"

        def __init__(self, responses: list[str]) -> None:
            self._responses = list(responses)
            self.seen: list[list[object]] = []

        def _init_client(self) -> None:
            pass

        def invoke(self, messages: list[object], **kwargs: object):  # type: ignore[override]
            self.seen.append(messages)
            from agenticflow.core.messages import AIMessage

            content = self._responses.pop(0) if self._responses else "ok"
            return AIMessage(content=content)

        async def ainvoke(self, messages: list[object], **kwargs: object):  # type: ignore[override]
            return self.invoke(messages, **kwargs)

        def bind_tools(self, tools: list[object], **kwargs: object):  # type: ignore[override]
            return self

    model = RecordingChatModel(["first", "second"])
    mem = Memory()
    agent = ReactiveAgent(name="ra", model=model, memory=mem).on("evt.test", tool=do_thing)

    # First reactive call saves to thread
    await agent.react(Event(name="evt.test", data={"x": 1}), task="t", context={"k": "v"}, thread_id="thr")

    assert agent.memory_manager is not None
    stored = await agent.memory_manager.get_thread_messages("thr")
    assert len(stored) == 2
    assert getattr(stored[0], "role", None) == "user"
    assert getattr(stored[1], "role", None) == "assistant"

    # Second call should include history messages
    await agent.react(Event(name="evt.test", data={"x": 2}), task="t", context={"k": "v"}, thread_id="thr")
    assert len(model.seen) >= 2
    second_call = model.seen[-1]
    # Expected: system + (history user/assistant) + new user envelope
    assert len(second_call) >= 4


@pytest.mark.asyncio
async def test_event_flow_passes_thread_id_to_agent_react() -> None:
    class CaptureAgent:
        def __init__(self) -> None:
            self.name = "cap"
            self.thread_ids: list[str | None] = []

        async def react(self, event, *, task: str, context: dict, thread_id: str | None = None):
            self.thread_ids.append(thread_id)
            return "ok"

    flow = EventFlow().thread_by_data("job_id")
    agent = CaptureAgent()
    flow.register(agent, [react_to("demo.start")])

    result = await flow.run(
        task="t",
        initial_event="demo.start",
        initial_data={"job_id": "job-123"},
        context={},
    )

    assert result.output == "ok"
    assert agent.thread_ids == ["job-123"]
