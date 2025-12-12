import pytest

from agenticflow.core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from agenticflow.models.base import convert_messages


def test_convert_messages_assigns_missing_tool_call_ids_and_pairs_tool_message() -> None:
    messages = [
        SystemMessage("sys"),
        HumanMessage("hi"),
        AIMessage(tool_calls=[{"id": "", "name": "search", "args": {"q": "x"}}]),
        ToolMessage("result", tool_call_id=""),
    ]

    out = convert_messages(messages)

    # Tool message should remain and have a tool_call_id.
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].get("tool_call_id")

    # The preceding assistant message must contain tool_calls with matching id.
    tool_index = out.index(tool_msgs[0])
    assert tool_index > 0
    prev = out[tool_index - 1]
    assert prev.get("role") == "assistant"
    assert prev.get("tool_calls")
    ids = [tc.get("id") for tc in prev.get("tool_calls")]
    assert tool_msgs[0]["tool_call_id"] in ids


def test_convert_messages_drops_unpaired_tool_message_when_not_adjacent() -> None:
    messages = [
        SystemMessage("sys"),
        HumanMessage("hi"),
        AIMessage(tool_calls=[{"id": "call_1", "name": "search", "args": {"q": "x"}}]),
        HumanMessage("intervening"),
        ToolMessage("result", tool_call_id="call_1"),
    ]

    out = convert_messages(messages)

    # Tool msg is not adjacent to assistant tool_calls; should be dropped to avoid 400.
    assert all(m.get("role") != "tool" for m in out)


def test_convert_messages_handles_multiple_tool_calls() -> None:
    """Multiple tool results after one assistant with tool_calls should all be kept."""
    messages = [
        SystemMessage("sys"),
        HumanMessage("hi"),
        AIMessage(tool_calls=[
            {"id": "call_1", "name": "search", "args": {"q": "x"}},
            {"id": "call_2", "name": "calc", "args": {"expr": "1+1"}},
        ]),
        ToolMessage("result1", tool_call_id="call_1"),
        ToolMessage("result2", tool_call_id="call_2"),
    ]

    out = convert_messages(messages)

    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "call_1"
    assert tool_msgs[1]["tool_call_id"] == "call_2"
