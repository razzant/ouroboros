from __future__ import annotations

import pytest

from ouroboros.copilot_acp_events import ACPEventTranslator
from ouroboros.gateways.copilot_acp import ACPError


def update(update_type, **fields):
    return {
        "jsonrpc": "2.0", "method": "session/update",
        "params": {"sessionId": "s", "update": {"sessionUpdate": update_type, **fields}},
    }


def text(value, **fields):
    return update("agent_message_chunk", content={"type": "text", "text": value}, **fields)


def test_final_answer_excludes_planning_and_child_messages():
    translator = ACPEventTranslator("s")
    translator.translate(text("I will investigate."))
    translator.translate(update("tool_call", toolCallId="read", kind="read", title="Read", rawInput={"path": "main.py"}, status="in_progress"))
    translator.translate(text("child chatter", _meta={"parentToolCallId": "read"}))
    translator.translate(update("tool_call_update", toolCallId="read", status="completed", rawOutput="source"))
    translator.translate(update("agent_thought_chunk", content={"type": "text", "text": "private thought"}))
    translator.translate(text("Fixed "))
    translator.translate(text("and tested."))
    assert translator.finish({"stopReason": "end_turn"}) == "Fixed and tested."


def test_copilot_tool_filter_notice_is_activity_not_the_final_answer():
    translator = ACPEventTranslator("s")
    notice = "Info: Disabled tools: bash, create, edit"
    assert translator.translate(text(notice))[0]["acp_update_type"] == "configuration_notice"
    translator.translate(text("ACP_SMOKE_OK"))
    assert translator.finish({"stopReason": "end_turn"}) == "ACP_SMOKE_OK"
    # An ordinary later mention is model text, not another startup notice.
    translator.translate(text(notice))
    assert translator.finish({"stopReason": "end_turn"}).endswith(notice)


def test_plan_diff_and_incremental_tool_fields_keep_full_content():
    translator = ACPEventTranslator("s")
    plan = [{"content": "Find regression", "status": "in_progress", "priority": "high"}]
    assert translator.translate(update("plan", entries=plan))[0]["plan"] == plan
    initial = translator.translate(update("tool_call", toolCallId="edit", kind="edit", title="Fix", rawInput={"path": "a.py"}, status="pending"))
    assert initial[0]["type"] == "tool_call_started"
    diff = {"type": "diff", "path": "a.py", "oldText": "old", "newText": "new\n" + "x" * 100000}
    rows = translator.translate(update("tool_call_update", toolCallId="edit", status="completed", content=[diff]))
    assert rows[0]["args"] == {"path": "a.py"}
    assert not rows[0]["is_error"]
    assert rows[1]["diff"] == diff
    assert rows[1]["text"].endswith("x" * 100000)
    assert "rawInput" not in translator.tools["edit"]
    assert translator.translate(update("tool_call_update", toolCallId="edit", status="completed")) == []


@pytest.mark.parametrize("output", [
    {"exitCode": 1, "content": "test failed"},
    {"success": False, "content": "test failed"},
    {"isError": True},
    "FAILED\n<shellId: 123 completed with exit code 1>",
])
def test_completed_tool_is_not_necessarily_success(output):
    rows = ACPEventTranslator("s").translate(update("tool_call", toolCallId="check", kind="execute", status="completed", rawOutput=output))
    assert rows[-1]["is_error"] is True


@pytest.mark.parametrize("frame", [
    update("tool_call", kind="edit"),
    update("tool_call", toolCallId="bad-status", status=[]),
    update("plan", entries="not a list"),
    update("plan", entries=[{}]),
    text(None),
    {"method": "session/update", "params": {"sessionId": "s", "update": None}},
])
def test_malformed_updates_are_visible_failures(frame):
    with pytest.raises(ACPError):
        ACPEventTranslator("s").translate(frame)


def test_unknown_and_other_session_updates_never_become_answers():
    translator = ACPEventTranslator("s")
    assert translator.translate(update("future_preview_feature", text="not final")) == []
    frame = text("foreign")
    frame["params"]["sessionId"] = "elsewhere"
    assert translator.translate(frame) == []
    with pytest.raises(ACPError, match="without a final answer"):
        translator.finish({"stopReason": "end_turn"})


@pytest.mark.parametrize("reason", ["cancelled", "max_tokens", "refusal", "", None])
def test_non_end_turn_is_not_completion(reason):
    translator = ACPEventTranslator("s")
    translator.translate(text("partial work"))
    with pytest.raises(ACPError):
        translator.finish({"stopReason": reason})


def test_unfinished_tools_and_oversized_answers_fail_closed():
    translator = ACPEventTranslator("s", max_answer_bytes=10)
    with pytest.raises(ACPError, match="complete frames remain"):
        translator.translate(text("x" * 11))
    translator = ACPEventTranslator("s")
    translator.translate(update("tool_call", toolCallId="pending", status="in_progress"))
    translator.translate(text("claiming done"))
    with pytest.raises(ACPError, match="unfinished tools"):
        translator.finish({"stopReason": "end_turn"})
