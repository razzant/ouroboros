"""Degenerate-turn guard: a single assistant turn cannot execute unbounded tool calls.

CyberGym r8 (2026-09-04, DeepSeek V4 Flash): two turns carried 872 and 1113
``run_command`` calls. Executing all of them appended ~1.7M tokens of tool
results, the next request overflowed the 1M window and both tasks died as
``context_overflow`` after 100+ productive rounds.
"""

from __future__ import annotations

import ouroboros.loop_tool_execution as lte
from ouroboros.loop_tool_execution import (
    cap_tool_call_burst,
    max_tool_calls_per_turn,
    tool_call_burst_notice,
)


def _calls(n: int) -> list[dict]:
    return [
        {"id": f"call_{i}", "type": "function",
         "function": {"name": "run_command", "arguments": "{}"}}
        for i in range(n)
    ]


def test_turn_within_limit_is_untouched():
    calls = _calls(5)
    msg = {"role": "assistant", "content": "", "tool_calls": list(calls)}
    kept, dropped = cap_tool_call_burst(msg, calls, limit=32)
    assert dropped == 0
    assert kept is calls
    assert msg["tool_calls"] == calls


def test_burst_keeps_emission_order_prefix_and_prunes_assistant_message():
    calls = _calls(1113)
    msg = {"role": "assistant", "content": "", "tool_calls": list(calls)}
    kept, dropped = cap_tool_call_burst(msg, calls, limit=32)
    assert dropped == 1081
    assert [tc["id"] for tc in kept] == [f"call_{i}" for i in range(32)]
    # Protocol invariant: the transcript's assistant turn advertises exactly the
    # calls that will receive tool results — no orphan tool_call_ids.
    assert [tc["id"] for tc in msg["tool_calls"]] == [tc["id"] for tc in kept]


def test_zero_or_negative_limit_disables_the_guard():
    calls = _calls(200)
    msg = {"role": "assistant", "tool_calls": list(calls)}
    for limit in (0, -1):
        kept, dropped = cap_tool_call_burst(msg, calls, limit=limit)
        assert (len(kept), dropped) == (200, 0)
    assert len(msg["tool_calls"]) == 200


def test_default_limit_comes_from_settings_then_env(monkeypatch):
    monkeypatch.delenv("OUROBOROS_MAX_TOOL_CALLS_PER_TURN", raising=False)
    monkeypatch.setattr(lte, "load_settings", lambda: {})
    assert max_tool_calls_per_turn() == 32
    monkeypatch.setattr(lte, "load_settings", lambda: {"OUROBOROS_MAX_TOOL_CALLS_PER_TURN": 8})
    assert max_tool_calls_per_turn() == 8
    monkeypatch.setenv("OUROBOROS_MAX_TOOL_CALLS_PER_TURN", "3")
    assert max_tool_calls_per_turn() == 3
    monkeypatch.setenv("OUROBOROS_MAX_TOOL_CALLS_PER_TURN", "junk")
    assert max_tool_calls_per_turn() == 32


def test_notice_tells_the_model_what_was_dropped():
    text = tool_call_burst_notice(32, 1081)
    assert "1113 tool calls" in text
    assert "first 32" in text
    assert "1081" in text and "DISCARDED" in text


def test_run_llm_loop_executes_only_the_capped_prefix_and_warns_the_model(tmp_path, monkeypatch):
    import queue

    import ouroboros.loop as loop_mod
    from ouroboros.loop import run_llm_loop
    from ouroboros.tools.registry import ToolRegistry

    monkeypatch.setenv("OUROBOROS_MAX_TOOL_CALLS_PER_TURN", "4")
    burst = {
        "role": "assistant",
        "content": None,
        "tool_calls": _calls(50),
        "response_id": "gen-burst",
    }
    requests: list[list[dict]] = []
    executed: list[list[str]] = []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call(_llm, request_messages, *_args, **_kwargs):
        requests.append([dict(item) for item in request_messages])
        if len(requests) == 1:
            return dict(burst), 0.0
        return {"role": "assistant", "content": "done"}, 0.0

    def fake_handle(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, _trace, _progress):
        executed.append([tc["id"] for tc in tool_calls])
        for tc in tool_calls:
            request_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "ok"})
        return 0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle)

    result, _usage, _trace = run_llm_loop(
        messages=[{"role": "user", "content": "go"}],
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="burst",
        drive_root=tmp_path,
    )

    assert result == "done"
    assert executed == [[f"call_{i}" for i in range(4)]]
    second = requests[1]
    assistant = next(item for item in second if item.get("response_id") == "gen-burst")
    assert [tc["id"] for tc in assistant["tool_calls"]] == [f"call_{i}" for i in range(4)]
    tool_ids = [item["tool_call_id"] for item in second if item.get("role") == "tool"]
    assert tool_ids == [f"call_{i}" for i in range(4)]
    notice = [item for item in second if item.get("role") == "user" and "DISCARDED" in str(item.get("content"))]
    assert len(notice) == 1 and "50 tool calls" in notice[0]["content"]
    events = (tmp_path / "events.jsonl").read_text() if (tmp_path / "events.jsonl").exists() else ""
    assert "tool_call_burst_truncated" in events
