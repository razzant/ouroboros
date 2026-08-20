"""Loop miscellaneous regressions: the message stream and pacing seams.

Consolidated from former ``test_loop_incoming_messages.py`` (image payload
preservation) and ``test_loop_skill_finalization.py``, then divided by theme:
the task-acceptance gate lives in ``test_loop_acceptance_gate.py``, the
self-authored skill finalization gate in ``test_loop_skill_finalization.py``,
the ``run_llm_loop`` round/finalization suite in ``test_run_llm_loop.py`` and
the image auto-attach seam in ``test_loop_image_attach.py``.

Kept as the home for loop micro-regressions that do not justify a standalone
file: message draining, owner directives, self-check and pacing injections,
the final-answer latch, the deadline-local finalize gate and the per-task
web gate.
"""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import ouroboros.loop as loop_mod
from ouroboros import loop_round_limits
from ouroboros.loop_acceptance import _latch_final_answer_marker, _server_web_allowed_by_task
from ouroboros.loop_messages import _initialize_owner_directives
from ouroboros.loop_round_limits import _drain_incoming_messages
from ouroboros.loop_nudges import _maybe_inject_self_check, _maybe_inject_time_budget_milestone


# ---------------------------------------------------------------------------
# _drain_incoming_messages — telegram image payload preservation
# ---------------------------------------------------------------------------


def test_drain_incoming_messages_preserves_image_payload():
    messages: list = []
    incoming_messages: queue.Queue = queue.Queue()
    incoming_messages.put({
        "text": "photo from telegram",
        "image_base64": "aW1hZ2U=",
        "image_mime": "image/png",
        "image_caption": "photo from telegram",
    })

    _drain_incoming_messages(
        messages=messages,
        incoming_messages=incoming_messages,
        drive_root=None,
        task_id="",
        event_queue=None,
        _owner_msg_seen=set(),
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "[Message from my human]: photo from telegram"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,aW1hZ2U="


def test_owner_directives_survive_compaction_without_control_prose(tmp_path):
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message

    ctx = SimpleNamespace()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Initial requirement verbatim"},
    ]
    _initialize_owner_directives(ctx, messages)
    incoming: queue.Queue = queue.Queue()
    incoming.put({"text": "direct follow-up", "client_message_id": "direct-1"})
    write_owner_message(tmp_path, "mailbox follow-up", task_id="root", msg_id="mail-1")
    write_owner_message(
        tmp_path, "deadline control", task_id="root", msg_id="control-1",
        kind=KIND_FINALIZE_NOW,
    )

    controls = _drain_incoming_messages(
        messages,
        incoming,
        tmp_path,
        "root",
        None,
        set(),
        owner_ctx=ctx,
    )

    assert controls == {"finalize_now": "deadline control"}
    assert [row["source"] for row in ctx._owner_directives] == [
        "initial_user", "direct_incoming", "owner_mailbox",
    ]
    assert ctx._owner_directives[0]["content"] == "Initial requirement verbatim"
    assert ctx._owner_directives[1]["msg_id"] == "direct-1"
    assert ctx._owner_directives[2] == {
        "source": "owner_mailbox",
        "content": "mailbox follow-up",
        "msg_id": "mail-1",
    }
    assert "deadline control" not in json.dumps(ctx._owner_directives)


def test_maybe_inject_self_check_handles_assistant_none_content():
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
    ]
    progress = []

    injected = _maybe_inject_self_check(
        15,
        30,
        messages,
        {"cost": 0.0},
        progress.append,
    )

    assert injected is True
    assert messages[-1]["role"] == "user"
    assert "[CHECKPOINT 1" in messages[-1]["content"]
    assert progress


def test_time_budget_milestone_injects_once_per_threshold(monkeypatch):
    messages = [{"role": "user", "content": "solve"}]
    ctx = SimpleNamespace(
        task_metadata={
            "created_at": "2026-06-10T00:00:00Z",
            "deadline_at": "2026-06-10T10:00:00Z",
        },
    )

    from datetime import datetime, timezone

    monkeypatch.setattr("ouroboros.task_pacing.utc_now", lambda: datetime(2026, 6, 10, 5, 1, tzinfo=timezone.utc))

    injected = _maybe_inject_time_budget_milestone(
        messages,
        SimpleNamespace(_ctx=ctx),
        event_queue=None,
        task_id="task-time",
        drive_logs=None,
    )
    injected_again = _maybe_inject_time_budget_milestone(messages, SimpleNamespace(_ctx=ctx))

    assert injected is True
    assert injected_again is False
    assert "[TIME BUDGET" in messages[-1]["content"]
    assert "50% remaining" in messages[-1]["content"]
    assert ctx._time_budget_milestones_seen == {"50%"}


def test_intrinsic_pacing_injects_without_deadline(monkeypatch):
    """No deadline_at: surface elapsed/rounds/cost once per interval bucket.
    v6.60.0: the FINAL ANSWER phrase appears ONLY when the task contract declares
    answer_protocol="final_answer_line" (marker phrases are protocol-gated)."""
    messages = [{"role": "user", "content": "solve"}]
    ctx = SimpleNamespace(task_metadata={"created_at": "2026-06-10T00:00:00Z"})  # no deadline_at
    from datetime import datetime, timezone

    monkeypatch.delenv("OUROBOROS_PACING_INTERVAL_SEC", raising=False)
    # 20 min elapsed, default interval 600s -> bucket 2.
    monkeypatch.setattr("ouroboros.task_pacing.utc_now", lambda: datetime(2026, 6, 10, 0, 20, tzinfo=timezone.utc))

    injected = _maybe_inject_time_budget_milestone(
        messages, SimpleNamespace(_ctx=ctx), round_idx=7,
        accumulated_usage={"cost": 1.25}, task_id="t",
    )
    injected_again = _maybe_inject_time_budget_milestone(
        messages, SimpleNamespace(_ctx=ctx), round_idx=8, accumulated_usage={"cost": 1.4},
    )

    assert injected is True
    assert injected_again is False  # same bucket -> not repeated
    assert "[PACING" in messages[-1]["content"]
    assert "Rounds so far: 7" in messages[-1]["content"]
    assert "FINAL ANSWER:" not in messages[-1]["content"]  # no protocol declared

    # With the protocol declared, the salvage phrase rides the SAME milestone.
    proto_ctx = SimpleNamespace(
        task_metadata={"created_at": "2026-06-10T00:00:00Z"},
        task_contract={"answer_protocol": "final_answer_line"},
    )
    proto_messages = [{"role": "user", "content": "solve"}]
    assert _maybe_inject_time_budget_milestone(
        proto_messages, SimpleNamespace(_ctx=proto_ctx), round_idx=7,
        accumulated_usage={"cost": 1.25}, task_id="t2",
    ) is True
    assert "FINAL ANSWER:" in proto_messages[-1]["content"]


def test_latch_final_answer_marker_captures_explicit_marker_only():
    trace = {"tool_calls": [{"tool": "read_file"}]}
    _latch_final_answer_marker(trace, "analysis\nFINAL ANSWER: 123")
    assert trace["best_valid_final_answer"] == "123"
    assert trace["best_valid_final_answer_tools"] == 1
    _latch_final_answer_marker(trace, "answer-ish prose without marker")
    assert trace["best_valid_final_answer"] == "123"


def test_latch_final_answer_marker_counts_same_turn_tool_calls():
    trace = {"tool_calls": [{"tool": "read_file"}]}
    current = [{"function": {"name": "run_command"}}, {"function": {"name": "verify_and_record"}}]
    _latch_final_answer_marker(trace, "FINAL ANSWER: draft", current_tool_calls=current)
    assert trace["best_valid_final_answer"] == "draft"
    # Same-turn tool calls are newer grounding and must invalidate this latch unless
    # the model re-emits the marker after those tools complete.
    assert trace["best_valid_final_answer_tools"] == 1


def test_server_web_allowed_respects_task_resource_contract():
    assert _server_web_allowed_by_task(SimpleNamespace(task_contract={})) is True
    assert _server_web_allowed_by_task(SimpleNamespace(task_contract={"allowed_resources": {"web": False}})) is False
    assert _server_web_allowed_by_task(SimpleNamespace(task_contract={"allowed_resources": {"network": False}})) is False
    assert _server_web_allowed_by_task(SimpleNamespace(task_contract={"disabled_tools": ["web_search"]})) is True


def test_intrinsic_pacing_disabled_when_interval_zero(monkeypatch):
    messages = [{"role": "user", "content": "solve"}]
    ctx = SimpleNamespace(task_metadata={"created_at": "2026-06-10T00:00:00Z"})
    from datetime import datetime, timezone

    monkeypatch.setenv("OUROBOROS_PACING_INTERVAL_SEC", "0")
    monkeypatch.setattr("ouroboros.task_pacing.utc_now", lambda: datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc))

    assert _maybe_inject_time_budget_milestone(messages, SimpleNamespace(_ctx=ctx), round_idx=3) is False


def test_deadline_local_finalize_gate(monkeypatch):
    """Self-finalize only when a REAL deadline is within the grace window."""
    from datetime import datetime, timezone

    captured = {}

    def _fake_final(ctx, *, prompt, fallback_text, reason_code):
        captured["reason_code"] = reason_code
        return ("BEST EFFORT", {"reason_code": reason_code}, {})

    monkeypatch.setattr(loop_mod, "_forced_final_answer", _fake_final)
    # v6.54.4: the gate consults the task_pacing effective reserve SSOT.
    monkeypatch.setattr("ouroboros.task_pacing.effective_finalization_reserve_sec", lambda ctx: 120.0)
    monkeypatch.setattr(loop_mod, "utc_now", lambda: datetime(2026, 6, 10, 9, 59, 0, tzinfo=timezone.utc))

    # Far from deadline (10:30 vs now 09:59 -> ~31 min left > 120s) -> no finalize.
    far = SimpleNamespace(_ctx=SimpleNamespace(task_metadata={"deadline_at": "2026-06-10T10:30:00Z"}))
    assert loop_round_limits._maybe_deadline_local_finalize(SimpleNamespace(), far) is None
    # Within grace (10:00 vs now 09:59 -> 60s < 120s) -> finalize best-effort.
    near = SimpleNamespace(_ctx=SimpleNamespace(task_metadata={"deadline_at": "2026-06-10T10:00:00Z"}))
    result = loop_round_limits._maybe_deadline_local_finalize(SimpleNamespace(), near)
    assert result is not None and result[0] == "BEST EFFORT"
    assert captured["reason_code"] == "deadline_local"
    # No deadline_at at all -> never fires (no synthesized deadline).
    none_ctx = SimpleNamespace(_ctx=SimpleNamespace(task_metadata={}))
    assert loop_round_limits._maybe_deadline_local_finalize(SimpleNamespace(), none_ctx) is None
