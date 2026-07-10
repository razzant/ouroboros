"""Loop miscellaneous regressions.

Consolidated from former ``test_loop_incoming_messages.py`` (image
payload preservation) and ``test_loop_skill_finalization.py``
(self-authored skill finalization gate). Both modules exercise
narrow corners of ``ouroboros.loop`` that did not justify standalone
files after Phase 5.

Kept here as one module so future loop micro-regressions have a
natural home instead of producing yet another single-test file.
"""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import ouroboros.loop as loop_mod
from ouroboros.loop import (
    _drain_incoming_messages,
    _latch_final_answer_marker,
    _maybe_inject_self_check,
    _maybe_inject_time_budget_milestone,
    _run_task_acceptance_review_once,
    _set_acceptance_decision,
    _skill_finalization_message,
    _skill_names_touched_by_trace,
    _task_acceptance_eligible,
    _server_web_allowed_by_task,
    run_llm_loop,
)
from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    save_enabled,
    save_review_state,
)


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


def test_set_acceptance_decision_preserves_agent_stance():
    trace = {
        "acceptance_decision": {
            "status": "rejected",
            "agent_disposition": "rejected",
            "agent_rationale": "Scope drift.",
        }
    }
    _set_acceptance_decision(trace, {
        "status": "accepted",
        "source": "task_acceptance_review",
        "rationale": "No actionable changes.",
    })

    assert trace["acceptance_decision"]["status"] == "accepted"
    assert trace["acceptance_decision"]["agent_disposition"] == "rejected"
    assert trace["acceptance_decision"]["agent_rationale"] == "Scope drift."


def test_task_acceptance_review_tool_result_lifts_agent_decision_into_trace():
    from ouroboros.loop_tool_execution import process_tool_results

    trace = {"tool_calls": []}
    messages = []
    result = {
        "request": {},
        "actors": [],
        "parsed_findings": [],
        "aggregate_signal": "PASS",
        "agent_decision": {
            "disposition": "deferred",
            "rationale": "Waiting for benchmark smoke.",
            "source": "agent_task_acceptance_review_tool",
        },
    }

    process_tool_results(
        [{
            "fn_name": "task_acceptance_review",
            "tool_call_id": "call-1",
            "result": json.dumps(result),
            "is_error": False,
            "args_for_log": {},
            "tool_args": {},
            "result_meta": {"status": "ok"},
        }],
        messages,
        trace,
        emit_progress=lambda _msg: None,
    )

    assert trace["acceptance_decision"]["agent_disposition"] == "deferred"
    assert trace["acceptance_decision"]["agent_rationale"] == "Waiting for benchmark smoke."


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
    assert loop_mod._maybe_deadline_local_finalize(SimpleNamespace(), far) is None
    # Within grace (10:00 vs now 09:59 -> 60s < 120s) -> finalize best-effort.
    near = SimpleNamespace(_ctx=SimpleNamespace(task_metadata={"deadline_at": "2026-06-10T10:00:00Z"}))
    result = loop_mod._maybe_deadline_local_finalize(SimpleNamespace(), near)
    assert result is not None and result[0] == "BEST EFFORT"
    assert captured["reason_code"] == "deadline_local"
    # No deadline_at at all -> never fires (no synthesized deadline).
    none_ctx = SimpleNamespace(_ctx=SimpleNamespace(task_metadata={}))
    assert loop_mod._maybe_deadline_local_finalize(SimpleNamespace(), none_ctx) is None


def test_task_acceptance_auto_is_llm_first_not_host_enforced(monkeypatch):
    trace = {
        "tool_calls": [
            {"tool": "write_file", "args": {"path": "x.py"}},
            {"tool": "run_command", "args": {"cmd": ["pytest"]}},
        ]
    }

    # auto stays LLM-first (host never enforces), regardless of effects.
    assert _task_acceptance_eligible("auto", trace, True)[0] is False
    # required is effect-gated: this trace has a workspace write -> eligible.
    assert _task_acceptance_eligible("required", trace, True)[0] is True
    assert _task_acceptance_eligible("off", trace, True)[0] is False

    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "required")
    ctx = SimpleNamespace(_task_acceptance_reviewed=False, is_direct_chat=False, drive_root="/tmp")
    reviewed_trace = {
        "tool_calls": [{"tool": "task_acceptance_review", "args": {}}],
        "review_runs": [{"request": {"surface": "task_acceptance"}, "aggregate_signal": "PASS"}],
    }
    assert _run_task_acceptance_review_once(
        tools=SimpleNamespace(_ctx=ctx),
        content="done",
        task_id="task1",
        task_type="task",
        llm_trace=reviewed_trace,
        drive_root=None,
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": "goal"}],
        emit_progress=lambda _msg: None,
    ) is False
    assert ctx._task_acceptance_reviewed is True
    assert reviewed_trace["review_decision"]["trigger"] == "agent_called_tool_result"


def test_task_acceptance_required_feeds_back_capsule(monkeypatch, tmp_path):
    """WA4 (v6.36.0): host-forced `required` review records the full verdict on
    the objective axis AND feeds the agent a COMPACT improvement capsule for a
    real best_effort/blocked_with_evidence (ONE bounded pass, anti-derailment
    framed). A solved/nothing-actionable result still finalizes with no injection."""
    import ouroboros.review_substrate as rs

    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "required")
    monkeypatch.setattr(rs, "reviewer_slots", lambda **k: [object(), object(), object()])

    # (a) CONTRACT-VALID solved PASS (a non-empty completion_coach, as the required
    # contract demands) with no actionable findings -> still NO injection, finalize.
    # A coach alone must not re-loop an already-solved deliverable.
    solved = rs.ReviewRunResult(
        request={"surface": "task_acceptance"},
        actors=[{"signal": "PASS", "slot_id": "s0",
                 "parsed": {"outcome_tier": "solved", "completion_coach": "ship it as-is"}}],
        parsed_findings=[], aggregate_signal="PASS",
    )
    monkeypatch.setattr(rs, "run_review_request", lambda *a, **k: solved)
    ctx = SimpleNamespace(_task_acceptance_reviewed=False, is_direct_chat=False, drive_root=str(tmp_path))
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "goal"}]
    result = _run_task_acceptance_review_once(
        tools=SimpleNamespace(_ctx=ctx), content="done", task_id="t", task_type="task",
        llm_trace=trace, drive_root=None, messages=messages, emit_progress=lambda _m: None,
    )
    assert result is False                                        # nothing to improve -> no extra round
    assert len(messages) == 2                                     # transcript NOT mutated
    assert trace["review_runs"][0]["aggregate_signal"] == "PASS"  # full verdict recorded (objective axis)

    # (b) blocked_with_evidence -> compact capsule fed back exactly once.
    blocked = rs.ReviewRunResult(
        request={"surface": "task_acceptance"},
        actors=[{"signal": "FAIL", "slot_id": "s0",
                 "parsed": {"outcome_tier": "blocked_with_evidence", "completion_coach": "run the real grader"}}],
        parsed_findings=[{"slot_id": "s0", "severity": "critical", "item": "fake test", "recommendation": "use the pre-existing suite"}],
        aggregate_signal="FAIL",
    )
    monkeypatch.setattr(rs, "run_review_request", lambda *a, **k: blocked)
    ctx2 = SimpleNamespace(_task_acceptance_reviewed=False, is_direct_chat=False, drive_root=str(tmp_path))
    trace2 = {"tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    messages2 = [{"role": "system", "content": ""}, {"role": "user", "content": "goal"}]
    tools2 = SimpleNamespace(_ctx=ctx2)
    result2 = _run_task_acceptance_review_once(
        tools=tools2, content="done", task_id="t", task_type="task",
        llm_trace=trace2, drive_root=None, messages=messages2, emit_progress=lambda _m: None,
    )
    assert result2 is True                                        # capsule -> one bounded re-loop
    # The capsule reaches the agent (appended/merged into the trailing user turn).
    assert "improvement note" in messages2[-1]["content"].lower()
    assert "Do not mention this review" in messages2[-1]["content"]
    # The CAPSULE is bounded (injected once), but the review is NOT yet terminal —
    # so the REVISED final deliverable still gets reviewed (round-4 state-machine fix).
    assert getattr(ctx2, '_task_acceptance_improvement_passes', 0) == 1  # v6.54.4: counter replaced the boolean latch
    assert getattr(ctx2, "_task_acceptance_reviewed", False) is False
    assert trace2["acceptance_decision"]["status"] == "revision_requested"

    # If the revised answer is accepted, the terminal decision overwrites the
    # earlier revision_requested state rather than leaving stale telemetry.
    monkeypatch.setattr(rs, "run_review_request", lambda *a, **k: solved)
    trace_ok = {"tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    messages_ok = [{"role": "system", "content": ""}, {"role": "user", "content": "goal"}]
    result_ok = _run_task_acceptance_review_once(
        tools=tools2, content="revised", task_id="t", task_type="task",
        llm_trace=trace_ok, drive_root=None, messages=messages_ok, emit_progress=lambda _m: None,
    )
    assert result_ok is False
    assert trace_ok["acceptance_decision"]["status"] == "accepted"
    tools2._ctx._task_acceptance_reviewed = False

    # (c) the revised final deliverable IS re-reviewed (verdict on the SHIPPED answer,
    # not the stale pre-revision one), and the one capsule is not injected again.
    monkeypatch.setattr(rs, "run_review_request", lambda *a, **k: blocked)
    trace3 = {"tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    messages3 = [{"role": "system", "content": ""}, {"role": "user", "content": "goal"}]
    result3 = _run_task_acceptance_review_once(
        tools=tools2, content="revised", task_id="t", task_type="task",
        llm_trace=trace3, drive_root=None, messages=messages3, emit_progress=lambda _m: None,
    )
    assert result3 is False                                       # capsule already spent -> finalize
    assert len(messages3) == 2                                    # no second capsule injected
    assert trace3["review_runs"][0]["aggregate_signal"] == "FAIL"  # final-deliverable verdict recorded
    assert ctx2._task_acceptance_reviewed is True                # now terminal


def test_required_review_blocked_commit_does_not_surface_prior_head(monkeypatch, tmp_path):
    """T1 (v6.35.0): a REVIEW_BLOCKED/GIT_ERROR commit attempt is is_error=False but
    carries a non-ok status, so it must NOT count as 'committed this turn' — else
    collect_turn_diff would surface an unrelated prior HEAD commit as evidence."""
    import ouroboros.review_evidence as re_mod
    import ouroboros.review_substrate as rs

    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "required")

    class _FakeResult:
        aggregate_signal = "PASS"
        request = {"surface": "task_acceptance"}

    monkeypatch.setattr(rs, "run_review_request", lambda *a, **k: _FakeResult())
    monkeypatch.setattr(rs, "reviewer_slots", lambda **k: [object(), object(), object()])

    captured = {}

    def _fake_collect(ctx, *, include_recent_commit=False, **k):
        captured["include_recent_commit"] = include_recent_commit
        return ""

    monkeypatch.setattr(re_mod, "collect_turn_diff", _fake_collect)

    ctx = SimpleNamespace(_task_acceptance_reviewed=False, is_direct_chat=False, drive_root=str(tmp_path))
    # A blocked commit attempt: is_error False, but structured status is "blocked".
    trace = {"tool_calls": [{"tool": "commit_reviewed", "is_error": False, "status": "blocked"}]}
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "goal"}]

    _run_task_acceptance_review_once(
        tools=SimpleNamespace(_ctx=ctx),
        content="done",
        task_id="t",
        task_type="task",
        llm_trace=trace,
        drive_root=None,
        messages=messages,
        emit_progress=lambda _m: None,
    )

    assert captured["include_recent_commit"] is False

    # A genuinely landed commit (status "ok") DOES surface the committed HEAD.
    captured.clear()
    trace_ok = {"tool_calls": [{"tool": "commit_reviewed", "is_error": False, "status": "ok"}]}
    ctx._task_acceptance_reviewed = False
    _run_task_acceptance_review_once(
        tools=SimpleNamespace(_ctx=ctx),
        content="done",
        task_id="t",
        task_type="task",
        llm_trace=trace_ok,
        drive_root=None,
        messages=messages,
        emit_progress=lambda _m: None,
    )
    assert captured["include_recent_commit"] is True


# ---------------------------------------------------------------------------
# Skill finalization gate (self-authored skills must reach ready+enabled
# before the loop accepts a final text response)
# ---------------------------------------------------------------------------


def _write_self_authored_skill(drive_root, name: str = "alpha"):
    skill_dir = drive_root / "skills" / "external" / name
    state_dir = drive_root / "state" / "skills" / name
    skill_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ntype: instruction\nversion: 0.1.0\n---\nbody\n",
        encoding="utf-8",
    )
    marker = {
        "schema_version": 1,
        "origin": "self_authored",
        "task_id": "task-1",
        "created_at": "2026-05-07T00:00:00+00:00",
    }
    (skill_dir / ".self_authored.json").write_text(json.dumps(marker), encoding="utf-8")
    (state_dir / "self_authored.json").write_text(json.dumps(marker), encoding="utf-8")
    return skill_dir


def test_skill_names_touched_by_trace_detects_data_skill_edits():
    trace = {
        "tool_calls": [
            {"tool": "write_file", "args": {"path": "skills/external/alpha/plugin.py"}},
            {"tool": "edit_text", "args": {"path": "data/skills/external/beta/SKILL.md"}},
            {"tool": "claude_code_edit", "args": {"cwd": "skills/external/gamma"}},
            {"tool": "write_file", "args": {"path": "SKILL.md", "bucket": "external", "skill_name": "delta"}},
        ]
    }

    assert _skill_names_touched_by_trace(trace) == ["alpha", "beta", "gamma", "delta"]


def test_skill_finalization_message_blocks_unreviewed_self_authored_skill(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_self_authored_skill(drive_root)
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "skills/external/alpha/SKILL.md"}}]}

    message = _skill_finalization_message(drive_root, trace)

    assert "SKILL_NOT_FINALIZED" in message
    assert "alpha" in message


def test_skill_finalization_message_allows_ready_self_authored_skill(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    skill_dir = _write_self_authored_skill(drive_root)
    content_hash = compute_content_hash(skill_dir)
    save_review_state(drive_root, "alpha", SkillReviewState(status="pass", content_hash=content_hash))
    save_enabled(drive_root, "alpha", True)
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "skills/external/alpha/SKILL.md"}}]}

    assert _skill_finalization_message(drive_root, trace) == ""


def test_run_llm_loop_preserves_assistant_tool_call_metadata(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    messages = [{"role": "user", "content": "inspect"}]
    assistant_metadata = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }],
        "reasoning": "I need the file first.",
        "reasoning_details": [{"type": "reasoning.text", "text": "I need the file first."}],
        "response_id": "gen-123",
    }
    seen_second_request = {}
    calls = {"count": 0}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, *_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return dict(assistant_metadata), 0.0
        seen_second_request["messages"] = [dict(item) for item in request_messages]
        return {"role": "assistant", "content": "done"}, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, _trace, _progress):
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "file"})
        return 0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)

    result, _usage, _trace = run_llm_loop(
        messages=messages,
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="roundtrip",
        drive_root=tmp_path,
    )

    assert result == "done"
    assistant_msg = next(item for item in seen_second_request["messages"] if item.get("response_id") == "gen-123")
    assert assistant_msg["tool_calls"] == assistant_metadata["tool_calls"]
    assert assistant_msg["reasoning"] == assistant_metadata["reasoning"]
    assert assistant_msg["reasoning_details"] == assistant_metadata["reasoning_details"]
    assert assistant_msg["response_id"] == "gen-123"


def test_run_llm_loop_narrates_reasoning_to_bubble_not_trace(tmp_path, monkeypatch):
    """Display-only contract: a pure tool-call round with no visible content narrates the
    provider's readable reasoning to the progress BUBBLE, but never records it in the durable
    trace (``reasoning_notes`` feeds build_trace_summary / task summaries) — so display-only
    reasoning cannot leak out of the display path."""
    from ouroboros.tools.registry import ToolRegistry

    messages = [{"role": "user", "content": "go"}]
    tool_round = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        "reasoning": "Let me read the file before answering.",
    }
    calls = {"count": 0}
    emitted: list = []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, *_a, **_k):
        calls["count"] += 1
        if calls["count"] == 1:
            return dict(tool_round), 0.0
        return {"role": "assistant", "content": "final answer"}, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _dl, _tid, _ex, request_messages, _tr, _pg):
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "file body"})
        return 0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)
    monkeypatch.setenv("OUROBOROS_REASONING_SUMMARY", "auto")

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda text: emitted.append(text),
        incoming_messages=queue.Queue(),
        task_id="narrate",
        drive_root=tmp_path,
    )

    assert result == "final answer"
    # the readable reasoning reached the display bubble...
    assert any("read the file before answering" in str(e) for e in emitted)
    # ...but did NOT leak into the durable trace (display-only).
    assert all("read the file before answering" not in str(n) for n in trace["reasoning_notes"])


def test_run_llm_loop_finalize_now_control_forces_best_effort_answer(tmp_path, monkeypatch):
    """A supervisor finalize_now control makes the loop extract one tool-less
    final answer and stamp the finalization_grace reason (typed best_effort
    gate downstream) — a deadline never returns emptiness."""
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
    from ouroboros.tools.registry import ToolRegistry

    write_owner_message(tmp_path, "deadline", task_id="graceful1", kind=KIND_FINALIZE_NOW)
    seen = {}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, _model, tools_arg, *_args, **_kwargs):
        seen["tools"] = tools_arg
        seen["messages"] = [dict(item) for item in request_messages]
        return {"role": "assistant", "content": "best effort summary"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    result, usage, _trace = run_llm_loop(
        messages=[{"role": "user", "content": "long job"}],
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="graceful1",
        drive_root=tmp_path,
    )

    assert result == "best effort summary"
    assert usage["reason_code"] == "finalization_grace"
    assert usage["execution_status"] == "failed"  # lifted to best_effort by the outcome gate
    assert usage["_best_effort_extracted"] is True  # typed fact: real model answer
    assert seen["tools"] is None  # tool-less final extraction
    joined = json.dumps(seen["messages"], ensure_ascii=False)
    assert "[FINALIZE_NOW]" in joined

    # End-to-end: the derived outcome lands on the typed best_effort shelf.
    from ouroboros.outcomes import EXECUTION_BEST_EFFORT, derive_loop_outcome
    outcome = derive_loop_outcome(result, usage, {"tool_calls": [], "reasoning_notes": []})
    assert outcome["outcome_axes"]["execution"]["status"] == EXECUTION_BEST_EFFORT


def test_run_llm_loop_keeps_task_model_override_across_tool_rounds(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    messages = [{"role": "user", "content": "inspect"}]
    seen_models: list[str] = []
    seen_use_local: list[bool] = []
    calls = {"count": 0}

    class FakeLLM:
        def default_model(self):
            return "default-model"

    def fake_call_llm_with_retry(_llm, request_messages, model, *_args, **kwargs):
        seen_models.append(model)
        seen_use_local.append(bool(kwargs.get("use_local")))
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            }, 0.0
        return {"role": "assistant", "content": "done"}, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, _trace, _progress):
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "file"})
        return 0

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_model_override = "subagent-light"
    registry._ctx.task_use_local_override = True
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)

    result, _usage, _trace = run_llm_loop(
        messages=messages,
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="subagent1",
        drive_root=tmp_path,
    )

    assert result == "done"
    assert seen_models == ["subagent-light", "subagent-light"]
    assert seen_use_local == [True, True]


def test_run_llm_loop_enforces_swarm_force_plan_before_final(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    messages = [{"role": "user", "content": "ship"}]
    calls = {"count": 0}
    seen_second_request = {}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, *_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"role": "assistant", "content": "premature final"}, 0.0
        if calls["count"] == 2:
            seen_second_request["messages"] = [dict(item) for item in request_messages]
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-plan",
                    "type": "function",
                    "function": {"name": "plan_task", "arguments": "{}"},
                }],
            }, 0.0
        return {"role": "assistant", "content": "done after plan"}, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, trace, _progress):
        trace["tool_calls"].append({
            "tool": tool_calls[0]["function"]["name"],
            "args": {},
            "result": "## Plan Review Results\n\nAGGREGATE: GREEN",
            "is_error": False,
            # v6.26.0: the force-plan gate reads this structured flag (captured
            # from the FULL tool result), not a substring of the 700-char preview.
            "plan_review_aggregate": True,
        })
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "## Plan Review Results\n\nAGGREGATE: GREEN"})
        return 0

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {"force_plan": True, "force_plan_source": "swarm"}
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="task1",
        drive_root=tmp_path,
    )

    assert result == "done after plan"
    assert calls["count"] == 3
    assert any("plan_task is required" in str(item.get("content") or "") for item in seen_second_request["messages"])
    assert trace["tool_calls"][0]["tool"] == "plan_task"


def test_run_llm_loop_does_not_accept_failed_plan_task_for_swarm_force_plan(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    messages = [{"role": "user", "content": "ship"}]
    calls = {"count": 0}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, _request_messages, *_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"role": "assistant", "content": "premature final"}, 0.0
        if calls["count"] == 2:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-plan",
                    "type": "function",
                    "function": {"name": "plan_task", "arguments": "{}"},
                }],
            }, 0.0
        return {"role": "assistant", "content": "still finalizing without a valid plan"}, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, trace, _progress):
        trace["tool_calls"].append({
            "tool": tool_calls[0]["function"]["name"],
            "args": {},
            "result": "ERROR: plan_task planning swarm failed closed: no planning subagent completed.",
            "is_error": False,
        })
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "ERROR: plan_task planning swarm failed closed."})
        return 0

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {"force_plan": True, "force_plan_source": "swarm"}
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)

    result, usage, trace = run_llm_loop(
        messages=messages,
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="task1",
        drive_root=tmp_path,
    )

    assert result.startswith("⚠️ SWARM_INITIATIVE_BLOCKED")
    assert calls["count"] == 4
    assert usage["reason_code"] == "swarm_force_plan_not_called"
    assert trace["tool_calls"][0]["tool"] == "plan_task"


def test_run_llm_loop_injects_subagent_handoff_before_final_text(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.registry import ToolRegistry

    write_task_result(
        tmp_path,
        "child1",
        STATUS_COMPLETED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        role="reviewer",
        result="child handoff",
    )
    messages = [{"role": "user", "content": "inspect"}]
    calls = {"count": 0}
    seen_second_request = {}
    progress = []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, *_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"role": "assistant", "content": "premature final"}, 0.0
        seen_second_request["messages"] = [dict(item) for item in request_messages]
        return {"role": "assistant", "content": "final after handoff"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=progress.append,
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )

    assert result == "final after handoff"
    assert calls["count"] == 2
    assert any("Subagent handoff status refreshed" in item for item in progress)
    assert any("Subagent handoff status refreshed" in item for item in trace["reasoning_notes"])
    second_text = "\n".join(str(item.get("content") or "") for item in seen_second_request["messages"])
    # C3.4: the parent now ABSORBS the child's FULL authored result before
    # finalizing (not just a 240-char preview), with a durable get_task_result pointer.
    assert "[SUBAGENT_RESULTS" in second_text
    assert "child child1" in second_text
    assert "child handoff" in second_text
    assert "get_task_result" in second_text


def test_run_llm_loop_appends_orphan_note_when_finalizing_with_unhandled_child(tmp_path, monkeypatch):
    """D#7 / P5: the subagent handoff reminder fires once per CHANGE (not every round, not
    suppressed by parsing the final prose). When the agent finalizes with a child still
    unhandled (not absorbed, not discarded/cancelled), the answer carries a LOUD orphan note
    instead of silently dropping the child."""
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolRegistry

    write_task_result(
        tmp_path,
        "child1",
        STATUS_RUNNING,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        role="reviewer",
        result="still collecting evidence",
    )
    messages = [{"role": "user", "content": "inspect"}]
    calls = {"count": 0}
    progress = []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, _request_messages, *_args, **_kwargs):
        calls["count"] += 1
        # The agent never absorbs/discards the child — it just keeps answering in prose.
        return {"role": "assistant", "content": "child1 is still running; I will finalize now."}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=progress.append,
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )

    # Round 1: child first seen -> reminder fires (signature changed) -> continue.
    # Round 2: signature unchanged + prose is NOT parsed -> finalize, with the orphan note.
    assert calls["count"] == 2
    assert sum(1 for item in progress if "Subagent handoff status refreshed" in item) == 1
    # The agent's prose is preserved AND the loud orphan note is appended (no silent loss).
    assert result.startswith("child1 is still running; I will finalize now.")
    assert "child1" in result and "NOTE: finalized" in result


def test_run_llm_loop_forces_best_effort_after_child_absorption_reminder(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolRegistry

    write_task_result(
        tmp_path,
        "child1",
        STATUS_RUNNING,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        role="reviewer",
        result="still collecting evidence",
    )
    messages = [{"role": "user", "content": "inspect"}]
    calls = {"count": 0}
    progress = []
    tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    tools._ctx.task_contract = {"delegation_budget": {"may_delegate": True, "may_fan_out": True}}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, _request_messages, *_args, **_kwargs):
        calls["count"] += 1
        return {"role": "assistant", "content": f"answer {calls['count']}"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    result, usage, trace = run_llm_loop(
        messages=messages,
        tools=tools,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=progress.append,
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )

    assert usage["reason_code"] == "children_unabsorbed"
    assert usage["_best_effort_extracted"] is True
    assert "Child absorption reminder injected" in "\n".join(progress)
    assert "Child absorption reminder injected" in "\n".join(trace["reasoning_notes"])
    assert "child task(s) not explicitly absorbed" in result
    assert calls["count"] == 4


def test_run_llm_loop_does_not_include_current_subagent_in_own_handoff(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolRegistry

    write_task_result(
        tmp_path,
        "child1",
        STATUS_RUNNING,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        role="reviewer",
        result="my own running mirror",
    )
    messages = [{"role": "user", "content": "inspect"}]
    calls = {"count": 0}
    progress = []
    tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    tools._ctx.task_metadata = {
        "parent_task_id": "parent1",
        "root_task_id": "parent1",
        "delegation_role": "subagent",
    }

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, _request_messages, *_args, **_kwargs):
        calls["count"] += 1
        return {"role": "assistant", "content": "subagent final"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=tools,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=progress.append,
        incoming_messages=queue.Queue(),
        task_id="child1",
        drive_root=tmp_path,
    )

    assert result == "subagent final"
    assert calls["count"] == 1
    assert not any("Subagent handoff status refreshed" in item for item in progress)
    assert not any("Subagent handoff status refreshed" in item for item in trace["reasoning_notes"])
