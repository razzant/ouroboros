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
import threading
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop_mod
from ouroboros.loop_llm_call import RETRY_WALL_EXHAUSTED_KEY
from ouroboros.loop import (
    _drain_incoming_messages,
    _initialize_owner_directives,
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


def _seed_acceptance_root(tmp_path, task_id: str, ctx: SimpleNamespace):
    """Mirror the production pre-loop canonical RUNNING admission."""
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    contract = build_task_contract({
        "id": task_id,
        "root_task_id": task_id,
        "delegation_role": "root",
    })
    write_task_result(
        tmp_path,
        task_id,
        STATUS_RUNNING,
        root_task_id=task_id,
        delegation_role="root",
        task_contract=contract,
        result="Task is running.",
    )
    metadata = getattr(ctx, "task_metadata", {})
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata.update({"root_task_id": task_id, "budget_drive_root": str(tmp_path)})
    ctx.task_id = task_id
    ctx.root_task_id = task_id
    ctx.delegation_role = "root"
    ctx.task_metadata = metadata
    ctx.task_contract = contract
    return contract


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
    from ouroboros import task_pacing
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.owner_mailbox import (
        KIND_FINALIZE_NOW,
        drain_owner_entries,
        write_owner_message,
    )

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
    control_entry = next(
        row for row in drain_owner_entries(tmp_path, "root")
        if row["msg_id"] == "control-1"
    )
    expected_deadline = (
        parse_deadline_ts(control_entry["ts"]).timestamp()
        + task_pacing.effective_finalization_reserve_sec(ctx)
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

    assert set(controls) == {"finalize_now", "finalize_deadline_ts"}
    assert controls["finalize_now"] == "deadline control"
    assert controls["finalize_deadline_ts"] == expected_deadline
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


def test_set_acceptance_decision_preserves_agent_stance():
    trace = {
        "acceptance_decision": {
            "agent_disposition": "rejected",
            "agent_rationale": "Scope drift.",
        }
    }
    _set_acceptance_decision(trace, {
        "status": "accepted",
        "reason": "no_actionable_changes",
        "source": "task_acceptance_review",
        "rationale": "No actionable changes.",
    })

    assert trace["acceptance_decision"]["status"] == "accepted"
    assert trace["acceptance_decision"]["reason"] == "no_actionable_changes"
    assert trace["acceptance_decision"]["agent_disposition"] == "rejected"
    assert trace["acceptance_decision"]["agent_rationale"] == "Scope drift."


def test_set_acceptance_decision_collapses_unknown_status_fail_closed():
    """v6.78.0 (P4.2): the merge point is the ONLY place a host acceptance status is
    minted, and it can only mint the canonical trio. A future writer that invents a
    fourth token gets `finalized_unaccepted` and its token survives as the reason —
    never a silent fourth owner-facing state, never a lost token."""
    from ouroboros.loop import ACCEPTANCE_DECISION_REASONS
    from ouroboros.outcomes import ACCEPTANCE_DECISION_STATUSES

    trace: dict = {}
    _set_acceptance_decision(trace, {"status": "some_future_state", "source": "x"})
    assert trace["acceptance_decision"]["status"] == "finalized_unaccepted"
    assert trace["acceptance_decision"]["reason"] == "some_future_state"

    _set_acceptance_decision(trace, {"status": "", "source": "x"})
    assert trace["acceptance_decision"]["status"] == "finalized_unaccepted"
    assert trace["acceptance_decision"]["reason"] == "unspecified"

    # Canonical status + typed reason passes through untouched.
    _set_acceptance_decision(trace, {"status": "accepted", "reason": "clean_pass"})
    assert trace["acceptance_decision"] == {"status": "accepted", "reason": "clean_pass"}
    assert ACCEPTANCE_DECISION_STATUSES == (
        "accepted", "revision_requested", "finalized_unaccepted",
    )
    assert "unspecified" in ACCEPTANCE_DECISION_REASONS


def test_every_host_acceptance_writer_emits_a_canonical_status_and_typed_reason():
    """Table-driven guard over the WHOLE writer inventory (v6.78.0): every
    `_set_acceptance_decision` call site must pass a canonical status constant and
    a reason from the closed set. Source-level so a new writer added without a
    reason fails here instead of silently shipping an untyped decision.

    Scans BOTH holders: the acceptance machinery moved into
    `acceptance_dialogue.py` while loop.py kept the rail-side writers, so reading
    only `loop_mod.__file__` would have gone quietly blind to most of them."""
    import pathlib
    import re

    import ouroboros.acceptance_dialogue as accept_mod
    from ouroboros.loop import ACCEPTANCE_DECISION_REASONS

    src = []
    for module in (loop_mod, accept_mod):
        src.extend(pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines())
    starts = [
        i for i, line in enumerate(src)
        if "_set_acceptance_decision(" in line and not line.lstrip().startswith("def ")
    ]
    # 8 rail-side writers in loop.py (owner follow-up, evidence refresh, delivery
    # binding, launch reserve, forced bypass, forced-rail revision, panel
    # infrastructure failure, and bounded fence exhaustion) + 12 in
    # acceptance_dialogue.py.
    assert len(starts) == 20, f"writer inventory changed: {len(starts)} call sites"
    allowed_status = {
        "ACCEPTANCE_ACCEPTED", "ACCEPTANCE_REVISION_REQUESTED",
        "ACCEPTANCE_FINALIZED_UNACCEPTED",
    }
    # The reason may be a literal OR an expression (a constant, or a conditional
    # picking between a constant and a literal). Both forms are checked: bare
    # literals against the closed set, and REASON_*/ACCEPTANCE_* names resolved
    # through the module that defines them.
    reason_names = {
        name: value for name, value in vars(accept_mod).items()
        if name.startswith(("REASON_", "ACCEPTANCE_REASON_")) and isinstance(value, str)
    }
    seen_expression_reasons = 0
    for start in starts:
        block = "\n".join(src[start:start + 30])
        status = re.findall(r'"status": ([A-Z_]+)', block)
        assert status and status[0] in allowed_status, f"line {start + 1}: {block[:120]}"
        assert '"reason"' in block, f"line {start + 1} has no typed reason"
        for reason in re.findall(r'"reason": "([a-z_]+)"', block):
            assert reason in ACCEPTANCE_DECISION_REASONS, reason
        for name in re.findall(r'\b(REASON_[A-Z_]+|ACCEPTANCE_REASON_[A-Z_]+)\b', block):
            if name not in reason_names:
                continue
            seen_expression_reasons += 1
            assert reason_names[name] in ACCEPTANCE_DECISION_REASONS, name
    # The widened regex really does catch expression-valued reasons: the two
    # `pass_reason if ... == REASON_REVIEW_CYCLES_EXHAUSTED` branches and the
    # A-material `REASON_IDENTICAL_ACCEPTANCE_REFUSED` writer.
    assert seen_expression_reasons >= 3, seen_expression_reasons


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


def test_root_acceptance_evidence_call_is_not_recorded_as_a_review_run():
    from ouroboros.loop_tool_execution import process_tool_results

    trace = {"tool_calls": []}
    payload = {
        "status": "deferred_to_host_acceptance",
        "authoritative": False,
        "evidence_revision": "a" * 64,
        "request": {"surface": "task_acceptance", "task_id": "root"},
        "evidence_refs": {"canonical_payload": {"sha256": "b" * 64}},
        "agent_decision": {
            "disposition": "accepted",
            "rationale": "Evidence is ready for the host panel.",
            "source": "agent_task_acceptance_review_tool",
        },
    }

    process_tool_results(
        [{
            "fn_name": "task_acceptance_review",
            "tool_call_id": "call-root",
            "result": json.dumps(payload),
            "is_error": False,
            "args_for_log": {},
            "tool_args": {},
            "result_meta": {"status": "ok"},
        }],
        [],
        trace,
        emit_progress=lambda _msg: None,
    )

    assert trace.get("review_runs") in (None, [])
    assert trace["acceptance_evidence_calls"] == [payload]
    assert trace["acceptance_decision"]["agent_disposition"] == "accepted"


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


def test_task_acceptance_agent_tool_is_advisory_before_auto_host_gate(monkeypatch, tmp_path):
    import ouroboros.review_substrate as rs

    trace = {
        "tool_calls": [
            {"tool": "write_file", "args": {"path": "x.py"}},
            {"tool": "run_command", "args": {"cmd": ["pytest"]}},
        ]
    }

    assert _task_acceptance_eligible("auto", trace, True) == (True, "auto_effect")
    assert _task_acceptance_eligible("required", trace, True)[0] is True
    assert _task_acceptance_eligible("off", trace, True)[0] is False

    clean = rs.ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"require_criterion_evidence": True}},
        actors=[{
            "signal": "PASS",
            "slot_id": "host-1",
            "parsed": {
                "outcome_tier": "solved",
                "completion_coach": "ship",
                "criteria_used": [{
                    "criterion": "owner request",
                    "status": "supported",
                    "evidence_refs": ["artifact:1"],
                }],
            },
        }],
        parsed_findings=[],
        aggregate_signal="PASS",
    )
    panel_state = {"calls": 0, "reviewed_at_dispatch": None}
    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "auto")
    monkeypatch.setattr(rs, "reviewer_slots", lambda **_kwargs: [object(), object(), object()])
    ctx = SimpleNamespace(
        _task_acceptance_reviewed=False,
        is_direct_chat=True,
        drive_root=str(tmp_path),
    )
    _seed_acceptance_root(tmp_path, "task1", ctx)

    def host_panel(*_args, **_kwargs):
        panel_state["calls"] += 1
        panel_state["reviewed_at_dispatch"] = ctx._task_acceptance_reviewed
        return clean

    monkeypatch.setattr(rs, "run_review_request", host_panel)
    reviewed_trace = {
        "tool_calls": [
            {"tool": "write_file", "args": {"path": "x.py"}},
            {"tool": "task_acceptance_review", "args": {}},
        ],
        "review_runs": [{"request": {"surface": "task_acceptance"}, "aggregate_signal": "PASS"}],
    }
    assert _run_task_acceptance_review_once(
        tools=SimpleNamespace(_ctx=ctx),
        content="done",
        task_id="task1",
        task_type="task",
        llm_trace=reviewed_trace,
        drive_root=tmp_path,
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": "goal"}],
        emit_progress=lambda _msg: None,
    ) is False
    assert panel_state == {"calls": 1, "reviewed_at_dispatch": False}
    assert ctx._task_acceptance_reviewed is True
    assert reviewed_trace["review_decision"]["trigger"] == "auto_effect_after_agent_advisory"
    assert len(reviewed_trace["review_runs"]) == 2
    assert reviewed_trace["review_runs"][0]["authority"] == "agent_advisory"
    assert reviewed_trace["review_runs"][0]["superseded_by_revision"] is True
    assert reviewed_trace["review_runs"][1]["authority"] == "host_root"

    # Defensive re-entry on the exact candidate/evidence/fence binding reapplies
    # the authoritative run but never pays for a second panel.
    ctx._task_acceptance_reviewed = False
    assert _run_task_acceptance_review_once(
        tools=SimpleNamespace(_ctx=ctx),
        content="done",
        task_id="task1",
        task_type="task",
        llm_trace=reviewed_trace,
        drive_root=tmp_path,
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": "goal"}],
        emit_progress=lambda _msg: None,
    ) is False
    assert panel_state["calls"] == 1
    assert reviewed_trace["review_decision"]["panel_reused"] is True
    assert len(reviewed_trace["review_runs"]) == 2


def _exercise_owner_followup_during_acceptance_panel(monkeypatch, tmp_path, *, direct: bool):
    import ouroboros.review_substrate as rs
    from ouroboros.owner_mailbox import drain_owner_entries
    from supervisor import events as events_mod
    from supervisor import queue as queue_mod

    root_id = "direct-root" if direct else "queued-root"
    chat_id = 17
    task = {
        "id": root_id,
        "type": "task",
        "chat_id": chat_id,
        "root_task_id": root_id,
        "delegation_role": "root",
        "drive_root": str(tmp_path),
    }
    pending = []
    running = {} if direct else {root_id: {"task": task}}
    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_mod, "QUEUE_SNAPSHOT_PATH", tmp_path / "state" / "queue_snapshot.json")
    monkeypatch.setattr(queue_mod, "PENDING", pending)
    monkeypatch.setattr(queue_mod, "RUNNING", running)
    monkeypatch.setattr(queue_mod, "ACCEPTANCE_FENCES", {})

    direct_agent = SimpleNamespace(
        _owner_message_admission_lock=threading.Lock(),
        _owner_message_generation=0,
        _busy=direct,
        _accepting_owner_messages=direct,
        _current_task_id=root_id if direct else "",
        _current_chat_id=chat_id,
        _current_task_metadata={},
    )
    token = ("a" if direct else "b") * 32

    def begin_fence(*, root_task_id, task_id):
        return queue_mod.transition_acceptance_fence(
            action="begin", token=token, root_task_id=root_task_id, task_id=task_id,
        )

    def inspect_fence(*, token):
        return queue_mod.transition_acceptance_fence(action="inspect", token=token)

    def end_fence(*, token, outcome, expected_generation=None):
        return queue_mod.transition_acceptance_fence(
            action="end", token=token, outcome=outcome,
            expected_generation=expected_generation,
        )

    acceptance_ctx = SimpleNamespace(
        _task_acceptance_reviewed=False,
        _task_acceptance_improvement_passes=0,
        is_direct_chat=direct,
        drive_root=str(tmp_path),
        task_id=root_id,
        task_metadata={"root_task_id": root_id},
        owner_message_admission_lock=direct_agent._owner_message_admission_lock,
        owner_message_admission_agent=direct_agent,
        begin_acceptance_fence=begin_fence,
        inspect_acceptance_fence=inspect_fence,
        end_acceptance_fence=end_fence,
    )
    task["task_contract"] = _seed_acceptance_root(tmp_path, root_id, acceptance_ctx)
    acknowledgements = []
    supervisor_ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING=running,
        PENDING=pending,
        get_chat_agent=lambda: direct_agent,
        persist_queue_snapshot=queue_mod.persist_queue_snapshot,
        bridge=SimpleNamespace(send_routing_ack=lambda *_a, **kw: acknowledgements.append(kw)),
    )
    clean = rs.ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"require_criterion_evidence": True}},
        actors=[{
            "signal": "PASS",
            "slot_id": "host-1",
            "parsed": {
                "outcome_tier": "solved",
                "completion_coach": "ship",
                "criteria_used": [{
                    "criterion": "owner request",
                    "status": "supported",
                    "evidence_refs": ["artifact:1"],
                }],
            },
        }],
        parsed_findings=[],
        aggregate_signal="PASS",
    )
    panel_calls = {"count": 0}

    def panel(*_args, **_kwargs):
        panel_calls["count"] += 1
        if panel_calls["count"] == 1:
            events_mod._handle_steer_task({
                "target_task_id": root_id,
                "message": "also satisfy the newly added criterion",
                "chat_id": chat_id,
                "client_message_id": f"owner-{root_id}",
            }, supervisor_ctx)
        return clean

    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "auto")
    monkeypatch.setattr(rs, "reviewer_slots", lambda **_kwargs: [object(), object(), object()])
    monkeypatch.setattr(rs, "run_review_request", panel)
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "goal"}]
    progress = []
    tools = SimpleNamespace(_ctx=acceptance_ctx)

    assert _run_task_acceptance_review_once(
        tools=tools, content="first answer", task_id=root_id, task_type="task",
        llm_trace=trace, drive_root=tmp_path, messages=messages, emit_progress=progress.append,
    ) is True
    assert acceptance_ctx._task_acceptance_reviewed is False
    assert root_id not in queue_mod.ACCEPTANCE_FENCES
    assert trace.get("root_phase_checkpoint") is None
    assert trace["review_runs"][0]["superseded_by_revision"] is True
    assert trace["review_runs"][0]["superseded_reason"] == "owner_followup_after_acceptance_evidence"
    assert trace["acceptance_decision"]["status"] == "revision_requested"
    assert (direct_agent._busy and direct_agent._accepting_owner_messages) if direct else root_id in running

    seen = set()
    _drain_incoming_messages(
        messages, queue.Queue(), tmp_path, root_id, None, seen, owner_ctx=acceptance_ctx,
    )
    assert "newly added criterion" in str(messages[-1]["content"])
    assert _run_task_acceptance_review_once(
        tools=tools, content="revised answer", task_id=root_id, task_type="task",
        llm_trace=trace, drive_root=tmp_path, messages=messages, emit_progress=progress.append,
    ) is False
    assert acceptance_ctx._task_acceptance_reviewed is True
    assert panel_calls["count"] == 2
    assert trace["review_runs"][-1].get("superseded_by_revision") is not True
    assert queue_mod.ACCEPTANCE_FENCES[root_id]["status"] == "sealed"
    assert not drain_owner_entries(tmp_path, root_id, seen_ids=seen)
    return queue_mod, events_mod, supervisor_ctx, acknowledgements, seen, root_id, chat_id


def test_direct_owner_followup_during_acceptance_panel_forces_fresh_review(monkeypatch, tmp_path):
    _exercise_owner_followup_during_acceptance_panel(monkeypatch, tmp_path, direct=True)


def test_queued_owner_followup_during_acceptance_panel_forces_fresh_review_and_sealed_rejects(
    monkeypatch, tmp_path,
):
    from ouroboros.owner_mailbox import drain_owner_entries

    queue_mod, events_mod, ctx, acknowledgements, seen, root_id, chat_id = (
        _exercise_owner_followup_during_acceptance_panel(monkeypatch, tmp_path, direct=False)
    )
    events_mod._handle_steer_task({
        "target_task_id": root_id,
        "message": "too late for the finalized run",
        "chat_id": chat_id,
        "client_message_id": "owner-after-seal",
    }, ctx)
    assert not drain_owner_entries(tmp_path, root_id, seen_ids=seen)
    assert acknowledgements[-1]["status"] == "needs_manual_target"
    assert queue_mod.ACCEPTANCE_FENCES[root_id]["status"] == "sealed"


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
                 "parsed": {"outcome_tier": "solved", "completion_coach": "ship it as-is",
                            "criteria_used": [{"criterion": "deliverable is verified",
                                               "status": "supported",
                                               "evidence_refs": ["verification_summary"]}]}}],
        parsed_findings=[], aggregate_signal="PASS",
    )
    monkeypatch.setattr(rs, "run_review_request", lambda *a, **k: solved)
    ctx = SimpleNamespace(_task_acceptance_reviewed=False, is_direct_chat=False, drive_root=str(tmp_path))
    _seed_acceptance_root(tmp_path, "t", ctx)
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
    _seed_acceptance_root(tmp_path, "t-blocked", ctx2)
    trace2 = {"tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    messages2 = [{"role": "system", "content": ""}, {"role": "user", "content": "goal"}]
    tools2 = SimpleNamespace(_ctx=ctx2)
    result2 = _run_task_acceptance_review_once(
        tools=tools2, content="done", task_id="t-blocked", task_type="task",
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
    # The pre-revision verdict remains authoritative until a replacement panel
    # result is ready; revision_requested alone must not erase it.
    assert trace2["review_runs"][0].get("superseded_by_revision") is not True

    monkeypatch.setattr(rs, "run_review_request", lambda *a, **k: solved)
    replacement = _run_task_acceptance_review_once(
        tools=tools2, content="revised", task_id="t-blocked", task_type="task",
        llm_trace=trace2, drive_root=None, messages=messages2, emit_progress=lambda _m: None,
    )
    assert replacement is False
    assert trace2["review_runs"][0]["superseded_by_revision"] is True
    assert trace2["review_runs"][0]["superseded_reason"] == "atomically_replaced_by_host_root_review"
    assert trace2["review_runs"][1]["authority"] == "host_root"
    tools2._ctx._task_acceptance_reviewed = False

    # If the revised answer is accepted, the terminal decision overwrites the
    # earlier revision_requested state rather than leaving stale telemetry.
    trace_ok = {"tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    messages_ok = [{"role": "system", "content": ""}, {"role": "user", "content": "goal"}]
    _seed_acceptance_root(tmp_path, "t-ok", tools2._ctx)
    result_ok = _run_task_acceptance_review_once(
        tools=tools2, content="revised", task_id="t-ok", task_type="task",
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
    _seed_acceptance_root(tmp_path, "t-blocked-alt", tools2._ctx)
    result3 = _run_task_acceptance_review_once(
        # A changed candidate creates a fresh binding; an unchanged candidate
        # must reuse the already-paid host panel under the v6.65 contract.
        tools=tools2, content="revised again", task_id="t-blocked-alt", task_type="task",
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
    _seed_acceptance_root(tmp_path, "t", ctx)
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
            {"tool": "write_file", "args": {"path": "SKILL.md", "bucket": "external", "skill_name": "delta"}},
        ]
    }

    assert _skill_names_touched_by_trace(trace) == ["alpha", "beta", "delta"]


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


def test_direct_final_admission_fence_consumes_followup_before_return(tmp_path, monkeypatch):
    import threading

    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.tools.registry import ToolRegistry

    class FakeLLM:
        def default_model(self):
            return "test-model"

    direct_agent = SimpleNamespace(
        _owner_message_admission_lock=threading.Lock(),
        _accepting_owner_messages=True,
        _busy=True,
        _current_task_id="direct-fence",
    )
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.owner_message_admission_lock = direct_agent._owner_message_admission_lock
    registry._ctx.owner_message_admission_agent = direct_agent
    calls = []

    def fake_call(_llm, request_messages, *_args, **_kwargs):
        calls.append([dict(row) for row in request_messages])
        if len(calls) == 1:
            write_owner_message(
                tmp_path,
                "Use FusionBrain images too",
                "direct-fence",
                msg_id="followup-1",
            )
            return {"role": "assistant", "content": "Initial draft"}, 0.0
        return {"role": "assistant", "content": "Revised with FusionBrain"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")

    result, _usage, _trace = run_llm_loop(
        messages=[{"role": "user", "content": "Build the AIRI report"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="direct-fence",
        drive_root=tmp_path,
    )

    assert result == "Revised with FusionBrain"
    assert len(calls) == 2
    assert any(
        row.get("role") == "user" and "FusionBrain" in str(row.get("content") or "")
        for row in calls[1]
    )
    assert direct_agent._accepting_owner_messages is False


def test_budget_rail_after_dispatch_is_terminal_without_provider_fallback(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.usage_accounting import AttemptRequest, BudgetExceeded, execute_physical_attempt

    class FakeLLM:
        def default_model(self):
            return "test-model"

    calls = {"primary": 0, "fallback": 0}

    def blocked(*_args, **_kwargs):
        calls["primary"] += 1
        raise BudgetExceeded(
            "root limit closed",
            limit_scope="root",
            root_task_id="budget-root",
        )

    def forbidden_fallback(**_kwargs):
        calls["fallback"] += 1
        raise AssertionError("budget rails must never enter model fallback")

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", blocked)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", forbidden_fallback)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    execute_physical_attempt(
        AttemptRequest(
            model="local/test",
            provider="local",
            drive_root=tmp_path,
            task_id="budget-task",
            root_task_id="budget-root",
        ),
        lambda: {"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    )
    events = queue.Queue()
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.root_task_id = "budget-root"
    registry._ctx.budget_drive_root = tmp_path

    result, usage, trace = run_llm_loop(
        messages=[{"role": "user", "content": "go"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        event_queue=events,
        task_id="budget-task",
        drive_root=tmp_path,
    )

    assert calls == {"primary": 1, "fallback": 0}
    assert result.startswith("🚫 Resource limit reached")
    assert usage["reason_code"] == "budget_exhausted"
    assert usage["resource_limit"] == trace["resource_limit"]
    assert usage["resource_limit"]["status"] == "resource_limited"
    assert usage["resource_limit"]["resume_policy"] == "cancel_or_new_run"
    checkpoint = events.get_nowait()["data"]
    assert checkpoint["checkpoint_kind"] == "budget_scope_paused"
    assert checkpoint["scope"] == "root"
    root_fence = events.get_nowait()
    assert root_fence["type"] == "budget_root_fence"
    assert root_fence["root_task_id"] == "budget-root"


def test_unknown_dispatched_outcome_skips_cross_model_fallback(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    class FakeLLM:
        def default_model(self):
            return "test-model"

    calls = {"primary": 0, "fallback": 0}

    def ambiguous(*args, **_kwargs):
        calls["primary"] += 1
        usage = args[10]
        usage["_last_llm_error"] = "provider outcome unknown"
        usage["_last_llm_error_kind"] = "provider_outcome_unknown"
        usage["_last_llm_retry_same_request"] = False
        return None, 0.0

    def forbidden_fallback(**_kwargs):
        calls["fallback"] += 1
        raise AssertionError("unknown physical work must stop the paid chain")

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", ambiguous)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", forbidden_fallback)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)

    result, usage, _trace = run_llm_loop(
        messages=[{"role": "user", "content": "go"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="unknown-provider-task",
        drive_root=tmp_path,
    )

    assert calls == {"primary": 1, "fallback": 0}
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert "no retry or paid fallback" in result


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
        return {
            "role": "assistant",
            "content": json.dumps({
                "delivery_control": "replace",
                "full_answer": "done after plan",
            }),
        }, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, trace, _progress):
        from ouroboros.task_results import STATUS_RUNNING, record_plan_review_wave, write_task_result

        fingerprint = "a" * 64
        write_task_result(tmp_path, "task1", STATUS_RUNNING, result="running")
        record_plan_review_wave(tmp_path, "task1", {
            "schema_version": 2, "cycle_index": 1, "request_fingerprint": fingerprint,
            "spec": {"goal": "g"}, "spec_hash": "b" * 64, "findings": [], "aggregate": "GREEN",
            "closed": True, "dispositions": [], "paid": True,
        })
        trace["tool_calls"].append({
            "tool": tool_calls[0]["function"]["name"],
            "args": {},
            "result": "## Plan Review Results\n\nAGGREGATE: GREEN",
            "is_error": False,
            "plan_review_outcome": "GREEN",
            "plan_review_closed": True,
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
    assert any("Call plan_task" in str(item.get("content") or "") for item in seen_second_request["messages"])
    assert trace["tool_calls"][0]["tool"] == "plan_task"


def test_force_plan_decision_does_not_treat_trace_marker_as_authority(tmp_path, monkeypatch):
    ctx = SimpleNamespace(
        task_metadata={"force_plan": True},
        is_ephemeral_turn=False,
        task_id="root1",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
    )
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")

    decision = loop_mod._force_plan_decision(ctx, {
        "tool_calls": [{
            "tool": "plan_task",
            "is_error": False,
            "plan_review_outcome": "GREEN",
            "plan_review_closed": True,
        }],
    })

    assert decision["allow"] is False
    assert decision["status"] == "absent"


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
        return {"role": "assistant", "content": "done despite unavailable plan"}, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, trace, _progress):
        from ouroboros.task_results import STATUS_RUNNING, record_plan_review_attempt, write_task_result

        write_task_result(tmp_path, "task1", STATUS_RUNNING, result="running")
        record_plan_review_attempt(tmp_path, "task1", fingerprint="d" * 64)
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

    assert result.startswith("done despite unavailable plan")
    assert "advisory enforcement" in result
    assert calls["count"] == 3
    assert usage.get("reason_code") != "swarm_force_plan_not_called"
    assert trace["tool_calls"][0]["tool"] == "plan_task"


def test_run_llm_loop_injects_subagent_handoff_before_final_text(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.registry import ToolRegistry
    from tests._delivery_candidate_shared import write_confirmed_disposition_fixture

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
        if calls["count"] == 2:
            write_confirmed_disposition_fixture(
                tmp_path,
                disposition="integrated",
                rationale="consumed in the final synthesis",
            )
        seen_second_request["messages"] = [dict(item) for item in request_messages]
        return {
            "role": "assistant",
            "content": '{"delivery_control":"replace","full_answer":"final after handoff"}',
        }, 0.0

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

    # This regression isolates the bounded handoff/orphan-note path; acceptance
    # quiescence has its own tests and would correctly wait for the running child.
    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "off")

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
        # The agent never absorbs/discards the child; it keeps through the
        # handoff control round, then answers the absorption reminder with
        # prose (the absorption round HOLDS the candidate — no JSON
        # instruction rides that round, so a typed keep is not requested).
        if calls["count"] == 1:
            content = "child1 is still running; I will finalize now."
        elif calls["count"] == 2:
            content = '{"delivery_control":"keep"}'
        else:
            content = "Best effort: child1 is still running."
        return {"role": "assistant", "content": content}, 0.0

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

    # Handoff, then one exact-disposition reminder, then honest forced best-effort.
    assert calls["count"] == 4
    assert sum(1 for item in progress if "Subagent handoff status refreshed" in item) == 1
    # The forced best-effort prose is preserved AND the loud orphan note is appended.
    assert result.startswith("Best effort: child1 is still running.")
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
    seen_requests = []
    progress = []
    tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    tools._ctx.task_contract = {"delegation_budget": {"may_delegate": True, "may_fan_out": True}}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, *_args, **_kwargs):
        calls["count"] += 1
        seen_requests.append([dict(m) for m in request_messages])
        # Call 2 answers the handoff control round with a typed keep; the
        # absorption round HOLDS the candidate (no JSON instruction), so the
        # reminder round is answered with prose like any ordinary round.
        content = (
            '{"delivery_control":"keep"}'
            if calls["count"] == 2
            else f"answer {calls['count']}"
        )
        return {"role": "assistant", "content": content}, 0.0

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
    # D2a: the absorption reminder round holds instead of arming — the new
    # messages of that round carry the reminder and NOT the JSON instruction
    # (the earlier handoff-armed instruction legitimately stays in history).
    absorption_round_delta = seen_requests[2][len(seen_requests[1]):]
    delta_text = "\n".join(
        str(m.get("content") or "") for m in absorption_round_delta
    )
    assert "[CHILD_ABSORPTION_REQUIRED]" in delta_text
    assert "[DELIVERY_FINALIZATION_CONTROL]" not in delta_text
    # D2b: the forced prompt names the CURRENT child listing, not a guess.
    forced_round_text = "\n".join(
        str(m.get("content") or "") for m in seen_requests[3]
    )
    assert "[FINALIZE_WITH_UNABSORBED_CHILDREN]" in forced_round_text
    assert "child1 [running]" in forced_round_text


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


def test_tool_results_carrying_auto_attach_image_get_the_image_same_round(tmp_path, monkeypatch):
    """A result whose JSON offers `auto_attach_image` (the unix_computer_use screenshot)
    must have its image attached in the SAME round, through the same implementation
    view_image uses — removing the mandatory second round per observation that consumed
    ~21% of the round budget on computer-use benches (v6.81.0 OSWorld: 3,830 view_image
    rounds after 3,893 screenshots). Failure is strictly non-fatal: a bad path must not
    turn a successful screenshot into a failed tool call."""
    import json as _json
    from types import SimpleNamespace

    from ouroboros.loop_tool_execution import process_tool_results

    attached = []

    def fake_attach(ctx, path):
        attached.append(path)
        ctx.messages.append({"role": "user", "content": [{"type": "image_url"}]})
        return True, "attached"

    import ouroboros.tools.vision as vision
    monkeypatch.setattr(vision, "attach_local_image_to_context", fake_attach)

    messages: list = []
    tools = SimpleNamespace(_ctx=SimpleNamespace(messages=messages, drive_root=str(tmp_path)))
    ok_result = _json.dumps({"ok": True, "path": "/x/shot.png",
                             "auto_attach_image": "/x/shot.png"})
    rows = [
        {"fn_name": "ext_1_r_unix_computer_use_screenshot", "tool_call_id": "c1",
         "result": ok_result, "is_error": False, "args_for_log": {}, "tool_args": {},
         "result_meta": {}},
        # An ERROR result never attaches, even if the field is present.
        {"fn_name": "ext_1_r_unix_computer_use_screenshot", "tool_call_id": "c2",
         "result": ok_result, "is_error": True, "args_for_log": {}, "tool_args": {},
         "result_meta": {}},
        # A result without the field never attaches.
        {"fn_name": "read_file", "tool_call_id": "c3",
         "result": "plain text", "is_error": False, "args_for_log": {}, "tool_args": {},
         "result_meta": {}},
    ]
    # An MCP-shaped result carrying the field must NOT attach: the capability is
    # defined for first-party extension tools; MCP results are untrusted
    # server-supplied data that must not drive automatic context mutation.
    rows.append({"fn_name": "mcp__someserver__screenshot", "tool_call_id": "c9",
                 "result": ok_result, "is_error": False, "args_for_log": {},
                 "tool_args": {}, "result_meta": {}})
    errors = process_tool_results(rows, messages, {"tool_calls": []},
                                  emit_progress=lambda _m: None, tools=tools)
    assert attached == ["/x/shot.png"], "exactly the opted-in successful result attaches"
    assert errors == 1
    # Ordering: in a multi-result round the image lands AFTER the round's complete
    # tool-message block, never between two tool messages answering one assistant
    # turn — contiguity by construction, not by transport repair.
    roles = [m["role"] for m in messages]
    first_image = roles.index("user")
    assert roles[:first_image] == ["tool"] * 4, roles
    # Attachment failure stays non-fatal and the tool result survives untouched.
    monkeypatch.setattr(vision, "attach_local_image_to_context",
                        lambda ctx, path: (_ for _ in ()).throw(RuntimeError("boom")))
    messages2: list = []
    tools2 = SimpleNamespace(_ctx=SimpleNamespace(messages=messages2, drive_root=str(tmp_path)))
    errors2 = process_tool_results(
        [dict(rows[0], tool_call_id="c4")], messages2, {"tool_calls": []},
        emit_progress=lambda _m: None, tools=tools2)
    assert errors2 == 0 and messages2[0]["role"] == "tool"
    # Legacy callers without `tools` keep exactly the old behavior.
    errors3 = process_tool_results(
        [dict(rows[0], tool_call_id="c5")], [], {"tool_calls": []},
        emit_progress=lambda _m: None)
    assert errors3 == 0


def test_a_tool_that_reports_its_own_failure_is_not_recorded_as_success():
    """Measured in the v6.81.1 OSWorld run: 329 tool calls returned `{"ok": false, ...}`
    in their JSON envelope and were recorded `is_error: false` / status "ok" — 302
    remote_exec, 20 screenshot, 5 key, 2 click. One agent killed the guest control
    server and then worked blind through 500-ing screenshots that all read as successes.
    The ⚠️-prefix convention only covers core-composed results; extension tools answer
    with JSON, so the failure has to be read from the payload."""
    from ouroboros.loop_tool_execution import (
        _extract_result_metadata,
        _is_tool_execution_failure,
        _structured_tool_failure,
    )

    fail = '{"ok": false, "error": "/screenshot failed: HTTPError: 500"}'
    ok = '{"ok": true, "path": "/x/shot.png"}'
    assert _structured_tool_failure(fail) is True
    assert _is_tool_execution_failure(True, fail) is True
    assert _extract_result_metadata("ext_1_r_x_screenshot", fail, False)["status"] == "tool_reported_failure"
    # Success and non-JSON prose are untouched.
    for benign in (ok, "plain text output", "", '["ok", false]', '{"ok": "false"}'):
        assert _structured_tool_failure(benign) is False, benign
        assert _is_tool_execution_failure(True, benign) is False, benign
    # A core ⚠️ result keeps its own typed status, not the new one.
    assert _extract_result_metadata("run_command", "⚠️ SHELL_EXIT_ERROR: 1", True)["status"] == "non_zero_exit"


def test_auto_attach_skips_a_result_that_declared_failure(tmp_path, monkeypatch):
    """A screenshot payload saying ok:false must not have an image lifted out of it."""
    import json as _json
    from types import SimpleNamespace

    from ouroboros.loop_tool_execution import _maybe_auto_attach_image

    attached = []
    import ouroboros.tools.vision as vision
    monkeypatch.setattr(vision, "attach_local_image_to_context",
                        lambda ctx, path: attached.append(path) or (True, "ok"))
    tools = SimpleNamespace(_ctx=SimpleNamespace(messages=[], drive_root=str(tmp_path)))
    failed = {"fn_name": "ext_1_r_unix_computer_use_screenshot", "is_error": False,
              "result": _json.dumps({"ok": False, "error": "boom",
                                     "auto_attach_image": "/x/shot.png"})}
    _maybe_auto_attach_image(failed, tools)
    assert attached == [], "an image was attached from a failed result"


def test_undecodable_image_fails_the_attach_not_the_provider_call():
    """A truncated PNG passes header checks; forwarding its bytes used to become a
    non-retryable provider 400 rounds later (5 task deaths in the v6.81.1 OSWorld
    run). The payload builder must raise at build time so the attach seam maps it
    to a tool-visible warning instead."""
    import io
    import pytest
    from PIL import Image

    from ouroboros.tools import vision

    buf = io.BytesIO()
    Image.new("RGB", (32, 16), (1, 2, 3)).save(buf, format="PNG")
    good = buf.getvalue()
    corrupt = good[:40] + b"\x00" * 400

    with pytest.raises(ValueError, match="IMAGE_UNDECODABLE"):
        vision._downscale_image_for_vlm(corrupt, "image/png")
    out, mime = vision._downscale_image_for_vlm(good, "image/png")
    assert out == good and mime == "image/png"


# ---------------------------------------------------------------------------
# OB-01 — provider death does not re-burn the budget on a second forced call
# ---------------------------------------------------------------------------


def _provider_death_ctx(tmp_path, accumulated):
    """A minimal forced-rail context. ``tools=None`` on purpose: every forced
    helper then takes its documented tool-less path, so the test observes the
    loop's own decisions instead of a registry stub's."""
    return loop_mod._RoundLimitContext(
        messages=[
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "PARTIAL RESULT: step one is done."},
        ],
        llm=SimpleNamespace(),
        active_model="openai/gpt-5.5",
        active_effort="medium",
        max_retries=3,
        drive_logs=tmp_path,
        task_id="task-provider-death",
        round_idx=7,
        event_queue=None,
        accumulated_usage=accumulated,
        task_type="task",
        active_use_local=False,
        max_rounds=200,
        drive_root=tmp_path,
    )


def _count_forced_helpers(monkeypatch):
    """Wrap the non-model work of the forced rail with call counters, keeping the
    REAL implementations so "it still runs" is observed, not simulated."""
    seen = {"model": [], "services": [], "drain": []}
    real_services = loop_mod._finalize_forced_services
    real_drain = loop_mod._drain_forced_owner_directives

    def _model(ctx):
        seen["model"].append(1)
        return "FRESH FORCED ANSWER"

    def _services(ctx, trace):
        seen["services"].append(1)
        return real_services(ctx, trace)

    def _drain(ctx, trace):
        seen["drain"].append(1)
        return real_drain(ctx, trace)

    monkeypatch.setattr(loop_mod, "_call_forced_model_once", _model)
    monkeypatch.setattr(loop_mod, "_finalize_forced_services", _services)
    monkeypatch.setattr(loop_mod, "_drain_forced_owner_directives", _drain)
    return seen


def test_provider_death_skips_the_forced_call_when_the_retry_wall_is_spent(tmp_path, monkeypatch):
    """The whole point of OB-01: the transport already spent the same-model retry
    wall, so the forced finalization must NOT re-burn the budget on a request that
    cannot land. Everything the forced rail owns besides the call still runs."""
    seen = _count_forced_helpers(monkeypatch)
    accumulated = {  # realistic transport stamps of an exhausted transient wall
        RETRY_WALL_EXHAUSTED_KEY: True,
        "execution_status": "infra_failed",
        "reason_code": "llm_api_error",
        "_last_llm_error_kind": "provider_transient",
    }
    ctx = _provider_death_ctx(tmp_path, accumulated)
    messages_before = [dict(m) for m in ctx.messages]

    text, usage, trace = loop_mod._handle_provider_unavailable(ctx)

    assert seen["model"] == []                    # ZERO llm calls from this rail
    assert len(seen["services"]) == 1             # services still finalized
    assert len(seen["drain"]) == 1                # exactly ONE directive drain
    # Salvage still delivered, and the delivery candidate still packaged.
    assert "PARTIAL RESULT: step one is done." in text
    assert trace["forced_finalization"]["reason_code"] == "provider_unavailable"
    # Replay durability: an unsent prompt must never reach the transcript, or a
    # resume would read it as a request the model ignored.
    assert [dict(m) for m in ctx.messages] == messages_before
    assert not any(
        "[PROVIDER_UNAVAILABLE]" in str(m.get("content") or "") for m in ctx.messages
    )
    # False completion: an outage is an INFRA FAILURE, and no model answer was
    # extracted, so the best_effort gate must not be handed a typed success fact.
    assert usage["reason_code"] == "provider_unavailable"
    assert usage["execution_status"] == "infra_failed"
    assert "_best_effort_extracted" not in usage


def test_provider_death_still_makes_the_forced_call_when_the_wall_is_unspent(tmp_path, monkeypatch):
    """The skip is not the new default. A PERMANENT refusal (auth/quota/bad
    request) fails fast and leaves the wall unspent, so the forced rail keeps the
    one chance its class is entitled to."""
    seen = _count_forced_helpers(monkeypatch)
    accumulated = {}  # no marker: the transport never exhausted its retries
    ctx = _provider_death_ctx(tmp_path, accumulated)

    text, usage, _trace = loop_mod._handle_provider_unavailable(ctx)

    assert seen["model"] == [1]
    assert "FRESH FORCED ANSWER" in text
    assert usage["execution_status"] == "infra_failed"
    assert any(
        "[PROVIDER_UNAVAILABLE]" in str(m.get("content") or "") for m in ctx.messages
    )


@pytest.mark.parametrize(
    "marker, expect_call",
    [
        ("unexpected-string", False),
        (1, False),
        ({"nested": "garbage"}, False),
        (0, True),
        ("", True),
        (None, True),
    ],
    ids=["truthy_str", "truthy_int", "truthy_dict", "zero", "empty_str", "none"],
)
def test_malformed_retry_wall_marker_reads_as_a_plain_truth_value(
    tmp_path, monkeypatch, marker, expect_call,
):
    """A shared usage dict can hold anything, so the read is `bool(...)` and
    nothing else: any truthy value means the wall is spent, any falsy value means
    it is not. No shape assumption, no crash, no silent third behaviour."""
    seen = _count_forced_helpers(monkeypatch)
    ctx = _provider_death_ctx(tmp_path, {
        RETRY_WALL_EXHAUSTED_KEY: marker,
        "execution_status": "infra_failed", "reason_code": "llm_api_error",
    })

    _text, usage, _trace = loop_mod._handle_provider_unavailable(ctx)

    assert seen["model"] == ([1] if expect_call else [])
    assert usage["execution_status"] == "infra_failed"


@pytest.mark.parametrize(
    "reason_code, marker",
    [
        ("round_limit", "[ROUND_LIMIT] wrap up"),
        ("finalization_grace", "[FINALIZE_NOW] wrap up"),
        ("deadline_local", "[DEADLINE] wrap up"),
        ("owner_requested_finalization", "[OWNER_STOP] wrap up"),
        ("budget_exhausted", "[BUDGET] wrap up"),
        ("children_unabsorbed", "[CHILDREN] wrap up"),
    ],
)
def test_other_forced_rails_never_skip_even_with_the_wall_marker_set(
    tmp_path, monkeypatch, reason_code, marker,
):
    """The skip is gated on the provider-death rail SPECIFICALLY. Every OTHER
    reason code `loop.py` passes to `_forced_final_answer` — round limit,
    finalization grace, deadline, owner stop, budget exhaustion and unabsorbed
    children — still makes its one forced call even when a transient wall was
    spent earlier in the same task: those rails end for their own reasons, not
    because the provider is unreachable. The list is exhaustive against the
    `reason_code=` literals in loop.py, so a new rail cannot silently inherit
    the skip."""
    seen = _count_forced_helpers(monkeypatch)
    ctx = _provider_death_ctx(tmp_path, {RETRY_WALL_EXHAUSTED_KEY: True})

    text, _usage, _trace = loop_mod._forced_final_answer(
        ctx, prompt=marker, fallback_text="fallback", reason_code=reason_code,
    )

    assert seen["model"] == [1]
    assert "FRESH FORCED ANSWER" in text
    assert any(marker in str(m.get("content") or "") for m in ctx.messages)
