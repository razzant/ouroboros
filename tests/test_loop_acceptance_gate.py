"""The task-acceptance gate of ``ouroboros.loop``.

Split out of ``tests/test_loop_misc.py`` when that module was divided by
theme; every moved block is verbatim. Covers the `_set_acceptance_decision`
merge point and its writer inventory, the agent-advisory acceptance tool
seam, the host acceptance panel in auto and required modes, owner follow-ups
racing the panel, and the commit-evidence gate.
"""
from __future__ import annotations

import json
import queue
import threading
from types import SimpleNamespace

import ouroboros.loop as loop_mod
from ouroboros.loop_acceptance import _set_acceptance_decision, _task_acceptance_eligible
from ouroboros.loop_acceptance_review import _run_task_acceptance_review_once
from ouroboros.loop_round_limits import _drain_incoming_messages


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
    `_set_acceptance_decision` call site in loop.py must pass a canonical status
    constant and a reason from the closed set. Source-level so a new writer added
    without a reason fails here instead of silently shipping an untyped decision."""
    import pathlib
    import re

    from ouroboros.loop_acceptance import ACCEPTANCE_DECISION_REASONS

    # The v7 L-B split spread the writers over loop.py and its leaves; the
    # inventory below is the union over the whole loop family, so a writer
    # cannot escape the guard by living in (or moving to) a leaf.
    loop_file = pathlib.Path(loop_mod.__file__)
    src = []
    for path in [loop_file, *sorted(loop_file.parent.glob("loop_*.py"))]:
        src.extend(path.read_text(encoding="utf-8").splitlines())
    starts = [
        i for i, line in enumerate(src)
        if "_set_acceptance_decision(" in line and not line.lstrip().startswith("def ")
    ]
    # 19th writer (F6 upstream sync): the A-material identical-acceptance
    # refusal joins the forced-rail bypass recorder and the forced
    # children_unabsorbed terminalizer.
    assert len(starts) == 19, f"writer inventory changed: {len(starts)} call sites"
    allowed_status = {
        "ACCEPTANCE_ACCEPTED", "ACCEPTANCE_REVISION_REQUESTED",
        "ACCEPTANCE_FINALIZED_UNACCEPTED",
    }
    # The reason may be a literal OR an expression (a constant, or a conditional
    # picking between a constant and a literal). Both forms are checked: bare
    # literals against the closed set, and REASON_*/ACCEPTANCE_* names resolved
    # through the modules that define them.
    import ouroboros.loop_acceptance as _accept_mod
    import ouroboros.loop_acceptance_review as _accept_review_mod
    from ouroboros import outcomes as _outcomes_mod
    reason_names = {}
    for module in (_accept_mod, _accept_review_mod, _outcomes_mod):
        reason_names.update({
            name: value for name, value in vars(module).items()
            if name.startswith(("REASON_", "ACCEPTANCE_REASON_")) and isinstance(value, str)
        })
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
        emit_progress=lambda _msg, *, incident=None: None,
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
        emit_progress=lambda _msg, *, incident=None: None,
    )

    assert trace.get("review_runs") in (None, [])
    assert trace["acceptance_evidence_calls"] == [payload]
    assert trace["acceptance_decision"]["agent_disposition"] == "accepted"

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
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **_kwargs: [object(), object(), object()])
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
        emit_progress=lambda _msg, *, incident=None: None,
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
        emit_progress=lambda _msg, *, incident=None: None,
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
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **_kwargs: [object(), object(), object()])
    monkeypatch.setattr(rs, "run_review_request", panel)
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "x.py"}}]}
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "goal"}]
    progress = []
    tools = SimpleNamespace(_ctx=acceptance_ctx)

    assert _run_task_acceptance_review_once(
        tools=tools, content="first answer", task_id=root_id, task_type="task",
        llm_trace=trace, drive_root=tmp_path, messages=messages, emit_progress=lambda text, *, incident=None: progress.append(text),
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
        llm_trace=trace, drive_root=tmp_path, messages=messages, emit_progress=lambda text, *, incident=None: progress.append(text),
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
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **k: [object(), object(), object()])

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
        llm_trace=trace, drive_root=None, messages=messages, emit_progress=lambda _m, *, incident=None: None,
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
        llm_trace=trace2, drive_root=None, messages=messages2, emit_progress=lambda _m, *, incident=None: None,
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
        llm_trace=trace2, drive_root=None, messages=messages2, emit_progress=lambda _m, *, incident=None: None,
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
        llm_trace=trace_ok, drive_root=None, messages=messages_ok, emit_progress=lambda _m, *, incident=None: None,
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
        llm_trace=trace3, drive_root=None, messages=messages3, emit_progress=lambda _m, *, incident=None: None,
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
    monkeypatch.setattr(rs, "triad_delivery_slots", lambda **k: [object(), object(), object()])

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
        emit_progress=lambda _m, *, incident=None: None,
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
        emit_progress=lambda _m, *, incident=None: None,
    )
    assert captured["include_recent_commit"] is True
