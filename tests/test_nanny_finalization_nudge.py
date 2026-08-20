"""The nanny postcondition (owner decision, 2026-08-07).

A child dispatched onto the delegated substrate (executor=harness) that reaches
finalization with ZERO delegate_start calls silently unmade a substrate decision:
the wave-0 incident (task 21d1d220, 2026-08-06) burned $8.89 of metered opus
tokens under a dispatch that promised subscription execution, and only its own
prose admitted it. The obligation used to be a prompt note alone; the seam below
makes the FACT structural (one re-loop) while the decision stays the child's.
"""

from types import SimpleNamespace

from ouroboros.loop_nudges import _maybe_inject_finalization_nudges


def _run(ctx_obj, msgs, tool_calls):
    return _maybe_inject_finalization_nudges(
        SimpleNamespace(_ctx=ctx_obj), None or __import__("pathlib").Path("."), "t",
        {"reasoning_notes": [], "tool_calls": tool_calls}, "done", msgs, lambda *_: None,
    )


def test_harness_child_finalizing_without_delegation_gets_one_nudge():
    ctx = SimpleNamespace(_nanny_route_dispatched=True, _nanny_finalization_injected=False)
    msgs: list = []
    assert _run(ctx, msgs, []) is True
    assert any("NANNY_DID_NOT_DELEGATE" in m.get("content", "") for m in msgs)
    # One-shot: the latch suppresses a second injection — the child may still
    # finalize with a stated reason, never a hard gate on its judgment (P5).
    assert _run(ctx, [], []) is False


def test_nanny_nudge_stays_out_of_owner_chat_progress(tmp_path):
    # Owner decision (2026-08-15): the nudge reaches the model as a [SYSTEM
    # REMINDER] and the durable trace, but never emit_progress (chat ⚠️ lines).
    # Observability rides a compact typed task_checkpoint on events.jsonl.
    import json
    import pathlib

    drive_logs = tmp_path / "logs"
    drive_logs.mkdir()
    ctx = SimpleNamespace(
        _nanny_route_dispatched=True, _nanny_finalization_injected=False,
        event_queue=None, drive_logs=drive_logs,
    )
    msgs: list = []
    progress: list = []
    trace = {"reasoning_notes": [], "tool_calls": []}
    assert _maybe_inject_finalization_nudges(
        SimpleNamespace(_ctx=ctx), pathlib.Path("."), "t", trace, "done", msgs,
        progress.append,
    ) is True
    assert progress == []
    assert any("NANNY_DID_NOT_DELEGATE" in m.get("content", "") for m in msgs)
    assert any("NANNY_DID_NOT_DELEGATE" in n for n in trace["reasoning_notes"])
    events = [json.loads(line) for line in
              (drive_logs / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e.get("type") == "task_checkpoint"
               and e.get("checkpoint_kind") == "nanny_finalization_nudge"
               and e.get("nanny_code") == "NANNY_DID_NOT_DELEGATE"
               for e in events)


def _tools(ctx_obj, available):
    return SimpleNamespace(_ctx=ctx_obj, available_tools=lambda: list(available))


def _custody_drive(tmp_path):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _run_full(tools, drive, task_id, msgs, tool_calls):
    return _maybe_inject_finalization_nudges(
        tools, drive, task_id,
        {"reasoning_notes": [], "tool_calls": tool_calls}, "done", msgs, lambda *_: None,
    )


def test_no_nudge_when_delegate_verbs_are_policy_hidden(tmp_path):
    # F4a (2026-08-10 saga): a child whose toolset does not carry the delegate
    # verbs cannot "choose" to delegate — accusing it is false. No reminder.
    ctx = SimpleNamespace(_nanny_route_dispatched=True, _nanny_finalization_injected=False)
    msgs: list = []
    assert _run_full(_tools(ctx, ["read_file", "web_search"]),
                     _custody_drive(tmp_path), "t", msgs, []) is False
    assert not any("NANNY" in m.get("content", "") for m in msgs)


def test_failed_delegated_run_gets_the_truthful_reminder(tmp_path):
    # F4b: a delegated run that STARTED but FAILED is an attempted route. The
    # durable custody evidence (not the per-execution trace) proves it, and the
    # reminder speaks the truth instead of accusing of zero attempts.
    from ouroboros import delegate_custody as custody

    drive = _custody_drive(tmp_path)
    assert custody.emit(drive, custody.STARTED, {
        "run_id": "run-1", "task_id": "child-1", "route": "claude", "max_seconds": 300,
    })
    assert custody.emit(drive, custody.SETTLED, {
        "run_id": "run-1", "task_id": "child-1", "route": "claude",
        "model": "claude-opus-5", "state": "failed", "cost_usd": 0.0,
        "cost_final": True, "spend_disclosed": True, "spend_estimated": False,
    })
    ctx = SimpleNamespace(_nanny_route_dispatched=True, _nanny_finalization_injected=False)
    msgs: list = []
    assert _run_full(_tools(ctx, ["delegate_start", "delegate_wait"]),
                     drive, "child-1", msgs, []) is True
    joined = "\n".join(m.get("content", "") for m in msgs)
    assert "NANNY_DELEGATED_RUN_FAILED" in joined
    assert "failed" in joined
    assert "NANNY_DID_NOT_DELEGATE" not in joined


def test_succeeded_delegated_run_suppresses_the_reminder(tmp_path):
    # A run that succeeded in an EARLIER execution (continuation reset the trace)
    # is a kept substrate decision — no reminder at all.
    from ouroboros import delegate_custody as custody

    drive = _custody_drive(tmp_path)
    assert custody.emit(drive, custody.STARTED, {
        "run_id": "run-1", "task_id": "child-1", "route": "claude", "max_seconds": 300,
    })
    assert custody.emit(drive, custody.SETTLED, {
        "run_id": "run-1", "task_id": "child-1", "route": "claude",
        "model": "claude-opus-5", "state": "succeeded", "cost_usd": 0.0,
        "cost_final": True, "spend_disclosed": True, "spend_estimated": False,
    })
    ctx = SimpleNamespace(_nanny_route_dispatched=True, _nanny_finalization_injected=False)
    msgs: list = []
    assert _run_full(_tools(ctx, ["delegate_start", "delegate_wait"]),
                     drive, "child-1", msgs, []) is False
    assert not any("NANNY" in m.get("content", "") for m in msgs)


def test_failed_run_nudges_even_with_delegate_start_in_this_trace(tmp_path):
    # Triad finding on e84475f2 (the saga's exact shape, all inside ONE
    # execution): delegate → the run dies → finish by hand → finalize. The
    # trace CONTAINS delegate_start, so the old outer guard skipped the nudge
    # entirely and the failure was never spoken. Custody evidence must win.
    from ouroboros import delegate_custody as custody

    drive = _custody_drive(tmp_path)
    assert custody.emit(drive, custody.STARTED, {
        "run_id": "run-1", "task_id": "child-1", "route": "codex", "max_seconds": 300,
    })
    assert custody.emit(drive, custody.SETTLED, {
        "run_id": "run-1", "task_id": "child-1", "route": "codex",
        "model": "gpt-5.6-sol", "state": "failed", "cost_usd": 0.0,
        "cost_final": True, "spend_disclosed": True, "spend_estimated": False,
    })
    ctx = SimpleNamespace(_nanny_route_dispatched=True, _nanny_finalization_injected=False)
    msgs: list = []
    assert _run_full(_tools(ctx, ["delegate_start", "delegate_wait"]),
                     drive, "child-1", msgs,
                     [{"tool": "delegate_start", "args": {}}]) is True
    joined = "\n".join(m.get("content", "") for m in msgs)
    assert "NANNY_DELEGATED_RUN_FAILED" in joined
    assert "NANNY_DID_NOT_DELEGATE" not in joined


def _split_root_ctx(parent, child):
    return SimpleNamespace(
        _nanny_route_dispatched=True, _nanny_finalization_injected=False,
        task_metadata={"budget_drive_root": str(parent)}, drive_root=str(child),
    )


def test_split_root_nanny_reads_custody_from_the_canonical_root(tmp_path):
    # Split-root fix (2026-08-10 amendments): custody rows are WRITTEN to the
    # canonical (budget) root — delegate_custody.custody_root — while a live
    # subagent's loop passes its isolated CHILD drive as drive_root. The nanny
    # read must resolve the same root as the writes: a succeeded run suppresses
    # the nudge, a started-but-failed run yields the truthful failure message —
    # both with the child drive passed exactly as production passes it.
    from ouroboros import delegate_custody as custody
    from ouroboros.loop_nudges import _nanny_finalization_message

    parent, child = tmp_path / "parent", tmp_path / "child"
    for root in (parent, child):
        (root / "logs").mkdir(parents=True)

    # (a) succeeded delegated run, rows on the CANONICAL root via the write path
    ctx = _split_root_ctx(parent, child)
    root = custody.custody_root(ctx)
    assert root == parent.resolve()
    assert custody.emit(root, custody.STARTED, {
        "run_id": "run-ok", "task_id": "child-ok", "route": "claude", "max_seconds": 300,
    })
    assert custody.emit(root, custody.SETTLED, {
        "run_id": "run-ok", "task_id": "child-ok", "route": "claude",
        "model": "claude-opus-5", "state": "succeeded", "cost_usd": 0.0,
        "cost_final": True, "spend_disclosed": True, "spend_estimated": False,
    })
    tools = _tools(ctx, ["delegate_start", "delegate_wait"])
    assert _nanny_finalization_message(tools, child, "child-ok") == ""

    # (b) started-but-failed run: the truthful failure message, not blindness
    assert custody.emit(root, custody.STARTED, {
        "run_id": "run-dead", "task_id": "child-dead", "route": "codex", "max_seconds": 300,
    })
    assert custody.emit(root, custody.SETTLED, {
        "run_id": "run-dead", "task_id": "child-dead", "route": "codex",
        "model": "gpt-5.6-sol", "state": "failed", "cost_usd": 0.0,
        "cost_final": True, "spend_disclosed": True, "spend_estimated": False,
    })
    message = _nanny_finalization_message(tools, child, "child-dead")
    assert "NANNY_DELEGATED_RUN_FAILED" in message
    assert "NANNY_DID_NOT_DELEGATE" not in message


def _emit_started(drive, run_id, task_id):
    from ouroboros import delegate_custody as custody

    assert custody.emit(drive, custody.STARTED, {
        "run_id": run_id, "task_id": task_id, "route": "claude", "max_seconds": 300,
    })


def _emit_settled(drive, run_id, task_id, state):
    from ouroboros import delegate_custody as custody

    assert custody.emit(drive, custody.SETTLED, {
        "run_id": run_id, "task_id": task_id, "route": "claude",
        "model": "claude-opus-5", "state": state, "cost_usd": 0.0,
        "cost_final": True, "spend_disclosed": True, "spend_estimated": False,
    })


def test_pending_run_gets_the_wait_reminder_not_a_failure_accusation(tmp_path):
    # PENDING ≠ FAILED (sol review on b49f8192): a STARTED row with no settled
    # receipt may simply still be executing. The old message called it failed and
    # told the child to retry — a duplicate concurrent delegated run — while
    # finalizing over it is exactly the orphan-result failure mode. The reminder
    # points at delegate_wait and accuses nothing.
    from ouroboros.loop_nudges import _nanny_finalization_message

    drive = _custody_drive(tmp_path)
    _emit_started(drive, "run-1", "child-1")
    ctx = SimpleNamespace(_nanny_route_dispatched=True, _nanny_finalization_injected=False)
    message = _nanny_finalization_message(
        _tools(ctx, ["delegate_start", "delegate_wait"]), drive, "child-1")
    assert "NANNY_DELEGATED_RUN_PENDING" in message
    assert "delegate_wait" in message
    assert "NANNY_DELEGATED_RUN_FAILED" not in message
    assert "NANNY_DID_NOT_DELEGATE" not in message


def test_mixed_failed_and_pending_runs_prefer_the_pending_reminder(tmp_path):
    # With one dead sibling AND one still in flight, "retry" is the wrong
    # instruction: the pending reminder wins, the earlier failure rides along
    # as a fact instead of being dropped.
    from ouroboros.loop_nudges import _nanny_finalization_message

    drive = _custody_drive(tmp_path)
    _emit_started(drive, "run-dead", "child-1")
    _emit_settled(drive, "run-dead", "child-1", "failed")
    _emit_started(drive, "run-live", "child-1")
    ctx = SimpleNamespace(_nanny_route_dispatched=True, _nanny_finalization_injected=False)
    message = _nanny_finalization_message(
        _tools(ctx, ["delegate_start", "delegate_wait"]), drive, "child-1")
    assert "NANNY_DELEGATED_RUN_PENDING" in message
    assert "failed" in message                     # the earlier death is still named
    assert "NANNY_DELEGATED_RUN_FAILED" not in message


def test_all_failed_runs_keep_the_failure_message(tmp_path):
    # All settled, none succeeded: the terminal non-success message stays.
    from ouroboros.loop_nudges import _nanny_finalization_message

    drive = _custody_drive(tmp_path)
    for run_id in ("run-1", "run-2"):
        _emit_started(drive, run_id, "child-1")
        _emit_settled(drive, run_id, "child-1", "failed")
    ctx = SimpleNamespace(_nanny_route_dispatched=True, _nanny_finalization_injected=False)
    message = _nanny_finalization_message(
        _tools(ctx, ["delegate_start", "delegate_wait"]), drive, "child-1")
    assert "NANNY_DELEGATED_RUN_FAILED" in message
    assert "NANNY_DELEGATED_RUN_PENDING" not in message


def test_closed_absent_run_counts_as_settled_not_pending(tmp_path):
    # A run the daemon says it does not have closed custody terminally without a
    # settlement row; the evidence must not read it as "still executing" forever.
    from ouroboros import delegate_custody as custody

    drive = _custody_drive(tmp_path)
    _emit_started(drive, "run-1", "child-1")
    assert custody.emit(drive, custody.CLOSED_ABSENT, {
        "run_id": "run-1", "task_id": "child-1", "route": "claude",
        "project_id": "", "reason": "reconcile_absent",
    })
    evidence = custody.task_execution_evidence(drive, "child-1")
    assert evidence["delegated_runs_settled"] == 1
    assert evidence["delegated_runs_succeeded"] == 0
    assert "closed_absent" in evidence["delegated_run_failure_states"]
    assert evidence["subscription_cost_usd"] is None  # spend undisclosed, never zero


def test_a_delegating_nanny_and_a_native_child_are_not_nudged():
    # A delegate_start in the trace with NO custody row yet (pending settlement
    # or an uncustodied start) is an attempt, not a choice — no accusation, and
    # the failure case is owned by custody evidence (see the test above).
    delegating = SimpleNamespace(_nanny_route_dispatched=True,
                                 _nanny_finalization_injected=False)
    assert _run(delegating, [], [{"tool": "delegate_start", "args": {}}]) is False

    native = SimpleNamespace(_nanny_route_dispatched=False,
                             _nanny_finalization_injected=False)
    assert _run(native, [], []) is False

    undispatched = SimpleNamespace()  # a ctx that never saw a dispatch at all
    assert _run(undispatched, [], []) is False


def _forced_run(tmp_path, nanny, tool_calls):
    import pathlib
    from unittest.mock import patch

    from ouroboros.loop_forced_finalization import _forced_final_answer
    from ouroboros.loop_round_limits import _RoundLimitContext

    class _Ctx:
        pass

    tools = SimpleNamespace(_ctx=_Ctx())
    tools._ctx._nanny_route_dispatched = nanny
    tools._ctx.drive_root = tmp_path
    tools._ctx.task_id = "t"
    messages: list = []
    ctx = _RoundLimitContext(
        messages=messages, llm=None, active_model="m", active_effort="low",
        max_retries=0, drive_logs=pathlib.Path("."), task_id="t", round_idx=1,
        event_queue=None, accumulated_usage={}, task_type="task",
        active_use_local=False, max_rounds=1,
    )
    ctx.tools = tools
    ctx.llm_trace = {"reasoning_notes": [], "tool_calls": tool_calls}
    with patch("ouroboros.loop_forced_finalization._call_forced_model_once", return_value="done"), \
         patch("ouroboros.loop._finalize_forced_services"), \
         patch("ouroboros.loop._forced_swarm_router_result", return_value=None), \
         patch("ouroboros.loop_forced_finalization._drain_forced_owner_directives", return_value=False):
        _forced_final_answer(ctx, prompt="wrap up", fallback_text="fb",
                             reason_code="round_limit")
    return "\n".join(m.get("content", "") for m in messages)


def test_forced_finalization_carries_the_nanny_note_instead_of_relooping(tmp_path):
    """Forced finalization may not re-loop (that is its whole point), so the
    substrate fact rides the one final prompt instead: a harness-dispatched child
    that made zero delegate_start calls sees the note and can state why."""
    assert "delegated substrate" in _forced_run(tmp_path, True, [])
    assert "delegated substrate" not in _forced_run(tmp_path, False, [])
    assert "delegated substrate" not in _forced_run(
        tmp_path, True, [{"tool": "delegate_start"}],
    )


def _emit_custody(tmp_path, kind, **payload):
    from ouroboros import delegate_custody as dc

    assert dc.emit(tmp_path, kind, {"task_id": "t", **payload})


def test_forced_note_consults_durable_custody_evidence(tmp_path):
    """The forced-path note is grounded in delegate_custody evidence on the
    custody root, not just the current trace: succeeded runs silence the note,
    unsettled runs get pending wording (no retry pressure), settled-without-
    success gets truthful failure wording."""
    from ouroboros import delegate_custody as dc

    # A SUCCEEDED run from an earlier execution: no note, no nag.
    _emit_custody(tmp_path, dc.STARTED, run_id="r1")
    _emit_custody(tmp_path, dc.SETTLED, run_id="r1", state="succeeded")
    out = _forced_run(tmp_path, True, [])
    assert "delegated substrate" not in out and "NOTE:" not in out

    # A started-but-unsettled run: pending wording, never "made no calls".
    pending_root = tmp_path / "pending"
    _emit_custody(pending_root, dc.STARTED, run_id="r2")
    out = _forced_run(pending_root, True, [])
    assert "not settled yet" in out and "made no delegate_start" not in out

    # A run that settled WITHOUT success (crashed in an earlier execution):
    # truthful failure wording instead of the false "made no calls" accusation.
    failed_root = tmp_path / "failed"
    _emit_custody(failed_root, dc.STARTED, run_id="r3")
    _emit_custody(failed_root, dc.SETTLED, run_id="r3", state="failed")
    out = _forced_run(failed_root, True, [])
    assert "settled WITHOUT success" in out and "made no delegate_start" not in out


def test_forced_note_never_accuses_over_unreadable_evidence(tmp_path):
    """An unreadable custody log must not be misread as 'zero runs': no note."""
    import os
    import platform

    import pytest

    from ouroboros import delegate_custody as dc

    if platform.system() == "Windows":
        pytest.skip("chmod-based permission test not portable to Windows")
    log_path = dc.event_log_path(tmp_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    os.chmod(log_path, 0)
    if os.geteuid() == 0:  # pragma: no cover — only hit in root CI
        pytest.skip("root user bypasses 0o000 chmod, cannot trigger OSError")
    try:
        assert "NOTE:" not in _forced_run(tmp_path, True, [])
    finally:
        os.chmod(log_path, 0o644)
