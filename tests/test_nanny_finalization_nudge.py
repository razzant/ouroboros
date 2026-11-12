"""A nanny at the finish door (audit 11.11, item 18).

The door used to return charter and run-state reminders (CONFIGURED_ACTOR_*, NANNY_DELEGATED_RUN_*). Now
the one pre-delivery notice states this task's own delegated runs that have not settled, once, and nothing
about runs that ended; what ending does to an open run is said where its state is read (delegate_wait's
``owner_end``). A forced finalization cannot re-loop, so its one prompt still carries the substrate note
(the wave-0 incident, task 21d1d220, 2026-08-06: $8.89 of metered tokens under a dispatch that promised
subscription execution).
"""

from types import SimpleNamespace

from ouroboros.loop_nudges import _maybe_inject_finalization_nudges


def _tools(ctx_obj, available):
    return SimpleNamespace(_ctx=ctx_obj, available_tools=lambda: list(available))


def _custody_drive(tmp_path):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _run_full(tools, drive, task_id, msgs, tool_calls, progress=None):
    tools._ctx.drive_root = drive
    return _maybe_inject_finalization_nudges(
        tools, drive, task_id, {"reasoning_notes": [], "tool_calls": tool_calls}, "done", msgs,
        (progress.append if progress is not None else (lambda *_: None)),
    )


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
    ctx = SimpleNamespace(_nanny_route_dispatched=True)
    msgs: list = []
    assert _run_full(_tools(ctx, ["delegate_start", "delegate_wait"]),
                     drive, "child-1", msgs, []) is False
    assert msgs == []


def _split_root_ctx(parent, child):
    return SimpleNamespace(
        _nanny_route_dispatched=True,
        task_metadata={"budget_drive_root": str(parent)}, drive_root=str(child),
    )


def test_split_root_nanny_reads_custody_from_the_canonical_root(tmp_path):
    # Split-root fix (2026-08-10 amendments): custody rows are WRITTEN to the
    # canonical (budget) root — delegate_custody.custody_root — while a live
    # subagent's loop passes its isolated CHILD drive as drive_root. The finish
    # door reads the same root as the writes: a settled run is no open run, a
    # started one without settlement is — both with the child drive passed
    # exactly as production passes it.
    from ouroboros import delegate_custody as custody
    from ouroboros.loop_nudges import _open_delegated_runs

    parent, child = tmp_path / "parent", tmp_path / "child"
    for root in (parent, child):
        (root / "logs").mkdir(parents=True)
    root = custody.custody_root(_split_root_ctx(parent, child))
    assert root == parent.resolve()
    assert custody.emit(root, custody.STARTED, {
        "run_id": "run-ok", "task_id": "child-ok", "route": "claude", "max_seconds": 300,
    })
    assert custody.emit(root, custody.SETTLED, {
        "run_id": "run-ok", "task_id": "child-ok", "route": "claude",
        "model": "claude-opus-5", "state": "succeeded", "cost_usd": 0.0,
        "cost_final": True, "spend_disclosed": True, "spend_estimated": False,
    })
    assert custody.emit(root, custody.STARTED, {
        "run_id": "run-live", "task_id": "child-live", "route": "codex", "max_seconds": 300,
    })
    assert _run_full(_tools(_split_root_ctx(parent, child), ["delegate_start"]), child, "child-ok", [], []) is False
    live = _tools(_split_root_ctx(parent, child), ["delegate_start", "delegate_wait"])
    assert _run_full(live, child, "child-live", [], []) is True
    assert _open_delegated_runs(live._ctx, "child-live") == 1
    assert custody.task_execution_evidence(child, "child-live")["delegated_runs_started"] == 0, "no rows there"


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


def _forced_run(tmp_path, nanny, tool_calls):
    from unittest.mock import patch

    from ouroboros.loop import _RoundLimitContext, _forced_final_answer

    class _Ctx:
        pass

    tools = SimpleNamespace(_ctx=_Ctx())
    tools._ctx._nanny_route_dispatched = nanny
    tools._ctx.drive_root = tmp_path
    tools._ctx.task_id = "t"
    messages: list = []
    ctx = _RoundLimitContext(
        messages=messages, llm=None, active_model="m", active_effort="low",
        max_retries=0, drive_logs=tmp_path / "logs", task_id="t", round_idx=1,
        event_queue=None, accumulated_usage={}, task_type="task",
        active_use_local=False, max_rounds=1,
    )
    ctx.tools = tools
    ctx.llm_trace = {"reasoning_notes": [], "tool_calls": tool_calls}
    with patch("ouroboros.loop._call_forced_model_once", return_value=("done", {})), \
         patch("ouroboros.loop._finalize_forced_services"), \
         patch("ouroboros.loop._drain_forced_owner_directives", return_value=False):
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


def test_a_nanny_is_told_once_about_its_open_runs_and_nothing_about_runs_that_ended(tmp_path):
    """A nanny whose delegated runs all failed finishes on its first finish with no return. One with a run
    still open beside a failed sibling gets the open-runs line once (this task's runs, counted from durable
    custody), and its next finish is accepted."""
    from ouroboros.loop_nudges import _open_delegated_runs, _pre_delivery_notice

    ended = _custody_drive(tmp_path / "ended")
    for run_id in ("run-1", "run-2"):
        _emit_started(ended, run_id, "child-1")
        _emit_settled(ended, run_id, "child-1", "failed")
    finished = SimpleNamespace(_nanny_route_dispatched=True)
    msgs: list = []
    assert _run_full(_tools(finished, ["delegate_start", "delegate_wait"]), ended, "child-1", msgs, []) is False
    assert msgs == [] and not hasattr(finished, "_completion_hold_cause")

    live = _custody_drive(tmp_path / "live")
    _emit_started(live, "run-dead", "child-1")
    _emit_settled(live, "run-dead", "child-1", "failed")
    _emit_started(live, "run-live", "child-1")
    nanny, progress = SimpleNamespace(_nanny_route_dispatched=True), []
    tools = _tools(nanny, ["delegate_start", "delegate_wait"])
    assert _run_full(tools, live, "child-1", msgs, [], progress) is True
    notice = _pre_delivery_notice(tools, live, "child-1", {"tool_calls": []}, [])
    assert notice and _open_delegated_runs(nanny, "child-1") == 1
    assert msgs[-1]["content"] == f"[SYSTEM REMINDER]\n{notice}"
    assert nanny._completion_hold_cause == "pre_delivery_notice"
    assert progress == [], "a nanny's notice writes no line into the owner's chat (owner, 15.08)"
    assert _run_full(tools, live, "child-1", msgs, [], progress) is False, "the next finish is accepted"
    plain, shown = SimpleNamespace(), []
    assert _run_full(_tools(plain, ["delegate_start", "delegate_wait"]), live, "child-1", [], [], shown) is True
    assert shown == ["Pre-delivery facts shown before final response."], "an ordinary task's progress line stays"
