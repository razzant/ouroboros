"""The operation that needs a finished run's product prepares it, without model polling (#1969).

A run whose engine terminal is observed may still owe its body collection (capture,
money). Apply/reject and Continue act on that execution fact, not on the money ledger.
A capture this call prepared, or one that finished while the call waited for the
shared lock, is returned for inspection and never applied in the same call; the next
apply decides that same retained capture. A capture ready before the call keeps the
ordinary contract: its caller decides it (no proof of reading is claimed). Live,
paused and failed preparations change nothing, and a pending ledger completes later.
"""
from __future__ import annotations

import json
import pathlib
import threading

import pytest

from ouroboros import delegate_collection as body
from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros.tools.subagent_integration import _integrate_delegated_patch
from tests import test_delegated_run_isolation as isolation
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _owned_gateway_uses_each_test_transport,
)
from tests.test_delegate_body_collection import terminal

INSPECT = "INTEGRATE_DELEGATED_INSPECT_CAPTURE"


@pytest.fixture
def money(monkeypatch):
    """Every subscription-ledger row the body writes."""
    from ouroboros import usage_accounting

    calls = []
    monkeypatch.setattr(usage_accounting, "record_subscription_session", lambda *a, **kw: calls.append((a, kw)))
    return calls


def _finished_unsettled(tmp_path, monkeypatch, *, state="succeeded", edit=True):
    """A real private snapshot whose run's terminal is OBSERVED; its body collection
    (capture, money) has not run, so custody is not settled."""
    target, ctx, handle = isolation.TestCaptureAndIntegrate()._provisioned(tmp_path, monkeypatch)
    if edit:
        (pathlib.Path(handle.path) / "tracked.txt").write_text("one\ntwo\nfinished work\n", encoding="utf-8")
    entry = isolation._isolated_entry(ctx, target, handle, settled=False)
    root = custody.custody_root(ctx)
    assert custody.record_started(root, entry)
    detail = terminal(entry.run_id)
    detail["summary"]["state"] = state
    assert body.observe_run(root, current.run(root, entry.run_id), detail)
    custody._CUSTODY.clear()  # every decision below reads the durable facts
    fresh = current.run(root, entry.run_id)
    assert fresh.terminal_state == state and not fresh.settled and not fresh.patch_captured
    return target, ctx, root, entry.run_id, handle


def _shown(text):
    assert text.startswith(f"⚠️ {INSPECT}") and "prepared during this call" in text, text
    return json.loads(text.split("\n", 1)[1])


def _rows(root, kind, rid):
    return [row for row in custody.custody_rows(root) if row.get("type") == kind and row.get("run_id") == rid]


@pytest.mark.parametrize("state", ["succeeded", "cancelled"])  # a Stop ends the run cancelled: its work is real
def test_apply_prepares_a_finished_runs_capture_returns_it_then_applies_that_capture(tmp_path, monkeypatch, money,
                                                                                     state):
    target, ctx, root, rid, _handle = _finished_unsettled(tmp_path, monkeypatch, state=state)
    try:
        shown = _shown(_integrate_delegated_patch(ctx, rid, "apply", "looks finished"))
        patch = pathlib.Path(shown["patch_artifact"])
        inspected = patch.read_bytes()
        assert shown["status"] == "ready_with_changes" and b"finished work" in inspected
        fresh = current.run(root, rid)
        assert fresh.patch_captured and not fresh.patch_disposed and not fresh.patch_apply_pending
        assert not _rows(root, custody.PATCH_APPLY_STARTED, rid) and not money  # nothing applied or paid yet
        assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"

        applied = _integrate_delegated_patch(ctx, rid, "apply", "inspected")
        assert "✅ Integrated" in applied, applied
        assert patch.read_bytes() == inspected  # the decision consumed the retained capture it showed
        assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\nfinished work\n"
        done = current.run(root, rid)
        assert done.patch_disposed == "applied" and not done.settled  # the ledger never held the result

        collected = body.collect_terminal(root, rid)  # the background body still owes only money
        assert collected["status"] == "collected" and len(money) == 1
        assert current.run(root, rid).settled and len(_rows(root, custody.PATCH_CAPTURED, rid)) == 1
    finally:
        custody._CUSTODY.clear()


def test_the_inspection_answer_is_no_error_and_no_integration_effect(tmp_path, monkeypatch):
    from ouroboros._outcome_tool_errors import _classify_tool_errors
    from ouroboros.loop_tool_execution import _typed_execution_failure, _typed_result_metadata
    from ouroboros.outcomes import build_verification_ledger, reviewable_effect_projection
    from ouroboros.tools.registry import ToolRegistry

    # Either work kind may integrate; this test isolates the capture-inspection guard.
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    _target, ctx, _root, rid, _handle = _finished_unsettled(tmp_path, monkeypatch)
    try:
        registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
        registry.set_context(ctx)
        shown = registry.execute_result("integrate_delegated_patch", {"run_id": rid})  # the public tool boundary
        assert (shown.status, shown.code) == ("ok", "INTEGRATION_INSPECTION_REQUIRED") and INSPECT in shown.text
        # The execution adapter's own reading: the protocol step is no error, and its bucket
        # stays outside the success set, so nothing counts as integrated or as a failure.
        is_error = _typed_execution_failure(True, shown)
        meta = _typed_result_metadata("integrate_delegated_patch", shown.text, is_error, shown)
        assert (is_error, meta["status"]) == (False, "inspection_required")
        call = {"tool": "integrate_delegated_patch", "args": {"run_id": rid}, "is_error": is_error, **meta}
        trace = {"tool_calls": [call]}
        assert reviewable_effect_projection(trace) == []  # nothing was integrated
        assert not any(_classify_tool_errors(trace).values())  # no error, recovery or denial to account for
        ledger = build_verification_ledger(task={"id": "inspect"}, loop_outcome={"outcome_axes": {}},
                                           llm_trace=trace, artifact_bundle={})
        assert ledger["summary"]["has_failures"] is False
    finally:
        custody._CUSTODY.clear()


def test_a_capture_the_collector_made_while_the_call_waited_is_returned_not_applied(tmp_path, monkeypatch, money):
    from ouroboros.tools import delegate_integration

    target, ctx, root, rid, _handle = _finished_unsettled(tmp_path, monkeypatch)
    original = delegate_integration.capture_terminal_patch_for_drive
    capturing, release = threading.Event(), threading.Event()

    def slow_capture(drive, entry, **kwargs):
        capturing.set()
        assert release.wait(10)
        return original(drive, entry, **kwargs)

    monkeypatch.setattr(delegate_integration, "capture_terminal_patch_for_drive", slow_capture)
    results = {}
    collector = threading.Thread(target=lambda: results.update(collected=body.collect_terminal(root, rid)))
    try:
        collector.start()
        assert capturing.wait(10)  # the collector holds the disposition lock mid-capture
        caller = threading.Thread(target=lambda: results.update(apply=_integrate_delegated_patch(ctx, rid, "apply", "")))
        caller.start()
        caller.join(0.5)
        assert caller.is_alive()  # the operation waits for the preparation; it does not answer "retry later"
        release.set()
        collector.join(10)
        caller.join(30)
        assert results["collected"]["status"] == "collected" and len(money) == 1
        assert _shown(results["apply"])["status"] == "ready_with_changes"
        assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"
        assert not current.run(root, rid).patch_disposed
        assert "✅ Integrated" in _integrate_delegated_patch(ctx, rid, "apply", "")
    finally:
        release.set()
        custody._CUSTODY.clear()


def test_limitation_a_capture_finished_before_the_call_is_the_callers_to_decide(tmp_path, monkeypatch, money):
    """Honest limit, unchanged contract: the collector may finish a capture after the caller's last read
    showed it pending; a capture ready before the call is applied by that call without being returned."""
    from ouroboros.delegate_collection import capture_projection

    target, ctx, root, rid, _handle = _finished_unsettled(tmp_path, monkeypatch)
    try:
        assert capture_projection(root, current.run(root, rid))["status"] == "pending"  # the caller's last read
        assert body.collect_terminal(root, rid)["status"] == "collected"  # captured and settled meanwhile
        applied = _integrate_delegated_patch(ctx, rid, "apply", "")
        assert "✅ Integrated" in applied and INSPECT not in applied
        assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\nfinished work\n"
    finally:
        custody._CUSTODY.clear()


def test_a_prepared_capture_is_rejected_in_one_call_and_its_patch_retained(tmp_path, monkeypatch):
    target, ctx, root, rid, handle = _finished_unsettled(tmp_path, monkeypatch)
    try:
        rejected = _integrate_delegated_patch(ctx, rid, "reject", "unwanted")  # prepared here, never applied
        assert "🚫 Rejected" in rejected and current.run(root, rid).patch_disposed == "rejected"
        assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\n"
        assert not pathlib.Path(handle.path).exists()
        assert len(_rows(root, custody.PATCH_CAPTURED, rid)) == 1  # the captured patch stays retained
    finally:
        custody._CUSTODY.clear()


def test_a_prepared_capture_without_changes_closes_itself_in_one_call(tmp_path, monkeypatch):
    """An empty result prepared by the call closes itself (no_changes): there is nothing to inspect,
    apply or reject, so the call answers OK at once and the copy and its pins are gone."""
    from ouroboros.subagent_worktrees import find_execution_snapshot

    _target, ctx, root, rid, handle = _finished_unsettled(tmp_path, monkeypatch, edit=False)
    try:
        out = _integrate_delegated_patch(ctx, rid, "apply", "")
        assert out.startswith("OK:") and "no_changes" in out and INSPECT not in out, out
        assert current.run(root, rid).patch_disposed == custody.NO_CHANGES
        assert find_execution_snapshot(handle.snapshot_id) is None and not pathlib.Path(handle.path).exists()
    finally:
        custody._CUSTODY.clear()


def test_live_paused_and_capture_failures_stay_typed_and_change_nothing(tmp_path, monkeypatch):
    from ouroboros.tools import delegate_integration

    target, ctx, handle = isolation.TestCaptureAndIntegrate()._provisioned(tmp_path, monkeypatch)
    (pathlib.Path(handle.path) / "tracked.txt").write_text("in progress\n", encoding="utf-8")
    entry = isolation._isolated_entry(ctx, target, handle, settled=False)
    root = custody.custody_root(ctx)
    try:
        assert custody.record_started(root, entry)
        paused = {"summary": {"runId": entry.run_id, "state": "running"}, "lastSeq": 2,
                  "pendingInteractions": [{"interactionId": "q1", "questions": [{"id": "f", "question": "which?"}]}]}
        assert body.observe_run(root, current.run(root, entry.run_id), paused)  # waiting on its question
        custody._CUSTODY.clear()
        for decision in ("apply", "reject"):
            assert "INTEGRATE_DELEGATED_NOT_TERMINAL" in _integrate_delegated_patch(ctx, entry.run_id, decision, "")
        assert not current.run(root, entry.run_id).patch_captured  # a live copy is never captured or released

        assert body.observe_run(root, current.run(root, entry.run_id), terminal(entry.run_id, seq=5))
        custody._CUSTODY.clear()
        monkeypatch.setattr(delegate_integration, "capture_terminal_patch_for_drive",
                            lambda *_a, **_k: {"status": "failed", "note": "fixture capture failure"})
        failed = _integrate_delegated_patch(ctx, entry.run_id, "apply", "")
        assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in failed and "fixture capture failure" in failed
        fresh = current.run(root, entry.run_id)
        assert not fresh.patch_captured and not fresh.patch_disposed and pathlib.Path(handle.path).exists()
        assert not _rows(root, custody.PATCH_APPLY_STARTED, entry.run_id)
    finally:
        custody._CUSTODY.clear()


def test_an_investigation_capture_owes_no_apply_or_reject_on_any_read(tmp_path, monkeypatch):
    from ouroboros.delegate_collection import capture_projection
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from tests.test_investigation_custody import _real_investigation

    _target, drive, entry, _copy, _handle = _real_investigation(tmp_path, monkeypatch)
    try:
        owed = capture_projection(drive, entry)
        assert owed["status"] == "pending" and owed["patch_disposition"] == "not_applicable"
        unretained = capture_terminal_patch_for_drive(drive, entry, retain_investigation=False)
        retained = capture_terminal_patch_for_drive(drive, entry)
        reread = capture_projection(drive, entry)
        for block in (unretained, retained, reread):
            assert "integrate_delegated_patch" not in block["note"] and "never applied" in block["note"]
        assert retained["disposition"] == reread["disposition"] == custody.EVIDENCE_RETAINED
    finally:
        custody._CUSTODY.clear()


# ----------------------------------------------------------------------- continuation

def test_continue_from_a_finished_run_whose_money_is_pending_starts_in_its_snapshot(tmp_path, monkeypatch):
    from ouroboros import usage_accounting
    from tests.test_delegate_continuation import _start

    monkeypatch.setattr(usage_accounting, "record_subscription_session",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("ledger busy")))
    first, _payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-first")
    snapshot_root = pathlib.Path(first["execution"]["workspaceRoot"])
    (snapshot_root / "first.txt").write_text("partial work\n", encoding="utf-8")
    detail = terminal("run-first")
    detail["summary"]["state"] = "failed"
    collected = body.collect_terminal(tmp_path, "run-first", detail=detail)  # money fails: still unsettled
    assert collected["status"] == "pending" and not current.run(tmp_path, "run-first").settled
    custody._CUSTODY.clear()

    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-next",
                                           start_kwargs={"continue_from": "run-first"})
    assert payload["status"] == "started", payload
    assert len(calls) == 1 and request["continueFrom"] == "run-first"
    assert request["execution"]["workspaceRoot"] == str(snapshot_root)
    assert payload["continuation"]["workspace"] == "same_snapshot" and payload["continuation"]["cause"] == "failed"
    prior = current.run(tmp_path, "run-first")
    assert prior.superseded_by == "run-next" and not prior.settled  # accounting finishes separately


def test_continue_still_refuses_a_run_without_an_observed_terminal(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ouroboros.delegate_continuation import REFUSAL_SOURCE_NOT_TERMINAL, bind_continuation
    from tests.test_delegate_continuation import _start

    _first, _payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-live")
    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-second",
                                           start_kwargs={"continue_from": "run-live"})
    # The open run is still an actor slot (early admission) and, by itself, a live source (the floor).
    assert request is None and calls == [] and payload["status"] != "started"
    assert payload["reason"] in {"replacement_requires_settlement", REFUSAL_SOURCE_NOT_TERMINAL}
    authority = SimpleNamespace(access="workspace_write", mode="agent", isolation="live")
    assert bind_continuation(ctx, tmp_path, "run-live", actor={}, route=SimpleNamespace(route_id=""),
                             authority=authority, target_root="")[1] == REFUSAL_SOURCE_NOT_TERMINAL


@pytest.mark.parametrize("held", [True, False])
def test_a_continuation_hand_over_waits_for_a_busy_snapshot_lock(tmp_path, monkeypatch, held):
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from tests.test_delegate_continuation import _start

    _first, _payload, _ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-first")
    assert body.observe_run(tmp_path, current.run(tmp_path, "run-first"), terminal("run-first"))
    custody._CUSTODY.clear()
    lock = custody.disposition_lock_path(tmp_path, current.run(tmp_path, "run-first"))
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire_exclusive_file_lock(lock, timeout_sec=0)
    assert fd is not None
    timer = None if held else threading.Timer(0.5, release_exclusive_file_lock, (lock, fd))
    if held:  # bound the hand-over's own wait so a lock that stays held answers quickly
        from ouroboros import platform_layer
        real = platform_layer.acquire_exclusive_file_lock
        monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock",
                            lambda path, **kw: real(path, **{**kw, "timeout_sec": min(kw.get("timeout_sec", 0), 0.5)}))
    else:
        timer.start()
    try:
        request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-next",
                                               start_kwargs={"continue_from": "run-first"})
    finally:
        if timer is not None:
            timer.join()
        else:
            release_exclusive_file_lock(lock, fd)
    if held:  # a hand-over that cannot claim the snapshot starts nothing and says so
        assert request is None and calls == [] and payload["reason"] == "continuation_handover_busy"
        assert not current.run(tmp_path, "run-first").superseded_by
    else:
        assert payload["status"] == "started" and request["continueFrom"] == "run-first"


def test_a_retry_successor_holds_a_finished_predecessor_product_before_its_money(tmp_path, monkeypatch):
    """The confirmed retry successor's product authority keys on the observed terminal."""
    from types import SimpleNamespace

    from ouroboros.delegate_shared import retry_result_status

    monkeypatch.setattr("ouroboros.delegate_shared._confirmed_retry_chain", lambda *_a, **_k: ("t-old", "t-new"))
    run = custody.RunCustody(run_id="run-p", task_id="t-old", route_id="r")
    assert custody.record_started(tmp_path, run)
    reader = SimpleNamespace(task_id="t-new")
    assert retry_result_status(reader, tmp_path, "run-p")[0] == custody.FOREIGN  # still live
    assert body.observe_run(tmp_path, current.run(tmp_path, "run-p"), terminal("run-p"))
    custody._CUSTODY.clear()
    status, _entry, predecessor = retry_result_status(reader, tmp_path, "run-p")
    assert (status, predecessor) == (custody.OWNED, "t-old") and not current.run(tmp_path, "run-p").settled
