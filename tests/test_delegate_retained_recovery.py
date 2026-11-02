"""Retained terminal facts survive transport loss and keep the same reader duties."""
from __future__ import annotations

from pathlib import Path
import threading

import pytest

from tests._delegate_capture_child import capture_child  # noqa: F401

from ouroboros import delegate_collection as body
from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.tools.delegate_terminal_evidence import collect_and_deliver_terminal, retained_terminal_result
from ouroboros.tools.registry import ToolContext
from tests.test_delegate_body_collection import _until, money, started, terminal  # noqa: F401


class AbsentGateway:
    store_identity = "store-a"

    def __init__(self, *, handshake_error=False):
        self.reads = 0
        self.closed = False
        self.handshake_error = handshake_error

    def handshake(self):
        if self.handshake_error:
            raise ClaudexorUnavailable("daemon_unreachable", "controlled transport gap")

    def get_run(self, rid):
        self.reads += 1
        raise ClaudexorUnavailable("not_found", "run gone", status_code=404)

    def close(self):
        self.closed = True


def recover(root, entry, factory, surface):
    if surface == "task":
        return custody.reconcile_task_runs(root, entry.task_id, gateway_factory=factory)
    if surface == "exit":
        return custody.release_task_runs(root, entry.task_id)
    return custody.reconcile_orphaned_runs(root, running_task_ids=set(), gateway_factory=factory)


@pytest.mark.parametrize("surface", ["task", "orphan", "exit"])
@pytest.mark.parametrize("transport", ["absent", "factory", "handshake"])
@pytest.mark.parametrize("pending", [False, True])
def test_settled_local_capture_recovers_without_remote_run(tmp_path, monkeypatch, money, surface, transport, pending):  # noqa: F811
    from ouroboros import claudexor_daemon
    from ouroboros import delegate_capture_job
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools import delegate_integration
    from tests.test_delegated_run_isolation import TestCaptureAndIntegrate, _isolated_entry
    from tests.test_exit_does_not_admit import INVOCATION, _pending, _pending_ids

    target, ctx, handle = TestCaptureAndIntegrate()._provisioned(tmp_path, monkeypatch)
    execution = Path(handle.path)
    (execution / "tracked.txt").write_text("retained work\n", encoding="utf-8")
    entry = _isolated_entry(ctx, target, handle, settled=False)
    root = custody.custody_root(ctx)
    assert custody.record_started(root, entry)
    with monkeypatch.context() as temporary:
        def unavailable(*args, **kwargs):
            raise OSError("controlled capture outage")
        temporary.setattr(delegate_integration, "capture_terminal_patch_for_drive", unavailable)
        body.collect_terminal(root, entry.run_id, detail=terminal(entry.run_id))
    before = current.run(root, entry.run_id)
    assert before.settled and body.collection_pending(before)
    if pending:
        _pending(root, task_id=entry.task_id)
    gateway = AbsentGateway(handshake_error=transport == "handshake")
    connections = []
    def factory():
        connections.append(True)
        if transport == "factory":
            raise ClaudexorUnavailable("daemon_unreachable", "controlled factory gap")
        return gateway
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **kw: factory())
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway",
                        lambda **kw: pytest.fail("collection at exit cannot start a daemon"))
    monkeypatch.setattr(gateway, "start_run",
                        lambda *a, **kw: pytest.fail("collection cannot replay a pending start"), raising=False)
    outcomes = recover(root, entry, None, surface)  # public default must attach only
    assert not current.run(root, entry.run_id).patch_captured
    if pending:
        assert _pending_ids(root, entry.task_id) == [INVOCATION]
        retained = [row for row in outcomes if row.get("action") == "invocation_retained"]
        assert bool(retained) == (surface == "orphan")
    capture = delegate_capture_job._capture
    entered, release = threading.Event(), threading.Event()
    def gated_capture(*args, **kwargs):
        entered.set()
        assert release.wait(10), "child capture gate was not released"
        return capture(*args, **kwargs)
    monkeypatch.setattr(delegate_capture_job, "_capture", gated_capture)
    runtime = body.start_body_runtime(root, gateway_factory=factory)
    try:
        _until(entered.is_set)
        assert not current.run(root, entry.run_id).patch_captured
        release.set()
        _until(lambda: current.run(root, entry.run_id).patch_captured)
    finally:
        release.set()
        runtime.close(wait=True)
    after = current.run(root, entry.run_id)
    assert after.patch_captured and not body.collection_pending(after)
    assert after.terminal_source == before.terminal_source
    assert after.output_source == before.output_source and not after.output_consumed
    assert not after.patch_disposed and find_execution_snapshot(handle.snapshot_id)
    assert (target / "tracked.txt").read_text(encoding="utf-8") != "retained work\n"
    assert gateway.reads == 0 and len(money) == 1
    assert not connections and not gateway.closed, "local capture neither opens nor closes an unused gateway"
    assert not any(row["type"] == custody.CLOSED_ABSENT for row in custody.custody_rows(root))


def test_unchanged_local_debt_emits_no_reconciliation_storm(tmp_path, monkeypatch, money):  # noqa: F811
    from ouroboros.tools import delegate_integration
    calls = []
    def unavailable(*args, **kwargs):
        calls.append(True)
        raise OSError("same capture outage")
    monkeypatch.setattr(delegate_integration, "capture_terminal_patch_for_drive", unavailable)
    entry = started(tmp_path, execution_root=str(tmp_path / "copy"), snapshot_id="snap")
    body.collect_terminal(tmp_path, entry.run_id, detail=terminal())
    gateway = AbsentGateway()
    first = recover(tmp_path, entry, lambda: gateway, "orphan")
    assert first == []
    after_first = list(custody.custody_rows(tmp_path))
    for _ in range(3):
        result = recover(tmp_path, entry, lambda: gateway, "orphan")
        assert result == []
    assert len(calls) == 1, "control reconciliation leaves local collection to the body"
    assert list(custody.custody_rows(tmp_path)) == after_first
    assert current.run(tmp_path, entry.run_id).settled and len(money) == 1
    assert body.collection_pending(current.run(tmp_path, entry.run_id))
    assert gateway.reads == 0


@pytest.mark.parametrize("review", [False, True])
def test_unsettled_absence_keeps_existing_recovery_semantics(tmp_path, review):
    entry = started(tmp_path, daemon_store_id="store-a",
                    **({"source": "review_substrate:slot"} if review else {}))
    assert entry.review_owned == review
    gateway = AbsentGateway()
    result = recover(tmp_path, entry, lambda: gateway, "orphan")
    assert result[0]["action"] == "absent" and gateway.reads == 1
    assert gateway.closed, "an acquired observation client is still closed"
    assert len([row for row in custody.custody_rows(tmp_path) if row["type"] == custody.CLOSED_ABSENT]) == 1


def test_retained_preview_records_same_unread_duty_once_and_eof_clears_it(tmp_path, monkeypatch, money):  # noqa: F811
    from ouroboros import delegate_output
    from tests.test_delegate_retry_output import read_whole
    monkeypatch.setattr(delegate_output, "_READ_COVERAGE", {})
    entry = started(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    body.collect_terminal(tmp_path, entry.run_id, detail=terminal(text="x" * 400_000))
    before = current.run(tmp_path, entry.run_id)
    assert before.settled and not before.unread_disclosed and not before.output_consumed
    fast = retained_terminal_result(ctx, entry.run_id)
    assert fast and not fast["output_delivery"]["complete"]
    assert "result_not_collected" in fast
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.unread_disclosed and not fresh.output_consumed
    ordinary = collect_and_deliver_terminal(ctx, entry.run_id, terminal(text="x" * 400_000),
                                             body._shape(fresh), fresh, None)
    assert ordinary["result_not_collected"] == fast["result_not_collected"]
    assert ordinary["output_delivery"]["artifact"]["sha256"] == fast["output_delivery"]["artifact"]["sha256"]
    assert retained_terminal_result(ctx, entry.run_id)["result_not_collected"] == fast["result_not_collected"]
    rows = [row for row in custody.custody_rows(tmp_path) if row["type"] == custody.SETTLED_UNREAD]
    assert len(rows) == 1 and rows[0]["source"] == before.output_source
    read_whole(ctx, fast["output_delivery"]["artifact"])
    after_eof = retained_terminal_result(ctx, entry.run_id)
    assert after_eof["output_delivery"]["consumed"] and "result_not_collected" not in after_eof
    assert current.run(tmp_path, entry.run_id).output_consumed


@pytest.mark.parametrize("complete", [False, True])
def test_retained_inline_or_incomplete_source_does_not_claim_unread_full_result(tmp_path, money, complete):  # noqa: F811
    entry = started(tmp_path)
    detail = terminal(text="brief answer")
    if not complete:
        detail["primaryOutput"].update(bytes=400_000, truncated=True)
    body.collect_terminal(tmp_path, entry.run_id, detail=detail)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    payload = retained_terminal_result(ctx, entry.run_id, complete_only=complete)
    assert payload is not None and payload["output_delivery"]["complete"] == complete
    assert "result_not_collected" not in payload
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.output_complete == complete and not fresh.output_consumed and not fresh.unread_disclosed
    assert not any(row["type"] == custody.SETTLED_UNREAD for row in custody.custody_rows(tmp_path))
