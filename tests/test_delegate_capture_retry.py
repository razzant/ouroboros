"""Retry facts keep terminal money/output live without repeating unchanged captures."""
from __future__ import annotations

import errno
import json

import pytest

from ouroboros import delegate_capture_job as job, delegate_collection as body
from ouroboros import delegate_custody_current as current, maintenance
from ouroboros.delegate_capture_attempt import collection_due
from ouroboros.delegate_capture_job import request_or_result
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.tools.registry import ToolContext
from tests._delegate_capture_child import capture_child  # noqa: F401
from tests.test_delegate_body_collection import started, terminal
from tests.test_delegate_collection_recovery import driven_runtime  # noqa: F401
from tests.test_delegated_directory import DirectoryEngine


@pytest.mark.parametrize("failure", ["transport", "problem", "refusal", "permission", "sharing"])
def test_directory_capture_uses_typed_retry_facts(tmp_path, monkeypatch, failure):
    from ouroboros import claudexor_daemon

    root, target = tmp_path / "data", tmp_path / "folder"
    target.mkdir()
    entry = started(root, resource_ref={"workspace_kind": "directory", "strategy": "copy"},
                    target_root=str(target))
    engine = DirectoryEngine(target)
    error = ClaudexorUnavailable("daemon_unreachable", "same diagnostic", observation_timeout=True)
    if failure in {"problem", "refusal"}:
        error = ClaudexorUnavailable("engine_answer", "same diagnostic", status_code=503)
        error.problem = {"retryable": failure == "problem"}
    elif failure in {"permission", "sharing"}:
        error = PermissionError(errno.EACCES, "same diagnostic")
        if failure == "sharing":
            error.winerror = 32  # The native Windows sharing signal, distinct from access denied.
    unavailable = [True]
    def gateway():
        if unavailable[0]:
            raise error
        return engine
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", gateway)
    monkeypatch.setattr(maintenance.child, "request", lambda: None)
    # Keep this test on the actual job protocol, without the barrier fixture's transport.
    monkeypatch.setattr(job, "request_or_result", request_or_result)
    monkeypatch.setattr(job, "GENERATION", "child-one")
    body.collect_terminal(root, entry.run_id, detail=terminal(entry.run_id), background=True, generation="body-one")
    assert job.run(root, lambda: False)["items"] == 1
    body.collect_terminal(root, entry.run_id, background=True, generation="body-one")
    failed = current.run(root, entry.run_id)
    transient = failure in {"transport", "problem", "sharing"}
    assert failed.capture_job["retry"] == ("transient" if transient else "changed_input")
    assert current.collection_pending(failed) and not failed.patch_captured
    assert not collection_due(failed, generation="body-one")
    assert collection_due(failed, generation="body-two") is transient
    assert job.run(root, lambda: False)["items"] == 0
    assert not job.due(root, {}, {}, False), "failed capture alone cannot spawn a new child"
    unavailable[0] = False
    monkeypatch.setattr(job, "GENERATION", "child-two")
    assert job.run(root, lambda: False)["items"] == int(transient)
    if not transient:
        assert current.run(root, entry.run_id).capture_job == failed.capture_job
        body.collect_terminal(root, entry.run_id, background=True, generation="body-two", recapture=True)
        assert job.run(root, lambda: False)["items"] == 1
    body.collect_terminal(root, entry.run_id, background=True, generation="body-two")
    final = current.run(root, entry.run_id)
    assert final.patch_captured and not current.collection_pending(final)
    assert not engine.posts and not engine.applies and not engine.decisions
    assert not job.due(root, {}, {}, True)


@pytest.mark.parametrize("debt", ["money", "output"])
def test_body_collects_changed_terminal_but_unchanged_debt_does_not_poll(
        tmp_path, monkeypatch, driven_runtime, debt):  # noqa: F811
    from ouroboros import usage_accounting

    runtime, clock, tick = driven_runtime
    entry = started(tmp_path)
    detail, attempts, reads, broken = [terminal()], [], [], [True]
    class Engine:
        def get_run(self, rid, **kwargs):
            reads.append(rid)
            return detail[0]
        def close(self): pass
    runtime.gateway_factory = Engine
    original = body._retain_output
    def retain(*args, **kwargs):
        attempts.append(debt)
        if broken[0]:
            raise OSError(errno.EBUSY, "retention unavailable")
        return original(*args, **kwargs) if debt == "output" else None
    monkeypatch.setattr(body if debt == "output" else usage_accounting,
                        "_retain_output" if debt == "output" else "record_subscription_session", retain)
    body.observe_run(tmp_path, entry, detail[0])
    tick()
    assert attempts == [debt] and current.run(tmp_path, entry.run_id).collection_attempt["retry"] == "transient"
    broken[0] = False
    clock[0] = 10000
    tick()
    assert not reads and attempts == [debt], "elapsed time earns neither observation nor retention"
    detail[0] = terminal(text="new terminal facts", seq=5)
    assert body.observe_run(tmp_path, entry, detail[0])
    tick()
    tick()
    assert attempts == [debt, debt]
    final = current.run(tmp_path, entry.run_id)
    assert final.settled and final.output_complete and not current.collection_pending(final)
    before = len(reads)
    tick()
    assert len(reads) == before and attempts == [debt, debt]


@pytest.mark.parametrize("surface", ["delivery", "wait", "supervised_wait"])
def test_explicit_wait_retries_owed_money_with_retained_output(tmp_path, monkeypatch, surface):
    from ouroboros import delegate_supervision, usage_accounting
    from ouroboros.tools import delegate
    from ouroboros.tools.delegate_terminal_evidence import collect_and_deliver_terminal, retained_terminal_result

    entry = started(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    attempts = []
    def money(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError(errno.EBUSY, "ledger unavailable")
    monkeypatch.setattr(usage_accounting, "record_subscription_session", money)
    body.collect_terminal(tmp_path, entry.run_id, detail=terminal())
    first = current.run(tmp_path, entry.run_id)
    assert first.output_complete and not first.settled
    assert retained_terminal_result(ctx, entry.run_id)["settlement"]["settled"] is False
    assert len(attempts) == 1, "a plain retained read is not a retry request"
    if surface == "delivery":
        result = collect_and_deliver_terminal(ctx, entry.run_id, terminal(), body._shape(first), first, None)
    elif surface == "wait":
        result = json.loads(delegate._delegate_wait(ctx, entry.run_id))
    else:
        result = json.loads(delegate_supervision.supervised_wait(ctx, entry.run_id).text)
    final = current.run(tmp_path, entry.run_id)
    assert result["settlement"]["settled"] and final.settled and len(attempts) == 2
    assert final.output_source == first.output_source
    assert final.collection_attempt == first.collection_attempt, "money-only retry is not a capture attempt"
    assert retained_terminal_result(ctx, entry.run_id)["settlement"]["settled"]
    assert len(attempts) == 2


def test_busy_disposition_lock_opens_no_gateway_and_release_resumes_collection(
        tmp_path, driven_runtime):  # noqa: F811
    from ouroboros import delegate_custody as custody
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    runtime, clock, tick = driven_runtime
    entry = started(tmp_path)
    body.observe_run(tmp_path, entry, terminal())
    opened, factory = [], runtime.gateway_factory
    runtime.gateway_factory = lambda: opened.append(True) or factory()
    lock = custody.disposition_lock_path(tmp_path, entry)
    fd = acquire_exclusive_file_lock(lock, owner_aware_stale=True)
    assert fd is not None
    try:
        for moment in range(10):
            clock[0] = moment * 3000
            tick()
        assert not opened
        assert not current.run(tmp_path, entry.run_id).collection_attempt
    finally:
        release_exclusive_file_lock(lock, fd)
    tick()
    assert opened == [True]
    assert not current.collection_pending(current.run(tmp_path, entry.run_id))
    tick()
    assert opened == [True]
