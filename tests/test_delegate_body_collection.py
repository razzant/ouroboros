"""The body records/collects a run while its semantic author is not executing."""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from tests._delegate_capture_child import capture_child  # noqa: F401

from ouroboros import delegate_collection as body
from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current


def started(root, rid="run-one", tid="task-one", **kwargs):
    entry = custody.RunCustody(run_id=rid, task_id=tid, invocation_id="inv-" + rid,
        route_id="test", model="test", access="readonly", mode="ask", ledger_root=str(root), **kwargs)
    assert custody.record_started(root, entry)
    return current.run(root, rid)


def terminal(rid="run-one", text="complete answer", seq=4):
    return {"summary": {"runId": rid, "state": "succeeded", "spendUsd": 0.25,
        "spendEstimated": False, "effectiveAccess": "readonly"}, "lastSeq": seq,
        "primaryOutput": {"text": text, "bytes": len(text.encode()), "truncated": False}}


@pytest.fixture
def money(monkeypatch):
    from ouroboros import usage_accounting
    calls = []
    monkeypatch.setattr(usage_accounting, "record_subscription_session", lambda *a, **kw: calls.append((a, kw)))
    return calls


def test_terminal_is_source_fact_before_money_and_survives_started_replay(tmp_path, money):
    entry = started(tmp_path, root_task_id="root-task", parent_task_id="parent-task")
    assert body.observe_run(tmp_path, entry, terminal())
    now = current.run(tmp_path, entry.run_id)
    assert now.terminal_state == "succeeded" and now.terminal_source
    assert not now.settled and not now.ledger_recorded and not money
    assert custody.record_started(tmp_path, entry)
    assert current.run(tmp_path, entry.run_id).terminal_source == now.terminal_source
    assert not body.observe_run(tmp_path, entry, {"summary": {"state": "running"}, "lastSeq": 3})
    assert current.run(tmp_path, entry.run_id).terminal_state == "succeeded"
    folded = current.fold(list(custody.custody_rows(tmp_path)))
    assert folded["runs"][entry.run_id][0]["terminal_source"] == now.terminal_source
    terminal_only = [row for row in custody.custody_rows(tmp_path) if row.get("type") == custody.TERMINAL_OBSERVED]
    reduced = current.fold(terminal_only)["runs"][entry.run_id][0]
    assert reduced["invocation_id"] == entry.invocation_id
    assert reduced["root_task_id"] == "root-task" and reduced["parent_task_id"] == "parent-task"


def test_background_retains_small_output_without_eof_or_retirement(tmp_path, money):
    entry = started(tmp_path, project_id="project", project_owned=True)
    result = body.collect_terminal(tmp_path, entry.run_id, detail=terminal())
    now = current.run(tmp_path, entry.run_id)
    assert result["status"] == "collected"
    assert now.settled and now.ledger_recorded and len(money) == 1
    assert now.output_source and now.output_complete and not now.output_consumed
    assert now.project_owned and not now.unread_disclosed
    assert body.retained_payload(tmp_path, now)["primary_output"]["text"] == "complete answer"
    body.collect_terminal(tmp_path, entry.run_id)
    assert len(money) == 1


def test_money_failure_does_not_erase_execution_or_output(tmp_path, monkeypatch):
    from ouroboros import usage_accounting
    monkeypatch.setattr(usage_accounting, "record_subscription_session",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("busy")))
    entry = started(tmp_path)
    result = body.collect_terminal(tmp_path, entry.run_id, detail=terminal())
    now = current.run(tmp_path, entry.run_id)
    assert result["status"] == "pending"
    assert now.terminal_state == "succeeded" and now.output_complete
    assert not now.ledger_recorded and not now.settled and not now.output_consumed
    assert entry.run_id in current.runs(tmp_path, "state=?", (current.HELD,))


def test_collection_reads_source_version_after_acquiring_the_result_lock(tmp_path, monkeypatch, money):
    from ouroboros import platform_layer
    entry = started(tmp_path)
    assert body.observe_run(tmp_path, entry, terminal(text="earlier answer"))
    acquire = platform_layer.acquire_exclusive_file_lock
    def refreshed(lock, **kwargs):
        assert body.observe_run(tmp_path, entry, terminal(text="latest answer", seq=5))
        return acquire(lock, **kwargs)
    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", refreshed)
    body.collect_terminal(tmp_path, entry.run_id)
    fresh = current.run(tmp_path, entry.run_id)
    assert body.retained_payload(tmp_path, fresh)["primary_output"]["text"] == "latest answer"
    assert fresh.output_terminal_sha == fresh.terminal_source["sha256"]


def test_question_observation_never_answers_and_terminal_clears_pending(tmp_path):
    entry = started(tmp_path)
    detail = {"summary": {"runId": entry.run_id, "state": "running"}, "lastSeq": 2,
              "pendingInteractions": [{"interactionId": "q1", "questions": [{"id": "file", "question": "which file?"}]}]}
    assert body.observe_run(tmp_path, entry, detail)
    now = current.run(tmp_path, entry.run_id)
    assert now.pending_interaction_ids == ["q1"] and now.interaction_source
    before = len(list(custody.custody_rows(tmp_path)))
    assert not body.observe_run(tmp_path, now, detail)
    assert len(list(custody.custody_rows(tmp_path))) == before
    assert body.observe_run(tmp_path, now, terminal(seq=3))
    assert current.run(tmp_path, entry.run_id).pending_interaction_ids == []


def test_foreign_observation_cannot_write_custody(tmp_path):
    entry = started(tmp_path)
    with pytest.raises(ValueError, match="another run"):
        body.observe_run(tmp_path, entry, terminal("run-foreign"))
    assert not current.run(tmp_path, entry.run_id).terminal_state


def test_collector_lock_is_a_quiet_join_not_an_integration_error(tmp_path, money):
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    entry = started(tmp_path)
    body.observe_run(tmp_path, entry, terminal())
    path = custody.disposition_lock_path(tmp_path, entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire_exclusive_file_lock(path, timeout_sec=0)
    assert fd is not None
    try:
        assert body.collect_terminal(tmp_path, entry.run_id) == {
            "status": "pending", "reason": "collection_in_progress"}
        assert not money
    finally:
        release_exclusive_file_lock(path, fd)
    assert body.collect_terminal(tmp_path, entry.run_id)["status"] == "collected"


def test_foreground_supervision_joins_collection_without_a_model_wake(tmp_path, monkeypatch, money):
    from ouroboros import delegate_supervision
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.tools.delegate_terminal_evidence import collect_and_deliver_terminal
    from ouroboros.tools.registry import ToolContext

    entry = started(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    path = custody.disposition_lock_path(tmp_path, entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire_exclusive_file_lock(path, timeout_sec=0)
    assert fd is not None
    monkeypatch.setattr(delegate_supervision, "_leaf_live_input", lambda *args: "unknown")
    monkeypatch.setattr(delegate_supervision, "_TICK_SEC", 0)
    observations = []
    def observed(*args, **kwargs):
        nonlocal fd
        if observations:
            release_exclusive_file_lock(path, fd)
            fd = None
        payload = collect_and_deliver_terminal(ctx, entry.run_id, terminal(), body._shape(entry), entry, None)
        observations.append(payload)
        return json.dumps(payload)
    try:
        result = delegate_supervision.supervised_wait(ctx, entry.run_id, wait_once=observed)
    finally:
        if fd is not None:
            release_exclusive_file_lock(path, fd)
    assert len(observations) == 2
    assert observations[0]["status"] == "observation_pending" and observations[0]["answered"]
    assert observations[0]["reason"] == "collection_in_progress"
    assert json.loads(result.text)["status"] == "terminal" and len(money) == 1


def test_foreground_supervision_wakes_on_terminal_custody_without_a_collector(tmp_path, monkeypatch):
    from ouroboros import delegate_supervision
    from ouroboros.tools.registry import ToolContext

    entry = started(tmp_path)
    assert body.observe_run(tmp_path, entry, terminal())
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    monkeypatch.setattr(delegate_supervision, "_leaf_live_input", lambda *args: "unknown")
    result = delegate_supervision.supervised_wait(ctx, entry.run_id, wait_once=lambda *_a, **_k:
        json.dumps({"status": "no_progress", "run_id": entry.run_id}))
    payload = json.loads(result.text)
    assert payload["status"] == "terminal_custody"
    assert payload["terminal_facts"] == [{"id": "run:" + entry.run_id,
                                          "reason": "run:" + entry.run_id, "state": "succeeded"}]


def test_collector_does_not_take_over_review_panel(tmp_path, money):
    entry = started(tmp_path, source="review_substrate_commit")
    assert entry.review_owned
    result = body.collect_terminal(tmp_path, entry.run_id, detail=terminal())
    assert result["reason"] == "collection_owner_mismatch" and not money
    assert not current.run(tmp_path, entry.run_id).terminal_state


@pytest.mark.parametrize("contended", [False, True])
def test_foreground_supervision_returns_review_collection_refusal(tmp_path, monkeypatch, money, contended):
    """A terminal execution cannot make a permanent collection refusal retryable."""
    from ouroboros import delegate_supervision
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.tools.delegate_terminal_evidence import collect_and_deliver_terminal
    from ouroboros.tools.registry import ToolContext

    entry = started(tmp_path, source="review_substrate_commit")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    monkeypatch.setattr(delegate_supervision, "_leaf_live_input", lambda *args: "unknown")
    monkeypatch.setattr(delegate_supervision, "_TICK_SEC", 0)
    observations = []

    def observed(*args, **kwargs):
        assert not observations, "permanent collection refusal was silently retried"
        payload = collect_and_deliver_terminal(ctx, entry.run_id, terminal(), body._shape(entry), entry, None)
        observations.append(payload)
        return json.dumps(payload)

    lock = custody.disposition_lock_path(tmp_path, entry)
    fd = None
    if contended:
        lock.parent.mkdir(parents=True, exist_ok=True)
        fd = acquire_exclusive_file_lock(lock, timeout_sec=0)
        assert fd is not None
    try:
        result = delegate_supervision.supervised_wait(ctx, entry.run_id, wait_once=observed)
    finally:
        if fd is not None:
            release_exclusive_file_lock(lock, fd)
    payload = json.loads(result.text)
    assert len(observations) == 1
    assert payload["status"] == "refused" and payload["ok"] is False
    assert result.code == payload["host_code"] == "TOOL_REPORTED_FAILURE"
    assert payload["reason"] == "collection_owner_mismatch"
    assert payload["collection"] == {"status": "unavailable", "reason": "collection_owner_mismatch"}
    assert payload["state"] == "succeeded" and payload["answered"] is True
    assert payload["run_id"] == entry.run_id and payload["last_seq"] == 4
    assert "review" in payload["detail"].lower()
    assert "primary_output" not in payload and "output_delivery" not in payload
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.review_owned and fresh.task_id == entry.task_id
    assert fresh.terminal_state == "succeeded" and fresh.terminal_source
    assert not fresh.output_source and not fresh.output_consumed
    assert not fresh.settled and not fresh.ledger_recorded and not money
    assert not fresh.patch_captured and not fresh.patch_disposed


def test_foreground_collection_preserves_missing_custody_refusal(tmp_path, monkeypatch, money):
    from ouroboros import platform_layer
    from ouroboros.tools.delegate_terminal_evidence import collect_and_deliver_terminal
    from ouroboros.tools.registry import ToolContext

    entry = started(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    acquire, read = platform_layer.acquire_exclusive_file_lock, current.run
    target_lock = custody.disposition_lock_path(tmp_path, entry)
    locked = False

    def acquired(path, **kwargs):
        nonlocal locked
        fd = acquire(path, **kwargs)
        if path == target_lock:
            locked = fd is not None
        return fd

    monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", acquired)
    monkeypatch.setattr(current, "run", lambda *args, **kwargs: None if locked else read(*args, **kwargs))
    result = collect_and_deliver_terminal(ctx, entry.run_id, terminal(), body._shape(entry), entry, None)
    assert locked and result["status"] == "refused" and result["ok"] is False
    assert result["reason"] == "custody_missing"
    assert result["collection"] == {"status": "unavailable", "reason": "custody_missing"}
    assert not money and "primary_output" not in result


def test_missing_retention_stays_pending_without_fabricated_output(tmp_path, monkeypatch, money):
    from ouroboros.tools.delegate_terminal_evidence import collect_and_deliver_terminal
    from ouroboros.tools.registry import ToolContext

    entry = started(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    monkeypatch.setattr(body, "_retain_output", lambda *args, **kwargs: {})
    result = collect_and_deliver_terminal(ctx, entry.run_id, terminal(), body._shape(entry), entry, None)
    assert result["status"] == "observation_pending" and result["collection"]["status"] == "pending"
    assert "primary_output" not in result and "output_delivery" not in result
    fresh = current.run(tmp_path, entry.run_id)
    assert not fresh.output_source and not fresh.output_complete and not fresh.output_consumed


def test_small_retained_inline_delivery_credits_only_the_actual_full_view(tmp_path, money):
    from ouroboros.delegate_output import acknowledge_staged_output_delivery
    from ouroboros.tools.delegate_terminal_evidence import _retained_terminal_payload
    entry = started(tmp_path)
    body.collect_terminal(tmp_path, entry.run_id, detail=terminal())
    ctx = SimpleNamespace(task_id=entry.task_id, drive_root=tmp_path, budget_drive_root=tmp_path,
                          task_metadata={})
    payload = _retained_terminal_payload(ctx, current.run(tmp_path, entry.run_id))
    assert payload["output_delivery"]["complete"]
    assert not current.run(tmp_path, entry.run_id).output_consumed
    acknowledge_staged_output_delivery(ctx, {}, {"result": json.dumps(payload), "result_partial": True})
    assert not current.run(tmp_path, entry.run_id).output_consumed
    acknowledge_staged_output_delivery(ctx, {}, {"result": json.dumps(payload), "result_partial": False})
    assert current.run(tmp_path, entry.run_id).output_consumed


def _until(check):
    end = time.monotonic() + 10
    while time.monotonic() < end:
        if check():
            return
        time.sleep(.01)
    raise AssertionError("physical work did not reach the expected barrier")


def test_slow_capture_does_not_block_another_terminal_observation(tmp_path, monkeypatch, money, capture_child):  # noqa: F811
    from ouroboros.tools import delegate_integration
    a = started(tmp_path, "run-a", execution_root=str(tmp_path / "copy-a"), snapshot_id="snap-a")
    b = started(tmp_path, "run-b")
    entered, release = threading.Event(), threading.Event()
    original = delegate_integration.capture_terminal_patch_for_drive

    def capture(root, entry, **kw):
        if entry.run_id == a.run_id:
            entered.set()
            assert release.wait(10)
            return None
        return original(root, entry, **kw)

    monkeypatch.setattr(delegate_integration, "capture_terminal_patch_for_drive", capture)

    class Gateway:
        def get_run(self, rid, **kwargs):
            return terminal(rid)
        def close(self):
            pass

    runtime = body.start_body_runtime(tmp_path, gateway_factory=Gateway, max_workers=2)
    try:
        assert runtime.ensure_runs([(a.run_id, a.invocation_id)], task_id=a.task_id)
        _until(lambda: current.run(tmp_path, a.run_id).terminal_state)
        runtime.tick()
        assert entered.wait(10)
        assert body_runtime_same(tmp_path, runtime)
        from ouroboros.tools.delegate_terminal_evidence import collect_and_deliver_terminal
        from ouroboros.tools.registry import ToolContext
        ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=a.task_id)
        available = collect_and_deliver_terminal(ctx, a.run_id, terminal(a.run_id), body._shape(a), a, None)
        assert available["status"] == "terminal" and available["output_delivery"]["complete"]
        assert available["workspace_capture"]["status"] == "pending"
        assert available["settlement"]["settled"] and not release.is_set()
        assert runtime.ensure_runs([(b.run_id, b.invocation_id)])
        _until(lambda: current.run(tmp_path, b.run_id).terminal_state)
        assert not release.is_set()
        body.stop_body_runtime(tmp_path, wait=False)
        assert not runtime.ensure_runs([(b.run_id, b.invocation_id)])
        assert any(thread.is_alive() for thread in capture_child.threads.values()), "the maintenance child still owns physical capture"
    finally:
        release.set()
        runtime.close(wait=True)
    assert body.body_runtime(tmp_path) is None


def body_runtime_same(root, runtime):
    return body.body_runtime(root) is runtime and runtime.identity()["active"]


def test_raw_custody_decode_does_not_hold_the_read_transaction(tmp_path, monkeypatch):
    from ouroboros import store
    from contextlib import contextmanager
    started(tmp_path)
    inside = False
    real = store.open_store
    @contextmanager
    def observed(*args, **kwargs):
        nonlocal inside
        with real(*args, **kwargs) as txn:
            inside = True
            try:
                yield txn
            finally:
                inside = False
    original = current._decode
    def decode(raw):
        assert not inside
        return original(raw)
    monkeypatch.setattr(store, "open_store", observed)
    monkeypatch.setattr(current, "_decode", decode)
    assert list(current.runs(tmp_path, "task_id=?", ("task-one",))) == ["run-one"]


def test_body_output_remains_readable_when_engine_is_gone(tmp_path, monkeypatch, money):
    from ouroboros.tools import delegate
    from ouroboros.gateways import claudexor as gateway
    from ouroboros.tools.registry import ToolContext
    entry = started(tmp_path)
    body.collect_terminal(tmp_path, entry.run_id, detail=terminal())
    contacted = []
    class Gone:
        def handshake(self, **kwargs):
            contacted.append(True)
            raise gateway.ClaudexorUnavailable("daemon_unreachable", "offline")
        def close(self):
            pass
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *a, **kw: Gone())
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    payload = json.loads(delegate._delegate_wait(ctx, entry.run_id, wait_sec=1))
    assert payload["status"] == "terminal"
    assert payload["primary_output"]["text"] == "complete answer"
    assert payload["current_observation"]["status"] == "retained" and not contacted
    assert not current.run(tmp_path, entry.run_id).output_consumed


def test_body_captures_real_snapshot_but_does_not_apply_or_dispose_it(tmp_path, monkeypatch, money):
    from pathlib import Path
    from tests.test_delegated_run_isolation import TestCaptureAndIntegrate, _isolated_entry
    from ouroboros.subagent_worktrees import find_execution_snapshot
    target, ctx, handle = TestCaptureAndIntegrate()._provisioned(tmp_path, monkeypatch)
    execution = Path(handle.path)
    (execution / "tracked.txt").write_text("body collected change\n", encoding="utf-8")
    entry = _isolated_entry(ctx, target, handle)
    root = custody.custody_root(ctx)
    assert custody.record_started(root, entry)
    result = body.collect_terminal(root, entry.run_id, detail=terminal(entry.run_id))
    fresh = current.run(root, entry.run_id)
    assert result["workspace_capture"]["status"] == "ready_with_changes"
    assert fresh.patch_captured and not fresh.patch_disposed
    assert find_execution_snapshot(handle.snapshot_id)
    assert (target / "tracked.txt").read_text(encoding="utf-8") != "body collected change\n"
    assert fresh.output_complete and not fresh.output_consumed


def test_one_unanswered_reader_does_not_hide_another_selected_run(tmp_path, monkeypatch, money):
    entered, release = threading.Event(), threading.Event()
    a, b = started(tmp_path, "run-a"), started(tmp_path, "run-b")
    class Gateway:
        def get_run(self, rid, **kwargs):
            if rid == a.run_id:
                entered.set()
                assert release.wait(10)
            return terminal(rid)
        def close(self):
            pass
    runtime = body.start_body_runtime(tmp_path, gateway_factory=Gateway, max_workers=2)
    try:
        assert entered.wait(10)
        _until(lambda: current.run(tmp_path, b.run_id).terminal_state)
        assert not current.run(tmp_path, a.run_id).terminal_state
        runtime.close(wait=False)
        assert body.start_body_runtime(tmp_path, gateway_factory=Gateway) is runtime
    finally:
        release.set()
        runtime.close(wait=True)


def test_observation_reuses_one_client_and_rebuilds_it_after_a_failed_read(tmp_path):
    """A handshake and an HTTP client per poll of every held run kept the server busy: the
    generation keeps one client, drops it on a failed read and closes it when it ends."""
    a, b = started(tmp_path, "run-a"), started(tmp_path, "run-b")
    made, closed, fail = [], [], {"once": True}
    class Gateway:
        def __init__(self):
            made.append(self)
        def get_run(self, rid, **kwargs):
            if rid == a.run_id and fail.pop("once", False):
                raise ConnectionError("the daemon moved")
            return {"summary": {"runId": rid, "state": "running"}, "lastSeq": 1, "pendingInteractions": []}
        def close(self):
            closed.append(self)
    stop = threading.Event()
    stop.set()  # no background ticks: the observations below are the only ones
    runtime = body.BodyRuntime(tmp_path, stop_event=stop, gateway_factory=Gateway, max_workers=2)
    runtime.thread.join(10)
    with pytest.raises(ConnectionError):
        runtime._observe(a)
    for entry in (a, b, b, a):
        runtime._observe(entry)
    assert len(made) == 2 and closed == [made[0]]
    runtime._drop_observer_gateway()
    assert closed == made


def test_automatic_availability_is_not_an_owner_omission(tmp_path, money):
    from ouroboros.context_health import _unread_output_line
    entry = started(tmp_path)
    body.collect_terminal(tmp_path, entry.run_id, detail=terminal())
    fresh = current.run(tmp_path, entry.run_id)
    assert _unread_output_line(tmp_path, fresh, entry.task_id).startswith("INFO: DELEGATED RESULT AVAILABLE")
    assert custody.record_settled_unread(tmp_path, fresh)
    assert _unread_output_line(tmp_path, fresh, entry.task_id).startswith("WARNING: DELEGATED RESULT NEVER READ")


def test_closing_runtime_does_not_start_later_collection_phases(tmp_path, monkeypatch, money):
    from ouroboros.tools import delegate_integration
    entry = started(tmp_path)
    stop = threading.Event()
    original = body._retain_output
    def retained(*a, **kw):
        answer = original(*a, **kw)
        stop.set()
        return answer
    monkeypatch.setattr(body, "_retain_output", retained)
    monkeypatch.setattr(delegate_integration, "capture_terminal_patch_for_drive",
        lambda *a, **kw: pytest.fail("capture started after runtime closed"))
    result = body.collect_terminal(tmp_path, entry.run_id, detail=terminal(), stopped=stop.is_set)
    assert result["status"] == "pending" and not money
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.terminal_source and fresh.output_complete and not fresh.settled


@pytest.mark.parametrize("strategy", ["direct", "copy"])
def test_directory_capture_keeps_complete_files_while_money_is_pending(tmp_path, monkeypatch, strategy):
    from pathlib import Path
    from ouroboros import usage_accounting
    from ouroboros.delegate_directory import capture_directory_result
    from tests.test_delegated_directory import DirectoryEngine, context, entry as directory_entry

    ctx, target = context(tmp_path, monkeypatch)
    engine, held = DirectoryEngine(target, strategy), directory_entry(ctx, target, strategy)
    held.settled = False
    held.invocation_id = "directory-invocation"
    held.authority_source = "external_workspace_root"
    assert custody.record_started(ctx.drive_root, held)
    with pytest.raises(ValueError, match="has not completed"):
        capture_directory_result(ctx.drive_root, held, engine)
    monkeypatch.setattr(usage_accounting, "record_subscription_session",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("ledger busy")))

    result = body.collect_terminal(ctx.drive_root, held.run_id, gateway=engine, detail=terminal(held.run_id))
    fresh = current.run(ctx.drive_root, held.run_id)
    captured = result["workspace_capture"]
    assert captured["status"] == "ready" and captured["capture_kind"] == "engine_directory"
    assert captured["complete"] is True
    assert Path(captured["file_outputs"][1]["path"]).read_bytes() == engine.body
    assert fresh.terminal_state == "succeeded" and fresh.patch_captured and fresh.output_complete
    assert not fresh.settled and not fresh.ledger_recorded and not fresh.output_consumed
    assert not fresh.patch_disposed and not engine.applies and not engine.decisions
    assert held.run_id in current.runs(ctx.drive_root, "state=?", (current.HELD,))
    assert not (target / ".git").exists() and not (target / "out.bin").exists()


def test_a_typed_capture_failure_stays_owed(tmp_path, monkeypatch, money):
    """A capture that reports ``status: failed`` (its exception was caught inside) is a failure
    the collector keeps owing, not a success that erases it; a later good capture clears it."""
    from ouroboros.tools import delegate_integration

    entry = started(tmp_path)
    outcomes = iter([{"status": "failed", "note": "patch capture failed (RuntimeError: Symlink loop)"},
                     {"status": "ready_with_changes"}])
    monkeypatch.setattr(delegate_integration, "capture_terminal_patch_for_drive", lambda *a, **kw: next(outcomes))
    failures = {}
    first = body.collect_terminal(tmp_path, entry.run_id, detail=terminal(), failures=failures)
    assert "capture:RuntimeError" in first["errors"] and "capture" in failures
    assert first["workspace_capture"]["status"] == "failed"
    second = body.collect_terminal(tmp_path, entry.run_id, detail=terminal(), failures=failures, recapture=True)
    assert not any(error.startswith("capture:") for error in second["errors"]) and "capture" not in failures
