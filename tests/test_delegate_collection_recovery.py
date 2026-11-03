"""Collection debt remains addressable; available source bytes never wait for transport."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._delegate_capture_child import capture_child  # noqa: F401

from ouroboros import delegate_collection as body, delegate_custody as custody, delegate_custody_current as current
from ouroboros.tools.delegate_terminal_evidence import collect_and_deliver_terminal
from ouroboros.tools.registry import ToolContext
from tests._delegated_transport_shared import _owned_gateway_uses_each_test_transport  # noqa: F401
from tests.test_delegate_body_collection import _until, started, terminal, money  # noqa: F401
from tests.test_delegated_directory import DirectoryEngine, context


@pytest.mark.parametrize("strategy", ["direct", "copy"])
@pytest.mark.parametrize("registration", ["existing", "fresh"])
@pytest.mark.parametrize("owner", ["body", "task_exit", "orphan"])
def test_real_directory_producer_recovers_owed_capture_under_each_existing_owner(
        tmp_path, monkeypatch, money, strategy, registration, owner):  # noqa: F811
    from ouroboros.gateways import claudexor
    from ouroboros.tools import delegate
    ctx, target = context(tmp_path, monkeypatch)
    engine = DirectoryEngine(target, strategy)
    if registration == "fresh":
        monkeypatch.setattr(engine, "find_project_id", lambda root: "")
        monkeypatch.setattr(engine, "register_project", lambda root: "created-folder", raising=False)
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *a, **kw: engine)
    produced = json.loads(delegate._delegate_start(ctx, "edit the documents", directory_strategy=strategy,
                                                   scope_paths=["."]).text)
    assert produced["status"] == "started"
    entry = current.run(ctx.drive_root, produced["run_id"])
    assert entry.project_owned == (registration == "fresh") and entry.project_persistent
    assert entry.resource_ref["strategy"] == strategy and not entry.snapshot_id
    original = engine.get_run
    capture_reads = []
    def detail(rid, **kwargs):
        capture_reads.append(rid)
        if len(capture_reads) == 1:
            raise OSError("controlled temporary capture outage")
        return {**original(rid), **terminal(rid)}
    monkeypatch.setattr(engine, "get_run", detail)
    first = body.collect_terminal(ctx.drive_root, entry.run_id, gateway=engine, detail=terminal(entry.run_id))
    assert first["workspace_capture"]["status"] == "failed"
    fresh = current.run(ctx.drive_root, entry.run_id)
    assert fresh.settled and fresh.output_complete and not fresh.patch_captured
    assert current.collection_pending is body.collection_pending and body.collection_pending(fresh)
    assert entry.run_id in current.runs(ctx.drive_root, "state=?", (current.HELD,))
    assert custody.open_runs(ctx.drive_root) == [], "financial open-runs semantics stay unchanged"
    if owner == "task_exit":
        assert custody.release_task_runs(ctx.drive_root, ctx.task_id, gateway_factory=lambda: engine) == []
    elif owner == "orphan":
        assert custody.reconcile_orphaned_runs(ctx.drive_root, running_task_ids={ctx.task_id},
                                               gateway_factory=lambda: engine) == []
        assert custody.reconcile_orphaned_runs(ctx.drive_root, running_task_ids=set(),
                                               gateway_factory=lambda: engine) == []
    assert not current.run(ctx.drive_root, entry.run_id).patch_captured
    assert len(capture_reads) == 1, "control reconciliation leaves bulk collection to its owner"
    runtime = body.start_body_runtime(ctx.drive_root, gateway_factory=lambda: engine)
    try:
        _until(lambda: current.run(ctx.drive_root, entry.run_id).patch_captured)
    finally:
        runtime.close(wait=True)
    fresh = current.run(ctx.drive_root, entry.run_id)
    assert fresh.patch_captured and not body.collection_pending(fresh)
    captured = body.capture_projection(ctx.drive_root, fresh)
    assert Path(captured["file_outputs"][1]["path"]).read_bytes() == engine.body
    assert not engine.applies and not engine.decisions and not fresh.output_consumed
    assert fresh.patch_disposed == ""
    assert (entry.run_id in current.runs(ctx.drive_root, "state=?", (current.HELD,))) == (strategy == "copy")


def test_actual_terminal_boundary_retries_money_without_an_extra_runtime(tmp_path, monkeypatch):
    from ouroboros import usage_accounting
    entry = started(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    calls = []
    def record(*a, **kw):
        calls.append(True)
        if len(calls) == 1:
            raise OSError("controlled ledger gap")
    monkeypatch.setattr(usage_accounting, "record_subscription_session", record)
    first = collect_and_deliver_terminal(ctx, entry.run_id, terminal(), body._shape(entry), entry, None)
    assert first["output_delivery"]["complete"] and not first["settlement"]["settled"]
    class Gateway:
        def handshake(self): pass
        def get_run(self, rid): return terminal(rid)
        def close(self): pass
    assert custody.release_task_runs(tmp_path, entry.task_id, gateway_factory=Gateway)
    assert len(calls) == 2 and current.run(tmp_path, entry.run_id).settled
    assert body.body_runtime(tmp_path) is None


def test_public_wait_delivers_verified_local_output_without_any_gateway(tmp_path, monkeypatch, money):  # noqa: F811
    from ouroboros import delegate_supervision
    from ouroboros.gateways import claudexor
    entry = started(tmp_path)
    body.collect_terminal(tmp_path, entry.run_id, detail=terminal())
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *a, **k: pytest.fail("available output opened transport"))
    result = json.loads(delegate_supervision.supervised_wait(ctx, entry.run_id).text)
    assert result["status"] == "terminal" and result["primary_output"]["text"] == "complete answer"
    assert result["current_observation"]["status"] == "retained"
    assert not current.run(tmp_path, entry.run_id).output_consumed


@pytest.mark.parametrize("source", ["missing", "incomplete"])
def test_public_wait_still_reads_engine_when_full_local_source_is_unavailable(tmp_path, monkeypatch, money, source):  # noqa: F811
    from ouroboros import delegate_supervision
    from ouroboros.gateways import claudexor
    entry = started(tmp_path)
    if source == "incomplete":
        detail = terminal()
        detail["primaryOutput"].update(truncated=True, bytes=1000)
        body.collect_terminal(tmp_path, entry.run_id, detail=detail)
        assert not current.run(tmp_path, entry.run_id).output_complete
    calls = []
    class Gateway:
        engine_version = "99.0.0"
        def agent_capabilities(self, **kwargs):
            return {"harnesses": [{"id": "test", "liveInput": "none"}]}
        def get_run(self, rid, **kwargs):
            calls.append(rid)
            return terminal(rid, seq=5)
        def close(self): pass
    monkeypatch.setattr(claudexor, "ClaudexorGateway", Gateway)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    result = json.loads(delegate_supervision.supervised_wait(ctx, entry.run_id).text)
    assert calls == [entry.run_id] and result["status"] == "terminal"
    assert result["output_delivery"]["complete"]


@pytest.mark.parametrize("kind", ["review", "ordinary", "retained_review"])
def test_registered_wait_preserves_collection_outcome(tmp_path, monkeypatch, money, kind):  # noqa: F811
    """The public string ABI must keep the native failure, not just wake the model."""
    from ouroboros import delegate_supervision
    from ouroboros.gateways import claudexor
    from ouroboros.tools import delegate
    from ouroboros.tools.tool_result import (
        _install_tool_result_sidecar, _published_tool_result, _restore_tool_result_sidecar,
    )

    entry = started(tmp_path, source="review_substrate_commit" if kind != "ordinary" else "")
    if kind == "retained_review":
        assert body.collect_terminal(tmp_path, entry.run_id, detail=terminal(), allow_review=True)["status"] == "collected"
    calls, closed = [], []

    class Gateway:
        engine_version = "99.0.0"
        def agent_capabilities(self, **kwargs):
            return {"harnesses": [{"id": "test", "liveInput": "none"}]}
        def get_run(self, rid, **kwargs):
            assert not calls, "the terminal collection result was silently retried"
            calls.append(rid)
            return terminal(rid)
        def close(self):
            closed.append(True)

    monkeypatch.setattr(claudexor, "ClaudexorGateway", Gateway)
    monkeypatch.setattr(delegate_supervision, "_TICK_SEC", 0)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    tool = next(item for item in delegate.get_tools() if item.name == "delegate_wait")
    sentinel = object()
    token = _install_tool_result_sidecar(ctx, sentinel)
    try:
        text = tool.handler(ctx, run_id=entry.run_id)
        native = _published_tool_result(ctx, sentinel)
    finally:
        _restore_tool_result_sidecar(token)
    assert native is not None and native.text == text
    payload = json.loads(text)
    fresh = current.run(tmp_path, entry.run_id)
    assert not fresh.output_consumed and not fresh.patch_captured and not fresh.patch_disposed
    assert fresh.review_owned == (kind != "ordinary")
    assert calls == ([] if kind == "retained_review" else [entry.run_id])
    assert bool(closed) == bool(calls)
    if kind == "review":
        assert native.code == "TOOL_REPORTED_FAILURE" and str(native.status) == "error"
        assert payload["status"] == "refused" and payload["ok"] is False
        assert payload["reason"] == "collection_owner_mismatch"
        assert payload["collection"]["status"] == "unavailable"
        assert "review" in payload["detail"].lower()
        assert not money and not fresh.ledger_recorded and not fresh.output_source
    else:
        assert native.code == "OK"
        assert payload["status"] == "terminal" and payload["primary_output"]["text"] == "complete answer"
        assert payload["output_delivery"]["complete"] and len(money) == 1


def test_ready_local_result_does_not_override_owner_control_or_foreign_authority(tmp_path, monkeypatch, money):  # noqa: F811
    from ouroboros import delegate_supervision
    from ouroboros.tools.delegate_terminal_evidence import retained_terminal_result
    entry = started(tmp_path)
    body.collect_terminal(tmp_path, entry.run_id, detail=terminal())
    foreign = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="foreign")
    assert retained_terminal_result(foreign, entry.run_id) is None
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    monkeypatch.setattr(delegate_supervision, "_control_wakes", lambda _: [{"type": "cancellation_intent"}])
    result = json.loads(delegate_supervision.supervised_wait(ctx, entry.run_id).text)
    assert result["status"] == "no_progress" and result["wake_events"] == [{"type": "cancellation_intent"}]
    assert not current.run(tmp_path, entry.run_id).output_consumed


@pytest.mark.parametrize("publication", [True, False])
def test_owned_unrecorded_start_delivers_exact_terminal_instead_of_renewing_forever(tmp_path, monkeypatch, money, publication):  # noqa: F811
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tools import delegate
    entry = custody.RunCustody(run_id="memo-only", task_id="memo-task", access="readonly", route_id="test")
    monkeypatch.setitem(custody._CUSTODY, entry.run_id, entry)
    original = custody.emit
    def emit(root, kind, row):
        if kind == custody.STARTED and not publication:
            return False
        return original(root, kind, row)
    monkeypatch.setattr(custody, "emit", emit)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=entry.task_id)
    result = collect_and_deliver_terminal(ctx, entry.run_id, terminal(entry.run_id), body._shape(entry), entry, None)
    assert result["status"] == "terminal"
    if publication:
        assert result["settlement"]["settled"] and current.run(tmp_path, entry.run_id).output_complete
    else:
        assert current.run(tmp_path, entry.run_id) is None and not money
        assert result["collection"]["reason"] == "custody_publication_unconfirmed"
        assert result["settlement"] == {"settled": False, "ledger_recorded": False}
        artifact = result["output_delivery"]["artifact"]
        full = json.loads(read_actor_source_bytes(tmp_path, entry.task_id, artifact["source"]))
        assert full["primary_output"]["text"] == "complete answer"
        assert not result["output_delivery"]["consumed"]
    assert delegate._owned_run(ctx, "delegate_wait", entry.run_id)[1] is not None


@pytest.fixture
def driven_runtime(tmp_path, monkeypatch, capture_child):  # noqa: F811
    """Keep the real driver parked while exercising its real tick and physical jobs on a controlled clock."""
    release, entered = threading.Event(), threading.Event()
    tick = body.BodyRuntime.tick
    def park(runtime):
        entered.set()
        assert release.wait(10)
    monkeypatch.setattr(body.BodyRuntime, "tick", park)
    clock = [0.0]
    monkeypatch.setattr(body, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    class Gateway:
        def get_run(self, rid, **kwargs): return terminal(rid)
        def close(self): pass
    runtime = body.start_body_runtime(tmp_path, gateway_factory=Gateway, max_workers=2)
    assert entered.wait(3)
    def step():
        tick(runtime)
        with runtime.lock:
            jobs = list(runtime.collections.values()) + list(runtime.reads.values())
        for job in jobs:
            job.result(timeout=5)
        if capture_child.drain(tmp_path):
            tick(runtime)  # server publishes the child's result under its fresh owner check
            for future in list(runtime.collections.values()):
                future.result(timeout=5)
    try:
        yield runtime, clock, step
    finally:
        runtime.close(wait=False)
        release.set()
        runtime.close(wait=True)


def test_unchanged_money_debt_rests_but_facts_and_explicit_request_are_immediate(
        tmp_path, monkeypatch, driven_runtime, caplog):
    from ouroboros import usage_accounting, config
    runtime, clock, tick = driven_runtime
    entry = started(tmp_path)
    body.observe_run(tmp_path, entry, terminal())
    fresh = current.run(tmp_path, entry.run_id)
    body._retain_output(tmp_path, fresh, terminal(), None)
    attempts = []
    def fail(*a, **kw):
        attempts.append(clock[0])
        raise OSError("stable unavailable ledger")
    monkeypatch.setattr(usage_accounting, "record_subscription_session", fail)
    caplog.set_level(logging.WARNING, logger=body.__name__)
    tick()
    assert attempts == [0.0]
    for second in (1.0, 3.0, 9.0, config.BODY_MAINTENANCE_INTERVAL_SEC - 1):
        clock[0] = second
        runtime.ensure_runs([(entry.run_id, entry.invocation_id)], task_id=entry.task_id)
        tick()
    assert attempts == [0.0], "passive cold ownership checks do not force bookkeeping retries"
    clock[0] += config.BODY_MAINTENANCE_INTERVAL_SEC
    tick()
    assert len(attempts) == 1, "elapsed time does not change retry eligibility"
    runtime._collect(entry.run_id, recapture=True)
    tick()
    assert len(attempts) == 2
    body.observe_run(tmp_path, entry, terminal(text="new exact result", seq=6))
    tick()
    assert len(attempts) == 3
    failures = [r for r in custody.custody_rows(tmp_path) if r.get("type") == custody.SETTLE_FAILED]
    assert len(failures) == 2, "one durable report for each changed failed source, no unchanged failure storm"
    warnings = [r for r in caplog.records if "money remains owed" in r.message and r.levelno >= logging.WARNING]
    assert len(warnings) == 2
    monkeypatch.setattr(usage_accounting, "record_subscription_session", lambda *a, **kw: None)
    runtime._collect(entry.run_id, recapture=True)
    tick()
    tick()
    assert current.run(tmp_path, entry.run_id).settled
    assert entry.run_id not in runtime.collection_failures
    assert current.run(tmp_path, entry.run_id).collection_attempt["retry"] == ""


def test_new_terminal_observation_is_independent_of_another_runs_paced_collection(tmp_path, monkeypatch, driven_runtime):
    from ouroboros import usage_accounting
    runtime, clock, tick = driven_runtime
    first = started(tmp_path, "first")
    body.observe_run(tmp_path, first, terminal(first.run_id))
    body._retain_output(tmp_path, current.run(tmp_path, first.run_id), terminal(first.run_id), None)
    calls = []
    monkeypatch.setattr(usage_accounting, "record_subscription_session",
        lambda *a, **kw: calls.append(a[0]) or (_ for _ in ()).throw(OSError("ledger unchanged")))
    tick()
    clock[0] = 1
    second = started(tmp_path, "second")
    tick()
    assert current.run(tmp_path, second.run_id).terminal_state == "succeeded"
    assert calls == [first.run_id]


def test_incomplete_output_does_not_republish_identical_preview_on_each_retry(tmp_path, money, driven_runtime):  # noqa: F811
    runtime, clock, tick = driven_runtime
    entry = started(tmp_path)
    detail = terminal()
    detail["primaryOutput"].update(truncated=True, bytes=1000)
    runtime.gateway_factory = lambda: SimpleNamespace(get_run=lambda *a, **k: detail, close=lambda: None)
    body.collect_terminal(tmp_path, entry.run_id, detail=detail)
    before = [r for r in custody.custody_rows(tmp_path) if r.get("type") == custody.OUTPUT_SPILLED]
    tick()
    clock[0] = 1
    tick()
    after = [r for r in custody.custody_rows(tmp_path) if r.get("type") == custody.OUTPUT_SPILLED]
    assert len(after) == len(before) == 1 and not current.run(tmp_path, entry.run_id).output_complete
    assert body.collection_pending(current.run(tmp_path, entry.run_id))
