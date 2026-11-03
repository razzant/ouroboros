"""Offline consumers for #2182: real recovery, capture, readiness, dispatch and control.

Tiny Git copies and the owned store are real. Engine transport, worker processes,
checkout publication and unrelated boot/maintenance duties are substituted; no
server ports, live daemon, provider or operator state are used.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._delegate_capture_child import capture_child  # noqa: F401

from ouroboros import delegate_collection as body, delegate_custody as c, delegate_custody_current as current
from ouroboros import review_checking as rc, review_checking_prune as prune, server_maintenance as maintenance
from ouroboros.subagents import delegated_run_shape
from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
from tests._budget_pause_exact_helpers import _install_queue
from tests._usage_store_testing import ledger_rows
from tests.test_delegate_body_collection import _until, terminal
from tests.test_headless_workspace_patch import _init_repo_with_file

pytestmark = pytest.mark.serial
ANSWER = "Recovered complete answer.\n" * 1024
EVIDENCE = b"\x00byte-exact checking output\xff\n" * 16


class Engine:
    engine_version = "99.0.0"

    def __init__(self, *, running=False):
        self.running, self.controls, self.reads = running, [], []

    def handshake(self):
        return {"compatible": True}

    def get_run(self, rid, **kwargs):
        self.reads.append(rid)
        detail = terminal(rid, ANSWER)
        if self.running:
            detail["summary"]["state"] = "running"
        return detail

    def cancel_run(self, rid, **kwargs):
        self.controls.append(rid)
        self.running = False
        return {"accepted": True}

    def close(self):
        pass


def prepared(tmp_path, root, kind="review", *, source_kind="git", operation="op-one"):
    subject = tmp_path / (kind + "-subject")
    _init_repo_with_file(subject)
    # The author's staged candidate differs from HEAD. A parent-commit fix
    # still attributes this edit to the reviewer, so the exact tree matters.
    (subject / "tracked.txt").write_bytes(b"author's staged candidate\n")
    subprocess.run(["git", "add", "."], cwd=subject, check=True, capture_output=True)
    if kind == "review":
        source = rc.freeze_checking_source(subject, data_root=root, subject=rc.SUBJECT_INDEX)
        if source_kind == "directory":
            from tests.test_review_checkout_lifecycle import directory_source
            folder = tmp_path / "skill"
            folder.mkdir()
            (folder / "check.py").write_bytes(b"author input\n")
            source = directory_source(folder, root)
        elif source_kind == "workspace":
            source = {"kind": "workspace", "root": str(subject)}
        shape = delegated_run_shape(False, "workspace_write")
        binding = rc.session_checking(Engine(), custody_drive=root, task_id=kind, operation_id=operation,
                                      root=str(subject), source=source, shape=shape)
        facts = binding["custody"]
    else:
        from ouroboros.subagent_worktrees import provision_execution_snapshot
        # The copy and its record live on the process roots, where every capture (in-process or a
        # maintenance child) and release reads them; conftest restores os.environ after the test.
        os.environ.update(OUROBOROS_DATA_DIR=str(root), OUROBOROS_SUBAGENT_WORKTREE_ROOT=str(tmp_path / "snapshots"))
        snapshot = provision_execution_snapshot(target_root=subject, task_id=kind,
            snapshot_id="ordinary-snapshot", worktree_root=tmp_path / "snapshots", data_dir=root)
        from ouroboros.delegate_registration_policy import INVESTIGATION_AUTHORITY

        facts = dict(snapshot_id=snapshot.snapshot_id, execution_root=snapshot.path,
                     target_root=snapshot.target_root, baseline_sha=snapshot.baseline_sha,
                     authority_source=INVESTIGATION_AUTHORITY if kind == "investigation" else "acting_constraint")
        source = {}
    area = Path(facts["execution_root"])
    (area / "check.py").write_bytes(b"# reviewer or delegate output\n")
    (area / rc.EVIDENCE_DIRNAME).mkdir(exist_ok=True)
    (area / rc.EVIDENCE_DIRNAME / "result.bin").write_bytes(EVIDENCE)
    entry = c.RunCustody(run_id="run-" + kind, task_id=kind, route_id="test", model="test",
        root_task_id=kind, source="review_substrate" if kind == "review" else "delegated_subagent",
        category="review" if kind == "review" else "subagent", ledger_root=str(root),
        invocation_id="inv-" + kind, access="workspace_write", mode="agent", **facts)
    assert c.record_started(root, entry)
    c._CUSTODY.clear()
    return current.run(root, entry.run_id), area, source


def quiet_custody(root, monkeypatch, engine):
    from ouroboros import claudexor_daemon, process_custody, process_scope, terminal_projection
    from supervisor import active_activity, terminal_delivery

    monkeypatch.setattr(maintenance, "DATA_DIR", root)
    monkeypatch.setattr(maintenance, "_supervisor_stop", threading.Event())
    monkeypatch.setattr(maintenance, "_restart_requested", threading.Event())
    monkeypatch.setattr(maintenance, "_installed_skill_names", lambda: set())
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda **kw: engine)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: engine)
    monkeypatch.setattr(claudexor_daemon, "get_owned_daemon", lambda: SimpleNamespace(
        clear_start_failure_latch=lambda **kw: False, stop_outcome=lambda: "not_running"))
    monkeypatch.setattr(process_custody, "reap_orphaned_processes", lambda *a, **kw: [])
    monkeypatch.setattr(process_scope, "settle_task_scopes", lambda *a, **kw: None)
    monkeypatch.setattr(terminal_projection, "reconcile_terminal_projections", lambda *a, **kw: None)
    monkeypatch.setattr(terminal_delivery, "replay_pending_deliveries", lambda *a, **kw: None)
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", active_activity.DirectActivityRegistry())
    for name in ("_prune_worktrees", "_prune_delegated_snapshots", "_prune_engine_runtimes"):
        monkeypatch.setattr(maintenance, name, lambda: None)


@pytest.mark.parametrize("source_kind", ["git", "directory", "workspace"])
def test_recovered_review_uses_native_subject_and_separate_terminal_bytes(tmp_path, monkeypatch, source_kind):
    root = tmp_path / "data"
    _install_queue(root, monkeypatch)
    entry, area, source = prepared(tmp_path, root, source_kind=source_kind)
    quiet_custody(root, monkeypatch, Engine())
    for _ in range(2):
        maintenance._startup_custody_sweep()
    saved = current.run(root, entry.run_id)
    assert saved.settled and saved.terminal_state == "succeeded" and saved.terminal_source
    assert body._read_source(root, saved, saved.terminal_source)["primaryOutput"]["text"] == ANSWER
    assert not saved.output_consumed and not saved.patch_captured and not saved.patch_disposed
    assert not current.collection_pending(saved), "the native review owner owes the files"
    assert capture_terminal_patch_for_drive(root, saved) is None
    assert not c.delegated_capture_dir(root, entry.task_id, c.capture_key(entry)).exists()
    assert len([row for row in ledger_rows(root) if row.get("kind") == "subscription_session"]) == 1
    assert area.is_dir()
    maintenance._prune_review_checkouts()
    receipt = rc._receipt(root, "review", "op-one")
    assert receipt["status"] == "retained" and not area.exists()
    evidence = Path(receipt["evidence_dir"])
    assert (evidence / "files" / rc.EVIDENCE_DIRNAME / "result.bin").read_bytes() == EVIDENCE
    if source_kind == "git":
        patch = (evidence / "delta.patch").read_text()
        assert "check.py" in patch and "tracked.txt" not in patch and rc.EVIDENCE_DIRNAME not in patch
        assert receipt["source"]["tree"] == source["tree"]
    elif source_kind == "directory":
        assert (evidence / "files" / "check.py").read_bytes() == b"# reviewer or delegate output\n"
    maintenance._startup_custody_sweep()
    maintenance._prune_review_checkouts()
    assert rc._receipt(root, "review", "op-one") == receipt
    assert len(list(evidence.parent.glob("evidence*"))) == 1


@pytest.mark.parametrize("basis", ["empty", "missing_field", "timeout", "missing_commit"])
def test_required_unknown_or_missing_basis_never_becomes_unborn(tmp_path, monkeypatch, basis):
    from ouroboros import workspace_preflight as preflight
    from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts

    repo = tmp_path / "repo"
    _init_repo_with_file(repo)
    git = {"head": ""}
    if basis == "timeout":
        run = subprocess.run
        def timed_out(cmd, **kw):
            if cmd == ["git", "rev-parse", "--verify", "HEAD^{commit}"]:
                raise subprocess.TimeoutExpired(cmd, 5)
            return run(cmd, **kw)
        with monkeypatch.context() as patch:
            patch.setattr(subprocess, "run", timed_out)
            git = preflight._git_snapshot(repo)
        assert git["head_state"] == "unknown"
    elif basis == "missing_field":
        git = {}
    elif basis == "missing_commit":
        git = {"head": "f" * 40}
    task = {"metadata": {"workspace_preflight": preflight.summarize_workspace_preflight({"git": git})}}
    artifacts, manifest = write_workspace_patch_artifacts(repo, tmp_path / "artifacts", task=task)
    assert manifest["status"] == "failed" and not manifest["base_is_empty_tree"]
    assert not any(row["kind"] == "workspace_patch" for row in artifacts)
    expected = "workspace_base_sha_missing" if basis == "missing_commit" else "workspace_base_unknown"
    assert expected in [row["type"] for row in manifest["errors"]]


def test_closed_absent_is_not_terminal_execution_and_terminal_does_not_wait_for_money(tmp_path, monkeypatch):
    root = tmp_path / "data"
    entry, area, _source = prepared(tmp_path, root)
    assert c.emit(root, c.CLOSED_ABSENT, {"run_id": entry.run_id, "task_id": entry.task_id})
    assert current.run(root, entry.run_id).settled
    assert prune.prune_review_checkouts(root, set(), now=10**11)["released"] == 0
    assert area.is_dir() and not rc._receipt(root, "review", "op-one")
    # Actual execution evidence suffices even while the ledger is unavailable.
    detail = terminal(entry.run_id, ANSWER)
    assert body.observe_run(root, entry, detail)
    assert prune.prune_review_checkouts(root, set())["released"] == 1


def test_review_capture_uses_execution_evidence_even_when_money_is_unsettled(tmp_path):
    root = tmp_path / "data"
    entry, area, _source = prepared(tmp_path, root)
    assert body.observe_run(root, entry, terminal(entry.run_id))
    assert not current.run(root, entry.run_id).settled
    assert prune.prune_review_checkouts(root, set())["released"] == 1 and not area.exists()
    assert not current.run(root, entry.run_id).settled


@pytest.mark.parametrize("missing", ["source", "tree"])
def test_review_missing_original_basis_retains_copy_until_basis_is_restored(tmp_path, missing):
    root = tmp_path / "data"
    entry, area, source = prepared(tmp_path, root)
    assert body.observe_run(root, entry, terminal(entry.run_id))
    marker_path = rc.seat_dir(root, "review", "op-one") / "checking.json"
    marker = json.loads(marker_path.read_text())
    marker["source"] = {} if missing == "source" else {k: v for k, v in source.items() if k != "tree"}
    marker_path.write_text(json.dumps(marker))
    for _ in range(2):
        assert prune.prune_review_checkouts(root, set(), now=10**11)["capture_failed"] == 1
        assert area.is_dir() and (area / "check.py").read_bytes() == b"# reviewer or delegate output\n"
    receipt = rc._receipt(root, "review", "op-one")
    assert receipt["status"] == "capture_failed" and not (Path(receipt["evidence_dir"]) / "delta.patch").exists()
    assert len(list(marker_path.parent.glob("evidence*"))) == 1
    marker["source"] = source
    marker_path.write_text(json.dumps(marker))
    assert prune.prune_review_checkouts(root, set(), now=10**11)["released"] == 1
    receipt = rc._receipt(root, "review", "op-one")
    assert (Path(receipt["evidence_dir"]) / "files" / rc.EVIDENCE_DIRNAME / "result.bin").read_bytes() == EVIDENCE


def test_relocated_failed_turn_is_captured_without_touching_active_thread(tmp_path, monkeypatch):
    from tests.test_review_checkout_lifecycle import TASK, _frozen, _thread, _work

    root, source = _frozen(tmp_path)
    area = _thread(root, source, "turn-one")
    _work(area)
    (area / "only-old.txt").write_bytes(EVIDENCE)
    compare = rc._directory_delta
    monkeypatch.setattr(rc, "_directory_delta", lambda *a: (_ for _ in ()).throw(OSError("temporary read failure")))
    assert rc.capture_seat_evidence(root, TASK, "turn-one")["status"] == "capture_failed"
    assert _thread(root, source, "turn-two") == area
    (area / "active.txt").write_bytes(b"live turn")
    old = rc.seat_dir(root, TASK, "turn-one") / "area.noindex"
    assert old.is_dir() and not (area / "only-old.txt").exists()
    monkeypatch.setattr(rc, "_directory_delta", compare)
    report = prune.prune_review_checkouts(root, {TASK}, now=10**11)
    assert report["released"] == 1 and not old.exists() and (area / "active.txt").read_bytes() == b"live turn"
    receipt = rc._receipt(root, TASK, "turn-one")
    assert (Path(receipt["evidence_dir"]) / "files" / "only-old.txt").read_bytes() == EVIDENCE
    assert rc.capture_seat_evidence(root, TASK, "turn-one") == receipt


def test_supervisor_reaches_ready_and_real_assignment_while_both_captures_are_blocked(tmp_path, monkeypatch):
    import server
    from ouroboros import server_liveness, subagent_worktrees as worktrees, usage_store
    from supervisor import queue, state, worker_assignment, workspace_preparation
    from tests.test_server_shutdown import _supervisor_harness
    from tests.worker_pool_support import affirm_ready

    root = tmp_path / "data"
    entries = [prepared(tmp_path, root, kind)[0] for kind in ("ordinary", "review")]
    quiet_custody(root, monkeypatch, Engine())
    drain = server._check_pending_restart_drain
    migrate_usage = usage_store.migrate_from_journal
    rec = _supervisor_harness(monkeypatch, root, ["ok"] * 1000)
    monkeypatch.setattr(usage_store, "migrate_from_journal", migrate_usage)
    _queue, _state, workers = _install_queue(root, monkeypatch)
    monkeypatch.setattr(server, "_startup_custody_sweep", maintenance._startup_custody_sweep)
    monkeypatch.setattr(server, "_periodic_supervisor_maintenance", maintenance._periodic_supervisor_maintenance)
    monkeypatch.setattr(server, "_startup_worker_pids", lambda root: set())
    monkeypatch.setattr(workers, "assign_tasks", worker_assignment.assign_tasks)
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(workers, "repo_writer_task_allowed", lambda task: True)
    for name in ("_run_terminal_duty", "_run_cold_custody", "_run_schedule_duty"):
        def finish(*args):
            next(arg for arg in args if hasattr(arg, "release")).release()
        monkeypatch.setattr(maintenance, name, finish)
    monkeypatch.setattr(maintenance, "_tick_maintenance_child", lambda: None)
    monkeypatch.setattr(maintenance, "_CUSTODY_SWEEP_LOCK", threading.Lock())
    entered = {kind: threading.Event() for kind in ("ordinary", "review")}
    release, body_stop, sent = threading.Event(), threading.Event(), []
    actual_patch, actual_review = worktrees.capture_result_commit, rc._git_delta

    def barrier(kind, action, *args, **kwargs):
        entered[kind].set()
        assert release.wait(10), "bounded fixture capture barrier was not released"
        return action(*args, **kwargs)

    monkeypatch.setattr(worktrees, "capture_result_commit",
                        lambda *a, **kw: barrier("ordinary", actual_patch, *a, **kw))
    monkeypatch.setattr(rc, "_git_delta", lambda *a, **kw: barrier("review", actual_review, *a, **kw))
    app_state = SimpleNamespace(supervisor_ready_event=rec.ready,
        supervisor_init_done_event=server._supervisor_init_done, get_supervisor_error=lambda: server._supervisor_error)

    def before_dispatch(ctx):
        assert drain(ctx) is False  # the real restart-drain decision allows assignment
        assert server_liveness.boot_document(root, app_state, jobs=False)["tasks_ready"]
        assert all(event.wait(5) for event in entered.values())
        return False

    monkeypatch.setattr(server, "_check_pending_restart_drain", before_dispatch)
    def delivered(task):
        sent.append(dict(task))
        rec.stop.set()
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None,
        in_q=SimpleNamespace(put=delivered)))
    task = {"id": "new-task", "type": "task", "chat_id": 0, "text": "Independent work", "admitted_dispatch": "none"}
    queue.PENDING.append(task)
    start = maintenance.bind_body_lifespan(root, body_stop, collection_ready=rec.ready.is_set)
    start()
    runtime = body.body_runtime(root)
    try:
        assert not any(event.is_set() for event in entered.values())
        server._run_supervisor({})  # actual ready latch, periodic owner and assignment-to-worker queue
        assert server._supervisor_error is None
        _until(lambda: any(row.get("id") == task["id"] for row in sent))
        assert task["id"] in workers.RUNNING and not queue.PENDING
        assert not release.is_set() and not current.run(root, entries[0].run_id).patch_captured
        assert not rc._receipt(root, "review", "op-one")
        # Another recovery observes the same settlements without joining either archive.
        maintenance._startup_custody_sweep()
        release.set()
        _until(lambda: current.run(root, entries[0].run_id).patch_captured)
        _until(lambda: rc._receipt(root, "review", "op-one").get("status") == "retained")
        _until(lambda: body.retained_payload(root, current.run(root, entries[0].run_id)) is not None)
        assert body.retained_payload(root, current.run(root, entries[0].run_id))["primary_output"]["text"] == ANSWER
        ordinary_evidence = c.delegated_capture_dir(root, "ordinary", c.capture_key(entries[0])) / "evidence"
        _until(lambda: (ordinary_evidence / "summary.json").exists())
        for evidence in (ordinary_evidence, Path(rc._receipt(root, "review", "op-one")["evidence_dir"])):
            assert (evidence / "files" / rc.EVIDENCE_DIRNAME / "result.bin").read_bytes() == EVIDENCE
        assert len([row for row in ledger_rows(root) if row.get("kind") == "subscription_session"]) == 2
    finally:
        release.set()
        body_stop.set()
        runtime.close(wait=True)
        workspace_preparation.pause_admission()
        _until(lambda: not maintenance._CUSTODY_SWEEP_LOCK.locked())


def test_body_stop_mid_capture_keeps_named_files_pending_for_the_next_generation(tmp_path, monkeypatch, capture_child):  # noqa: F811
    from ouroboros import subagent_worktrees as worktrees

    root = tmp_path / "data"
    _install_queue(root, monkeypatch)
    entry, area, _source = prepared(tmp_path, root, "ordinary")
    engine = Engine()
    quiet_custody(root, monkeypatch, engine)
    maintenance._startup_custody_sweep()
    actual = worktrees.capture_result_commit
    entered, release = threading.Event(), threading.Event()
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return actual(*args, **kwargs)
    monkeypatch.setattr(worktrees, "capture_result_commit", blocked)
    runtime = body.start_body_runtime(root, gateway_factory=lambda: engine)
    try:
        assert entered.wait(5)
        body.stop_body_runtime(root, wait=False)
        assert not release.is_set() and not runtime.active
        assert any(thread.is_alive() for thread in capture_child.threads.values())
    finally:
        release.set()
        runtime.close(wait=True)
        capture_child.drain(root)
    saved = current.run(root, entry.run_id)
    assert saved.capture_job["state"] == "ready" and current.collection_pending(saved, root) and area.is_dir()
    assert body._read_source(root, saved, saved.capture_job["source"])["capture_interrupted"]
    assert not body.saved_named_evidence(root, saved)[1]
    # A new body resumes just the owed retention from the existing copy/receipts.
    monkeypatch.setattr(worktrees, "capture_result_commit", lambda *a, **kw: pytest.fail("product patch already retained"))
    runtime = body.start_body_runtime(root, gateway_factory=lambda: engine)
    try:
        _until(lambda: not current.collection_pending(current.run(root, entry.run_id), root))
    finally:
        runtime.close(wait=True)
    evidence = c.delegated_capture_dir(root, "ordinary", c.capture_key(entry)) / "evidence"
    assert (evidence / "files" / rc.EVIDENCE_DIRNAME / "result.bin").read_bytes() == EVIDENCE
    assert len([row for row in ledger_rows(root) if row.get("kind") == "subscription_session"]) == 1


@pytest.mark.parametrize("kind", ["ordinary", "review"])
@pytest.mark.parametrize("door", ["stop", "restart"])
def test_stop_and_owner_restart_leave_capture_to_existing_owners(tmp_path, monkeypatch, kind, door):
    from ouroboros import claudexor_daemon, owned_shutdown, server_restart
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.task_results import write_task_result
    from tests.test_manual_restart_execution import _ctx, _restart

    root = tmp_path / "data"
    queue, _state, _workers = _install_queue(root, monkeypatch)
    entry, area, _source = prepared(tmp_path, root, kind)
    engine = Engine(running=True)
    quiet_custody(root, monkeypatch, engine)
    write_task_result(root, kind, "running", result="work in progress")
    # A capture on either control chain fails loudly; later use the actual collector.
    with monkeypatch.context() as blocked:
        blocked.setattr("ouroboros.subagent_worktrees.capture_result_commit",
                        lambda *a, **kw: pytest.fail("bulk capture entered a control chain"))
        blocked.setattr(rc, "_git_delta", lambda *a, **kw: pytest.fail("review capture entered a control chain"))
        if door == "stop":
            request_cancel(root, kind, reason="owner stop", source="owner", requested_by="owner")
            assert queue.cancel_task_custody(kind, deliver=False) == "cancelled"
        else:
            monkeypatch.setattr(server_restart, "DATA_DIR", root)
            monkeypatch.setattr(server_restart, "_OWNER_STOP_BEGAN", threading.Event())
            monkeypatch.setattr(owned_shutdown, "begin_owned_stop", lambda root, **_kw: None)
            monkeypatch.setattr(owned_shutdown, "stop_owned_work", lambda root: None)
            monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway",
                                lambda **kw: pytest.fail("Restart must only attach"))
            calls = []
            def killed(**kw):
                assert kw["reconcile_delegate_custody"] is False
                write_task_result(root, kind, "cancelled", result="Owner restart stopped the fixture worker")
                return True
            ctx = _ctx(calls, kill=killed, running={kind: {"task": {"id": kind}}})
            ctx.PENDING = []
            assert _restart(ctx, monkeypatch) == [True]
            assert _restart(ctx, monkeypatch) == [True], "another Restart keeps the pending capture"
        assert engine.controls == [entry.run_id]
        assert not current.run(root, entry.run_id).patch_captured and area.exists()
        # Repeated recovery before collection stays cheap and preserves the obligation.
        c.reconcile_orphaned_runs(root, set(), gateway_factory=lambda: engine)
        c.reconcile_orphaned_runs(root, set(), gateway_factory=lambda: engine)
    saved = current.run(root, entry.run_id)
    assert saved.terminal_source and saved.settled and not saved.output_consumed
    assert len([row for row in ledger_rows(root) if row.get("kind") == "subscription_session"]) == 1
    if kind == "review":
        assert prune.prune_review_checkouts(root, set())["released"] == 1
        evidence = Path(rc._receipt(root, kind, "op-one")["evidence_dir"])
    else:
        runtime = body.start_body_runtime(root, gateway_factory=lambda: engine)
        try:
            _until(lambda: current.run(root, entry.run_id).patch_captured)
            _until(lambda: (c.delegated_capture_dir(root, kind, c.capture_key(entry)) / "evidence" / "summary.json").exists())
        finally:
            runtime.close(wait=True)
        evidence = c.delegated_capture_dir(root, kind, c.capture_key(entry)) / "evidence"
    assert (evidence / "files" / rc.EVIDENCE_DIRNAME / "result.bin").read_bytes() == EVIDENCE
