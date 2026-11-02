"""Capture-required completion survives custody contention without replaying work."""

import json
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import headless, obligations
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_status import effective_task_result
from ouroboros.test_evidence import diagnostic_directory, write_record
from tests._store_shared import result_signature, unreadable_result
from tests.test_terminal_file_boundary import (  # noqa: F401 -- shared finite worker/reaper fixtures
    file_recovery,
    terminal_context,
    worker,
)


@contextmanager
def _busy_custody(root, task_id):
    """Hold the real publication lock in another thread; barriers, no timed race."""
    from ouroboros.task_custody import task_custody_lock

    acquired, release = threading.Event(), threading.Event()
    errors = []

    def hold():
        try:
            with task_custody_lock(root, task_id) as locked:
                assert locked
                acquired.set()
                assert release.wait(60), "custody test did not release its publisher"
        except BaseException as exc:
            errors.append(exc)
            acquired.set()

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert acquired.wait(60) and not errors
        yield
    finally:
        release.set()
        thread.join(60)
        assert not thread.is_alive() and not errors


def _consumers(root, task_id, expected):
    """Use the actual tool and HTTP handlers in process, with no listening port."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.tasks import api_task_get, api_tasks_list
    from ouroboros.tools.control_task_results import _wait_for_task

    before = result_signature(root, task_id)
    ctx = SimpleNamespace(task_id="observer", drive_root=root, budget_drive_root=root, task_metadata={})
    answer = _wait_for_task(ctx, task_id, timeout_sec=0)
    assert ("Task wait completed" in answer) is (expected == "completed"), answer
    app = Starlette(routes=[Route("/api/tasks", api_tasks_list), Route("/api/tasks/{task_id}", api_task_get)])
    app.state.drive_root = root
    with TestClient(app) as client:
        listing, detail = client.get("/api/tasks"), client.get(f"/api/tasks/{task_id}")
    assert listing.status_code == detail.status_code == 200
    listed = next(row for row in listing.json()["tasks"] if row["task_id"] == task_id)
    assert listed["status"] == detail.json()["status"] == expected
    assert result_signature(root, task_id) == before, "status consumers must not perform capture"
    evidence = diagnostic_directory(root / "capture-evidence", "workspace-capture")
    with (evidence / "consumers.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"expected": expected, "wait_task": answer,
                                 "task_list": listed, "task_detail": detail.json()}) + "\n")
    print(json.dumps({"consumer_status": expected, "wait": answer.splitlines()[0],
                      "list_status": listed["status"], "detail_status": detail.json()["status"]}))


@pytest.fixture
def promoted(worker, file_recovery, tmp_path_factory, monkeypatch):  # noqa: F811
    from ouroboros.agent import OuroborosAgent
    from ouroboros.agent_task_pipeline import _store_task_result
    from supervisor import events, workers
    from supervisor import queue as task_queue

    ctx, slot, pushed, jobs, returned = file_recovery
    root, tid = worker.root, "prepared"
    ctx.DRIVE_ROOT = root
    running = ctx.RUNNING.pop(tid)
    slot.busy_task_id = None
    task_queue.init(root)
    pending = []
    for module in (task_queue, workers):
        monkeypatch.setattr(module, "DRIVE_ROOT", root)
        monkeypatch.setattr(module, "PENDING", pending)
        monkeypatch.setattr(module, "RUNNING", ctx.RUNNING)
    monkeypatch.setattr(workers, "REPO_DIR", root.parent / "repo")
    monkeypatch.setattr(workers, "_broadcast_task_named", lambda *_a: None)
    ctx.PENDING = pending
    ctx.enqueue_task = lambda task: pending.append(task) or task
    ctx.persist_queue_snapshot = lambda **_kw: True
    ctx.load_state = lambda: {"owner_chat_id": 1}
    workspace = tmp_path_factory.mktemp("capture-workspace")
    subprocess.run(["git", "init", str(workspace)], check=True, capture_output=True)
    (workspace / "answer.txt").write_text("The completed workspace work.\n", encoding="utf-8")
    fork_attempts = []

    def failed_fork(*args, **kwargs):
        fork_attempts.append(args[1])
        raise OSError("test fork storage unavailable")

    with monkeypatch.context() as promotion:
        promotion.setattr(headless, "prepare_task_drive", failed_fork)
        # Toolchain probing is unrelated; the capture below reads the real unborn Git tree.
        promotion.setattr("ouroboros.workspace_admission.bounded_workspace_preflight",
                          lambda *_a: {"git": {"head": "(unborn)"}})
        admitted = events._handle_promote_chat_to_task({
            "task_id": tid, "routing_token": "capture-admission", "chat_id": 1,
            "objective": "Write the report", "workspace_root": str(workspace),
        }, ctx)
    assert admitted["status"] == "scheduled", admitted
    assert fork_attempts == [tid] and len(pending) == 1
    assert not pending[0].get("drive_root"), "exercise the canonical fallback after fork failure"
    assert not load_task_result(root, tid).get("artifact_status")
    worker.task.clear()
    worker.task.update(pending.pop(), _attempt=3)
    ctx.RUNNING[tid] = running
    slot.busy_task_id = tid
    ctx.RUNNING[tid]["task"] = worker.task
    ctx.RUNNING[tid]["worker_id"] = 0
    ctx.WORKERS.clear()
    ctx.WORKERS[0] = slot
    env = SimpleNamespace(drive_root=root, repo_dir=workspace, drive_path=lambda rel: root / rel)
    OuroborosAgent._persist_running_record(SimpleNamespace(env=env), worker.task)
    obligations.drive_started(root, worker.task)
    _store_task_result(env, worker.task, "Authored answer retained.", {"rounds": 1}, {"tool_calls": []})
    # Post-work has ended; this test starts at its ordinary file-publication boundary.
    write_task_result(root, tid, "completed", root_phase_checkpoint={"post_task_synthesis": "completed"})
    worker.events[:] = [{"type": "task_done", "task_id": tid, "status": "completed"}]
    monkeypatch.setattr(headless, "PUBLICATION_LOCK_SEC", 0)
    return SimpleNamespace(root=root, tid=tid, task=worker.task, worker=worker, ctx=ctx,
                           slot=slot, pushed=pushed, jobs=jobs, returned=returned, workspace=workspace)


def _restart_recovery(root):
    """A fresh interpreter has no RUNNING row, in-memory recovery or DutyRest cache."""
    script = """
import json, sys
from pathlib import Path
from ouroboros import headless, obligations
from ouroboros.startup_task_files import recover_terminal_task_files
from ouroboros.task_results import load_task_result
root = Path(sys.argv[1])
headless.PUBLICATION_LOCK_SEC = 0
captures = []
capture = headless.write_workspace_patch_artifacts
def measured(*args, **kwargs):
    captures.append(str(args[0]))
    return capture(*args, **kwargs)
headless.write_workspace_patch_artifacts = measured
reports = [recover_terminal_task_files(root, set()) for _ in range(2)]
print(json.dumps(dict(reports=reports, captures=captures,
    pending=sorted(obligations.members(root, 'pending_drives')),
    result=load_task_result(root, 'prepared'))))
"""
    run = subprocess.run([sys.executable, "-c", script, str(root)], capture_output=True, text=True, check=True)
    observed = json.loads(run.stdout)
    print(json.dumps({"restart": observed, "stderr": run.stderr}))
    return observed


@pytest.mark.parametrize("recovery", ["health", "restart"])
def test_promoted_fork_failure_busy_stamped_completion_recovers_once(promoted, monkeypatch, recovery):
    from ouroboros.store import close_connections_under
    from supervisor import events, task_reaper

    s = promoted
    original = load_task_result(s.root, s.tid)
    assert not original.get("artifact_status")
    captures = []
    capture = headless.write_workspace_patch_artifacts

    def measured(*args, **kwargs):
        captures.append(args[0])
        return capture(*args, **kwargs)

    monkeypatch.setattr(headless, "write_workspace_patch_artifacts", measured)
    with _busy_custody(s.root, s.tid):
        s.worker.run()
        assert s.worker.calls == [s.tid] and not s.worker.crashes
        frame = s.worker.output.get_nowait()
        assert frame["_files_prepared_attempt"] == 3 and frame["worker_id"] == 0
        events.dispatch_event(frame, s.ctx)
        _consumers(s.root, s.tid, "running")
        print(json.dumps({"stamped_attempt": frame["_files_prepared_attempt"],
                          "running": sorted(s.ctx.RUNNING), "queued_recovery": s.jobs.qsize(),
                          "pending_drives": sorted(obligations.members(s.root, "pending_drives")),
                          "published_statuses": [row.get("status") for row in s.pushed],
                          "captures": len(captures)}))
        assert s.tid in s.ctx.RUNNING and s.slot.busy_task_id == s.tid
        assert s.jobs.qsize() == 1 and not s.pushed
        assert s.tid in obligations.members(s.root, "pending_drives")
        for _ in range(2):
            task_reaper._recover_terminal_files(s.jobs.get_nowait())
            assert s.returned.empty() and not captures
            task_reaper.retry_terminal_file_recoveries()
            assert s.jobs.qsize() == 1, "only the existing health cadence retries"
        _consumers(s.root, s.tid, "running")
        assert not load_task_result(s.root, s.tid).get("artifact_status")
        if recovery == "restart":
            s.ctx.RUNNING.clear()
            s.ctx.WORKERS.clear()
            s.jobs.get_nowait()  # discard the old process's last in-memory job
            close_connections_under(s.root)
            blocked = _restart_recovery(s.root)
            assert blocked["pending"] == [s.tid] and not blocked["captures"]
            assert all(report["unresolved"] == [s.tid] for report in blocked["reports"])
    if recovery == "restart":
        recovered = _restart_recovery(s.root)
        assert len(recovered["captures"]) == 1
        assert recovered["reports"][0]["recovered"] == [s.tid]
        assert recovered["reports"][1]["recovered"] == [] and recovered["pending"] == []
    else:
        task_reaper._recover_terminal_files(s.jobs.get_nowait())
        events.dispatch_event(s.returned.get_nowait(), s.ctx)
        task_reaper.retry_terminal_file_recoveries()
        assert s.jobs.empty() and len(captures) == 1
        assert s.tid not in s.ctx.RUNNING and s.slot.busy_task_id is None
        assert s.pushed[-1]["status"] == "completed"
    result = load_task_result(s.root, s.tid)
    assert result["artifact_status"] == "ready_with_changes"
    assert result["result"] == original["result"]
    assert result["outcome_axes"]["objective"] == original["outcome_axes"]["objective"]
    assert result["outcome_axes"]["review"] == original["outcome_axes"]["review"]
    assert result.get("accounted_upper_bound_usd") == original.get("accounted_upper_bound_usd")
    patch = headless.task_artifacts_dir(s.root, s.tid) / "workspace.patch"
    assert "+The completed workspace work." in patch.read_text(encoding="utf-8")
    assert s.tid not in obligations.members(s.root, "pending_drives")
    _consumers(s.root, s.tid, "completed")
    assert s.worker.calls == [s.tid], "file recovery never replays the model"
    evidence = diagnostic_directory(s.root / "capture-evidence", "workspace-capture")
    for name in ("workspace.patch", "workspace_patch.json"):
        shutil.copyfile(patch.parent / name, evidence / name)
    write_record(evidence / "capture.json", {"recovery": recovery, "before": original, "after": result,
                 "capture_count": len(recovered["captures"]) if recovery == "restart" else len(captures),
                 "worker_calls": s.worker.calls, "pending_drives": sorted(obligations.members(s.root, "pending_drives"))})


@pytest.mark.parametrize("rested", [False, True])
def test_startup_prepares_addressed_canonical_pending_capture(tmp_path, monkeypatch, rested):
    """Independent negative control: positive readiness alone cannot repair startup."""
    from ouroboros.startup_task_files import recover_terminal_task_files

    task = {"id": "canonical", "workspace_root": str(tmp_path / "missing-workspace")}
    write_task_result(tmp_path, task["id"], "completed", workspace_root=task["workspace_root"],
                      artifact_status="pending", result="Retained answer")
    if rested:
        from ouroboros.startup_task_files import _drive_stamps

        previous = obligations.DutyRest(tmp_path, "drive_recovery", {
            task["id"]: {"drives": _drive_stamps(tmp_path, task["id"])}})
        previous.wait(task["id"], "child_drive_absent", report={"unresolved": False})
        previous.commit()
    monkeypatch.setattr(headless, "PUBLICATION_LOCK_SEC", 0)
    with _busy_custody(tmp_path, task["id"]):
        for _ in range(2):
            report = recover_terminal_task_files(tmp_path, set())
            assert report["unresolved"] == [task["id"]]
            assert task["id"] in obligations.members(tmp_path, "pending_drives")
    report = recover_terminal_task_files(tmp_path, set())
    assert report["recovered"] == [task["id"]]
    row = load_task_result(tmp_path, task["id"])
    assert row["artifact_status"] == "failed", "missing workspace cannot be a successful capture"
    assert row["artifact_error"] and row["result"] == "Retained answer"
    assert row["artifact_bundle"]["status"] == "failed"
    assert recover_terminal_task_files(tmp_path, set())["recovered"] == []
    _consumers(tmp_path, task["id"], "completed")


@pytest.mark.parametrize("artifact_status,ready", [
    (None, False), ("", False), ("pending", False), ("finalizing", False), ("future_status", False),
    ("ready", True), ("ready_with_changes", True), ("ready_no_changes", True), ("failed", True), ("missing", True),
])
def test_readers_share_positive_capture_finality(tmp_path, artifact_status, ready):
    task = {"id": "capture", "workspace_root": str(tmp_path / "workspace")}
    raw = write_task_result(tmp_path, task["id"], "completed", workspace_root=task["workspace_root"],
                            artifact_status=artifact_status, result="Authored answer")
    before = result_signature(tmp_path, task["id"])
    assert headless.terminal_task_files_ready(tmp_path, task, raw) is ready
    effective = effective_task_result(tmp_path, raw, materialize_artifacts=False)
    assert (effective["status"] == "completed") is ready
    assert result_signature(tmp_path, task["id"]) == before
    assert load_task_result(tmp_path, task["id"]).get("artifact_status") == raw.get("artifact_status")
    if artifact_status in {"failed", "missing"}:
        from ouroboros.outcomes import artifact_bundle_from_result, normalize_outcome_axes

        assert artifact_bundle_from_result(effective)["status"] == artifact_status
        assert normalize_outcome_axes(effective)["objective"]["status"] != "pass"


@pytest.mark.parametrize("fields", [
    {},
    {"workspace_root": "/unused", "delegation_role": "subagent",
     "task_constraint": {"mode": "local_readonly_subagent"}},
    {"workspace_root": "/unused", "execution_owner": {"kind": "presence"},
     "metadata": {"presence": {"binding_id": "b"}}},
    {"workspace_root": "/unused", "_is_direct_chat": True},
])
def test_no_capture_exceptions_still_require_terminal_source(tmp_path, monkeypatch, fields):
    task = {"id": "exempt", **fields}
    monkeypatch.setattr(headless, "finalize_task_artifacts", lambda *_a: pytest.fail("exempt capture"))
    for status in ("running", "completed"):
        raw = write_task_result(tmp_path, task["id"], status, result="Own answer", **fields)
        prepared = headless.prepare_terminal_task_files(tmp_path, task)
        assert prepared["terminal_source_present"] is (status == "completed")
        assert headless.terminal_task_files_ready(tmp_path, task, raw) is (status == "completed")
        assert effective_task_result(tmp_path, raw, materialize_artifacts=False)["status"] == status
        assert prepared["error"] == "" if status == "completed" else prepared["error"]
        assert not load_task_result(tmp_path, task["id"]).get("artifact_status")


@pytest.mark.parametrize("metadata,direct", [({"presence": {}}, False), ({"presence": {}, "source": "presence"}, True)])
def test_pooled_owner_requires_capture_even_with_presence_origin(tmp_path, metadata, direct):
    raw = write_task_result(tmp_path, "pooled", "completed", workspace_root=str(tmp_path / "workspace"),
                            metadata=metadata, _is_direct_chat=direct, execution_owner={"kind": "pooled"})
    assert not headless.terminal_task_files_ready(tmp_path, {**raw, "id": "pooled"}, raw)
    assert effective_task_result(tmp_path, raw, materialize_artifacts=False)["status"] == "running"


def test_unreadable_terminal_keeps_membership_and_raw_capture_unknown(tmp_path, monkeypatch):
    from ouroboros.startup_task_files import recover_terminal_task_files

    task = {"id": "unreadable", "workspace_root": str(tmp_path / "workspace")}
    write_task_result(tmp_path, task["id"], "completed", workspace_root=task["workspace_root"], artifact_status="future_status")
    obligations.drive_started(tmp_path, task)
    before = result_signature(tmp_path, task["id"])
    with monkeypatch.context() as fault:
        unreadable_result(fault, task["id"], root=tmp_path)
        fault.setattr(headless, "finalize_task_artifacts", lambda *_a: pytest.fail("unreadable source capture"))
        report = headless.prepare_terminal_task_files(tmp_path, task)
        assert report["terminal_source_present"] is None and report["error"]
        assert task["id"] in obligations.members(tmp_path, "pending_drives")
        report = recover_terminal_task_files(tmp_path, set())
        assert report["unresolved"] == [task["id"]] and report["errors"]
    assert result_signature(tmp_path, task["id"]) == before


def test_workspace_free_child_still_exports_memory_after_adoption(tmp_path, monkeypatch):
    child = headless.prepare_task_drive(tmp_path, "memory", "empty")
    write_task_result(child, "memory", "completed", result="Child answer")
    task = {"id": "memory", "drive_root": str(child), "memory_mode": "empty"}
    monkeypatch.setattr(headless, "write_workspace_patch_artifacts", lambda *_a, **_kw: pytest.fail("workspace-free capture"))
    prepared = headless.prepare_terminal_task_files(tmp_path, task)
    assert prepared["error"] == "" and prepared["terminal_source_present"] is True
    assert prepared["result"]["result"] == "Child answer"
    assert headless.terminal_task_files_ready(tmp_path, task, prepared["result"])
    assert (headless.task_artifacts_dir(tmp_path, "memory") / "memory_export.json").is_file()


def test_actual_presence_task_does_not_capture_its_workspace(tmp_path, monkeypatch):
    from ouroboros.presence_runner import _build_task
    from tests.test_presence_runner import _admission, _event

    task = _build_task(_admission(), _event(), drive_root=tmp_path, staged_files=())
    task["workspace_root"] = str(tmp_path / "unused")
    row = write_task_result(tmp_path, task["id"], "completed", **{k: v for k, v in task.items() if k != "id"})
    monkeypatch.setattr(headless, "finalize_task_artifacts", lambda *_a: pytest.fail("inline Presence capture"))
    assert headless.prepare_terminal_task_files(tmp_path, task)["error"] == ""
    assert headless.terminal_task_files_ready(tmp_path, task, row)
    assert effective_task_result(tmp_path, row, materialize_artifacts=False)["status"] == "completed"


def test_startup_leaves_protected_and_unaddressed_history_untouched(tmp_path, monkeypatch):
    """No migration scan. A later write can re-enroll old work for capture of its then-current tree."""
    from ouroboros import store
    from ouroboros.startup_task_files import recover_terminal_task_files

    for tid in ("protected", "historical"):
        write_task_result(tmp_path, tid, "completed", workspace_root=str(tmp_path / "workspace"))
    # Model a row whose old completion already retired membership. Recovery is
    # addressed; a later write can re-enroll this row, but startup does not scan for it.
    with store.open_store(tmp_path, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET pending_drive=0 WHERE task_id='historical'")
        txn.delete_row("pending_drives", "historical")
    before = {tid: result_signature(tmp_path, tid) for tid in ("protected", "historical")}
    monkeypatch.setattr(headless, "prepare_terminal_task_files", lambda *_a: pytest.fail("unowned recovery"))
    monkeypatch.setattr("ouroboros.task_results.list_task_results", lambda *_a, **_kw: pytest.fail("history scan"))
    report = recover_terminal_task_files(tmp_path, {"protected"})
    assert report == {"recovered": [], "unresolved": [], "protected": ["protected"], "errors": []}
    assert set(obligations.members(tmp_path, "pending_drives")) == {"protected"}
    assert {tid: result_signature(tmp_path, tid) for tid in before} == before


def test_later_write_reenrolls_history_and_discloses_recovery_time_capture(tmp_path):
    from ouroboros import store
    from ouroboros.startup_task_files import recover_terminal_task_files
    from ouroboros.utils import utc_now_iso

    root, workspace = tmp_path / "data", tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", str(workspace)], check=True, capture_output=True)
    (workspace / "answer.txt").write_text("The task's original work.\n", encoding="utf-8")
    original = write_task_result(root, "historical", "completed", result="Original answer, unchanged.",
        workspace_root=str(workspace), metadata={"workspace_preflight": {"git": {"head": "(unborn)"}}},
        started_at="2026-01-01T00:00:00Z", completed_at="2026-01-01T00:10:00Z",
        outcome_axes={"execution": {"status": "ok"}, "objective": {"status": "fail"},
                      "review": {"status": "fail", "aggregate_signals": ["FAIL"]}})
    with store.open_store(root, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET pending_drive=0 WHERE task_id='historical'")
        txn.delete_row("pending_drives", "historical")
    assert "historical" not in obligations.members(root, "pending_drives")
    write_task_result(root, "historical", "completed", project_id="later-placement")
    assert "historical" in obligations.members(root, "pending_drives")
    (workspace / "later-unrelated.txt").write_text("Written after that task ended.\n", encoding="utf-8")
    before_capture = utc_now_iso()
    report = recover_terminal_task_files(root, set())
    after_capture = utc_now_iso()
    assert report == {"recovered": ["historical"], "unresolved": [], "protected": [], "errors": []}
    stored = load_task_result(root, "historical")
    assert stored["result"] == original["result"]
    assert stored["outcome_axes"]["objective"] == original["outcome_axes"]["objective"]
    assert stored["outcome_axes"]["review"] == original["outcome_axes"]["review"]
    assert stored["completed_at"] == original["completed_at"]
    artifacts = headless.task_artifacts_dir(root, "historical")
    manifest = json.loads((artifacts / "workspace_patch.json").read_text())
    assert manifest["status"] == "ready_with_changes"
    assert before_capture <= manifest["created_at"] <= after_capture
    assert manifest["created_at"] > stored["completed_at"]
    assert manifest["base_provenance"] == "admission_head"
    assert "Current workspace at capture time" in manifest["base_explanation"]
    assert "later unrelated changes" in manifest["base_explanation"]
    assert "does not attribute authorship" in manifest["base_explanation"]
    assert "later-unrelated.txt" in manifest["untracked_included"]
    assert "+Written after that task ended." in (artifacts / "workspace.patch").read_text(encoding="utf-8")
    assert "historical" not in obligations.members(root, "pending_drives")
