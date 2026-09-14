"""Startup recovers saved child work before materialization or source cleanup."""

import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros import headless, observability, server_maintenance as maintenance
from ouroboros.task_results import load_task_result, write_task_result

from tests._cancel_intents_shared import _LiveProc
from tests._cancel_intents_shared import (  # noqa: F401  (autouse fixture applies on import)
    _reap_spawned_live_procs,
)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    from ouroboros import config, post_task_checkpoint
    from supervisor import queue, state, workers, active_activity

    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(maintenance, "DATA_DIR", root)
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", root / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "PENDING", queue.PENDING)
    monkeypatch.setattr(workers, "RUNNING", queue.RUNNING)
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    monkeypatch.setattr(state, "DRIVE_ROOT", root)
    monkeypatch.setattr(state, "STATE_PATH", root / "state/state.json")
    monkeypatch.setattr(post_task_checkpoint, "POST_TASK_SYNTHESIS_INFLIGHT", {})
    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: registry)
    monkeypatch.setenv("OUROBOROS_GC_RETENTION_DAYS", "0")
    return root, tmp_path / "repo"


def _terminal(root, task_id="saved", *, family="headless", phase="completed"):
    if family == "headless":
        child = headless.prepare_task_drive(root, task_id, "empty")
    else:
        child = root / "task_drives" / task_id
        child.mkdir(parents=True)
    ref = observability.persist_call(child, task_id=task_id, call_id="response", call_type="llm_response",
                                     payload={"answer": "full retained answer"})["manifest_ref"]
    write_task_result(child, task_id, "completed", result="full retained answer", artifact_status="ready",
                      trace_refs={"response": ref}, memory_mode="empty", drive_root=str(child),
                      root_phase_checkpoint={"post_task_synthesis": phase})
    write_task_result(root, task_id, "completed", child_drive_root=str(child),
                      root_phase_checkpoint={"post_task_synthesis": phase})
    return child


def _recovery(root, repo):
    return maintenance._run_startup_task_recovery(root, repo, skip_live_data=False, prior_worker_pids=set())


@pytest.mark.parametrize("family", ["headless", "task_drives"])
def test_saved_body_and_sources_precede_orphan_reader_and_actual_prune(roots, monkeypatch, family):
    root, repo = roots
    child = _terminal(root, family=family)
    seen = []
    def orphan_reader(_root, *, exclude_task_ids, expired_quizzes=None):
        row = load_task_result(root, "saved", strict=True)
        assert row["result"] == "full retained answer"
        manifest = observability.read_call_manifest_ref(root, row["trace_refs"]["response"], task_id="saved")
        assert observability.read_blob_ref(root, manifest["full_payload_ref"])["answer"] == row["result"]
        seen.append("orphan")
    monkeypatch.setattr("ouroboros.task_status.reconcile_orphaned_running_tasks", orphan_reader)
    monkeypatch.setattr("ouroboros.agent_task_pipeline.recover_pending_root_post_task_synthesis",
                        lambda *a, **k: seen.append("synthesis"))
    report = _recovery(root, repo)
    assert report["recovered"] == ["saved"] and report["unresolved"] == []
    assert seen == ["orphan", "synthesis"]
    with monkeypatch.context() as saved:
        saved.setattr(headless, "prepare_terminal_task_files", lambda *a: pytest.fail("already saved task recopied"))
        assert _recovery(root, repo)["recovered"] == []
    row = load_task_result(root, "saved")
    monkeypatch.setattr("ouroboros.retention.age_cutoff", lambda *a, **k: 4_000_000_000)
    maintenance._startup_prune_sweeps()
    assert not child.exists()
    manifest = observability.read_call_manifest_ref(root, row["trace_refs"]["response"], task_id="saved")
    assert observability.read_blob_ref(root, manifest["full_payload_ref"])["answer"] == "full retained answer"


def test_live_open_synthesis_and_pending_owner_wait_survive_while_dead_child_heals(roots, monkeypatch):
    from ouroboros import post_task_checkpoint
    from supervisor import queue

    root, repo = roots
    child = _terminal(root, "dead", phase="running")
    waiting_child = _terminal(root, "waiting", phase="running")
    queue.PENDING.append({"id": "waiting", "_owner_wait_resume": {"wait_id": "owner-question"}})
    live = write_task_result(root, "live", "completed", result="answer already delivered",
                             root_phase_checkpoint={"post_task_synthesis": "running"})
    post_task_checkpoint.POST_TASK_SYNTHESIS_INFLIGHT[(str(root.resolve()), "live")] = SimpleNamespace(closed=False)
    waiting_bytes = (waiting_child / "task_results/waiting.json").read_bytes()
    reader = []
    monkeypatch.setattr("ouroboros.task_status.load_effective_task_result",
                        lambda root, tid: reader.append(tid) or pytest.fail("terminal rows need no orphan materialization"))
    report = _recovery(root, repo)
    assert report["protected"] == ["live", "waiting"]
    assert report["recovered"] == ["dead"]
    assert load_task_result(root, "live") == live
    assert (waiting_child / "task_results/waiting.json").read_bytes() == waiting_bytes
    assert load_task_result(root, "waiting")["root_phase_checkpoint"]["post_task_synthesis"] == "running"
    assert load_task_result(root, "dead")["root_phase_checkpoint"]["post_task_synthesis"] == "degraded"
    assert load_task_result(root, "dead")["result"] == "full retained answer"
    assert child.exists() and reader == []


def test_orphan_exclusion_filters_before_effective_materialization(roots, monkeypatch):
    from ouroboros.task_status import reconcile_orphaned_running_tasks
    root, _ = roots
    for tid in ["live", "dead"]:
        write_task_result(root, tid, "running", result="original")
    read = []
    def effective(root, tid):
        read.append(tid)
        return {"task_id": tid, "status": "failed", "result": "proven orphan"}
    monkeypatch.setattr("ouroboros.task_status.load_effective_task_result", effective)
    assert reconcile_orphaned_running_tasks(root, exclude_task_ids={"live"}) == 1
    assert read == ["dead"]
    assert load_task_result(root, "live")["status"] == "running"
    assert load_task_result(root, "dead")["status"] == "failed"


def test_startup_recovery_skips_symlinked_external_task_drive(roots):
    root, repo = roots
    external = root.parent / "external-task-drive"
    external.mkdir()
    write_task_result(external, "evil", "completed", result="outside data")
    drives = root / "task_drives"
    drives.mkdir()
    (drives / "evil").symlink_to(external, target_is_directory=True)
    report = _recovery(root, repo)
    assert report["recovered"] == []
    assert report["unresolved"] == []
    assert not (root / "task_results" / "evil.json").exists()


@pytest.mark.parametrize("family", ["headless", "task_drives"])
@pytest.mark.parametrize("status", ["failed", "cancelled", "rejected_duplicate"])
@pytest.mark.parametrize("child_status", [None, "running"])
def test_host_terminal_without_child_terminal_does_not_disable_retention(
    roots, monkeypatch, family, status, child_status,
):
    root, repo = roots
    task_id = "host-stopped"
    child = (headless.prepare_task_drive(root, task_id, "empty") if family == "headless"
             else root / "task_drives" / task_id)
    child.mkdir(parents=True, exist_ok=True)
    if child_status:
        write_task_result(child, task_id, child_status, result="unfinished work")
    stored = write_task_result(root, task_id, status, result="Host ended this execution",
                               child_drive_root=str(child))
    # Repeated boots do not turn confirmed absence of a child terminal into
    # a permanent save obligation that blocks unrelated startup retention.
    for _ in range(2):
        report = _recovery(root, repo)
        assert report["unresolved"] == report["errors"] == report["protected"] == []
        assert load_task_result(root, task_id, strict=True) == stored
    monkeypatch.setattr("ouroboros.retention.age_cutoff", lambda *a, **k: 4_000_000_000)
    maintenance._startup_prune_sweeps(preserve_task_sources=bool(
        report["unresolved"] or report["protected"] or report["errors"]))
    assert not child.exists()
    assert load_task_result(root, task_id, strict=True) == stored


def test_no_provider_unrestored_wait_is_preserved_but_other_saved_work_recovers(roots, monkeypatch):
    root, repo = roots
    _terminal(root, "dead")
    waiting = _terminal(root, "waiting", phase="running")
    (root / "state/queue_snapshot.json").write_text(json.dumps({"pending": [{"task": {
        "id": "waiting", "_owner_wait_resume": {"wait_id": "saved-question"}}}]}))
    before = load_task_result(root, "waiting")
    report = _recovery(root, repo)
    assert report["protected"] == ["waiting"] and report["recovered"] == ["dead"]
    assert load_task_result(root, "waiting") == before and waiting.exists()


def test_live_data_test_guard_precedes_every_recovery_read(roots, monkeypatch):
    root, repo = roots
    monkeypatch.setattr(maintenance, "_migrate_startup_cancel_latches", lambda *a: pytest.fail("live read"))
    assert maintenance._run_startup_task_recovery(root, repo, skip_live_data=True)["recovered"] == []


def test_failed_first_save_preserves_child_and_followup_attachment_through_prune(roots, monkeypatch):
    from ouroboros.artifacts import stage_task_attachments, task_artifact_dir_path
    from ouroboros.owner_mailbox import write_owner_message, _mailbox_path

    root, repo = roots
    child = _terminal(root)
    upload = root / "uploads" / ("a" * 32 + "_answer.txt")
    upload.parent.mkdir()
    upload.write_text("accepted follow-up file")
    manifest = stage_task_attachments(child, "saved", [str(upload)] * 26)
    assert len(manifest) == 26 and all(row["status"] == "staged" for row in manifest)
    assert write_owner_message(child, "use all attached material", "saved", attachment_manifest=manifest)
    early = load_task_result(root, "saved")
    called = []
    monkeypatch.setattr("ouroboros.task_status.reconcile_orphaned_running_tasks",
                        lambda *a, **k: called.append(k["exclude_task_ids"]))
    monkeypatch.setattr("ouroboros.agent_task_pipeline.recover_pending_root_post_task_synthesis",
                        lambda *a, **k: called.append(k["exclude_task_ids"]))
    with monkeypatch.context() as failure:
        failure.setattr(headless, "write_task_result", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        report = _recovery(root, repo)
    assert report["unresolved"] == ["saved"]
    assert called == [{"saved"}, {"saved"}]
    assert load_task_result(root, "saved") == early
    maintenance._startup_prune_sweeps(preserve_task_sources=True)
    assert child.exists() and _mailbox_path(child, "saved").exists()
    recovered = _recovery(root, repo)
    assert recovered["recovered"] == ["saved"]
    assert load_task_result(root, "saved")["child_ref_promotion"]["pending_refs"] == []
    for row in manifest:
        assert (task_artifact_dir_path(root, "saved") / row["relpath"]).read_text() == "accepted follow-up file"


@pytest.mark.parametrize("pids", [None, {777}])
def test_unknown_or_live_prior_worker_defers_without_sleep_or_capture(roots, monkeypatch, pids):
    root, repo = roots
    child = _terminal(root)
    monkeypatch.setattr("ouroboros.platform_layer.pid_is_alive", lambda pid: True)
    monkeypatch.setattr(headless, "prepare_terminal_task_files", lambda *a: pytest.fail("live owner capture"))
    monkeypatch.setattr("ouroboros.task_status.reconcile_orphaned_running_tasks", lambda *a, **k: pytest.fail("live materialization"))
    report = maintenance._run_startup_task_recovery(root, repo, skip_live_data=False, prior_worker_pids=pids)
    assert report["errors"] == ["prior_worker_ownership_unconfirmed"] and child.exists()


def test_worker_pid_evidence_is_captured_before_new_pool_overwrites_it(roots, monkeypatch):
    root, _ = roots
    path = root / "state/worker_pids.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"server_pid": 444, "workers": [{"pid": 555}]}))
    monkeypatch.setattr("ouroboros.process_custody._read_ledger_strict",
                        lambda root: (True, [{"pid": 666, "purpose": "worker:1"}, {"pid": 999, "purpose": "claudexor"}]))
    captured = maintenance._startup_worker_pids(root)
    path.write_text(json.dumps({"workers": [{"pid": 777}]}))
    assert captured == {444, 555, 666}
    path.write_text("{")
    assert maintenance._startup_worker_pids(root) is None


def test_missing_or_nonterminal_child_does_not_resume_model_work(roots, monkeypatch):
    root, repo = roots
    child = _terminal(root)
    (child / "task_results/saved.json").unlink()
    monkeypatch.setattr(headless, "prepare_terminal_task_files", lambda *a: pytest.fail("no terminal source"))
    excluded = []
    monkeypatch.setattr("ouroboros.task_status.reconcile_orphaned_running_tasks", lambda *a, **k: excluded.append(k["exclude_task_ids"]))
    monkeypatch.setattr("ouroboros.agent_task_pipeline.recover_pending_root_post_task_synthesis", lambda *a, **k: excluded.append(k["exclude_task_ids"]))
    assert _recovery(root, repo)["unresolved"] == ["saved"]
    write_task_result(child, "saved", "running", result="unfinished")
    assert _recovery(root, repo)["unresolved"] == ["saved"]
    assert excluded == [{"saved"}] * 4


def _started_split_root(root, task_id="split", *, canonical_status="scheduled", started=True):
    from ouroboros.utils import utc_now_iso

    child = headless.prepare_task_drive(root, task_id, "empty")
    write_task_result(root, task_id, canonical_status, delegation_role="root",
                      result="admitted", ts="2020-01-01T00:00:00+00:00")
    write_task_result(child, task_id, "running", delegation_role="root", drive_root=str(child),
                      budget_drive_root=str(root), result="unfinished child work",
                      ts="2020-01-01T00:00:01+00:00",
                      **({"started_at": "2020-01-01T00:00:01+00:00"} if started else {}))
    (root / "state/queue_snapshot.json").write_text(json.dumps({
        "ts": utc_now_iso(), "pending": [], "running": [],
    }))
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs/events.jsonl").write_text(json.dumps({
        "ts": "2020-01-01T00:01:00+00:00", "type": "worker_boot",
    }) + "\n")
    return child


def test_actual_split_root_start_is_canonical_and_existing_orphan_sweep_settles_it(roots, monkeypatch):
    from ouroboros.agent import OuroborosAgent
    from ouroboros.task_status import reconcile_orphaned_running_tasks

    root, _ = roots
    child = _started_split_root(root)
    actor = SimpleNamespace(env=SimpleNamespace(drive_root=child, budget_drive_root=root),
                            _task_started_ts=1577836801.0)
    OuroborosAgent._persist_running_record(actor, {
        "id": "split", "delegation_role": "root", "budget_drive_root": str(root),
        "drive_root": str(child), "_is_direct_chat": False,
    })
    started = load_task_result(root, "split")
    assert started["status"] == "running" and started["child_drive_root"] == str(child)
    assert started["started_at"] == "2020-01-01T00:00:01+00:00"
    assert reconcile_orphaned_running_tasks(root) == 1
    settled = load_task_result(root, "split")
    assert settled["status"] == "failed"
    assert settled["reason_code"] == "orphaned_running_after_worker_restart"
    assert settled["outcome_axes"]["execution"]["status"] == "infra_failed"


def test_legacy_scheduled_split_root_is_rebound_only_from_proven_child_start(roots, monkeypatch):
    from supervisor import queue, workers

    root, repo = roots
    child = _started_split_root(root)
    monkeypatch.setattr("ouroboros.agent.run_llm_loop", lambda *a, **k: pytest.fail("no task replay"))
    monkeypatch.setattr("ouroboros.agent_task_pipeline.recover_pending_root_post_task_synthesis", lambda *a, **k: None)
    report = _recovery(root, repo)
    assert report["rebound"] == ["split"]
    assert report["unresolved"] == report["errors"] == []
    result = load_task_result(root, "split")
    assert result["status"] == "failed"
    assert result["reason_code"] == "orphaned_running_after_worker_restart"
    assert result["child_drive_root"] == str(child)
    assert "unfinished child work" in result["result"]
    assert queue.PENDING == [] and queue.RUNNING == {} and workers.WORKERS == {}
    before = (root / "task_results/split.json").read_bytes()
    assert not _recovery(root, repo).get("rebound")
    assert (root / "task_results/split.json").read_bytes() == before


@pytest.mark.parametrize("condition", ["pending", "owner_wait", "live", "no_start", "missing_queue", "stale_queue", "no_boot", "cancel_pending"])
def test_legacy_split_start_recovery_preserves_unproven_or_owned_work(roots, monkeypatch, condition):
    from supervisor import queue
    from ouroboros.cancel_intents import request_cancel

    root, repo = roots
    child = _started_split_root(root, started=condition != "no_start")
    if condition in {"pending", "owner_wait"}:
        queue.PENDING.append({"id": "split", **({"_owner_wait_resume": {"wait_id": "question"}} if condition == "owner_wait" else {})})
    elif condition == "live":
        queue.RUNNING["split"] = {"task": {"id": "split"}}
    elif condition == "missing_queue":
        (root / "state/queue_snapshot.json").unlink()
    elif condition == "stale_queue":
        (root / "state/queue_snapshot.json").write_text('{"ts":"2020-01-01T00:02:00+00:00","pending":[],"running":[]}')
    elif condition == "no_boot":
        (root / "logs/events.jsonl").write_text("")
    elif condition == "cancel_pending":
        request_cancel(root, "split", reason="owner stop", source="test")
    before = (root / "task_results/split.json").read_bytes()
    child_before = (child / "task_results/split.json").read_bytes()
    monkeypatch.setattr("ouroboros.agent_task_pipeline.recover_pending_root_post_task_synthesis", lambda *a, **k: None)
    report = _recovery(root, repo)
    assert not report.get("rebound")
    assert (root / "task_results/split.json").read_bytes() == before
    assert (child / "task_results/split.json").read_bytes() == child_before


def test_late_split_root_start_cannot_replace_canonical_terminal(roots):
    from ouroboros.agent import OuroborosAgent

    root, _ = roots
    child = _started_split_root(root, canonical_status="cancelled")
    actor = SimpleNamespace(env=SimpleNamespace(drive_root=child, budget_drive_root=root),
                            _task_started_ts=1577836801.0)
    task = {"id": "split", "delegation_role": "root", "budget_drive_root": str(root), "drive_root": str(child)}
    before = (root / "task_results/split.json").read_bytes()
    OuroborosAgent._persist_running_record(actor, task)
    assert (root / "task_results/split.json").read_bytes() == before


@pytest.mark.parametrize("failure", [False, True])
def test_periodic_bulk_work_does_not_block_drain_or_duplicate_sweep(roots, monkeypatch, failure):
    root, _ = roots
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    lock = threading.Lock()
    clock = [100.0]
    calls = []
    monkeypatch.setattr(maintenance, "_CANCEL_INTENT_SWEEP_LOCK", lock)
    monkeypatch.setattr(maintenance, "_LAST_CANCEL_INTENT_SWEEP", [0.0])
    monkeypatch.setattr(maintenance, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr("supervisor.task_lifecycle.sweep_cancel_intents", lambda: calls.append("cancel"))
    monkeypatch.setattr("supervisor.terminal_delivery.replay_pending_deliveries", lambda root: calls.append("delivery"))
    def bulk(root):
        calls.append("refs")
        entered.set()
        assert release.wait(3)
        done.set()
        if failure:
            raise OSError("copy failed")
    monkeypatch.setattr(observability, "retry_pending_child_ref_promotions", bulk)
    maintenance._periodic_supervisor_maintenance([100.0], [100.0])
    assert entered.wait(2)
    try:
        clock[0] = 125
        maintenance._periodic_supervisor_maintenance([125.0], [125.0])
        assert calls == ["cancel", "delivery", "refs"]  # drain returned while I/O is still held
    finally:
        release.set()
    assert done.wait(2) and lock.acquire(timeout=2)
    lock.release()
    maintenance._periodic_supervisor_maintenance([125.0], [125.0])
    assert lock.acquire(timeout=2)
    lock.release()
    assert calls == ["cancel", "delivery", "refs"] * 2


def test_thread_start_failure_releases_maintenance_latch(roots, monkeypatch):
    lock = threading.Lock()
    monkeypatch.setattr(maintenance, "_CANCEL_INTENT_SWEEP_LOCK", lock)
    monkeypatch.setattr(maintenance, "_LAST_CANCEL_INTENT_SWEEP", [0.0])
    monkeypatch.setattr(maintenance, "time", SimpleNamespace(time=lambda: 100.0))
    monkeypatch.setattr(maintenance, "threading", SimpleNamespace(Thread=lambda **k: (_ for _ in ()).throw(RuntimeError("thread unavailable"))))
    maintenance._periodic_supervisor_maintenance([100.0], [100.0])
    assert lock.acquire(blocking=False)
    lock.release()


def test_real_supervisor_orders_custody_recovery_before_prune(roots, monkeypatch, tmp_path):
    import server
    from tests.test_server_shutdown import _supervisor_harness
    from supervisor import queue, workers

    _supervisor_harness(monkeypatch, tmp_path, ["stop"])
    order = []
    monkeypatch.setattr(server, "_migrate_startup_cancel_latches", lambda root: order.append("migrate"))
    monkeypatch.setattr(server, "_startup_worker_pids", lambda root: order.append("capture-pids") or {777})
    monkeypatch.setattr(queue, "restore_pending_from_snapshot",
                        lambda **_kw: order.append("restore") or 0)
    monkeypatch.setattr(workers, "kill_workers", lambda **k: order.append("kill"))
    monkeypatch.setattr(workers, "spawn_workers", lambda n: order.append("spawn"))
    monkeypatch.setattr(server, "_startup_custody_sweep", lambda: order.append("custody"))
    def recover(root, repo, **kw):
        assert kw["prior_worker_pids"] == {777}
        order.append("recover")
        return {"unresolved": ["saved"], "protected": [], "errors": []}
    monkeypatch.setattr(server, "_run_startup_task_recovery", recover)
    monkeypatch.setattr(server, "_startup_prune_sweeps", lambda **kw: order.append(("prune", kw["preserve_task_sources"])))
    server._run_supervisor({})
    assert server._supervisor_error is None
    assert order == ["migrate", "capture-pids", "restore", "kill", "spawn", "custody", "recover", ("prune", True)]


def test_supervisor_init_failure_keeps_boot_recovery_owner():
    import ast
    import inspect
    import server
    tree = ast.parse(inspect.getsource(server._run_supervisor))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "_run_startup_task_recovery"]
    assert len(calls) == 2
    assert any(any(isinstance(parent, ast.ExceptHandler) for parent in ast.walk(tree))
               for _ in calls)


def test_lifespan_does_not_race_recovery_against_provider_supervisor():
    import ast
    import inspect
    import textwrap
    import server

    tree = ast.parse(textwrap.dedent(inspect.getsource(server.lifespan)))
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "not has_startup_ready_provider(settings)"]
    assert len(branches) == 1
    recovery = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "_run_startup_task_recovery"]
    assert len(recovery) == 1 and recovery[0] in list(ast.walk(branches[0]))
    assert "skip_live_data=pytest_default_real_data_dir" in ast.unparse(recovery[0])


def test_orphan_reconcile_closes_the_open_quiz_and_its_paired_wait(roots, monkeypatch):
    """The healer writes a terminal OFF the task-done seam, so it owes that seam's
    domain reconciliation itself: a task whose record says 'ended' while its card
    still shows an open question, and whose wait never releases, is the class."""
    from ouroboros import owner_quiz
    from ouroboros.task_status import reconcile_orphaned_running_tasks
    root, _ = roots

    for tid in ("ghost-root", "ghost-child"):
        write_task_result(root, tid, "running", result="original")
        owner_quiz.record_asked(root, tid, quiz_id=f"{tid}-q", question="Which folder?",
                                options=["A", "B"], wait_for_answer=True)
        write_task_result(root, tid, "running", owner_wait={"state": "waiting", "quiz_id": f"{tid}-q"})
    monkeypatch.setattr(
        "ouroboros.task_status.load_effective_task_result",
        lambda _root, tid: {"task_id": tid, "status": "failed", "result": "proven orphan"},
    )

    assert reconcile_orphaned_running_tasks(root) == 2

    for tid in ("ghost-root", "ghost-child"):
        stored = load_task_result(root, tid)
        assert stored["status"] == "failed"
        assert owner_quiz.quiz_states(root, tid)[f"{tid}-q"]["state"] == "expired_terminal"
        assert stored["owner_wait"]["state"] == "expired_terminal"

    # Idempotent against the task-done seam running the same legs afterwards.
    from supervisor.queue_transitions import reconcile_terminal_task_projections

    reconcile_terminal_task_projections(root, "ghost-root")
    assert owner_quiz.quiz_states(root, "ghost-root")["ghost-root-q"]["state"] == "expired_terminal"


SERVER_STOPPED_CANCEL = "Task cancelled: the server stopped while this task was still running."


def _interrupted_running_row(root, task_id, *, chat_id=1, age_sec=120.0, **fields):
    """A row the previous generation left RUNNING, written the way a boot finds it.

    The snapshot the shutdown left still names it (restore reads that list), the
    heartbeat is older than the healer's grace window, and a worker_boot row
    after it is the healer's positive death evidence.
    """
    import datetime as dt

    from ouroboros.utils import append_jsonl, utc_now_iso

    stamp = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age_sec))
    write_task_result(root, task_id, "running", chat_id=chat_id,
                      ts=stamp.isoformat().replace("+00:00", "Z"), **fields)
    append_jsonl(root / "logs" / "events.jsonl",
                 {"ts": utc_now_iso(), "type": "worker_boot", "worker_id": 1})
    _snapshot_naming_running(root, task_id, chat_id=chat_id)


def _snapshot_naming_running(root, task_id, *, chat_id=1):
    """The snapshot a shutdown left behind, naming one row as still running."""
    from ouroboros.utils import utc_now_iso

    (root / "state").mkdir(parents=True, exist_ok=True)
    (root / "state" / "queue_snapshot.json").write_text(json.dumps({
        "ts": utc_now_iso(), "pending": [], "acceptance_fences": [], "budget_root_fences": [],
        "running": [{"id": task_id, "task": {"id": task_id, "chat_id": chat_id}}],
    }), encoding="utf-8")


def test_the_boot_healer_leaves_a_fenced_row_to_cancellation_custody(roots, monkeypatch):
    """The real boot order: restore mints the fence, the startup re-persist empties
    the snapshot, and startup recovery runs inside the watchdog's ten-second
    minimum age. Healing there would settle the row as infra_failed and the later
    sweep would answer already_settled, leaving a Failed card under a boot line
    that promised a cancellation (owner Q11=A)."""
    import time

    from ouroboros.task_status import reconcile_orphaned_running_tasks
    from supervisor import queue as queue_module, task_lifecycle
    root, _ = roots

    _interrupted_running_row(root, "ghost-fenced")
    fenced: list = []
    assert queue_module.restore_pending_from_snapshot(terminalized=fenced) == 0
    assert fenced == ["ghost-fenced"]
    queue_module.persist_queue_snapshot(reason="startup")

    assert reconcile_orphaned_running_tasks(root) == 0
    assert load_task_result(root, "ghost-fenced")["status"] == "running"

    assert task_lifecycle.sweep_cancel_intents(now=time.time() + 60)["ghost-fenced"] == "cancelled"
    stored = load_task_result(root, "ghost-fenced")
    assert stored["status"] == "cancelled" and stored["result"] == SERVER_STOPPED_CANCEL


def test_the_healer_tells_rendered_cards_their_question_expired(roots, monkeypatch):
    """The durable projection is only half of it: the seam that normally expires a
    quiz also sends the live frame, and the surfaces Ouroboros runs on have no
    reload affordance. A healed terminal owes the same frame."""
    from ouroboros import owner_quiz
    from supervisor import message_bus, queue as queue_module
    root, repo = roots

    write_task_result(root, "ghost-asked", "running", chat_id=1)
    owner_quiz.record_asked(root, "ghost-asked", quiz_id="q9", question="Which folder?",
                            options=["A", "B"], wait_for_answer=True)
    _interrupted_running_row(root, "ghost-asked")
    queue_module.persist_queue_snapshot(reason="startup")
    frames: list = []
    monkeypatch.setattr(message_bus, "get_bridge",
                        lambda: SimpleNamespace(send_quiz_state=lambda *args: frames.append(args)))

    _recovery(root, repo)

    assert load_task_result(root, "ghost-asked")["status"] == "failed"
    assert owner_quiz.quiz_states(root, "ghost-asked")["q9"]["state"] == "expired_terminal"
    assert frames == [("q9", "ghost-asked", "expired_terminal")]


def _restore_rows(root, row_type):
    path = root / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("type") == row_type]


def test_a_running_row_with_no_durable_result_is_never_fenced(roots):
    """Custody settles an intent for an id with no durable row as not_found and
    writes nothing, so fencing that row would put a cancellation in the boot line
    that never happens.

    A failed RUNNING mirror is not this branch: admission writes the durable
    scheduled row, so that row stays readable and is fenced like any other. This
    branch is a missing or deleted admission record, logged and never fenced.
    """
    from ouroboros import cancel_intents
    from supervisor import queue as queue_module
    root, _ = roots

    _interrupted_running_row(root, "has-a-row")
    snapshot = json.loads((root / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    snapshot["running"].append({"id": "never-recorded", "task": {"id": "never-recorded", "chat_id": 1}})
    (root / "state" / "queue_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    fenced: list = []
    assert queue_module.restore_pending_from_snapshot(terminalized=fenced) == 0
    assert fenced == ["has-a-row"]
    assert cancel_intents.active_intent(root, "never-recorded") is None
    unrecorded = _restore_rows(root, "queue_restore_running_row_without_result")
    assert unrecorded and unrecorded[-1]["task_ids"] == ["never-recorded"]


def test_a_fail_closed_restore_still_records_the_rows_it_fenced(roots):
    """The two fail-closed exits return before the restore ledger row, so the
    boot line named fenced ids that no durable row recorded. The fence happens
    either way, so its record must too."""
    from supervisor import queue as queue_module
    root, _ = roots

    _interrupted_running_row(root, "ghost-in-a-broken-snapshot")
    snapshot = json.loads((root / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    snapshot["acceptance_fences"] = ["not-a-fence-object"]
    (root / "state" / "queue_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")

    fenced: list = []
    assert queue_module.restore_pending_from_snapshot(terminalized=fenced) == 0
    assert fenced == ["ghost-in-a-broken-snapshot"]
    assert _restore_rows(root, "queue_restore_invalid_acceptance_fences")
    recorded = _restore_rows(root, "queue_restored_from_snapshot")
    assert recorded and recorded[-1]["terminalized_running"] == ["ghost-in-a-broken-snapshot"]


def test_the_healer_gates_a_root_on_its_real_liveness_evidence(roots):
    """Assignment now mirrors RUNNING for roots, so this sweep decides their fate
    too and its REAL gates have to be exercised, not a monkeypatched projection:
    live queue ownership, an unusable snapshot and the grace window each keep the
    row, and only the boot shape (fresh snapshot without it, stale heartbeat, a
    worker booted after it) settles it."""
    from ouroboros.task_status import reconcile_orphaned_running_tasks
    from supervisor import queue as queue_module
    root, _ = roots

    _interrupted_running_row(root, "root-ghost", root_task_id="root-ghost",
                             result="Assigned to a worker.")

    # The snapshot the shutdown left still names it: that is live ownership.
    assert reconcile_orphaned_running_tasks(root) == 0
    assert load_task_result(root, "root-ghost")["status"] == "running"

    # No snapshot at all cannot prove a dead owner either (the work order's
    # "with no snapshot" case is a refusal, not the reconciling one).
    (root / "state" / "queue_snapshot.json").unlink()
    assert reconcile_orphaned_running_tasks(root) == 0
    assert load_task_result(root, "root-ghost")["status"] == "running"

    # The boot re-persist leaves a fresh snapshot without the row.
    queue_module.persist_queue_snapshot(reason="startup")
    assert reconcile_orphaned_running_tasks(root) == 1
    healed = load_task_result(root, "root-ghost")
    assert healed["status"] == "failed"
    assert healed["reason_code"] == "orphaned_running_after_worker_restart"

    # A root assigned seconds ago is never reconciled: the grace window holds
    # even once the snapshot no longer names it.
    _interrupted_running_row(root, "root-just-assigned", age_sec=1.0,
                             result="Assigned to a worker.")
    queue_module.persist_queue_snapshot(reason="startup")
    assert reconcile_orphaned_running_tasks(root) == 0
    assert load_task_result(root, "root-just-assigned")["status"] == "running"


def test_the_boot_healer_still_settles_a_running_row_nothing_owns(roots):
    """The skip is the intent, not the shape: an orphan with no cancel intent is
    reconciled exactly as before."""
    from ouroboros.task_status import reconcile_orphaned_running_tasks
    from supervisor import queue as queue_module
    root, _ = roots

    _interrupted_running_row(root, "ghost-unowned")
    queue_module.persist_queue_snapshot(reason="startup")

    assert reconcile_orphaned_running_tasks(root) == 1
    stored = load_task_result(root, "ghost-unowned")
    assert stored["status"] == "failed"
    assert stored["reason_code"] == "orphaned_running_after_worker_restart"


LIVE_WORKER_CANCEL = "Running task cancelled and worker terminated."


def _surviving_worker(root, task_id, *, chat_id=1):
    """A REAL child process behind the worker surface, recorded the way the pool
    records the owner of a running task.

    A worker is a session leader, so SIGTERM to the server does not end it: the
    process outlives the shutdown and is still alive while the next generation
    restores the queue. That is the state this stands in for.
    """
    from supervisor import queue as queue_module, workers

    proc = _LiveProc()
    workers.WORKERS[0] = SimpleNamespace(wid=0, proc=proc, busy_task_id=task_id, reaping=False)
    queue_module.RUNNING[task_id] = {"task": {"id": task_id, "chat_id": chat_id}, "worker_id": 0}
    return proc


@pytest.mark.serial
def test_a_worker_that_survived_the_shutdown_is_killed_before_the_terminal_is_written(
    roots, monkeypatch,
):
    """Custody claims and kills first, and only a confirmed-dead worker gets a
    terminal row: that is why the boot order (restore, then the reap) is not a
    correctness condition and why the fence never races a second writer.

    The cause is one producer for both lanes. In a real boot the pool is empty
    when restore runs (spawn_workers comes after kill_workers, server.py
    :658-660), so a fence ordinarily settles through the miss lane, which is
    what test_the_boot_healer_leaves_a_fenced_row_to_cancellation_custody pins.
    A worker that outlived SIGTERM and is still claimable settles here instead,
    and the owner must read the SAME sentence either way."""
    import time

    from ouroboros import cancel_intents, task_results
    from supervisor import queue as queue_module, task_lifecycle, workers
    root, _ = roots

    task_id = "shutdown-survivor"
    _interrupted_running_row(root, task_id)
    proc = _surviving_worker(root, task_id)
    # Slot hygiene AFTER the durable boundary: a real respawn would start a
    # second worker process, which this test neither observes nor owns.
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)

    writes: list = []
    real_write = task_results.write_task_result

    def observe(drive_root, written_id, status, **fields):
        if str(written_id) == task_id:
            writes.append({"status": str(status), "worker_alive": proc.is_alive(),
                           "result": str(fields.get("result") or "")})
        return real_write(drive_root, written_id, status, **fields)

    monkeypatch.setattr(task_results, "write_task_result", observe)

    fenced: list = []
    assert queue_module.restore_pending_from_snapshot(terminalized=fenced) == 0
    assert fenced == [task_id]
    assert proc.is_alive(), "restore mints an intent; it never kills or writes"
    assert writes == []
    assert cancel_intents.active_intent(root, task_id)["reason"] == "server_shutdown"

    assert task_lifecycle.sweep_cancel_intents(now=time.time() + 60)[task_id] == "cancelled"

    assert not proc.is_alive(), "custody must confirm the death it reports"
    terminal = [row for row in writes if row["status"] == "cancelled"]
    assert len(terminal) == 1, f"exactly one terminal writer, saw {writes}"
    assert terminal[0]["worker_alive"] is False, "the kill precedes the terminal write"
    stored = load_task_result(root, task_id)
    assert stored["status"] == "cancelled"
    assert cancel_intents.active_intent(root, task_id) is None
    assert stored["result"] == terminal[0]["result"] == SERVER_STOPPED_CANCEL


@pytest.mark.serial
def test_an_ordinary_cancel_of_a_live_worker_keeps_stating_the_kill(roots, monkeypatch):
    """The shared producer speaks for a shutdown fence and for nothing else.

    An owner (or parent) cancel of a running task has no shutdown cause to state,
    so the kill path keeps the only sentence it has ever written there."""
    from ouroboros import cancel_intents
    from supervisor import task_lifecycle, workers
    root, _ = roots

    task_id = "owner-cancelled-live"
    _interrupted_running_row(root, task_id)
    proc = _surviving_worker(root, task_id)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)

    cancel_intents.request_cancel(root, task_id, reason="no longer needed",
                                  requested_by="owner")

    assert task_lifecycle.cancel_task_custody(task_id) == task_lifecycle.CANCEL_CANCELLED

    assert not proc.is_alive()
    stored = load_task_result(root, task_id)
    assert stored["status"] == "cancelled"
    assert stored["result"] == LIVE_WORKER_CANCEL


@pytest.mark.serial
def test_the_fence_is_the_same_on_either_side_of_the_reap(roots):
    """Restore never consults process liveness, so whether the previous
    generation's worker died before or after the fence was minted must not change
    anything the owner sees.

    This is the real boot shape: the pool is empty when restore runs (workers_init
    creates no process and spawn_workers comes after kill_workers, server.py
    :658-660), so the worker that outlived SIGTERM is an orphan of the previous
    generation, not a slot this process owns. The live order (restore, then
    kill_workers) stays exactly as it is, pinned by
    tests/test_server_shutdown.py::test_supervisor_startup_restores_queue_before_worker_reset.
    """
    import time

    from ouroboros import cancel_intents
    from supervisor import queue as queue_module, task_lifecycle, workers
    root, _ = roots

    def reap(proc):
        proc.terminate()
        proc.join(timeout=5)
        assert not proc.is_alive()

    def boot(task_id, *, reap_first):
        survivor = _LiveProc()
        _interrupted_running_row(root, task_id)
        if reap_first:
            reap(survivor)
        notice: list = []
        restored = queue_module.restore_pending_from_snapshot(terminalized=notice)
        alive_at_mint = survivor.is_alive()
        workers.kill_workers(preserve_pending=True)
        if not reap_first:
            reap(survivor)
        minted = cancel_intents.active_intent(root, task_id) or {}
        outcome = task_lifecycle.sweep_cancel_intents(now=time.time() + 60).get(task_id)
        settled = load_task_result(root, task_id)
        # A later boot re-reading the same pre-restart snapshot must not fence a
        # row custody already settled, and must not touch its terminal text.
        _snapshot_naming_running(root, task_id)
        replay: list = []
        queue_module.restore_pending_from_snapshot(terminalized=replay)
        return alive_at_mint, {
            "restored_pending": restored,
            "boot_notice": len(notice),
            "fence_reason": (minted.get("reason"), minted.get("source")),
            "outcome": outcome,
            "status": settled.get("status"),
            "result": settled.get("result"),
            "replay_notice": replay,
            "replay_intent": cancel_intents.active_intent(root, task_id),
            "replay_result": (load_task_result(root, task_id) or {}).get("result"),
        }

    alive_at_mint, after_the_fence = boot("reaped-after-restore", reap_first=False)
    dead_at_mint, before_the_fence = boot("reaped-before-restore", reap_first=True)

    # The two compositions really are the two sides of the reap.
    assert (alive_at_mint, dead_at_mint) == (True, False)
    assert after_the_fence == before_the_fence
    assert after_the_fence == {
        "restored_pending": 0,
        "boot_notice": 1,
        "fence_reason": ("server_shutdown", "snapshot_restore"),
        "outcome": "cancelled",
        "status": "cancelled",
        "result": SERVER_STOPPED_CANCEL,
        "replay_notice": [],
        "replay_intent": None,
        "replay_result": SERVER_STOPPED_CANCEL,
    }
