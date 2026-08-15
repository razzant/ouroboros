"""The donor's admission spec, re-pointed at OUR authorities (RWS v2 §3.3).

The donor (`tests/test_remote_admission.py`, 1787 lines) proved its admission invariants
against a THIRD task population — a `requested` state machine in `supervisor/task_lifecycle.py`
holding tasks that were neither pending nor running. This branch deliberately has no such
state: durable transitions stay in `supervisor/queue.py` under `_queue_lock`, and a task is
either in the queue or it is not. So the invariants are kept and the mechanism is not.

Each test below names the donor case it descends from. What the two architectures share is
the hard part: cancellation must own an admission that is still in flight, fences must be
evaluated at the moment of the durable transition rather than at request time, a restart
must not leave a task in limbo, and a sealed placement must not be editable after the fact.
"""
from __future__ import annotations

import pathlib
import subprocess
import threading
import time

import pytest


def _init_git_repo(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(path), check=True)
    (path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(path), check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@local", "commit", "-qm", "init"],
        cwd=str(path), check=True,
    )


@pytest.fixture()
def live_queue(tmp_path):
    """A live queue bound to a data drive of its own."""
    from supervisor import queue, state as supervisor_state

    drive = tmp_path / "data"
    drive.mkdir()
    supervisor_state.init(drive)
    queue.init(drive, 600, 1800)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    return queue, drive, pending


def _admitted_task(task_id: str = "adm-1", **extra) -> dict:
    task = {
        "id": task_id,
        "type": "task",
        "chat_id": 1,
        "text": "work",
        "description": "work",
        "root_task_id": task_id,
        "metadata": {},
    }
    task.update(extra)
    return task


def _write_scheduled(drive: pathlib.Path, task: dict) -> None:
    """What every creation surface does BEFORE it enqueues: persist the record."""
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result

    write_task_result(
        drive, str(task["id"]), STATUS_SCHEDULED,
        chat_id=task.get("chat_id"),
        root_task_id=str(task["id"]),
        description=str(task.get("description") or ""),
        result="Scheduled.",
    )


# --- (b) cancellation races ---------------------------------------------------

def test_cancel_during_admission_owns_the_task_before_it_is_inserted(live_queue):
    """Donor `test_cancel_owns_inflight_admission_and_late_completion_cannot_enqueue`.

    Sequence: the surface persists SCHEDULED → the owner cancels → the surface finishes
    resolving and enqueues. The late insert MUST be refused: a task that shows up in the
    queue after the owner cancelled it is the worst kind of ghost, one they believe is gone.
    """
    from ouroboros.task_results import STATUS_CANCEL_REQUESTED, load_task_result

    queue, drive, pending = live_queue
    task = _admitted_task()
    _write_scheduled(drive, task)

    # The cancel lands while the admission is still in flight — no queue row exists yet,
    # so it must latch the intent durably rather than silently no-op.
    assert queue.cancel_task_by_id("adm-1") is True
    assert load_task_result(drive, "adm-1")["status"] == STATUS_CANCEL_REQUESTED

    admitted = queue.enqueue_task(task)
    assert admitted["_admission_blocked"] == "task_cancelled_during_admission"
    assert pending == []
    # The cancel intent is not overwritten by the late admission attempt.
    assert load_task_result(drive, "adm-1")["status"] == STATUS_CANCEL_REQUESTED


def test_cancel_after_insert_removes_the_task_and_its_id_is_not_reusable(live_queue):
    """Donor `test_admission_id_rejects_stale_completion_and_terminal_task_id_reuse`.

    The donor needed an `admission_id` generation token because its completion arrived
    asynchronously. Our transition is synchronous under the queue lock, so the DURABLE
    RECORD is the generation: once terminal, that id can never enter the queue again.
    """
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result

    queue, drive, pending = live_queue
    task = _admitted_task()
    _write_scheduled(drive, task)
    queue.enqueue_task(task)
    assert [row["id"] for row in pending] == ["adm-1"]

    assert queue.cancel_task_by_id("adm-1") is True
    assert pending == []
    assert load_task_result(drive, "adm-1")["status"] == STATUS_CANCELLED

    # A retry/duplicate of the SAME id cannot resurrect it.
    replayed = queue.enqueue_task(_admitted_task())
    assert replayed["_admission_blocked"] == "task_cancelled_during_admission"
    assert replayed["_durable_status"] == STATUS_CANCELLED
    assert pending == []


def test_cancel_of_an_unknown_id_stays_a_refusal(live_queue):
    """The latch is for tasks that EXIST durably; an unknown id is still not cancellable,
    so the admission-window branch cannot be used to mint state for arbitrary ids."""
    queue, _drive, _pending = live_queue
    assert queue.cancel_task_by_id("never-existed") is False


# --- (c) project fences ------------------------------------------------------

def test_project_deletion_during_admission_refuses_the_late_insert(live_queue):
    """Donor `test_project_quiescence_and_cascade_include_requested_admissions`, re-pointed.

    The donor needed unsealed admissions to be VISIBLE to project quiescence, because its
    third population could hold work the project fence would otherwise miss. We have no
    such population, so the invariant is carried by the fence instead of by visibility:
    a project whose deletion started can admit nothing, whenever the admission began.
    """
    from ouroboros.projects_registry import begin_project_deletion, create_project

    queue, drive, pending = live_queue
    create_project(drive, "doomed", name="Doomed", origin="test")
    task = _admitted_task(project_id="doomed")
    _write_scheduled(drive, task)

    # The deletion fence closes mid-admission.
    begin_project_deletion(drive, "doomed")

    admitted = queue.enqueue_task(task)
    assert admitted["_admission_blocked"] == "project_routing_fence"
    assert admitted["_project_lifecycle"] == "deleting"
    assert pending == []


def test_acceptance_fence_opened_during_admission_refuses_the_late_insert(live_queue):
    """Donor `test_acceptance_fence_is_rechecked_before_requested_task_becomes_pending`:
    a fence opened AFTER admission started still applies, because the fences are evaluated
    at the durable transition, not when the request arrived."""
    queue, drive, pending = live_queue
    task = _admitted_task("child-1", root_task_id="root-1")
    _write_scheduled(drive, task)
    queue.ACCEPTANCE_FENCES["root-1"] = {"root_task_id": "root-1", "status": "active", "token": "tok-1"}

    admitted = queue.enqueue_task(task)
    assert admitted["_admission_blocked"] == "task_acceptance_fence"
    assert admitted["_acceptance_fence_token"] == "tok-1"
    assert pending == []
    queue.ACCEPTANCE_FENCES.clear()


# --- (d) restart recovery ----------------------------------------------------

def test_a_crash_mid_admission_leaves_no_queue_zombie(live_queue):
    """Donor `test_restart_restores_requested_state_for_broker_rebind_without_enqueuing`
    and `test_restore_drops_pending_duplicate_of_requested_admission`.

    Both donor cases exist to reconcile TWO populations that can disagree after a restart.
    Here the claim is stronger and simpler, and this test is its proof: a task is in the
    queue or it is not. A crash between "persist SCHEDULED" and "enqueue" leaves nothing in
    the snapshot to restore and nothing to resurrect — and the owner is not stuck with an
    unkillable record, because that durable `scheduled` row is exactly what the
    admission-window cancel can latch.
    """
    from ouroboros.task_results import STATUS_CANCEL_REQUESTED, load_task_result

    queue, drive, pending = live_queue
    interrupted = _admitted_task("crashed-1")
    _write_scheduled(drive, interrupted)
    # ... the process dies here, before enqueue_task.
    queue.persist_queue_snapshot(reason="crash")

    pending.clear()
    assert queue.restore_pending_from_snapshot() == 0
    assert pending == []
    # No third state to recover: the queue has nothing, and the durable record is the only
    # trace — cancellable, so the task cannot become an immortal phantom.
    assert queue.cancel_task_by_id("crashed-1") is True
    assert load_task_result(drive, "crashed-1")["status"] == STATUS_CANCEL_REQUESTED


def test_restore_does_not_resurrect_a_task_cancelled_while_the_server_was_down(live_queue):
    """Donor `test_stale_requested_snapshot_cannot_cancel_same_id_already_in_pending` /
    `test_terminal_failure_survives_...`: the durable record is authoritative and the
    snapshot is derived, so a snapshot row whose task has since gone terminal is dropped."""
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result

    queue, drive, pending = live_queue
    task = _admitted_task("gone-1")
    _write_scheduled(drive, task)
    queue.enqueue_task(task)
    queue.persist_queue_snapshot(reason="before_shutdown")

    # The server is down; the task is terminalized out of band.
    write_task_result(drive, "gone-1", STATUS_CANCELLED, result="cancelled while down")
    pending.clear()

    assert queue.restore_pending_from_snapshot() == 0
    assert pending == []


def test_restored_snapshot_placement_survives_a_restart_unchanged(live_queue):
    """The sealed placement is durable, so a restart is placement-identical (C-3 #6)."""
    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY, workspace_ref_for

    queue, drive, pending = live_queue
    sealed = {"kind": "ssh", "connection_id": "conn-1", "remote_root": "/srv/app", "workspace_id": "ws-1"}
    task = _admitted_task("placed-1", workspace_root="/srv/app", workspace_mode="external")
    task["metadata"][SEALED_WORKSPACE_REF_KEY] = dict(sealed)
    _write_scheduled(drive, task)
    queue.enqueue_task(task)
    queue.persist_queue_snapshot(reason="before_restart")

    pending.clear()
    assert queue.restore_pending_from_snapshot() == 1
    restored = pending[0]
    assert restored["metadata"][SEALED_WORKSPACE_REF_KEY] == sealed
    ref = workspace_ref_for(restored)
    assert ref.kind == "ssh" and ref.remote_root == "/srv/app"


# --- (g) admission timeout ---------------------------------------------------

def test_admission_preflight_is_bounded_and_the_cut_is_disclosed():
    """Donor `test_project_admission_timeout_cancels_and_closes_late_transport`, adapted.

    The donor's timeout guarded a BROKER handshake (Lane 1 owns that clock). The Home-side
    admission clock we own is the preflight cap: a hanging workspace probe must not stall
    admission, and the degraded result must SAY it was cut rather than look complete.
    """
    from ouroboros.workspace_admission import bounded_workspace_preflight
    import ouroboros.workspace_preflight as preflight_module

    original = preflight_module.collect_workspace_preflight
    preflight_module.collect_workspace_preflight = lambda root: time.sleep(30)
    try:
        started = time.monotonic()
        summary = bounded_workspace_preflight("/tmp/whatever", timeout_sec=1.0)
    finally:
        preflight_module.collect_workspace_preflight = original
    assert time.monotonic() - started < 10
    assert "cap at admission" in summary["error"]
    assert summary["workspace_root"] == "/tmp/whatever"


def test_remote_admission_refuses_without_waiting_on_a_transport(tmp_path, monkeypatch):
    """The ssh branch's own clock: with no broker it refuses TYPED and immediately.

    Both refusals are pinned, because they are different answers: an unknown
    connection is a MALFORMED request (there is nothing to reach), while a known
    connection with no broker in this process is an UNAVAILABLE placement. Neither
    may block, and neither may become a Home path.
    """
    import ouroboros.workspace_admission as admission
    from ouroboros import config, connection_store

    store = tmp_path / "connections.json"
    monkeypatch.setattr(config, "REMOTE_CONNECTIONS_PATH", store)
    kwargs = {"system_repo_dir": tmp_path / "sys", "drive_root": tmp_path / "data"}

    monkeypatch.setattr(admission, "known_connections", lambda: {"conn-1": "host-1"})
    started = time.monotonic()
    with pytest.raises(admission.WorkspaceRootError) as unknown:
        admission.validate_workspace_root(
            {"kind": "ssh", "connection_id": "conn-1", "remote_root": "/srv/app", "workspace_id": "ws-1"},
            **kwargs,
        )
    assert "not an active remote connection" in str(unknown.value)
    assert not isinstance(unknown.value, admission.RemoteWorkspaceUnavailableError)

    row = connection_store.add_connection(name="Box", ssh_alias="box", path=store)
    monkeypatch.setattr(admission, "known_connections", lambda: {row["id"]: ""})
    with pytest.raises(admission.RemoteWorkspaceUnavailableError):
        admission.validate_workspace_root(
            {"kind": "ssh", "connection_id": row["id"], "remote_root": "/srv/app", "workspace_id": "ws-1"},
            **kwargs,
        )
    assert time.monotonic() - started < 5


# --- (h) idempotency ---------------------------------------------------------

def test_readmitting_the_same_request_is_idempotent_not_additive(tmp_path, monkeypatch):
    """Donor `test_admission_id_rejects_stale_completion_...`, request half: the same
    creation request twice must not produce two tasks, and re-resolving the same placement
    must produce the byte-identical sealed ref (so a retry cannot drift the target)."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    import types

    import supervisor.workers as workers

    from ouroboros.gateway.tasks import api_tasks_create
    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY

    workspace = tmp_path / "ws"
    _init_git_repo(workspace)
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    (data / "memory" / "identity.md").write_text("seed", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    captured: list = []
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: captured.append(dict(task)) or task)
    # Both must model a READY server, or v6.82's admission answers 503 before the
    # placement this case is about is ever resolved: `/api/tasks` reserves with
    # `require_worker_pool=True`, and `_enqueue_api_task_durably` treats a snapshot
    # result that is not exactly `True` as a failed durable write and rolls back.
    monkeypatch.setattr(workers, "WORKERS", {0: types.SimpleNamespace()})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr("ouroboros.workspace_admission.bootstrap_process_path", lambda: [])

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    app.state.supervisor_ready = True
    client = TestClient(app)
    body = {"task_id": "fixed-1", "description": "work", "workspace_root": str(workspace)}

    first = client.post("/api/tasks", json=body)
    assert first.status_code == 200, first.text
    replay = client.post("/api/tasks", json=body)
    assert replay.status_code == 409
    assert "already exists" in replay.json()["error"]
    assert len(captured) == 1

    # An independent request against the same folder seals the IDENTICAL placement.
    second = client.post("/api/tasks", json={"description": "work", "workspace_root": str(workspace)})
    assert second.status_code == 200, second.text
    assert (
        captured[1]["metadata"][SEALED_WORKSPACE_REF_KEY]
        == captured[0]["metadata"][SEALED_WORKSPACE_REF_KEY]
    )


# --- (i) immutability of the queued placement --------------------------------

def test_a_queued_placement_cannot_be_rewritten_by_its_creator(live_queue):
    """Donor `test_requested_task_and_public_snapshots_are_deeply_immutable`.

    The queue keeps a SHALLOW copy of a task, so without isolation the creator's dict and
    the queued row share the sealed payload — and the creator could retarget a task that is
    already queued. Reads are frozen dataclasses either way; this pins the write side.
    """
    from ouroboros.workspace_ref import SEALED_WORKSPACE_REF_KEY, workspace_ref_for

    queue, drive, pending = live_queue
    task = _admitted_task("sealed-1", workspace_root="/srv/original", workspace_mode="external")
    task["metadata"][SEALED_WORKSPACE_REF_KEY] = {
        "kind": "ssh", "connection_id": "conn-1", "remote_root": "/srv/original", "workspace_id": "ws-1",
    }
    _write_scheduled(drive, task)
    queue.enqueue_task(task)

    # The creator keeps mutating the dict it handed over.
    task["metadata"][SEALED_WORKSPACE_REF_KEY]["remote_root"] = "/srv/caller-mutated"
    assert workspace_ref_for(pending[0]).remote_root == "/srv/original"

    # And the snapshot persists the placement that was admitted, not the mutation.
    queue.persist_queue_snapshot(reason="after_mutation")
    import json

    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    row = snap["pending"][0]["task"]
    assert row["metadata"][SEALED_WORKSPACE_REF_KEY]["remote_root"] == "/srv/original"

    # A read of the placement is a FROZEN value: no consumer can mutate it in place.
    ref = workspace_ref_for(pending[0])
    with pytest.raises(Exception):
        ref.remote_root = "/srv/consumer-mutated"  # type: ignore[misc]


# --- (h) linearization -------------------------------------------------------

def test_admission_is_linearized_with_the_snapshot_writer(live_queue):
    """Donor `test_snapshot_writers_are_linearized_with_admission_completion`.

    Same requirement, same lock: while a snapshot writer holds `_queue_lock`, an admission
    must WAIT rather than interleave. The donor needed this because its transition touched
    two populations; we need it because the fences and the append must be one instant.
    """
    queue, drive, pending = live_queue
    entered = threading.Event()
    release = threading.Event()

    def _slow_writer():
        with queue._queue_lock:
            entered.set()
            release.wait(5)

    writer = threading.Thread(target=_slow_writer, daemon=True)
    writer.start()
    assert entered.wait(5)

    task = _admitted_task("linear-1")
    _write_scheduled(drive, task)
    admitting = threading.Thread(target=lambda: queue.enqueue_task(task), daemon=True)
    admitting.start()
    time.sleep(0.1)
    assert admitting.is_alive(), "admission interleaved with a snapshot writer holding the queue lock"
    assert pending == []

    release.set()
    admitting.join(5)
    writer.join(5)
    assert [row["id"] for row in pending] == ["linear-1"]
