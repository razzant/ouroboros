"""Temporary authority faults keep the same accepted row waiting, never replayed or failed."""
from __future__ import annotations

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ouroboros import projects_registry as registry
from ouroboros import task_results as results
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, workers
from tests.test_hurry_initial_lifecycle import pool  # noqa: F401
from tests.test_main_project_consistency import owner_turn
from tests.test_project_hold_recovery import accepted, restore_unreadable, worker
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


def test_registry_loss_blocks_admission_until_reconstruction_with_new_identity(tmp_path, monkeypatch):
    from ouroboros.project_admission import project_scope_admission
    from supervisor import state

    state.init(tmp_path)
    project = registry.create_project(tmp_path, "target")
    original_basis = registry.project_admission_view(tmp_path, "target", frozen=True)
    (tmp_path / "projects" / "target").mkdir(parents=True)  # a store reconcile could rebuild from
    registry._registry_witness_path(tmp_path).unlink()  # committed by a build without the witness
    registry.reconcile_projects(tmp_path)
    registry._registry_path(tmp_path).unlink()
    writers = (lambda: registry.create_project(tmp_path, "other"),
               lambda: registry.update_project(tmp_path, "target", name="renamed"),
               lambda: registry.bind_task_to_project(tmp_path, "task", "other", origin={"absent": "system"}),
               lambda: registry.begin_project_deletion(tmp_path, "target"),
               lambda: project_scope_admission(tmp_path, project_id="target"))
    for write in writers:  # absence after a commit is unavailable authority, never zero rooms
        with pytest.raises(FileNotFoundError):
            write()
    direct, receipts = owner_turn(tmp_path, monkeypatch, project["chat_id"], "web")
    assert not direct and receipts[-1]["status"] == "project_unavailable"
    assert registry.reconcile_projects(tmp_path) == 1
    restored = project_scope_admission(tmp_path, project_id="target")
    assert restored["project"]["chat_id"] == project["chat_id"]
    assert restored["project"]["routing_incarnation"] != project["routing_incarnation"]
    with pytest.raises(registry.ProjectAdmissionError, match="changed"):
        with registry.project_admission_guard(tmp_path, original_basis):
            pytest.fail("A reconstructed room cannot authorize the previous assignment")


def test_never_committed_install_still_creates_and_reconciles_rooms(tmp_path):
    (tmp_path / "projects" / "legacy").mkdir(parents=True)
    assert registry.reconcile_projects(tmp_path) == 1  # legacy stores register on a fresh install
    assert registry._registry_witness_path(tmp_path).exists()
    assert registry.create_project(tmp_path, "fresh")["created"] is True
    assert {row["id"] for row in registry.list_reserved_projects(tmp_path, strict=True)} == {"legacy", "fresh"}


def test_registry_loss_is_refused_only_once_a_witness_stamp_lands(tmp_path, monkeypatch):
    """The exact guarantee, not universal no-loss: a save whose stamp failed and is lost
    before any re-stamp reads as a first boot (disclosed gap); the next save stamps it."""
    witness = registry._registry_witness_path
    monkeypatch.setattr(registry, "_registry_witness_path", lambda root: tmp_path / "unwritable" / "witness")
    registry.create_project(tmp_path, "first")  # the commit lands; only its stamp fails
    assert registry._registry_path(tmp_path).exists() and not witness(tmp_path).exists()
    registry._registry_path(tmp_path).unlink()
    assert registry.list_reserved_projects(tmp_path, strict=True) == []  # the disclosed gap
    monkeypatch.setattr(registry, "_registry_witness_path", witness)
    assert registry.create_project(tmp_path, "second")["created"] is True  # this save stamps it
    assert witness(tmp_path).exists()
    registry._registry_path(tmp_path).unlink()
    with pytest.raises(FileNotFoundError):  # from here a loss is unavailable, never zero rooms
        registry.list_reserved_projects(tmp_path, strict=True)


@pytest.mark.parametrize("veto", [None, "possible", "started"])
def test_first_restore_keeps_accepted_project_row_waiting_through_unreadable_result(host, tmp_path, monkeypatch, veto):  # noqa: F811
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.project_admission import project_hold_fact

    prepared = copy.deepcopy(accepted(host, tmp_path))
    accepted(host, tmp_path, tid="sibling")
    assert prepared["admitted_dispatch"] == "none" and "_project_admission_restore_hold" not in prepared
    result = host.root / "task_results" / "held.json"
    original = result.read_bytes()
    assert queue.persist_queue_snapshot()
    result.write_text("{torn", encoding="utf-8")
    for _restart in range(2):  # the persisted hold survives a second restart during the fault
        host.pending.clear()
        assert queue.restore_pending_from_snapshot() == 2
        held = next(row for row in host.pending if row["id"] == "held")
        assert not held.get("_terminalization_retry")
        assert project_hold_fact(held)["label"] == "Waiting for Project verification"
        assert queue.persist_queue_snapshot()
    census = {row["activity_id"]: row for row in _chat_activities_snapshot_safe(host.root, availability={})}
    # The same accepted id stays visible; its unreadable result is also unreadable
    # Pause authority, so Batch4 reports `unknown` rather than a guessed queue phase.
    assert census["held"]["phase"] == "unknown"
    assert census["held"]["project_admission_hold"]["label"] == "Waiting for Project verification"
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["sibling"]  # a healthy neighbor is not blocked
    queue.RUNNING.clear()
    workers.WORKERS[0].busy_task_id = None
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["sibling"] and [row["id"] for row in host.pending] == ["held"]
    result.write_bytes(original)
    if veto == "possible":
        write_task_result(host.root, "held", "scheduled", admitted_dispatch="possible")
    elif veto == "started":
        write_task_result(host.root, "held", "scheduled", started_at="2026-01-01T00:00:00Z")
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == (["sibling"] if veto else ["sibling", "held"])
    assert load_task_result(host.root, "held")["status"] == ("scheduled" if veto else "running")  # never failed
    assert not host.attempts
    if veto:
        assert host.pending[0]["_project_admission_restore_hold"]
    else:
        assert sent[1]["_project_admission"] == prepared["_project_admission"]
        assert sent[1]["workspace_root"] == prepared["workspace_root"]
        assert sent[1]["drive_root"] == prepared["drive_root"]


def _uncertain_child(host, monkeypatch):  # noqa: F811
    """The real child admission whose committed receipt cannot be read back (`_admission_uncertain`)."""
    from supervisor import task_admission

    write_task_result(host.root, "parent", "running", chat_id=1)
    child = {"id": "held", "type": "task", "text": "Child work", "chat_id": 1, "project_id": "target",
             "root_task_id": "parent", "parent_task_id": "parent", "delegation_role": "subagent", "depth": 1,
             "_project_admission": registry.project_admission_view(host.root, "target", frozen=True)}
    committed, write = [], task_admission.write_task_result

    def write_then_tear(root, tid, *args, **kwargs):
        stored = write(root, tid, *args, **kwargs)
        path = results.task_result_path(root, tid)
        committed.append(path.read_bytes())
        path.write_text("{torn", encoding="utf-8")  # committed, then unreadable before its readback
        return stored

    monkeypatch.setattr(task_admission, "write_task_result", write_then_tear)
    admitted, reason, _detail, _persist = task_admission.enqueue_subagent_with_scheduled_result(
        host.ctx, child, result_fields={"chat_id": 1}, admitted_task_contract={}, admitted_depth_provenance={},
        direct_child_count=0, pending_ref=host.pending)
    monkeypatch.setattr(task_admission, "write_task_result", write)
    assert not reason and admitted["_admission_uncertain"] and admitted in host.pending
    return copy.deepcopy(admitted), committed[0]


@pytest.mark.parametrize("producer", ["promotion", "child_uncertain"])
@pytest.mark.parametrize("veto", [None, "stop", "terminal"])
def test_live_accepted_project_row_waits_through_its_first_unreadable_result(host, tmp_path, monkeypatch,  # noqa: F811
                                                                              producer, veto):
    """R2: a normally accepted, never-restored Project row whose own result first becomes
    unreadable keeps its id, payload and resources under the existing hold instead of failed
    custody; the healthy neighbor proceeds. Once readable, the original receipt releases it
    exactly once, while Stop or a terminal result settles it without a handoff."""
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.gateway.state import _chat_activities_snapshot_safe

    accepted(host, tmp_path, tid="sibling")
    if producer == "promotion":
        prepared = copy.deepcopy(accepted(host, tmp_path))
        result = host.root / "task_results" / "held.json"
        original = result.read_bytes()
        result.write_text("{torn", encoding="utf-8")
    else:
        prepared, original = _uncertain_child(host, monkeypatch)
        result = host.root / "task_results" / "held.json"
    assert prepared["admitted_dispatch"] == "none" and "_project_admission_restore_hold" not in prepared
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["sibling"]  # a healthy neighbor is not blocked
    held = next(row for row in host.pending if row["id"] == "held")
    assert held["_project_admission_restore_hold"] and not held.get("_terminalization_retry")
    if producer == "promotion":  # the census lists roots; a child's wait shows on its parent's tree
        census = {row["activity_id"]: row for row in _chat_activities_snapshot_safe(host.root, availability={})}
        assert census["held"]["project_admission_hold"]["label"] == "Waiting for Project verification"
    snapshot = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert next(row for row in snapshot["pending"] if row["id"] == "held")["task"]["_project_admission_restore_hold"]
    queue.RUNNING.clear()
    workers.WORKERS[0].busy_task_id = None
    result.write_bytes(original)  # the original receipt is readable again
    if veto == "stop":
        request_cancel(host.root, "held", reason="owner stopped")
    elif veto == "terminal":
        write_task_result(host.root, "held", "failed", result="Settled by its own owner.")
    for _pass in range(3):
        workers.assign_tasks()
        queue.RUNNING.clear()
        workers.WORKERS[0].busy_task_id = None
    assert [row["id"] for row in sent] == (["sibling"] if veto else ["sibling", "held"])  # never replayed
    assert "held" not in {row["id"] for row in host.pending} and not host.attempts
    stored = load_task_result(host.root, "held")
    # A dispatched child without a prepared drive keeps its receipt status (no RUNNING mirror).
    assert stored["status"] == ({"stop": "cancelled", "terminal": "failed"}.get(veto)
                                or ("running" if producer == "promotion" else "scheduled"))
    if not veto:
        for key in ("_project_admission", "workspace_root", "drive_root", "text"):
            assert sent[1].get(key) == prepared.get(key)


RESOLVER = "update_assisted_merge_test"


def _resolver(root):
    """The apply's fresh transaction: no resolver submitted yet, so its first admission is provable."""
    from supervisor import update_merge

    tx = {"task_id": RESOLVER, "phase": "assisted_resolution", "owner_chat_id": 1,
          "target_sha": "target", "pre_update_sha": "baseline", "resolver_submitted_id": ""}
    update_merge.write_update_tx(tx)
    assert update_merge.enqueue_assisted_resolution_task(tx) == RESOLVER
    return tx


def _drain(slot):
    sent = []
    while not slot.in_q.empty():
        sent.append(slot.in_q.get_nowait()["id"])
    return sent


@pytest.mark.parametrize("prior", [None, "possible"])
def test_assisted_resolver_recovers_same_id_after_restore_scope_hold(pool, prior):  # noqa: F811
    """Unreadable bindings hold the restored resolver; once scope is verifiable it runs once,
    under its first-dispatch receipt or, if it may have reached a worker, boot recovery's resume."""
    from supervisor import update_merge

    tx = _resolver(pool.root)
    [row] = queue.PENDING
    assert row["_project_scope_none"] is True and row["admitted_dispatch"] == "none"
    receipt = load_task_result(pool.root, RESOLVER, strict=True)
    assert receipt["status"] == "scheduled" and receipt["host_admission"]["status"] == "accepted"
    assert queue.persist_queue_snapshot()
    if prior == "possible":  # the resolver may already have reached a worker
        write_task_result(pool.root, RESOLVER, "scheduled", admitted_dispatch="possible")
    bindings = registry._bindings_path(pool.root)
    bindings.parent.mkdir(parents=True, exist_ok=True)
    bindings.write_text("{torn", encoding="utf-8")
    queue.PENDING.clear()
    queue.restore_pending_from_snapshot()
    assert update_merge.enqueue_assisted_resolution_task(tx) == RESOLVER  # boot recovery refreshes it
    [held] = queue.PENDING
    assert held["id"] == RESOLVER and held["_project_admission_restore_hold"]
    workers.assign_tasks()
    assert not _drain(pool.slot)
    bindings.write_text('{"bindings": {}}', encoding="utf-8")
    workers.assign_tasks()
    workers.assign_tasks()
    assert _drain(pool.slot) == [RESOLVER] and not queue.PENDING
    stored = load_task_result(pool.root, RESOLVER, strict=True)
    assert stored["host_admission"] == receipt["host_admission"]  # no fresh receipt was minted
    assert queue.RUNNING[RESOLVER]["task"]["admitted_dispatch"] == "possible"
    assert "_managed_update_resume" not in queue.RUNNING[RESOLVER]["task"]


def test_resolver_boot_reenqueue_of_an_existing_id_fabricates_no_receipt(pool):  # noqa: F811
    write_task_result(pool.root, RESOLVER, "interrupted", result="mid-flight restart",
                      admitted_dispatch="possible")
    before = results.task_result_path(pool.root, RESOLVER).read_bytes()
    _resolver(pool.root)
    assert [row["id"] for row in queue.PENDING] == [RESOLVER]
    assert queue.PENDING[0]["admitted_dispatch"] == "possible"  # never a first-dispatch claim
    assert results.task_result_path(pool.root, RESOLVER).read_bytes() == before
    workers.assign_tasks()
    assert _drain(pool.slot) == [RESOLVER]  # the managed resume still runs its resolver


def _unadmitted(root):
    rows = [json.loads(line) for line in (root / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    return [row["reason"] for row in rows if row["type"] == "managed_update_assisted_resolver_unadmitted"]


@pytest.mark.parametrize("prior", ["torn", "future", "foreign"])
def test_resolver_never_mints_a_receipt_over_an_unreadable_prior(pool, prior):  # noqa: F811
    from ouroboros.contracts.schema_versions import SCHEMA_VERSION_KEY
    from ouroboros.task_result_schema import TASK_RESULT_SCHEMA_VERSION
    from supervisor import update_merge

    path = results.task_result_path(pool.root, RESOLVER)
    body = "{torn" if prior == "torn" else json.dumps({
        "task_id": RESOLVER if prior == "future" else "other", "status": "running",
        SCHEMA_VERSION_KEY: TASK_RESULT_SCHEMA_VERSION + (prior == "future")})
    path.write_text(body, encoding="utf-8")
    tx = {"task_id": RESOLVER, "phase": "assisted_resolution", "owner_chat_id": 1, "target_sha": "target"}
    update_merge.write_update_tx(tx)
    assert update_merge.enqueue_assisted_resolution_task(tx) == ""  # possibly sent before: no replay
    assert not queue.PENDING and path.read_text(encoding="utf-8") == body  # the unknown bytes keep their custody
    assert tx["task_id"] == RESOLVER and _unadmitted(pool.root) == ["task_result_unreadable"]


@pytest.mark.parametrize("prior", [None, "interrupted"])
def test_resolver_refused_by_its_admission_is_never_claimed_started(pool, prior):  # noqa: F811
    from supervisor import update_merge

    if prior:
        write_task_result(pool.root, RESOLVER, prior, admitted_dispatch="possible")
    before = load_task_result(pool.root, RESOLVER)
    bindings = registry._bindings_path(pool.root)
    bindings.parent.mkdir(parents=True, exist_ok=True)
    bindings.write_text("{torn", encoding="utf-8")
    tx = {"task_id": RESOLVER, "phase": "assisted_resolution", "owner_chat_id": 1, "target_sha": "target",
          "resolver_submitted_id": ""}  # the apply's transaction: nothing submitted yet
    update_merge.write_update_tx(tx)
    assert update_merge.enqueue_assisted_resolution_task(tx) == ""
    assert not queue.PENDING and load_task_result(pool.root, RESOLVER) == before  # no receipt, no row
    assert _unadmitted(pool.root) == ["project_routing_fence_lookup_failed"]
    assert update_merge.read_update_tx()["resolver_submitted_id"] == ""  # a proven refusal submitted nothing
    bindings.write_text('{"bindings": {}}', encoding="utf-8")
    assert update_merge.enqueue_assisted_resolution_task(tx) == RESOLVER  # the next attempt admits it
    workers.assign_tasks()
    assert _drain(pool.slot) == [RESOLVER]


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("loss", [None, "removed", "quarantined", "legacy"])
def test_resolver_whose_dispatch_evidence_is_lost_is_never_replayed(pool, loss, queued):  # noqa: F811
    """A resolver that reached a worker and then lost its result may already have run:
    the managed resume neither sends it again nor mints a first-dispatch receipt, whether
    boot finds no row or a row restored from a snapshot older than that handoff. The
    control (no loss, restart before the handoff) resumes the same id exactly once."""
    from supervisor import update_merge

    path = results.task_result_path(pool.root, RESOLVER)
    _resolver(pool.root)
    assert update_merge.read_update_tx()["resolver_submitted_id"] == RESOLVER  # written ahead
    assert load_task_result(pool.root, RESOLVER)["status"] == "scheduled"  # its provable first
    older = queue.QUEUE_SNAPSHOT_PATH.read_bytes() if queue.persist_queue_snapshot() else b""
    if loss:
        workers.assign_tasks()
        assert _drain(pool.slot) == [RESOLVER]  # the first resolver reached a worker
        assert load_task_result(pool.root, RESOLVER)["admitted_dispatch"] == "possible"
        queue.RUNNING.clear()  # the restart took it down with its worker
        pool.slot.busy_task_id = None
        if loss == "quarantined":  # a fail-soft reader moves the torn bytes aside, then reports absence
            path.write_text("{torn", encoding="utf-8")
            assert load_task_result(pool.root, RESOLVER) is None
            assert list((pool.root / "task_results" / "quarantine").glob(RESOLVER + "*.json"))
        else:
            path.unlink()
        if loss == "legacy":  # a transaction written before submissions were recorded
            update_merge.write_update_tx({key: value for key, value in update_merge.read_update_tx().items()
                                          if key != "resolver_submitted_id"})
        assert not path.exists()
    tx = update_merge.read_update_tx()
    queue.PENDING.clear()
    if queued:  # boot restores the row from the snapshot persisted before the handoff
        queue.QUEUE_SNAPSHOT_PATH.write_bytes(older)
        assert queue.restore_pending_from_snapshot() == 1
    assert update_merge.enqueue_assisted_resolution_task(tx) == ("" if loss else RESOLVER)
    workers.assign_tasks()
    workers.assign_tasks()
    assert _drain(pool.slot) == ([] if loss else [RESOLVER])  # at most one handoff per id
    if not loss:
        return
    assert _unadmitted(pool.root) == ["task_result_missing"] and not path.exists()  # no invented receipt
    assert update_merge.read_update_tx().get("resolver_submitted_id") == (None if loss == "legacy" else RESOLVER)
    if queued:  # the same accepted row, its authority refreshed, waits visibly: never replayed
        [held] = queue.PENDING
        assert held["id"] == RESOLVER and held["_project_admission_restore_hold"]
        assert update_merge.assisted_task_metadata_authorizes(tx, held["metadata"])
    else:
        assert not queue.PENDING


def _boot_recovery(pool, tmp_path, monkeypatch):  # noqa: F811
    """A live materialized assisted merge whose boot recovery reaches the REAL resolver admission."""
    from supervisor import git_ops, update_merge
    from tests import test_update_merge_assisted as tua

    repo, _head, _plan, tx = tua._materialized_conflict_tx(tmp_path, monkeypatch)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", pool.root)  # one root for results, receipts and logs
    update_merge.write_update_tx({**tx, "task_id": RESOLVER, "owner_chat_id": 1, "resolver_submitted_id": ""})
    return repo


def _supervisor_types(root):
    return [json.loads(line)["type"] for line in (root / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("fault", ["bindings", "result"])
def test_boot_recovery_claims_no_resume_until_its_resolver_is_admitted(pool, tmp_path, monkeypatch, fault):  # noqa: F811
    from supervisor import update_merge

    repo = _boot_recovery(pool, tmp_path, monkeypatch)
    if fault == "bindings":
        broken = registry._bindings_path(pool.root)
        broken.parent.mkdir(parents=True, exist_ok=True)
        original = '{"bindings": {}}'.encode()
    else:  # the prior resolver's own result is unreadable: it may have run
        broken = results.task_result_path(pool.root, RESOLVER)
        write_task_result(pool.root, RESOLVER, "interrupted", admitted_dispatch="possible")
        original = broken.read_bytes()
    broken.write_text("{torn", encoding="utf-8")
    assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True) == {
        "finalized": False, "resumed": False, "resolution_attempts": 1}
    types = _supervisor_types(pool.root)
    assert "managed_update_assisted_resume_unadmitted" in types and "managed_update_assisted_resumed" not in types
    stored = update_merge.read_update_tx()
    assert stored["phase"] == "assisted_resolution" and stored["resolution_attempts"] == 1  # retried next boot
    assert update_merge._merge_head_sha() == stored["target_sha"]  # never rolled back
    assert (repo / "a.txt").read_text(encoding="utf-8") == "the resolver's precious resolution\n"
    assert not queue.PENDING and not queue.RUNNING
    broken.write_bytes(original)
    assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True) == {
        "finalized": False, "resumed": True, "resolution_attempts": 2}
    assert [row["id"] for row in queue.PENDING] == [RESOLVER]
    workers.assign_tasks()
    assert _drain(pool.slot) == [RESOLVER]


@pytest.mark.parametrize("veto", [None, "stop", "terminal", "torn", "bound", "foreign_tx"])
def test_boot_recovery_resumes_a_resolver_restored_from_its_claimed_handoff_once(pool, tmp_path, monkeypatch, veto):  # noqa: F811
    """R1: the server dies after the handoff, before its assign snapshot. Restore holds the
    claimed 'possible' row beside the worker's readable result; the NEXT real boot recovery's
    same-id resume then releases that exact row once. Stop, a terminal or unreadable result,
    a scope change and another transaction each keep it from ever being sent again."""
    from ouroboros.cancel_intents import request_cancel
    from supervisor import update_merge

    _boot_recovery(pool, tmp_path, monkeypatch)
    assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True)["resumed"] is True
    receipt = load_task_result(pool.root, RESOLVER, strict=True)["host_admission"]
    claimed, put = [], pool.slot.in_q.put
    pool.slot.in_q.put = lambda row, *a, **k: (claimed.append(queue.QUEUE_SNAPSHOT_PATH.read_bytes()), put(row, *a, **k))
    workers.assign_tasks()
    assert _drain(pool.slot) == [RESOLVER] and len(claimed) == 1
    write_task_result(pool.root, RESOLVER, "running", started_at="2026-01-01T00:00:00Z")  # the worker ran
    queue.RUNNING.clear()
    pool.slot.busy_task_id = None
    queue.PENDING.clear()
    queue.QUEUE_SNAPSHOT_PATH.write_bytes(claimed[0])  # the post-handoff snapshot never landed
    assert queue.restore_pending_from_snapshot() == 1
    [held] = queue.PENDING
    assert held["_project_admission_restore_hold"] and held["admitted_dispatch"] == "possible"
    workers.assign_tasks()
    assert not _drain(pool.slot)  # restore alone never replays a possibly dispatched row
    path = results.task_result_path(pool.root, RESOLVER)
    original = path.read_bytes()
    if veto == "stop":
        request_cancel(pool.root, RESOLVER, reason="owner stopped")
    elif veto == "terminal":  # shutdown custody settled it: boot recovery mints a fresh id
        write_task_result(pool.root, RESOLVER, "cancelled", cancel_origin={"source": "snapshot_restore",
                                                                           "reason": "server_shutdown"})
    elif veto == "torn":
        path.write_text("{torn", encoding="utf-8")
    elif veto == "bound":  # the unscoped resolver's assignment changed after its admission
        registry.create_project(pool.root, "room")
        registry.bind_task_to_project(pool.root, RESOLVER, "room", origin={"absent": "system"})
    resumed = update_merge.finalize_managed_update_on_boot(supervisor_ready=True)["resumed"]
    assert resumed is (veto != "torn")
    if veto == "foreign_tx":  # the transaction that granted this resume is no longer live
        assert update_merge.clear_update_tx()
    for _pass in range(3):
        workers.assign_tasks()
    fresh = update_merge.read_update_tx().get("task_id")
    assert _drain(pool.slot) == {None: [RESOLVER], "terminal": [fresh]}.get(veto, [])  # at most once
    waiting = [row["id"] for row in queue.PENDING if row.get("_project_admission_restore_hold")]
    assert waiting == ([RESOLVER] if veto in {"torn", "foreign_tx"} else [])
    if veto == "torn":  # unknown dispatch waits; the next boot's resume, not readability, releases it
        path.write_bytes(original)
        workers.assign_tasks()
        assert not _drain(pool.slot) and queue.PENDING[0]["_project_admission_restore_hold"]
        assert update_merge.finalize_managed_update_on_boot(supervisor_ready=True)["resumed"] is True
        workers.assign_tasks()
        workers.assign_tasks()
        assert _drain(pool.slot) == [RESOLVER] and not queue.PENDING
        return
    stored = load_task_result(pool.root, RESOLVER, strict=True)
    assert stored["status"] == {None: "running", "stop": "cancelled", "terminal": "cancelled",
                                "bound": "failed", "foreign_tx": "running"}[veto]
    if veto is None:  # the same accepted row continues: 'possible' kept, no receipt minted
        assert queue.RUNNING[RESOLVER]["task"]["admitted_dispatch"] == "possible"
        assert "_managed_update_resume" not in queue.RUNNING[RESOLVER]["task"]
        assert stored["host_admission"] == receipt and stored["admitted_dispatch"] == "possible"


@pytest.mark.parametrize("bindings", ["torn", "healthy"])
def test_assisted_apply_reports_started_only_for_an_admitted_resolver(pool, monkeypatch, bindings):  # noqa: F811
    import ouroboros.reviewer_slot_config as reviewer_slot_config
    import supervisor.worker_chat_lane as worker_chat_lane
    from ouroboros.gateway import control
    from supervisor import git_ops, state, update_merge

    base = "a" * 40

    def capture(cmd):
        if "--abbrev-ref" in cmd:
            return 0, "ouroboros", ""
        if "MERGE_HEAD" in cmd:
            return 1, "", ""
        return (0, base, "") if cmd[:3] == ["git", "rev-parse", "--verify"] else (0, "", "")

    txs, rollbacks = [], []
    monkeypatch.setattr(reviewer_slot_config, "review_pool_slots", lambda **_kw: [])
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    monkeypatch.setattr(git_ops, "git_capture", capture)
    monkeypatch.setattr(git_ops, "_create_rescue_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr(git_ops, "_collect_repo_sync_state", lambda: {})
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 10.0)
    monkeypatch.setattr(update_merge, "write_update_tx", lambda tx: txs.append(dict(tx)))
    monkeypatch.setattr(update_merge, "ensure_assisted_resolver_ready", lambda _sha: True)
    monkeypatch.setattr(update_merge, "materialize_assisted_merge_live", lambda *_a: (True, "ok", "m0tree"))
    monkeypatch.setattr(update_merge, "rollback_managed_update",
                        lambda reason, **_k: rollbacks.append(reason) or (True, "rolled back"))
    monkeypatch.setattr(worker_chat_lane, "preload_owner_control_path", lambda: [])
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda _reason: None)
    monkeypatch.setattr(control, "_respawn_workers_after_failed_update", lambda: None)
    if bindings == "torn":
        path = registry._bindings_path(pool.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{torn", encoding="utf-8")
    plan = {"base_sha": base, "target_sha": "b" * 40, "local_snapshot": base, "kind": "conflicting",
            "code_conflict_paths": ["ouroboros/config.py"], "doc_conflict_paths": []}
    response = control._start_assisted_merge_fenced(
        plan, {"phase": "stashing_local_work", "stash_sha": "", "local_work_carrier": "none"})
    body = json.loads(response.body)
    task_id = txs[-1]["task_id"]
    # The fresh transaction records its submission ahead of the one admission attempt;
    # a proven refusal appended nothing, so it records the id unsubmitted again.
    assert [tx.get("resolver_submitted_id") for tx in txs] == ["", "", task_id] + [""] * (bindings == "torn")
    if bindings == "torn":
        assert response.status_code == 409 and body["rolled_back"] is True and "status" not in body
        assert rollbacks == ["assisted_worker_start_failed"]
        assert not queue.PENDING and load_task_result(pool.root, task_id) is None
        assert _unadmitted(pool.root) == ["project_routing_fence_lookup_failed"]
    else:
        assert response.status_code == 200 and body["status"] == "assisted_started" and body["task_id"] == task_id
        assert not rollbacks
        [row] = queue.PENDING
        assert row["id"] == task_id and row["admitted_dispatch"] == "none"  # a fresh id's provable first
        assert load_task_result(pool.root, task_id, strict=True)["status"] == "scheduled"
        workers.assign_tasks()
        assert _drain(pool.slot) == [task_id]


def _child_of(host, tmp_path, parent):  # noqa: F811
    """A real Project admission queued as the child of ``parent`` (whose shutdown state varies)."""
    child = accepted(host, tmp_path, tid="child")
    child.update(parent_task_id="parent", root_task_id="parent", delegation_role="subagent")
    write_task_result(host.root, "parent", "running" if parent == "interrupted" else "completed", chat_id=1)
    assert queue.persist_queue_snapshot()
    if parent == "interrupted":  # the parent was RUNNING when the server stopped
        snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        snap["running"] = [{"id": "parent", "task": {"id": "parent", "chat_id": 1}}]
        queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap), encoding="utf-8")
    return host.root / "task_results" / "parent.json"


@pytest.mark.parametrize("parent", ["interrupted", "unreadable", "completed"])
def test_held_project_child_never_starts_behind_an_interrupted_parent(host, tmp_path, monkeypatch, parent):  # noqa: F811
    parent_path = _child_of(host, tmp_path, parent)
    parent_bytes = parent_path.read_bytes()
    if parent == "unreadable":
        parent_path.write_text("{torn", encoding="utf-8")
    result = host.root / "task_results" / "child.json"
    original = result.read_bytes()
    result.write_text("{torn", encoding="utf-8")  # the child's own receipt is unknown at restore
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    [held] = host.pending
    assert held["_project_admission_restore_hold"] and not held.get("_terminalization_retry")
    sent = worker(host, monkeypatch)
    result.write_bytes(original)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == (["child"] if parent == "completed" else [])
    if parent == "interrupted":  # the restore's shutdown custody, never a later revival
        assert load_task_result(host.root, "child")["status"] == "cancelled" and not host.pending
    elif parent == "unreadable":  # unknown lineage keeps the same accepted row waiting
        assert host.pending[0]["_project_admission_restore_hold"]
        assert not host.pending[0].get("_terminalization_retry")
        parent_path.write_bytes(parent_bytes)
        workers.assign_tasks()
        workers.assign_tasks()
        assert [row["id"] for row in sent] == ["child"]


@pytest.mark.parametrize("parent", ["completed", "shutdown", "unreadable"])
def test_readable_project_child_waits_for_its_unreadable_parent(host, tmp_path, monkeypatch, parent):  # noqa: F811
    """Only the parent is unknown at restore: the child's receipt and the bindings are healthy."""
    parent_path = _child_of(host, tmp_path, "completed")
    parent_bytes = parent_path.read_bytes()
    parent_path.write_text("{torn", encoding="utf-8")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    [held] = host.pending
    assert held["id"] == "child" and held["_project_admission_restore_hold"]
    assert not held.get("_terminalization_retry")  # neither interruption nor permission
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and load_task_result(host.root, "child")["status"] == "scheduled"
    if parent == "completed":
        parent_path.write_bytes(parent_bytes)
    elif parent == "shutdown":  # an earlier boot handed the parent to shutdown custody
        parent_path.unlink()
        write_task_result(host.root, "parent", "cancelled", chat_id=1,
                          cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})
    for _pass in range(3):
        workers.assign_tasks()
    assert [row["id"] for row in sent] == (["child"] if parent == "completed" else [])  # exactly once
    if parent == "shutdown":
        assert load_task_result(host.root, "child")["status"] == "cancelled" and not host.pending
    elif parent == "unreadable":
        assert host.pending[0]["_project_admission_restore_hold"] and not host.attempts


def _fault_parent(host, path, fault):  # noqa: F811
    if fault == "missing":
        path.unlink()
    elif fault == "torn":
        path.write_text("{torn", encoding="utf-8")
    elif fault == "never_admitted":  # a refusal receipt cannot be the parent that spawned it
        path.unlink()
        write_task_result(host.root, "parent", "failed", chat_id=1, admission_outcome="never_admitted")
    elif fault == "shutdown":  # an earlier boot handed the parent to shutdown custody
        path.unlink()
        write_task_result(host.root, "parent", "cancelled", chat_id=1,
                          cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})


def _passes(count=4, busy=""):
    for _pass in range(count):  # each dispatched row finishes before the next pass
        workers.WORKERS[0].busy_task_id = busy or None  # busy: holds are revalidated, nothing starts
        workers.assign_tasks()
        queue.RUNNING.clear()
        workers.WORKERS[0].busy_task_id = None


# A queued parent whose own result tears AFTER restore is terminalized by assignment's
# result-authority owner, so the torn queued case is exercised at restore only.
@pytest.mark.parametrize("where,fault,phase", [
    *((where, fault, phase) for phase in ("restore", "release") for where, fault in (
        ("queued", None), ("queued", "missing"), ("queued", "shutdown"),
        ("external", "missing"), ("external", "never_admitted"))),
    ("queued", "torn", "restore")])
def test_healthy_child_waits_for_an_unknown_parent_inside_or_outside_the_snapshot(
        host, tmp_path, monkeypatch, where, fault, phase):  # noqa: F811
    """A child's own evidence is healthy; only its parent's is not. A parent queued beside
    it or known only by its result, whose result is missing, unreadable or a refusal,
    keeps the child waiting at restore and at hold release; a parent the shutdown stopped
    cancels it; a readable parent lets each id start exactly once."""
    if where == "queued":
        accepted(host, tmp_path, tid="parent")  # restored beside the child, with its own receipt
    child = accepted(host, tmp_path, tid="child")
    child.update(parent_task_id="parent", root_task_id="parent", delegation_role="subagent")
    parent_path = host.root / "task_results" / "parent.json"
    if where == "external":
        write_task_result(host.root, "parent", "completed", chat_id=1)
    assert queue.persist_queue_snapshot()
    parent_bytes = parent_path.read_bytes()
    child_path = host.root / "task_results" / "child.json"
    child_bytes = child_path.read_bytes()
    if phase == "restore":
        _fault_parent(host, parent_path, fault)
    else:  # the child's own receipt is what is unknown at restore; its parent is healthy
        child_path.write_text("{torn", encoding="utf-8")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    sent = worker(host, monkeypatch)
    if phase == "release":
        assert next(row for row in host.pending if row["id"] == "child")["_project_admission_restore_hold"]
        _fault_parent(host, parent_path, fault)  # the parent's evidence fails before release
        child_path.write_bytes(child_bytes)
        _passes(1, busy="busy")
    rows = {row["id"]: row for row in host.pending}
    if fault == "shutdown":  # the restore's shutdown custody, never a later revival
        _passes()
        assert not sent and load_task_result(host.root, "child")["status"] == "cancelled"
        assert "child" not in {row["id"] for row in host.pending}
        return
    if fault:  # neither interruption nor permission: the same accepted row waits
        assert rows["child"]["_project_admission_restore_hold"]
        assert not rows["child"].get("_terminalization_retry")
        # A queued parent restored healthy is itself dispatchable; it must not start here.
        _passes(busy="busy" if where == "queued" and phase == "release" else "")
        assert "child" not in [row["id"] for row in sent]
        assert load_task_result(host.root, "child")["status"] == "scheduled"
        if fault == "never_admitted":
            assert host.pending[-1]["_project_admission_restore_hold"] and not host.attempts
            return
        parent_path.write_bytes(parent_bytes)  # the parent's evidence returns
    else:  # a readable parent is permission: the child is never held for it
        assert not rows["child"].get("_project_admission_restore_hold")
    _passes()
    assert [row["id"] for row in sent] == (["parent", "child"] if where == "queued" else ["child"])
    assert not host.pending and not host.attempts


@pytest.mark.parametrize("veto", [None, "forged", "revoked"])
def test_exact_pause_of_held_project_row_continues_only_under_its_resume_grant(host, tmp_path, monkeypatch, veto):  # noqa: F811
    from ouroboros import budget_pause
    from tests._budget_pause_exact_helpers import _pause

    prepared = copy.deepcopy(accepted(host, tmp_path))
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["held"]
    running = queue.RUNNING.pop("held")["task"]
    workers.WORKERS[0].busy_task_id = None
    _pause(host.root, monkeypatch, task_id="held")  # the dispatched run paused mid-flight
    budget_pause.end_dispatch_fence("held")
    row = budget_pause.budget_pause_row(host.root, "held")
    host.pending.append({**running, "_budget_pause": budget_pause.exact_pause_marker(row, default_root="held")})
    budget_pause.set_budget_pause(host.root, "held", {**row, "state": budget_pause.STATE_PAUSED})
    path, original = restore_unreadable(host)
    path.write_bytes(original)
    workers.assign_tasks()
    assert len(sent) == 1 and host.pending[0]["_project_admission_restore_hold"]  # paused, never replayed
    assert queue.resume_budget_paused_task("held")["ok"] is True
    [held] = host.pending
    if veto == "forged":
        held["_budget_pause_resume"]["grant_id"] = "forged"
    elif veto == "revoked":
        live = budget_pause.budget_pause_row(host.root, "held")
        budget_pause.set_budget_pause(host.root, "held", {**live, "grant": {**live["grant"], "revoked_at": "now"}})
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == (["held"] if veto else ["held", "held"])
    assert load_task_result(host.root, "held")["status"] != "failed" and not host.attempts
    if veto:
        assert host.pending[0]["_project_admission_restore_hold"]
    else:
        assert sent[1]["_budget_pause_resume"]["grant_id"] and "_budget_pause" not in sent[1]
        assert sent[1]["_project_admission"] == prepared["_project_admission"]
        assert sent[1]["workspace_root"] == prepared["workspace_root"]


def _consciousness_review(monkeypatch, window):
    from ouroboros import consciousness_allowance
    from ouroboros.consciousness_authority import consciousness_origin_metadata

    entered, release, reads = threading.Event(), threading.Event(), []

    def contended_ledger(*_args, **_kwargs):
        entered.set()
        assert release.wait(10)
        reads.append(True)
        return window

    monkeypatch.setattr(consciousness_allowance, "allowance_window", contended_ledger)
    origin = consciousness_origin_metadata({"initiator": "consciousness", "consciousness_autonomy": "full"})
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(queue.queue_deep_self_review_task, "consciousness:review",
                                 force=True, chat_id=1, origin=origin)
        assert entered.wait(10)
        free = queue._queue_lock.acquire(timeout=2)
        if free:
            queue._queue_lock.release()
        release.set()
        return future.result(timeout=30), free, reads


@pytest.mark.parametrize("status", ["available", "exhausted"])
def test_consciousness_host_producer_reads_its_allowance_off_the_queue_lock(host, monkeypatch, status):  # noqa: F811
    from ouroboros.consciousness_allowance import STATUS_AVAILABLE

    notices = []
    monkeypatch.setattr(queue, "send_with_budget", lambda _chat, text, **_k: notices.append(text))
    window = {"status": STATUS_AVAILABLE if status == "available" else "exhausted", "limit_usd": 5.0,
              "settled_usd": 0.0 if status == "available" else 5.0,
              "accounted_usd": 0.0 if status == "available" else 5.0, "remaining_usd": 5.0,
              "resets_at": "", "unknown_unmetered": 0}
    tid, free, reads = _consciousness_review(monkeypatch, window)
    assert free, "a contended usage ledger held the global queue lock"
    assert reads == [True]  # one exact admission read, still the money door
    if status == "available":
        assert [row["id"] for row in host.pending] == [tid]
        assert load_task_result(host.root, tid)["host_admission"]["status"] == "accepted"
    else:
        assert tid is None and not host.pending
        assert "consciousness_allowance_exhausted" in notices[-1]
