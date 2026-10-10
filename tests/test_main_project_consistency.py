"""Known non-Project work retains its original scope across authority outages."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from ouroboros import projects_registry as registry
from ouroboros.task_results import load_task_result, write_task_result
from supervisor import queue, workers
from tests.test_project_hold_recovery import resume_after_app_stop, worker
from tests.test_swarm_host_admission import host  # noqa: F401

pytestmark = pytest.mark.serial


def admit_main(host, producer="promotion", tid="main"):  # noqa: F811
    if producer == "promotion":
        from supervisor.events_project_routing import _handle_promote_chat_to_task
        result = _handle_promote_chat_to_task({"task_id": tid, "routing_token": tid + "-token",
            "objective": "Ordinary Main work", "chat_id": 1}, host.ctx)
        assert result["status"] == "scheduled"
    else:
        from starlette.requests import Request

        from ouroboros.gateway.tasks import _create_task_from_body
        request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(
            drive_root=host.root, repo_dir=workers.REPO_DIR))})
        response = _create_task_from_body(request, {"task_id": tid, "description": "Ordinary Main work"})
        assert response.status_code == 200, response.body
    return next(row for row in host.pending if row["id"] == tid)


@pytest.mark.parametrize("producer", ["promotion", "api"])
def test_known_none_recovers_same_id_after_bindings_outage(host, monkeypatch, producer):  # noqa: F811
    from ouroboros.project_admission import project_hold_fact

    original = copy.deepcopy(admit_main(host, producer))
    assert "_project_admission" not in original
    assert queue.persist_queue_snapshot()
    path = registry._bindings_path(host.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{torn", encoding="utf-8")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and not host.attempts
    assert project_hold_fact(host.pending[0])["label"] == "Waiting for task scope verification"
    assert queue.persist_queue_snapshot()
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    path.write_text('{"bindings": {}}', encoding="utf-8")
    registry._registry_path(host.root).write_text("{still torn", encoding="utf-8")
    resume_after_app_stop(host, original["id"])
    workers.assign_tasks()
    workers.assign_tasks()
    assert [row["id"] for row in sent] == [original["id"]]
    assert sent[0].get("drive_root") == original.get("drive_root")
    assert "_project_admission" not in sent[0]
    assert not host.attempts


@pytest.mark.parametrize("producer", ["promotion", "api"])
def test_unscoped_producers_preserve_absence_and_stale_restore_policy(host, producer):  # noqa: F811
    row = admit_main(host, producer)
    assert "_project_admission" not in row
    assert "_project_admission" not in load_task_result(host.root, "main")
    assert queue.persist_queue_snapshot()
    snap = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert "_project_admission" not in snap["pending"][0]["task"]
    snap["ts"] = "2000-01-01T00:00:00Z"
    queue.QUEUE_SNAPSHOT_PATH.write_text(json.dumps(snap), encoding="utf-8")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    # Owner 2026-10-08 (#1563): accepted work never expires with snapshot age; the
    # stale row waits for an explicit Resume and keeps its unscoped absence.
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact

    [held] = host.pending
    assert budget_hold_fact(held)["reason"] == HOLD_SAVED_WORK and "_project_admission" not in held


@pytest.mark.parametrize("veto", ["legacy", "null", "malformed", "dispatch", "stopped", "binding", "origin", "folder"])
def test_no_scope_recovery_requires_original_positive_authority(host, monkeypatch, veto):  # noqa: F811
    row = admit_main(host)
    if veto == "legacy":
        row.pop("_project_scope_none", None)
    elif veto in {"null", "malformed"}:
        row["_project_admission"] = None if veto == "null" else []
    elif veto == "dispatch":
        write_task_result(host.root, "main", "scheduled", admitted_dispatch="possible")
    elif veto == "stopped":
        write_task_result(host.root, "main", "cancelled")
    elif veto == "origin":
        row["origin_message_ref"] = {"chat_id": 1, "client_message_id": "origin"}
    elif veto == "folder":
        row["workspace_root"] = str(host.root)
    row["_project_admission_restore_hold"] = {"reason": "project_routing_fence_lookup_failed"}
    if veto in {"binding", "origin"}:
        registry.create_project(host.root, "new-room")
        registry.bind_task_to_project(host.root, "main" if veto == "binding" else "sibling", "new-room",
            origin={"absent": "system"} if veto == "binding" else {"ref": {
                "chat_id": 1, "client_message_id": "origin", "ts": "2026-09-28T00:00:00Z",
                "text_sha256": hashlib.sha256(b"Owner work").hexdigest()}, "text": "Owner work"})
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and not host.attempts
    assert not any(r.get("project_id") == "new-room" for r in host.pending)


def test_restore_never_retargets_unscoped_row_to_a_later_origin_binding(host, monkeypatch):  # noqa: F811
    row = admit_main(host)
    row["origin_message_ref"] = {"chat_id": 1, "client_message_id": "origin"}
    assert queue.persist_queue_snapshot()
    registry.create_project(host.root, "new-room")
    registry.bind_task_to_project(host.root, "sibling", "new-room", origin={"ref": {
        "chat_id": 1, "client_message_id": "origin", "ts": "2026-09-28T00:00:00Z",
        "text_sha256": hashlib.sha256(b"Owner work").hexdigest()}, "text": "Owner work"})
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and not any(r.get("project_id") == "new-room" for r in host.pending)
    stored = load_task_result(host.root, "main")  # a semantic change, visibly terminal, never rerouted
    assert stored["status"] == "failed" and stored["reason_code"] == "project_routing_fence_changed"


def test_old_main_snapshot_cannot_replay_after_handoff(host, monkeypatch):  # noqa: F811
    admit_main(host)
    assert queue.persist_queue_snapshot()
    old = queue.QUEUE_SNAPSHOT_PATH.read_bytes()
    sent = worker(host, monkeypatch)
    monkeypatch.setattr("supervisor.worker_assignment._mirror_assigned_running_status", lambda _t: None)
    workers.assign_tasks()
    assert len(sent) == 1
    assert load_task_result(host.root, "main")["admitted_dispatch"] == "possible"
    queue.RUNNING.clear()
    workers.WORKERS[0].busy_task_id = None
    queue.QUEUE_SNAPSHOT_PATH.write_bytes(old)
    registry._bindings_path(host.root).write_text("{torn", encoding="utf-8")
    queue.restore_pending_from_snapshot()
    registry._bindings_path(host.root).write_text('{"bindings": {}}', encoding="utf-8")
    workers.assign_tasks()
    assert len(sent) == 1


@pytest.mark.parametrize("project_room", [False, True])
def test_corrupt_registry_keeps_main_dialogue_available(tmp_path, monkeypatch, project_room):
    import server
    from tests.test_project_routing_v664 import _ctx, _ImmediateThread

    project = registry.create_project(tmp_path, "target")
    chat_id = project["chat_id"] if project_room else 1
    direct, receipts = [], []
    ctx = _ctx(tmp_path, direct=lambda *_a, **_k: direct.append(True))
    registry._registry_path(tmp_path).write_text("{torn", encoding="utf-8")
    monkeypatch.setattr("ouroboros.server_owner_routing.threading", SimpleNamespace(Thread=_ImmediateThread))
    class Bridge:
        def get_updates(self, **_kwargs):
            return [{"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": 1},
                "text": "Repair the project registry", "source": "web", "client_message_id": "repair"}}]
        def send_routing_ack(self, *_args, **kwargs):
            receipts.append(kwargs)
        def broadcast(self, _payload):
            pass
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)
    server._process_bridge_updates(Bridge(), 0, ctx)
    assert direct == ([] if project_room else [True])
    if project_room:
        assert receipts[-1]["status"] == "project_unavailable"


@pytest.mark.parametrize("project_room", [False, True])
def test_main_steers_healthy_foreign_task_despite_corrupt_registry(host, monkeypatch, project_room):  # noqa: F811
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.tools.control import _steer_task
    from tests.test_steer_relay import _live_tool_ctx, _supervisor_ctx

    project = registry.create_project(host.root, "target")
    ctx = _supervisor_ctx(host.root, [])
    ctx.RUNNING["t-target"]["task"]["chat_id"] = 7
    issuer = _live_tool_ctx(host.root, ctx, [], task_id="owner-turn", metadata={
        "client_message_id": "steer", "origin_message_text": "Continue original task"})
    if project_room:
        issuer.current_chat_id = project["chat_id"]
        issuer.task_metadata["origin_message_ref"]["chat_id"] = project["chat_id"]
    registry._registry_path(host.root).write_text("{torn", encoding="utf-8")
    _steer_task(issuer, "t-target", "Continue original task")
    assert drain_owner_messages(host.root, "t-target") == ([] if project_room else ["Continue original task"])


@pytest.mark.parametrize("producer", ["promotion", "api"])
@pytest.mark.parametrize("authority", ["registry", "bindings"])
def test_fresh_main_admission_distinguishes_irrelevant_registry_from_unknown_scope(host, producer, authority):  # noqa: F811
    registry.create_project(host.root, "neighbor")
    path = registry._registry_path(host.root) if authority == "registry" else registry._bindings_path(host.root)
    path.write_text("{torn", encoding="utf-8")
    if authority == "registry":
        row = admit_main(host, producer)
        assert row["_project_scope_none"] is True and row["admitted_dispatch"] == "none"
    else:
        from starlette.requests import Request

        from ouroboros.gateway.tasks import _create_task_from_body
        from supervisor.events_project_routing import _handle_promote_chat_to_task
        if producer == "promotion":
            result = _handle_promote_chat_to_task({"task_id": "main", "routing_token": "ours",
                "objective": "Ordinary work", "chat_id": 1}, host.ctx)
            assert result["status"] != "scheduled"
        else:
            request = Request({"type": "http", "app": SimpleNamespace(state=SimpleNamespace(
                drive_root=host.root, repo_dir=workers.REPO_DIR))})
            assert _create_task_from_body(request, {"task_id": "main", "description": "Ordinary work"}).status_code >= 400
        assert not host.pending and not host.attempts


@pytest.mark.parametrize("reason", ["source", "pool", "attachments"])
@pytest.mark.parametrize("stub", [False, True])
def test_early_promotion_refusals_have_positive_never_admitted_evidence(host, reason, stub):  # noqa: F811
    from ouroboros.terminal_projection import _settled
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    event = {"task_id": "refused", "routing_token": "ours", "objective": "Original work", "chat_id": 1}
    if stub:
        write_task_result(host.root, "refused", "requested", promotion_admission={
            "status": "emitted", "routing_token": "ours"})
    if reason == "source":
        event["_source_error"] = "Source checkout failed"
    elif reason == "pool":
        host.ctx.WORKERS.clear()
    else:
        event["attachment_uploads"] = [{"path": str(host.root / "missing.png"), "label": "missing.png"}]
    outcome = _handle_promote_chat_to_task(event, host.ctx)
    assert outcome["status"] == "needs_manual_target"
    stored = load_task_result(host.root, "refused")
    assert stored["admission_outcome"] == "never_admitted"
    assert not _settled(stored) and not host.pending and not host.attempts


def test_admission_and_revalidation_read_bindings_once_per_operation(host, monkeypatch):  # noqa: F811
    from supervisor.task_admission import revalidate_project_holds

    reads = []
    original = registry._load_bindings
    def counted(root, **kwargs):
        reads.append(kwargs.get("strict"))
        return original(root, **kwargs)
    monkeypatch.setattr(registry, "_load_bindings", counted)
    for i in range(8):
        row = {"id": f"main-{i}", "type": "task", "chat_id": 1, "root_task_id": f"main-{i}",
               "origin_message_ref": {"chat_id": 1, "client_message_id": f"origin-{i}"}}
        reads.clear()
        admitted = queue.enqueue_task(row)
        assert not admitted.get("_admission_blocked") and reads == [True]
        admitted["_project_admission_restore_hold"] = {"reason": "project_routing_fence_lookup_failed"}
        write_task_result(host.root, row["id"], "scheduled")
    reads.clear()
    with queue._queue_lock:
        revalidate_project_holds()
    assert reads == [True]
    assert all(not row.get("_project_admission_restore_hold") for row in host.pending)


def _crash_after_handoff(host, monkeypatch, snapshot):  # noqa: F811
    """Hand the row to a worker, then lose RUNNING before its snapshot as a crash does."""
    sent, claimed = worker(host, monkeypatch), []
    put = workers.WORKERS[0].in_q.put
    workers.WORKERS[0].in_q = SimpleNamespace(
        put=lambda row: (claimed.append(queue.QUEUE_SNAPSHOT_PATH.read_bytes()), put(row)))
    monkeypatch.setattr("supervisor.worker_assignment._mirror_assigned_running_status", lambda _t: None)
    workers.assign_tasks()
    assert len(sent) == 1
    queue.RUNNING.clear()
    workers.WORKERS[0].busy_task_id = None
    queue.QUEUE_SNAPSHOT_PATH.write_bytes(claimed[0] if snapshot is None else snapshot)
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    return sent


@pytest.mark.parametrize("producer", ["promotion", "api"])
@pytest.mark.parametrize("window", ["claimed_snapshot", "older_snapshot"])
def test_unscoped_possible_handoff_never_replays_with_healthy_bindings(host, monkeypatch, producer, window):  # noqa: F811
    from ouroboros.project_admission import project_hold_fact

    admit_main(host, producer)
    assert queue.persist_queue_snapshot()
    older = queue.QUEUE_SNAPSHOT_PATH.read_bytes() if window == "older_snapshot" else None
    sent = _crash_after_handoff(host, monkeypatch, older)
    assert load_task_result(host.root, "main")["admitted_dispatch"] == "possible"
    workers.assign_tasks()
    workers.assign_tasks()
    assert len(sent) == 1 and not host.attempts
    hold = project_hold_fact(host.pending[0])
    assert hold["reason"] == "project_dispatch_unconfirmed"
    assert hold["label"] == "Waiting: previous run unconfirmed"


@pytest.mark.parametrize("producer", ["promotion", "api"])
def test_never_dispatched_main_restores_once_and_live_retry_still_assigns(host, monkeypatch, producer):  # noqa: F811
    admit_main(host, producer)
    assert queue.persist_queue_snapshot()
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent, "fresh accepted work after an application stop requires Resume"
    resume_after_app_stop(host, "main")
    workers.assign_tasks()
    assert [row["id"] for row in sent] == ["main"]
    queue.RUNNING.clear()
    workers.WORKERS[0].busy_task_id = None
    retry = queue.enqueue_task({**sent[0], "_attempt": 2}, front=True)
    assert retry["admitted_dispatch"] == "possible" and retry["_project_scope_none"] is True
    workers.assign_tasks()
    assert [row["_attempt"] for row in sent] == [1, 2]


def _deep_review(host, monkeypatch):  # noqa: F811
    monkeypatch.setattr(queue, "send_with_budget", lambda *_a, **_k: None)
    return queue.queue_deep_self_review_task("owner:/review", force=True, chat_id=1)


def _evolution(host, monkeypatch):  # noqa: F811
    from supervisor import evolution_lifecycle, state

    state.save_state({"owner_chat_id": 1, "evolution_mode_enabled": True, "evolution_owner_stopped": False})
    evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    monkeypatch.setattr(queue, "send_with_budget", lambda *_a, **_k: None)
    monkeypatch.setattr(queue, "budget_remaining", lambda *_a, **_k: 100.0)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    queue.enqueue_evolution_task_if_needed()
    return host.pending[0]["id"]


@pytest.mark.parametrize("producer", [_deep_review, _evolution], ids=["deep_review", "evolution"])
@pytest.mark.parametrize("veto", [None, "unreadable", "possible", "legacy"])
def test_host_producer_row_recovers_once_after_bindings_fault(host, monkeypatch, producer, veto):  # noqa: F811
    tid = producer(host, monkeypatch)
    [row] = host.pending
    assert row["id"] == tid and row["_project_scope_none"] is True and row["admitted_dispatch"] == "none"
    receipt = load_task_result(host.root, tid)
    assert receipt["status"] == "scheduled" and receipt["host_admission"]["status"] == "accepted"
    assert "admitted_dispatch" not in receipt
    assert queue.persist_queue_snapshot()
    path = registry._bindings_path(host.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{torn", encoding="utf-8")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert not sent and host.pending[0]["_project_admission_restore_hold"]
    assert host.pending[0]["_queue_seq"] > 0 and host.pending[0]["queued_at"]  # order survives the hold
    if veto == "unreadable":
        (host.root / "task_results" / f"{tid}.json").write_text("{torn", encoding="utf-8")
    elif veto == "possible":
        write_task_result(host.root, tid, "scheduled", admitted_dispatch="possible")
    elif veto == "legacy":
        for key in ("_project_scope_none", "admitted_dispatch"):
            host.pending[0].pop(key)
    path.write_text('{"bindings": {}}', encoding="utf-8")
    if not veto:
        resume_after_app_stop(host, tid)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ([] if veto else [tid]) and not host.attempts
    if veto:
        assert host.pending[0]["_project_admission_restore_hold"]
    else:
        assert queue.persist_queue_snapshot()
        queue.RUNNING.clear()
        workers.WORKERS[0].busy_task_id = None
        host.pending.clear()
        queue.restore_pending_from_snapshot()
        workers.assign_tasks()
        assert len(sent) == 1


def test_host_producer_refuses_when_its_admission_receipt_is_not_written(host, monkeypatch):  # noqa: F811
    notices = []
    monkeypatch.setattr(queue, "send_with_budget", lambda _chat, text, **_k: notices.append(text))
    monkeypatch.setattr("supervisor.task_admission.write_task_result", lambda *_a, **_k: None)
    assert queue.queue_deep_self_review_task("owner:/review", force=True, chat_id=1) is None
    assert not host.pending and "could not be queued" in notices[-1]


EXTERNAL_CHAT = 3141592  # synthetic external owner chat id (>= 1000, no room's derived id)


def _bind_external_owner():
    from supervisor import state

    state.save_state({"owner_id": 1, "owner_chat_id": 1, "owner_external_id": 4242,
                      "owner_external_chat_id": EXTERNAL_CHAT})


def _committed_room(root, fault):
    """The target room's chat id; a fresh install never committed any registry."""
    if fault == "fresh":
        from ouroboros.contracts.chat_id_policy import project_chat_id

        return project_chat_id("target")
    return registry.create_project(root, "target")["chat_id"]


def _registry_fault(root, fault):
    """After its first commit the registry becomes unreadable (torn) or absent (missing)."""
    path = registry._registry_path(root)
    if fault == "torn":
        path.write_text("{torn", encoding="utf-8")
    elif fault == "missing":
        path.unlink()


def _sender_chat(sender, room_chat):
    return {"project_room": room_chat, "external_owner": EXTERNAL_CHAT}.get(sender, EXTERNAL_CHAT + 1)


def owner_turn(root, monkeypatch, chat_id, source):
    """One owner message through the real bridge ingress: (direct Main turns, routing receipts)."""
    import server
    from tests.test_project_routing_v664 import _ctx, _ImmediateThread

    direct, receipts = [], []
    ctx = _ctx(root, direct=lambda *_a, **_k: direct.append(True))
    monkeypatch.setattr("ouroboros.server_owner_routing.threading", SimpleNamespace(Thread=_ImmediateThread))

    class Bridge:
        def get_updates(self, **_kwargs):
            return [{"update_id": 1, "message": {"chat": {"id": chat_id}, "from": {"id": 4242},
                "text": "Continue with Main", "source": source, "client_message_id": "external"}}]

        def send_routing_ack(self, *_args, **kwargs):
            receipts.append(kwargs)

        def broadcast(self, _payload):
            pass
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)
    server._process_bridge_updates(Bridge(), 0, ctx)
    return direct, receipts


@pytest.mark.parametrize("fault", ["torn", "missing", "fresh"])
@pytest.mark.parametrize("sender", ["external_owner", "unbound_external", "project_room"])
def test_external_owner_main_dialogue_survives_registry_fault(tmp_path, monkeypatch, sender, fault):
    from supervisor import state

    state.init(tmp_path)
    chat_id = _sender_chat(sender, _committed_room(tmp_path, fault))
    _bind_external_owner()
    _registry_fault(tmp_path, fault)
    direct, receipts = owner_turn(tmp_path, monkeypatch, chat_id,
                                  "web" if sender == "project_room" else "skill:telegram")
    # Only a fresh install's absence is positive: no Project room was ever committed.
    main = sender == "external_owner" or fault == "fresh"
    assert direct == ([True] if main else [])
    if not main:
        assert receipts[-1]["status"] == "project_unavailable"


@pytest.mark.parametrize("fault", ["torn", "missing", "fresh"])
@pytest.mark.parametrize("sender", ["external_owner", "unbound_external", "project_room"])
def test_external_owner_steers_despite_registry_fault(host, monkeypatch, sender, fault):  # noqa: F811
    from ouroboros.owner_mailbox import drain_owner_messages
    from ouroboros.tools.control import _steer_task
    from tests.test_steer_relay import _live_tool_ctx, _supervisor_ctx

    chat_id = _sender_chat(sender, _committed_room(host.root, fault))
    _bind_external_owner()
    ctx = _supervisor_ctx(host.root, [])
    ctx.RUNNING["t-target"]["task"]["chat_id"] = 7
    issuer = _live_tool_ctx(host.root, ctx, [], task_id="owner-turn", metadata={
        "client_message_id": "steer", "origin_message_text": "Continue original task"})
    issuer.current_chat_id = chat_id
    issuer.task_metadata["origin_message_ref"]["chat_id"] = chat_id
    _registry_fault(host.root, fault)
    _steer_task(issuer, "t-target", "Continue original task")
    delivered = drain_owner_messages(host.root, "t-target")
    allowed = sender == "external_owner" or fault == "fresh"
    assert delivered == (["Continue original task"] if allowed else [])
