"""Input custody and late result observation are independent of a failed turn."""
import asyncio
import contextlib
import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from ouroboros.gateway import host_service
from ouroboros.presence_runner import PresenceTurnGate, presence_turn_task_id, run_presence_turn
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_host_service_responsiveness import _event, _presence_app, _request, _turn
from tests.test_presence_lane_rows import _checkpoint
from tests.test_presence_moved_lifecycle import _boot_window_row, _seed_and_built, root as root
from tests.test_presence_runner import _admission, _event as _runner_event


def _poll(app, binding, reference):
    request = _request(app)
    request.method = "GET"
    request.path_params = {"work_ref": reference}
    request.query_params = {"binding_id": binding}
    response = asyncio.run(host_service._api_presence_work(request))
    return response.status_code, json.loads(response.body)


def _host(tmp_path, *, cause="unknown", child=False, proof=False):
    started = []

    class Agent:
        def handle_task(self, task):
            started.append(task["id"])
            meta = dict(task["metadata"])
            if meta["presence"]["event"]["source_event_id"] == "old":
                if child:
                    meta["presence_work_ref"] = "child"
                    write_task_result(tmp_path, "child", "running", metadata={"presence": meta["presence"]})
                reason = "resource_refusal_no_resend" if cause == "quota" else "provider_unavailable"
                if cause == "unknown":
                    meta["presence_unknown_outcome"] = {"source": "provider_outcome_unknown_no_resend"}
                if proof:
                    meta["presence_retry_proof"] = {
                        "kind": "first_round_engine_not_started", "task_id": task["id"],
                        "event_identity": meta["presence_event_identity"],
                        "reset_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                    }
                write_task_result(tmp_path, task["id"], "failed", metadata=meta, reason_code=reason,
                                  terminal_origin="model_final", result="unselected draft")
            else:
                meta.update(presence_outcome="message", presence_result_text="new reply")
                write_task_result(tmp_path, task["id"], "completed", metadata=meta,
                                  terminal_origin="model_final", result="new reply")
            return [{"type": "presence_result", "outcome": "message", "text": "new reply"}]

    app, binding, ctx = _presence_app(tmp_path, lambda **kwargs: run_presence_turn(
        repo_dir=tmp_path, drive_root=tmp_path, agent_factory=lambda **kw: Agent(),
        gate=PresenceTurnGate(1), **kwargs))
    return app, binding, ctx, started


def test_retained_receipt_releases_next_event_preserves_original_and_late_child(tmp_path):
    app, binding, ctx, started = _host(tmp_path, child=True)
    old = presence_turn_task_id(binding, "old")
    first = asyncio.run(host_service._api_presence_turn(_request(app, {
        "binding_id": binding, "event": _event("old"), "delivery_reporting_version": 1})))
    receipt = json.loads(first.body)
    assert first.status_code == 409 and receipt["disposition"] == "retained"
    assert receipt["turn_ref"] == old and receipt["work_ref"] == "child"
    assert receipt["delivery_reporting_version"] == 1 and "outcome" not in receipt
    assert ctx.presence_turns.live() == []
    later = asyncio.run(_turn(app, binding, "new"))
    assert json.loads(later.body)["text"] == "new reply"
    again = asyncio.run(_turn(app, binding, "old"))
    assert json.loads(again.body)["disposition"] == "retained"
    assert started == [old, presence_turn_task_id(binding, "new")]
    status, ended = _poll(app, binding, old)
    assert status == 200 and ended["status"] == "failed"  # it can no longer change: observation ends
    assert ended["turn_ref"] == ended["work_ref"] == old and ended["child_work_ref"] == "child"
    assert (ended["outcome"], ended["text"]) == ("silent", "") and ended["code"] == "presence_attempt_outcome_unknown"
    assert _poll(app, binding, "child")[1]["status"] == "pending"
    child = load_task_result(tmp_path, "child")
    write_task_result(tmp_path, "child", "completed", metadata={**child["metadata"],
                      "presence_outcome": "message", "presence_result_text": "child answer"},
                      terminal_origin="model_final", result="child answer")
    assert _poll(app, binding, "child")[1]["text"] == "child answer"
    assert _poll(app, binding, old)[1]["status"] == "failed"
    # Projection guarantee only: a canonical answer stored on the row is served first (no production
    # writer adds one after an attempt ended; conforming transports stopped at the terminal above).
    # A canonical late publication may enrich the existing failed result with its
    # source-bound selected answer; task lifecycle remains monotonic. Observation
    # neither regenerates the old event nor rewrites it as a silent completion.
    previous = load_task_result(tmp_path, old)
    metadata = dict(previous["metadata"])
    metadata["presence_unknown_outcome"] = None  # canonical producer supersedes the unknown marker
    metadata.update(presence_outcome="message", presence_result_text="late original")
    write_task_result(tmp_path, old, "failed", metadata=metadata, reason_code="",
                      terminal_origin="model_final", result="late original")
    assert _poll(app, binding, old)[1]["text"] == "late original"
    assert len(started) == 2


@pytest.mark.parametrize("proof,disposition", [(False, "retained"), (True, "retry")])
def test_only_source_bound_no_start_proof_keeps_quota_event_retryable(tmp_path, proof, disposition):
    app, binding, _, started = _host(tmp_path, cause="quota", proof=proof)
    for _ in range(2):
        response = asyncio.run(_turn(app, binding, "old"))
        body = json.loads(response.body)
        assert response.status_code == 409 and body["disposition"] == disposition
        assert body["code"] == "presence_resources_unavailable"
    assert len(started) == 1



def test_work_poll_distinguishes_live_then_retained_lost_turn_and_late_terminal(tmp_path):
    """One real Host author: live pending, lost retained uncertainty, then its canonical late answer."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from starlette.testclient import TestClient
    from tests.test_presence_orphan_replay import _answering_agent, _complete, _host as orphan_host, _host_body

    agents, calls = [], []
    app, binding = orphan_host(tmp_path, agents)
    body = _host_body(binding)
    reference = presence_turn_task_id(binding, body["event"]["source_event_id"])
    started, release = Event(), Event()

    def hold(_task):
        started.set()
        assert release.wait(60)

    agents.append(_answering_agent(calls, "", tmp_path, during=hold, lost=True))
    with TestClient(app) as client:
        headers = {"X-Skill-Token": "presence-token"}
        def post():
            return client.post("/presence/turn", headers=headers, json=body)
        def poll():
            return client.get(f"/presence/work/{reference}", headers=headers, params={"binding_id": binding})

        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(post)
            try:
                assert started.wait(60)
                live = poll()
                assert live.status_code == 202 and live.json()["status"] == "pending"
                assert "presence_recovery_recorded" not in load_task_result(tmp_path, reference)
            finally:
                release.set()
            assert running.result(timeout=60).status_code == 500

        refused = post()
        assert refused.status_code == 409 and refused.json()["disposition"] == "retained"
        lost = load_task_result(tmp_path, reference)
        assert lost["status"] == "running" and lost["presence_recovery_recorded"]
        ended = poll()
        facts = ended.json()
        assert ended.status_code == 200 and facts["status"] == "failed"  # the gate proved it lost
        assert facts["code"] == "presence_attempt_outcome_unknown" and "disposition" not in facts
        assert facts["turn_ref"] == facts["work_ref"] == reference
        assert facts["source_event_id"] == body["event"]["source_event_id"]
        assert facts["recorded_status"] == "running" and facts["delivery_reporting_version"] == 1
        assert (facts["outcome"], facts["text"]) == ("silent", "")
        assert poll().json() == facts

        # Projection guarantee only, as above: a canonical answer stored after the end is served first.
        _complete(calls[0], "Late canonical answer", tmp_path)
        completed = poll()
        assert completed.status_code == 200 and completed.json()["status"] == "completed"
        assert completed.json()["outcome"] == "message" and completed.json()["text"] == "Late canonical answer"
        assert poll().json() == completed.json()
        assert post().json()["text"] == "Late canonical answer"
        assert load_task_result(tmp_path, reference)["presence_recovery_recorded"] == lost["presence_recovery_recorded"]
    assert len(calls) == 1 and agents == []


@pytest.fixture(autouse=True)
def _focus_admission_enabled(monkeypatch):
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance", lambda _root: {"status": "available", "remaining_usd": 10})


# --- a moved turn's queue custody, when the queue or its binding cannot be read (#2566) ---------------------------

def _returned_turn(root, monkeypatch):
    """Production recovery and enqueue after the owner's Restart: a moved turn that yielded its conversation keeps its old
    running row, its controller gone, beside the PENDING successor the queue now holds. Returns its id and stored row."""
    from ouroboros.task_results import load_task_result
    from ouroboros.task_store import patch_document
    from supervisor import queue as q, queue_snapshot, workers
    from supervisor.restart_retention import recover_saved_running

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    monkeypatch.setattr("ouroboros.review_operation.controller_state", lambda _controller: "dead")
    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: False)
    _boot_window_row(root, tid, built)
    patch_document(root, tid, lambda current: {**current, "presence_continuation": {
        "initial": {"status": "continuing", "outcome": "deferred", "text": "", "output_ref": "", "work_ref": ""},
        "continuation_version": 1, "event_identity": built["metadata"]["presence_event_identity"]}})
    _checkpoint(root, tid)
    recovered = recover_saved_running([{"task": built, "attempt": 1}], root,
                                      transaction={"return_ids": [tid], "task_ids": [tid]}, cause="owner_restart")
    admitted = q.enqueue_task(recovered[0], restoring_snapshot=True)
    assert not admitted.get("_admission_blocked") and any(row["id"] == tid for row in workers.PENDING)
    stored = load_task_result(root, tid)
    assert stored["status"] == "running" and stored["execution_owner"]["kind"] == "presence"
    return tid, stored


@contextlib.contextmanager
def _queue_lock_contended(monkeypatch):
    """The real queue lock held by another thread; only the reader's bounded wait is shortened from its 5 s."""
    from supervisor import queue as q

    real = q._queue_lock
    entered, release = threading.Event(), threading.Event()

    def hold():
        with real:
            entered.set()
            release.wait(30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert entered.wait(2)

    class ShortWait:
        def acquire(self, timeout=-1):
            return real.acquire(timeout=min(timeout, 0.05))

        def release(self):
            real.release()

    try:
        with monkeypatch.context() as patched:
            patched.setattr(q, "_queue_lock", ShortWait())
            yield
    finally:
        release.set()
        holder.join(5)


def test_a_real_queue_lock_timeout_never_ends_a_returned_turn_the_queue_still_holds(root, monkeypatch):
    """The review's reproduction: the old running row beside its PENDING successor, the queue read timing out. An
    unreadable observation is no confirmed absence: the poll stays pending, the next turn is not told the author is gone."""
    from ouroboros.gateway.host_service_places import _queue_custody_refusal
    from ouroboros.presence_continuation import continuation_status
    from ouroboros.presence_events import PresenceTurnError
    from ouroboros.presence_recovery import CUSTODY_QUEUED, CUSTODY_UNKNOWN, attempt_ended, queue_custody, retained_work_view

    tid, stored = _returned_turn(root, monkeypatch)
    identity = stored["metadata"]["presence_event_identity"]
    assert queue_custody(root, tid, stored) == CUSTODY_QUEUED and attempt_ended(stored, root) is False  # readable: the control
    with _queue_lock_contended(monkeypatch):
        assert queue_custody(root, tid, stored) == CUSTODY_UNKNOWN
        assert attempt_ended(stored, root) is False
        status, body = retained_work_view(stored, tid, root)
        assert (status, body["status"]) == (202, "pending")
        assert continuation_status(stored, root) == "running"
        with pytest.raises(PresenceTurnError) as refused:
            _queue_custody_refusal(root, tid, identity)
        assert (refused.value.code, refused.value.custody) == ("presence_turn_custody_unknown", "unknown")
    assert queue_custody(root, tid, stored) == CUSTODY_QUEUED  # the read answers again, and the queue still holds it


def test_a_worker_reads_no_live_queue_so_a_moved_turn_stays_unknown_and_the_next_turn_hears_no_end(root, monkeypatch):
    """A conversation-lane worker building the next turn cannot see the server's queue. A moved author the queue may hold
    is never presented there as gone, and no read in that worker reconciles its row as lost."""
    from ouroboros.presence_context import _previous_turn_line
    from ouroboros.presence_continuation import continuation_status
    from ouroboros.presence_recovery import CUSTODY_UNKNOWN, attempt_ended, queue_custody
    from ouroboros.presence_runner import current_previous_turn, record_continuing_turn
    from ouroboros.task_status import _presence_attempt_lost
    from ouroboros.utils import WORKER_PROCESS_ENV, utc_now_iso

    tid, stored = _returned_turn(root, monkeypatch)
    key = _runner_event().conversation_key
    record_continuing_turn(root, key, tid, outcome="deferred", message="", work_ref="", lent_at=utc_now_iso())
    monkeypatch.setenv(WORKER_PROCESS_ENV, "1")
    assert queue_custody(root, tid, stored) == CUSTODY_UNKNOWN and attempt_ended(stored, root) is False
    assert continuation_status(stored, root) == "running"
    assert _presence_attempt_lost(root, stored) is False
    previous, _open_turns = current_previous_turn(root, key)
    assert previous["task_id"] == tid and previous["continuing"] is True and "author_status" not in previous
    assert "nothing restarts it" not in _previous_turn_line(previous)


def test_a_stop_is_retried_while_custody_is_unreadable_and_an_unmoved_turn_keeps_its_own_rules(root, monkeypatch):
    from ouroboros import presence_recovery
    from ouroboros.gateway.tasks import _record_stop
    from ouroboros.presence_continuation import continuation_status
    from ouroboros.presence_recovery import attempt_ended, queue_custody, retained_work_view
    from ouroboros.task_results import load_task_result

    tid, _stored = _returned_turn(root, monkeypatch)
    unmoved = tid + "-unmoved"
    _seed, built = _seed_and_built(root)
    _boot_window_row(root, unmoved, {**built, "id": unmoved}, move=False)
    monkeypatch.setattr(presence_recovery, "_queue_row", lambda *_a: None)
    with pytest.raises(OSError):  # the door's 503: retry, never "not found"
        _record_stop(tid, cascade=True, stop_action_id="stop-unreadable")
    # A turn never moved can never be queue-held: an unreadable queue changes nothing for it, whatever the record reads.
    row = load_task_result(root, unmoved)
    assert queue_custody(root, unmoved, row) == "" and attempt_ended(row, root) is True
    status, body = retained_work_view(row, unmoved, root)
    assert (status, body["status"]) == (200, "failed")
    assert continuation_status({**row, "presence_continuation": {"initial": {}}}, root) == "interrupted"
    assert _record_stop(unmoved, cascade=True, stop_action_id="stop-unmoved") is None


def test_an_unreadable_saved_queue_or_binding_confirms_no_absence(root, monkeypatch):
    """The other unreadable observations the same rule reads: a broken saved document in the boot window, and a binding
    that cannot say whether the turn moved. Where the queue itself answers, its answer is the custody."""
    from ouroboros import projects_registry
    from ouroboros.presence_recovery import CUSTODY_QUEUED, CUSTODY_UNKNOWN, attempt_ended, queue_custody, retained_work_view
    from ouroboros.task_results import load_task_result
    from ouroboros.utils import WORKER_PROCESS_ENV
    from supervisor import queue_snapshot, workers

    tid, stored = _returned_turn(root, monkeypatch)
    with monkeypatch.context() as boot:
        boot.setattr(queue_snapshot, "saved_queue_unowned", lambda: True)
        (root / "state").mkdir(exist_ok=True)
        (root / "state" / "queue_snapshot.json").write_text("{broken", encoding="utf-8")
        assert queue_custody(root, tid, stored) == CUSTODY_UNKNOWN and attempt_ended(stored, root) is False
        assert retained_work_view(stored, tid, root)[0] == 202

    def unreadable(*_a, **_k):
        raise OSError("bindings unreadable")

    monkeypatch.setattr(projects_registry, "project_binding_for_task", unreadable)
    assert queue_custody(root, tid, stored) == CUSTODY_QUEUED and attempt_ended(stored, root) is False  # the queue answers
    with monkeypatch.context() as worker:  # neither the binding nor the queue can answer here
        worker.setenv(WORKER_PROCESS_ENV, "1")
        assert queue_custody(root, tid, stored) == CUSTODY_UNKNOWN and attempt_ended(stored, root) is False
    workers.PENDING.clear()
    assert queue_custody(root, tid, stored) == "" and attempt_ended(load_task_result(root, tid), root) is True


def test_a_carriers_retry_of_a_lost_moved_turn_is_its_lost_attempt_and_what_the_queue_keeps_is_parked(root, monkeypatch):
    """A carrier reads no live queue, but its lane row is the queue's word on the id (``dispatch_presence_turn`` admits none
    the queue holds): a retried moved turn stays the retained lost attempt, with its room note. Durable park facts still
    answer there, and an uncarried read that cannot confirm custody still refuses as unknown."""
    from ouroboros.presence_events import PresenceTurnError
    from ouroboros.presence_runner import run_presence_turn
    from ouroboros.task_results import load_task_result
    from ouroboros.utils import WORKER_PROCESS_ENV
    from supervisor import queue_snapshot

    _seed, built = _seed_and_built(root)
    tid = built["id"]
    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: False)
    _boot_window_row(root, tid, built)
    monkeypatch.setenv(WORKER_PROCESS_ENV, "1")
    no_author = lambda **_kw: pytest.fail("no author is built for a lost or kept attempt")  # noqa: E731
    args = dict(admission=_admission(), event=_runner_event(), repo_dir=root, drive_root=root, agent_factory=no_author)
    with pytest.raises(PresenceTurnError) as uncarried:
        run_presence_turn(**args)
    assert uncarried.value.code == "presence_turn_custody_unknown"
    assert not load_task_result(root, tid).get("presence_recovery_recorded")
    carrier = {"kind": "presence", "lane": "conversation", "executor": "worker"}
    with pytest.raises(PresenceTurnError) as lost:
        run_presence_turn(**args, carrier_owner=carrier)
    assert (lost.value.code, lost.value.disposition) == ("presence_attempt_outcome_unknown", "retained")
    assert load_task_result(root, tid).get("presence_recovery_recorded")
    _boot_window_row(root, tid, built, status="scheduled",  # the same id, now saved by its exact pause
                     pause={"state": "paused", "pause_id": "p", "source_ref": "source", "task_attempt": 1})
    with pytest.raises(PresenceTurnError) as kept:
        run_presence_turn(**args, carrier_owner=carrier)
    assert (kept.value.code, kept.value.custody) == ("presence_turn_parked", "queued")


def test_a_resumed_saved_turn_whose_row_cannot_be_read_returns_to_its_exact_pause_for_the_owner(root, monkeypatch):
    """Review R2 F1 for the owner's Resume (a restart's return: ``test_a_saved_turn_whose_row_cannot_be_read_...`` in
    ``test_presence_moved_lifecycle``): the failed read raises into the worker's rail, which writes no terminal inside
    the exact pause's window (``budget_pause.death_returns_to_pause``), and nothing releases the attempt with a frame.
    The exited worker's recovery revokes the grant nothing consumed and parks the SAME id on its exact pause again: the
    queue keeps it, the Host's poll answers it pending, and the owner can Resume it again."""
    import os
    from types import SimpleNamespace

    from ouroboros import presence_runner
    from ouroboros.presence_events import PresenceTurnError
    from ouroboros.presence_recovery import CUSTODY_QUEUED, queue_custody, retained_work_view
    from ouroboros.store import StoreBusy
    from supervisor import queue_snapshot, task_reaper, workers
    from supervisor.worker_assignment import _mirror_assigned_running_status
    from supervisor.worker_health import recover_confirmed_dead_worker
    from supervisor.worker_process import run_worker_task
    from tests.test_presence_moved_allowance import _moved_paused, _resume_over_http
    from tests.test_presence_moved_lifecycle import _slot

    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: False)
    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **_k: pytest.fail("a failed read builds no agent"))
    _seed, built = _seed_and_built(root)
    tid = built["id"]
    _moved_paused(root, monkeypatch, built, tid)
    assert _resume_over_http(tid).status_code == 200
    [task] = [row for row in workers.PENDING if row["id"] == tid]
    assert task.get("_budget_pause_resume")
    workers.PENDING.remove(task)  # the queue assigns it to a project worker
    task["drive_root"] = str(root)
    _mirror_assigned_running_status(task)
    slot = _slot(workers, 91, os.getpid(), tid, lane="project")
    meta = workers.RUNNING[tid] = {"task": task, "worker_id": 91, "attempt": 1}
    frames = []

    def unreadable(*_args, **_kwargs):
        raise StoreBusy("task store acquisition timed out", reason="contention")

    with monkeypatch.context() as outage, pytest.raises(PresenceTurnError):
        outage.setattr(presence_runner, "load_task_result", unreadable)
        run_worker_task(SimpleNamespace(_event_queue=None, owner_wait_callback=None), task, wid=91,
                        out_q=SimpleNamespace(put=frames.append), repo_dir=str(root), drive_root=str(root))
    assert frames == [] and workers.RUNNING[tid] is meta and slot.busy_task_id == tid
    assert load_task_result(root, tid)["status"] == "running"
    slot.proc = SimpleNamespace(pid=4242, is_alive=lambda: False, exitcode=0)
    monkeypatch.setattr(workers, "_reconcile_confirmed_dead_review_owner", lambda *_a: None)
    monkeypatch.setattr(task_reaper, "_respawn_after_reap", lambda *_a, **_k: None)
    recover_confirmed_dead_worker({"kind": "confirmed_dead_worker", "worker_id": 91, "worker": slot, "task_id": tid,
                                   "meta": meta, "task": dict(task), "attempt": 1, "exitcode": 0,
                                   "drive_root": str(root)})
    [parked] = [row for row in workers.PENDING if row["id"] == tid]
    assert parked.get("_budget_pause") and not parked.get("_budget_pause_resume") and tid not in workers.RUNNING
    stored = load_task_result(root, tid)
    assert stored["status"] not in {"completed", "failed", "cancelled"}
    assert queue_custody(root, tid, stored) == CUSTODY_QUEUED and retained_work_view(stored, tid, root)[0] == 202
    assert _resume_over_http(tid).status_code == 200
