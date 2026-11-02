"""A direct turn whose server process died is found at boot and settled by its existing owners.

A machine that dies mid-turn can leave ``state/direct_roots.json`` (and even
``state/queue_snapshot.json``) not naming a direct turn whose durable result
still says ``running``: the roster is a projection the dead process never
rewrote. The contributor's report (task ``1f7eef5f``, 2026-10-09: progress to
22:03, machine dead by 22:05, the next boot restored nothing) is that class.

Queue init adds such a row to the roster's handover only on a positive fact:
its execution owner names another server generation while this process holds
the data root's admission. Snapshot restore then hands it to the owners every
caught direct root has: saved work and pauses wait for Resume, the rest is
fenced for cancellation custody, which finalizes from an answer the host
already holds instead of cancelling it. These tests drive the real seams
(queue init, snapshot restore, startup publication, the cancel sweep, outbox
replay, Resume) and read the real consumers (effective result, the public task
payload, the history annotator, the delivery registry, Continue eligibility).
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents as ci
from ouroboros import process_custody
from ouroboros.task_results import STATUS_CANCELLED, STATUS_FAILED, STATUS_RUNNING, load_task_result, write_task_result
from supervisor import terminal_delivery as td
from tests._cancel_intents_shared import _CaptureQueue, qenv as _qenv
from tests.test_recovery_registered_answer import _register
from tests.test_restart_child_orphans import SERVER_STOPPED_CANCEL, _direct_roots_fragment, _restart_snapshot

# Re-bound through a module attribute: a fixture imported by name that reappears
# as a test parameter is an F811 redefinition under the CI ruff gate.
qenv = _qenv

TID = "crash-direct-turn"
OLD_TS = "2026-10-09T22:03:00+00:00"
ANSWER = "the authored answer"  # tests.test_recovery_registered_answer._event's text
NO_OUTPUT = "no salvageable agent output"


def _crashed_turn(drive, *, owner=None, answer="none", task_id=TID):
    """The durable row the dead process's live writer left, stamped the way ``agent.py`` stamps it."""
    from ouroboros.task_status import execution_owner_record

    stamp = execution_owner_record(drive, {"id": task_id, "_attempt": 0}, "direct") if owner is None else owner
    write_task_result(drive, task_id, STATUS_RUNNING, chat_id=1, task_attempt=0, ts=OLD_TS, started_at=OLD_TS,
                      execution_owner=stamp, _is_direct_chat=True)
    if answer != "none":
        _register(drive, task_id, answer)
        registry = drive / "state" / "terminal_deliveries.json"
        if answer == "owed":  # registered before the machine died, not a moment before this boot
            document = json.loads(registry.read_text(encoding="utf-8"))
            for row in document["pending"].values():
                row["registered_at"] = "2026-10-09T22:04:00+00:00"
            registry.write_text(json.dumps(document), encoding="utf-8")


def _next_generation(monkeypatch, drive, *, admitted=True):
    """The process that boots after the crash: a new custody generation holding the root's admission."""
    from ouroboros.server_entrypoint import admit_data_root

    monkeypatch.setattr(process_custody, "_SESSION_ID", "next-server-generation")
    if admitted:
        admit_data_root(drive, wait_sec=0)  # released by conftest's autouse admission fixture


def _bus(qenv, monkeypatch):
    bus, sent = _CaptureQueue(), []
    monkeypatch.setattr(qenv.workers, "DRIVE_ROOT", qenv.drive, raising=False)
    monkeypatch.setattr(qenv.workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(qenv.workers, "RUNNING", qenv.q.RUNNING, raising=False)
    monkeypatch.setattr(qenv.workers, "_EVENT_Q_SHUTDOWN", False, raising=False)
    monkeypatch.setattr(qenv.workers, "get_event_q", lambda create=True: bus, raising=False)
    monkeypatch.setattr("supervisor.message_bus.send_with_budget",
                        lambda *a, **k: sent.append({"text": " ".join(str(x) for x in a), **k}))
    return bus, sent


def _snapshot(qenv, monkeypatch, shape):
    state = qenv.drive / "state"
    state.mkdir(parents=True, exist_ok=True)
    if shape == "valid":
        _restart_snapshot(qenv, monkeypatch, running=[])
        return
    monkeypatch.setattr(qenv.q, "QUEUE_SNAPSHOT_PATH", state / "queue_snapshot.json", raising=False)
    if shape == "zero_byte":
        (state / "queue_snapshot.json").write_bytes(b"")


def _boot(qenv, bus=None):
    """server.py's order: queue init, snapshot restore, startup publication, the startup custody
    sweep's outbox replay, then the watchdog's cancel sweep a minute later."""
    from supervisor import queue_snapshot

    qenv.q.init(qenv.drive)
    handed = list(qenv.q.PRIOR_DIRECT_ROOTS.get("task_ids") or [])
    fenced: list = []
    qenv.q.restore_pending_from_snapshot(terminalized=fenced)
    queue_snapshot.persist_queue_snapshot(reason="startup")
    if bus is not None:
        td.replay_pending_deliveries(qenv.drive, event_queue=bus)
    swept = qenv.tl.sweep_cancel_intents(now=time.time() + 60)
    return handed, fenced, swept


def _history(drive, *, answered):
    from ouroboros.gateway.history import _annotate_terminal_task_truth

    rows = [{"role": "user", "text": "question", "ts": "2026-10-09T22:02:00+00:00", "chat_id": 1},
            {"role": "assistant", "task_id": TID, "is_progress": True, "text": "reading", "ts": OLD_TS, "chat_id": 1}]
    if answered:
        rows.append({"role": "assistant", "task_id": TID, "text": ANSWER, "ts": "2026-10-09T22:04:00+00:00",
                     "chat_id": 1})
    _annotate_terminal_task_truth(rows, drive)
    return rows[1:]


def _readers(drive, *, answered=False):
    from ouroboros.outcomes import public_task_result
    from ouroboros.owner_continue import continuation_eligibility
    from ouroboros.task_status import load_effective_task_result

    effective = load_effective_task_result(drive, TID, materialize_artifacts=False)
    return SimpleNamespace(
        effective=effective,
        public=public_task_result(load_effective_task_result(drive, TID)),
        history=_history(drive, answered=answered),
        receipts=td.terminal_answer_receipts(drive, TID),
        cont=continuation_eligibility(load_task_result(drive, TID) or {}, TID),
    )


@pytest.mark.parametrize("roster", ["missing", "empty"])
@pytest.mark.parametrize("snapshot", ["valid", "absent", "zero_byte"])
def test_a_dead_turn_without_roster_or_snapshot_is_an_honest_interruption(qenv, monkeypatch, roster, snapshot):
    bus, _sent = _bus(qenv, monkeypatch)
    _crashed_turn(qenv.drive)
    _snapshot(qenv, monkeypatch, snapshot)
    if roster == "empty":
        _direct_roots_fragment(qenv.drive, [])
    _next_generation(monkeypatch, qenv.drive)

    handed, fenced, swept = _boot(qenv)

    assert handed == [TID] and fenced == [TID] and swept[TID] == "cancelled"
    seen = _readers(qenv.drive)
    assert seen.effective["status"] == STATUS_CANCELLED and seen.public["status"] == STATUS_CANCELLED
    assert seen.effective["result"] == SERVER_STOPPED_CANCEL
    assert seen.effective["cancel_origin"]["source"] == "snapshot_restore"
    assert [row.get("task_terminal_status") for row in seen.history] == [STATUS_CANCELLED]
    assert seen.history[0].get("task_phase") != "unfinished"
    assert seen.cont["eligible"] is True, "a technical interruption offers Continue by the ordinary rules"
    assert any(e.get("type") == "task_done" and e.get("task_id") == TID for e in bus.events)
    log = [json.loads(line) for line in (qenv.drive / "logs" / "supervisor.jsonl").read_text().splitlines()]
    assert {"type": "direct_roots_adopted_orphans", "task_ids": [TID]}.items() <= next(
        row for row in log if row.get("type") == "direct_roots_adopted_orphans").items()


@pytest.mark.parametrize("roster", ["missing", "listed"])
@pytest.mark.parametrize("snapshot", ["valid", "absent"])
@pytest.mark.parametrize("answer", ["owed", "delivered", "exhausted"])
def test_a_registered_answer_survives_the_crash_without_a_contradicting_notice(qenv, monkeypatch, roster,
                                                                              snapshot, answer):
    from ouroboros.working_checkpoint import REASON_FINALIZATION_INCOMPLETE

    bus, sent = _bus(qenv, monkeypatch)
    _crashed_turn(qenv.drive, answer=answer)
    final_id = td.delivery_id_for(TID, ANSWER)
    before = td.terminal_answer_receipts(qenv.drive, TID)
    _snapshot(qenv, monkeypatch, snapshot)
    if roster == "listed":
        _direct_roots_fragment(qenv.drive, [{"task_id": TID, "chat_id": 1}])
    _next_generation(monkeypatch, qenv.drive)

    handed, fenced, swept = _boot(qenv, bus)

    assert handed == [TID] and fenced == [TID] and swept[TID] == "already_settled"
    seen = _readers(qenv.drive, answered=answer != "exhausted")
    assert seen.effective["status"] == STATUS_FAILED and seen.public["status"] == STATUS_FAILED
    assert seen.effective["reason_code"] == REASON_FINALIZATION_INCOMPLETE
    assert seen.effective["result"] == ANSWER, "the answer the host holds stays the result"
    assert "the application stopped" in seen.effective["terminal_host_notice"]
    assert "cancel_origin" not in seen.effective
    progress = seen.history[0]
    assert progress["task_terminal_status"] == STATUS_FAILED and progress["outcome_final"] is True
    assert all(row.get("task_phase") != "unfinished" for row in seen.history)
    # The original delivery keeps its own state; no cancellation receipt is added beside it.
    assert seen.receipts["owed_delivery_ids"] == before["owed_delivery_ids"]
    assert [row["delivery_id"] for row in seen.receipts["delivered"]] == [
        row["delivery_id"] for row in before["delivered"]]
    assert not [did for did in seen.receipts["owed_delivery_ids"] if did.startswith("cancel:")]
    assert all(NO_OUTPUT not in str(e.get("text") or "") for e in bus.events + sent)
    done = [e for e in bus.events if e.get("type") == "task_done" and e.get("task_id") == TID]
    assert [e["status"] for e in done] == [STATUS_FAILED]
    assert seen.cont["eligible"] is False, "an answered turn is not offered Continue"
    # The boot replay sends an owed answer itself, once; nothing is sent for one already settled.
    assert [(e["delivery_id"], e["text"]) for e in bus.events if e.get("type") == "send_message"] == (
        [(final_id, ANSWER)] if answer == "owed" else [])


def test_an_unreadable_delivery_registry_keeps_the_ordinary_interruption(qenv, monkeypatch):
    """Adoption needs a positively registered answer; an unreadable registry proves none."""
    _bus(qenv, monkeypatch)
    _crashed_turn(qenv.drive, answer="unreadable")
    _snapshot(qenv, monkeypatch, "absent")
    _next_generation(monkeypatch, qenv.drive)

    _handed, fenced, _swept = _boot(qenv)

    assert fenced == [TID]
    stored = load_task_result(qenv.drive, TID)
    assert stored["status"] == STATUS_CANCELLED and stored.get("reason_code") != "finalization_incomplete"


def _install(tmp_path, monkeypatch):
    from tests._budget_pause_exact_helpers import _install_queue

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(workers, "get_event_q", lambda create=True: _CaptureQueue(), raising=False)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    return queue, workers


def _crash_without_projections(tmp_path, monkeypatch, queue):
    """The process dies: no roster, no snapshot; then the next generation boots and restores."""
    (tmp_path / "state" / "direct_roots.json").unlink(missing_ok=True)
    (tmp_path / "state" / "queue_snapshot.json").unlink(missing_ok=True)
    _next_generation(monkeypatch, tmp_path)
    queue.PENDING.clear()
    queue.init(tmp_path)
    fenced: list = []
    queue.restore_pending_from_snapshot(terminalized=fenced)
    return fenced


@pytest.mark.serial
def test_a_saved_owner_pause_waits_for_resume_without_roster_or_snapshot(tmp_path, monkeypatch):
    from ouroboros.task_status import execution_owner_record
    from tests.test_warm_restart_retention import _park

    queue, workers = _install(tmp_path, monkeypatch)
    _ctx, limit, wait, _task = _park(tmp_path, workers, "owner_pause", direct=True, attempt=2)
    write_task_result(tmp_path, "saved-warm", STATUS_RUNNING, execution_owner=execution_owner_record(
        tmp_path, {"id": "saved-warm", "_attempt": 2}, "direct"))
    workers.RUNNING.clear()  # a direct turn is never a RUNNING row

    fenced = _crash_without_projections(tmp_path, monkeypatch, queue)

    assert fenced == [] and not ci.has_active_intent(tmp_path, "saved-warm", strict=True)
    task = next(row for row in workers.PENDING if row["id"] == "saved-warm")
    pause = load_task_result(tmp_path, "saved-warm")["budget_pause"]
    assert task["_attempt"] == pause["task_attempt"] == 2 and pause["state"] == "paused"
    from ouroboros.artifacts import read_actor_source_bytes

    assert json.loads(read_actor_source_bytes(tmp_path, "saved-warm", pause["source_ref"]))["messages"] == limit.messages
    outcome = queue.resume_budget_paused_task("saved-warm")
    assert outcome["ok"], outcome


@pytest.mark.serial
@pytest.mark.parametrize("attempt", [0, 1])
def test_saved_working_state_is_held_for_resume_without_roster_or_snapshot(tmp_path, monkeypatch, attempt):
    from ouroboros import working_checkpoint as wc
    from ouroboros.task_status import execution_owner_record
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact
    from tests._budget_pause_exact_helpers import _loop_ctx

    queue, workers = _install(tmp_path, monkeypatch)
    # The real direct-chat writer uses zero on its first (non-pooled) attempt.
    write_task_result(tmp_path, TID, STATUS_RUNNING, chat_id=1, task_attempt=attempt, _is_direct_chat=True,
                      execution_owner=execution_owner_record(tmp_path, {"id": TID, "_attempt": attempt}, "direct"))
    _ctx, limit = _loop_ctx(tmp_path, TID, direct=True, attempt=attempt)
    wc.save_round(limit, "pre_effect")

    fenced = _crash_without_projections(tmp_path, monkeypatch, queue)

    assert fenced == [] and not ci.has_active_intent(tmp_path, TID, strict=True)
    task = next(row for row in workers.PENDING if row["id"] == TID)
    hold = budget_hold_fact(task)
    assert hold and hold["reason"] == HOLD_SAVED_WORK and hold["door"] == "app_stop"
    assert task["_working_recovery"]["from_attempt"] == 1 and task["_attempt"] == 2
    assert load_task_result(tmp_path, TID)["status"] != STATUS_CANCELLED
    assert not task.get("_project_admission_restore_hold")
    assert queue.resume_budget_paused_task(TID)["ok"]
    assert budget_hold_fact(task) is None and task["_working_recovery"]["source_ref"]
    from supervisor.task_admission import _working_resume_granted, record_project_dispatch_possible

    assert record_project_dispatch_possible(task) is True
    stored = load_task_result(tmp_path, TID)
    assert stored["admitted_dispatch_attempt"] == 2 and stored["task_attempt"] == attempt
    assert record_project_dispatch_possible(task) is False
    assert _working_resume_granted(task, tmp_path) is False


@pytest.mark.serial
def test_registered_answer_takes_precedence_over_saved_work(tmp_path, monkeypatch):
    from ouroboros import working_checkpoint as wc
    from ouroboros.task_status import execution_owner_record
    from supervisor.task_lifecycle import sweep_cancel_intents
    from tests._budget_pause_exact_helpers import _loop_ctx

    queue, _workers = _install(tmp_path, monkeypatch)
    write_task_result(tmp_path, TID, STATUS_RUNNING, chat_id=1, task_attempt=1, _is_direct_chat=True,
                      execution_owner=execution_owner_record(tmp_path, {"id": TID, "_attempt": 1}, "direct"))
    _ctx, limit = _loop_ctx(tmp_path, TID, direct=True, attempt=1)
    wc.save_round(limit, "pre_effect")
    _register(tmp_path, TID, "delivered")
    _crash_without_projections(tmp_path, monkeypatch, queue)
    sweep_cancel_intents(now=time.time() + 60)
    row = load_task_result(tmp_path, TID)
    assert not any(task["id"] == TID for task in queue.PENDING)
    assert row["result"] == ANSWER
    assert row["reason_code"] == "finalization_incomplete"


def test_pooled_restore_keeps_its_existing_outcome(qenv, monkeypatch):
    from ouroboros.task_status import execution_owner_record

    _bus(qenv, monkeypatch)
    owner = execution_owner_record(qenv.drive, {"id": TID}, "pooled")
    _crashed_turn(qenv.drive, owner=owner, answer="delivered")
    write_task_result(qenv.drive, TID, STATUS_RUNNING, _is_direct_chat=False)
    _restart_snapshot(qenv, monkeypatch, running=[{"id": TID, "chat_id": 1, "type": "task"}])
    _next_generation(monkeypatch, qenv.drive)
    handed, fenced, swept = _boot(qenv)
    assert handed == [] and fenced == [TID] and swept[TID] == "cancelled"
    assert load_task_result(qenv.drive, TID)["status"] == STATUS_CANCELLED
    assert _readers(qenv.drive).cont["eligible"] is True


def _live_actor(drive):
    from supervisor.active_activity import get_direct_activity_registry

    registry = get_direct_activity_registry()
    registry.register(TID, 1, actor=SimpleNamespace(env=SimpleNamespace(drive_root=drive)))
    return registry


@pytest.mark.parametrize("case", [
    "same_process_revival", "live_actor", "unstamped_legacy_owner", "owner_bound_elsewhere", "admission_not_held",
])
def test_no_positive_end_of_its_process_leaves_the_turn_alone(qenv, monkeypatch, tmp_path_factory, case):
    """Absence from the roster, age, a worker boot or a supervisor revival is no death."""
    from ouroboros.task_status import execution_owner_record

    _bus(qenv, monkeypatch)
    owner = execution_owner_record(qenv.drive, {"id": TID}, "direct")
    if case == "unstamped_legacy_owner":
        owner.pop("process")
    elif case == "owner_bound_elsewhere":
        owner["data_root"] = str(tmp_path_factory.mktemp("other-root").resolve())
    _crashed_turn(qenv.drive, owner=owner)
    _snapshot(qenv, monkeypatch, "absent")
    registry = _live_actor(qenv.drive) if case == "live_actor" else None
    if case == "same_process_revival":
        from ouroboros.server_entrypoint import admit_data_root

        admit_data_root(qenv.drive, wait_sec=0)  # this very process, whose supervisor revives
        qenv.q.init(qenv.drive)
    else:
        _next_generation(monkeypatch, qenv.drive, admitted=case != "admission_not_held")
    try:
        handed, fenced, _swept = _boot(qenv)
    finally:
        if registry is not None:
            registry.unregister(TID)

    assert handed == [] and fenced == []
    assert ci.active_intent(qenv.drive, TID) is None
    assert load_task_result(qenv.drive, TID)["status"] == STATUS_RUNNING


def test_a_revival_whose_roster_names_only_live_turns_restores_nothing_without_a_snapshot(qenv, monkeypatch):
    """No caught root, no restore: an unrelated acknowledged Restart transaction is not consumed."""
    from ouroboros import delegate_recovery as dr
    from supervisor.restart_retention import fresh_return_transaction, prepare_restart_returns

    _bus(qenv, monkeypatch)
    _crashed_turn(qenv.drive)
    _direct_roots_fragment(qenv.drive, [{"task_id": TID, "chat_id": 1}])
    _snapshot(qenv, monkeypatch, "absent")
    prepare_restart_returns(qenv.drive, {}, [], transaction_id="tx-unrelated")
    row = dr._read_restart_transaction(qenv.drive, "tx-unrelated")
    dr._write_restart_transaction(qenv.drive, {**row, "status": "normal_exit_acknowledged", "exit_code": 42})
    assert fresh_return_transaction(qenv.drive)
    registry = _live_actor(qenv.drive)
    try:
        handed, fenced, _swept = _boot(qenv)
    finally:
        registry.unregister(TID)

    assert handed == [TID] and fenced == []
    assert fresh_return_transaction(qenv.drive), "a boot that caught nothing decided no returns"
    assert load_task_result(qenv.drive, TID)["status"] == STATUS_RUNNING


def test_an_owner_stop_before_the_crash_keeps_its_own_outcome(qenv, monkeypatch):
    """The owner's Stop is already the custody of this row: the fence adds nothing, adoption never runs."""
    _bus(qenv, monkeypatch)
    _crashed_turn(qenv.drive, answer="delivered")
    ci.request_cancel(qenv.drive, TID, reason="stopped from the app", source="http_single")
    _snapshot(qenv, monkeypatch, "absent")
    _next_generation(monkeypatch, qenv.drive)

    handed, fenced, swept = _boot(qenv)

    assert handed == [TID] and fenced == [] and swept[TID] == "cancelled"
    stored = load_task_result(qenv.drive, TID)
    assert stored["status"] == STATUS_CANCELLED and stored["cancel_origin"]["source"] == "http_single"
    from ouroboros.owner_continue import continuation_eligibility

    assert continuation_eligibility(stored, TID)["refusal"] == "stopped_by_owner"


def test_a_failed_discovery_read_adds_nothing(qenv, monkeypatch):
    _crashed_turn(qenv.drive)
    _direct_roots_fragment(qenv.drive, [{"task_id": "listed-turn", "chat_id": 1}])
    _next_generation(monkeypatch, qenv.drive)

    def unreadable(*_a, **_k):
        raise OSError("store unreadable")

    monkeypatch.setattr("ouroboros.task_results.list_task_results", unreadable)
    qenv.q.init(qenv.drive)

    assert qenv.q.PRIOR_DIRECT_ROOTS["task_ids"] == ["listed-turn"], "the roster's own handover stands"
    assert load_task_result(qenv.drive, TID)["status"] == STATUS_RUNNING


@pytest.mark.parametrize("case", ["direct", "pooled", "owner_missing", "owner_attempt", "boolean", "marked", "later"])
def test_zero_attempt_is_only_the_unmarked_direct_first_launch(case):
    from supervisor.task_admission import _working_attempt_unstarted

    task = {"id": TID, "_attempt": 2}
    stored = {"task_id": TID, "status": STATUS_RUNNING, "task_attempt": 0,
              "execution_owner": {"kind": "direct", "task_attempt": 0}}
    source = {"task_attempt": 1, "retained_work_dispatch_protocol": 1}
    if case == "pooled":
        stored["execution_owner"]["kind"] = "pooled"
    elif case == "owner_missing":
        stored.pop("execution_owner")
    elif case == "owner_attempt":
        stored["execution_owner"]["task_attempt"] = 1
    elif case == "boolean":
        stored["task_attempt"] = False
    elif case == "marked":
        stored["admitted_dispatch_attempt"] = 2
    elif case == "later":
        task["_attempt"] = 3
    assert _working_attempt_unstarted(task, stored, source) is (case == "direct")


def test_only_a_direct_owner_names_its_process_generation():
    """Pooled ownership keeps its exact shape: assignment compares the whole record."""
    from ouroboros.task_status import execution_owner_record

    direct = execution_owner_record("/tmp", {"id": "t", "_attempt": 1}, "direct")
    pooled = execution_owner_record("/tmp", {"id": "t", "_attempt": 1}, "pooled")
    assert direct["process"] == process_custody.current_custody_session_id()
    assert "process" not in pooled
