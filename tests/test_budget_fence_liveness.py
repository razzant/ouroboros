"""Money admission rechecks the current dispatch file and keeps ordinary lock contention cancellable."""
from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib

import pytest

from ouroboros import queue_observation
from ouroboros import usage_accounting as ua

ROOT, CHILD = "root-1", "child-1"
OLD_TS = "2020-01-01T00:00:00+00:00"


def _write(root, name, payload):
    path = pathlib.Path(root) / "state" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _publish(root, *, fences=(), alive_generation="g1"):
    _write(root, "queue_dispatch.json", {"generation": "g1", "ts": OLD_TS, "budget_root_fences": list(fences), "members": {}})
    if alive_generation is not None:
        _write(root, "queue.alive", {"generation": alive_generation, "ts": OLD_TS, "heartbeats": {}})


def _reserve(root, task_id):
    with ua.usage_scope(ua.UsageScope(drive_root=root, task_id=task_id, root_task_id=ROOT)):
        return ua.reserve_attempt(ua.AttemptRequest(model="m", provider="p", task_id=task_id))


@pytest.fixture(autouse=True)
def _tree(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    monkeypatch.setattr(ua, "_reservation_cost", lambda _request: 0.01)
    monkeypatch.setattr(queue_observation, "_sleep", lambda _sec: None)
    write_task_result(tmp_path, ROOT, STATUS_RUNNING, chat_id=0, root_task_id=ROOT)
    write_task_result(tmp_path, CHILD, STATUS_RUNNING, chat_id=0, root_task_id=ROOT, parent_task_id=ROOT)


def test_an_old_but_certified_publication_admits_an_unfenced_send(tmp_path):
    _publish(tmp_path)  # both stamps are years old: age never gates a send
    assert _reserve(tmp_path, CHILD).attempt_id


def test_an_active_fence_refuses_whatever_the_stamp_age(tmp_path):
    _publish(tmp_path, fences=[{"root_task_id": ROOT, "status": "active", "fence_id": "F1"}])
    with pytest.raises(ua.BudgetExceeded, match="paused pending explicit resume"):
        _reserve(tmp_path, CHILD)


@pytest.mark.parametrize("alive_generation", [None, "g0"])
def test_dispatch_is_independent_of_missing_or_mismatched_stamp(tmp_path, alive_generation):
    _publish(tmp_path, alive_generation=alive_generation)
    assert _reserve(tmp_path, CHILD).attempt_id


@pytest.mark.parametrize("damage", ["malformed", "missing", "non_object", "no_generation", "unreadable"])
def test_an_unreadable_dispatch_document_refuses_typed(tmp_path, monkeypatch, caplog, damage):
    _publish(tmp_path)
    path = tmp_path / "state/queue_dispatch.json"
    if damage == "missing":
        path.unlink()
    elif damage == "unreadable":
        read = pathlib.Path.read_text
        def refuse(target, *args, **kwargs):
            if target == path:
                raise PermissionError("fixture dispatch read failed")
            return read(target, *args, **kwargs)
        monkeypatch.setattr(pathlib.Path, "read_text", refuse)
    else:
        path.write_text({"malformed": "{not json", "non_object": "[]", "no_generation": "{}"}[damage], encoding="utf-8")
    reads, waits, load = [], [], queue_observation.load_queue_document
    def observe(*args):
        reads.append(args)
        return load(*args)
    monkeypatch.setattr(queue_observation, "load_queue_document", observe)
    monkeypatch.setattr(queue_observation, "_sleep", waits.append)
    with pytest.raises(ua.UsageAccountingError, match="root budget fence authority unavailable"):
        _reserve(tmp_path, CHILD)
    assert len(reads) == 1 and waits == []
    assert len([record for record in caplog.records if record.exc_info]) == int(damage in {"malformed", "unreadable"})


@pytest.fixture(params=[1, 2], ids=["reserve", "dispatch"])
def reservation_race(tmp_path, monkeypatch, request):
    """One contended acquisition, with the transaction unwound before retry."""
    from ouroboros.store import StoreBusy

    _publish(tmp_path)
    held, acquisitions = [], []
    original = ua._locked

    @contextlib.contextmanager
    def locked(*args, **kwargs):
        acquisitions.append(True)
        if len(acquisitions) == request.param:
            raise StoreBusy("another writer", reason="contention")
        with original(*args, **kwargs) as view:
            held.append(True)
            try:
                yield view
            finally:
                held.pop()

    monkeypatch.setattr(ua, "_locked", locked)
    return held, acquisitions, request.param


@pytest.mark.parametrize(("provider_fails", "asynchronous"), [(False, False), (True, True)])
def test_reservation_race_keeps_one_physical_send(tmp_path, monkeypatch, reservation_race, asynchronous, provider_fails):
    held, _, revoke_on = reservation_race
    waits, prepared, sent = [], [], []

    def publish(_seconds):
        assert not held
        rows = ua.read_usage_records(tmp_path)
        assert [row["state"] for row in rows] == (["reserved"] if revoke_on == 2 else [])
        assert len(prepared) == revoke_on - 1 and sent == []
        waits.append(True)
        _publish(tmp_path)

    def send():
        sent.append(True)
        if provider_fails:
            raise TimeoutError("provider already received the call")
        return {"usage": {"cost": 0.02}}

    async def send_async():
        return send()

    monkeypatch.setattr(queue_observation, "_sleep", publish)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=CHILD, root_task_id=ROOT)), ua.physical_attempt_limit(1):
        request = ua.AttemptRequest(model="m", provider="p", task_id=CHILD)
        with pytest.raises(TimeoutError) if provider_fails else contextlib.nullcontext():
            if asynchronous:
                asyncio.run(ua.execute_physical_attempt_async(request, send_async, before_dispatch=lambda r: prepared.append(r.attempt_id)))
            else:
                ua.execute_physical_attempt(request, send, before_dispatch=lambda r: prepared.append(r.attempt_id))
        assert ua._PHYSICAL_LIMIT.get().used == 1
    rows = ua.read_usage_records(tmp_path)
    assert len(waits) == len(prepared) == len(sent) == len(rows) == 1
    assert rows[0]["attempt_id"] == prepared[0]
    assert rows[0]["state"] == ("unresolved" if provider_fails else "settled")
    assert rows[0].get("cost_usd") == (None if provider_fails else 0.02)


def test_reservation_race_observes_a_new_fence(tmp_path, monkeypatch, reservation_race):
    held, acquisitions, revoke_on = reservation_race
    sent = []

    def publish_fence(_seconds):
        assert not held
        _publish(tmp_path, fences=[{"root_task_id": ROOT, "status": "active", "fence_id": "F2"}])

    monkeypatch.setattr(queue_observation, "_sleep", publish_fence)
    with pytest.raises(ua.BudgetExceeded, match="paused pending explicit resume"):
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=CHILD, root_task_id=ROOT)):
            ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="p"), lambda: sent.append(True))
    assert revoke_on <= len(acquisitions) <= revoke_on + 2 and sent == []
    assert [row["state"] for row in ua.read_usage_records(tmp_path)] == (["released"] if revoke_on == 2 else [])


@pytest.mark.parametrize("reason", ["cancelled", "deadline"])
def test_reservation_publication_wait_honors_owner_controls(tmp_path, monkeypatch, reservation_race, reason):
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.model_wait import task_model_wait_scope

    held, acquisitions, revoke_on = reservation_race
    waits, control, sent = [], [], []

    def stop(_seconds):
        assert not held
        waits.append(True)
        control.append(reason)

    monkeypatch.setattr(queue_observation, "_sleep", stop)
    with task_model_wait_scope(task={"id": CHILD}, drive_root=tmp_path, event_queue=None,
                               worker_slot_held=False, owner_control=lambda: control[0] if control else None):
        with pytest.raises(PhysicalDispatchInterrupted) as error:
            with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=CHILD, root_task_id=ROOT)):
                ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="p"), lambda: sent.append(True))
    assert error.value.control_reason == reason
    assert len(waits) == 1 and len(acquisitions) == (1 if revoke_on == 1 else 3) and sent == []
    assert [row["state"] for row in ua.read_usage_records(tmp_path)] == (["released"] if revoke_on == 2 else [])


@pytest.mark.parametrize(("stage", "operation"), [("write", "reserve"), ("commit", "dispatch"), ("exit", "reserve")])
def test_permanent_mutation_or_exit_error_is_never_retried(tmp_path, monkeypatch, stage, operation):
    from ouroboros.usage_ledger import UsageAccountingError

    _publish(tmp_path)
    reservation = _reserve(tmp_path, CHILD) if operation == "dispatch" else None
    acquisitions, writes = [], []
    original_lock, original_write, original_commit = ua._locked, ua.usage_store.Txn.write, ua.usage_store.Txn.commit

    def write(view, row, *args, **kwargs):
        result = original_write(view, row, *args, **kwargs)
        writes.append(row["attempt_id"])
        if stage == "write":
            raise UsageAccountingError("after mutation")
        return result

    def commit(view, **kwargs):
        original_commit(view, **kwargs)
        if stage == "commit" and view.writable:
            raise UsageAccountingError("after commit")

    @contextlib.contextmanager
    def locked(*args, **kwargs):
        with original_lock(*args, **kwargs) as view:
            acquisitions.append(True)
            yield view
            if stage == "exit":
                raise UsageAccountingError("during exit")

    monkeypatch.setattr(ua.usage_store.Txn, "write", write)
    monkeypatch.setattr(ua.usage_store.Txn, "commit", commit)
    monkeypatch.setattr(ua, "_locked", locked)
    with pytest.raises(UsageAccountingError, match="after mutation|after commit|during exit"):
        ua.mark_dispatched(reservation) if reservation else _reserve(tmp_path, CHILD)
    assert len(acquisitions) == len(writes) == 1
    rows = ua.read_usage_records(tmp_path)
    expected = (["dispatched"] if stage != "write" else ["reserved"]) if reservation else (
        ["reserved"] if stage == "commit" else [])
    assert [row["state"] for row in rows] == expected


@pytest.mark.parametrize("reservation_race", [2], indirect=True)
def test_dispatch_reacquisition_refreshes_the_attempt_after_a_custody_write(tmp_path, monkeypatch, reservation_race):
    """A custody write that lands while the dispatch waits out a publication moves the reserved row's
    revision; the reacquired dispatch transitions that CURRENT row, keeping the write, and sends once.
    (The billing it sends under is the binding as recorded: no owner amendment overlays it any more.)"""
    held, _, _ = reservation_race
    prepared, sent = [], []

    def publish(_seconds):
        assert not held
        [attempt_id] = prepared
        # A real concurrent custody write changes the reserved row's revision.
        with ua.usage_store.hold(tmp_path) as view:
            row = view.attempt(attempt_id)
            assert row["state"] == "reserved"
            assert view.record_recovery(attempt_id, {"observation": "during_publication"},
                                        expected_revision=row["revision"])
        _publish(tmp_path)

    def send():
        [row] = ua.read_usage_records(tmp_path)
        assert row["state"] == "dispatched" and row["revision"] == 3
        assert row["recovery"] == {"observation": "during_publication"}
        sent.append(row["attempt_id"])
        return {"usage": {"cost": 0.02}}

    monkeypatch.setattr(queue_observation, "_sleep", publish)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=CHILD, root_task_id=ROOT)):
        ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="p"), send,
                                    before_dispatch=lambda row: prepared.append(row.attempt_id))
    assert sent == prepared and len(sent) == len(ua.read_usage_records(tmp_path)) == 1


@pytest.mark.parametrize("reservation_race", [2], indirect=True)
def test_dispatch_reacquisition_observes_a_new_member_fence(tmp_path, monkeypatch, reservation_race):
    from ouroboros import owner_pause
    from ouroboros.llm_attempt import _PhysicalSendNotStarted

    held, _, _ = reservation_race
    sent = []

    def publish(_seconds):
        assert not held
        owner_pause.install_fence(tmp_path, ROOT, request_id="pause-during-publication")
        _publish(tmp_path)

    monkeypatch.setattr(queue_observation, "_sleep", publish)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=CHILD, root_task_id=ROOT)):
        with pytest.raises(_PhysicalSendNotStarted, match="owner_pause"):
            ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="p"), lambda: sent.append(True))
    [row] = ua.read_usage_records(tmp_path)
    assert row["state"] == "released" and sent == []


@pytest.mark.parametrize("reservation_race", [2], indirect=True)
@pytest.mark.parametrize("axis", ["global", "root", "group"])
def test_dispatch_reacquisition_rechecks_known_spend(tmp_path, monkeypatch, reservation_race, axis):
    held, _, _ = reservation_race
    sent = []

    def publish(_seconds):
        assert not held
        ua.record_subscription_session("late-spend", drive_root=tmp_path, route="subscription",
                                       task_id=ROOT, root_task_id=ROOT, spend_usd=1.0)
        _publish(tmp_path)

    monkeypatch.setattr(queue_observation, "_sleep", publish)
    scope = ua.UsageScope(drive_root=tmp_path, task_id=CHILD, root_task_id=ROOT,
                         root_limit_usd=1.0 if axis == "root" else None,
                         billing_group_id=ROOT, billing_group_limit_usd=1.0 if axis == "group" else None)
    with ua.usage_scope(scope), pytest.raises(ua.BudgetExceeded) as error:
        ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="p",
                                    global_limit_usd=1.0 if axis == "global" else None), lambda: sent.append(True))
    assert error.value.limit_scope == ("root" if axis == "group" else axis) and sent == []
    assert ("whole-work budget exhausted" if axis == "group" else f"{axis} model budget changed") in str(error.value)
    rows = ua.read_usage_records(tmp_path)
    assert sorted(row["state"] for row in rows) == ["released", "settled"]
