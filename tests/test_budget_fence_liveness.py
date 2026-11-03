"""The root budget fence reads the dispatch document of the last successful publication.

A send needs the dispatch document's generation certified by ``queue.alive`` — never a
fresh stamp: a stalled supervisor tick leaves the certificate in place and must not
stop a single call. A publication in flight (the publisher unlinks the stamp first) is
re-read briefly; a certificate that stays missing or mismatched (a failed publication)
refuses typed, and an active fence still refuses whatever the stamp's age.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import threading
from types import SimpleNamespace

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
def test_a_failed_publication_refuses_typed(tmp_path, alive_generation):
    _publish(tmp_path, alive_generation=alive_generation)  # unlinked stamp, or one of an older publication
    with pytest.raises(ua.UsageAccountingError, match="root budget fence authority unavailable"):
        _reserve(tmp_path, CHILD)


@pytest.mark.parametrize("alive_generation", [None, "g0"])
def test_a_publication_in_flight_is_waited_out(tmp_path, monkeypatch, alive_generation):
    _publish(tmp_path, alive_generation=alive_generation)
    waits = []

    def publisher_finishes(_sec):
        waits.append(_sec)
        _write(tmp_path, "queue.alive", {"generation": "g1", "ts": OLD_TS, "heartbeats": {}})

    monkeypatch.setattr(queue_observation, "_sleep", publisher_finishes)
    assert _reserve(tmp_path, CHILD).attempt_id and waits


def test_a_publication_slower_than_the_former_window_is_waited_out(tmp_path, monkeypatch):
    _publish(tmp_path, alive_generation=None)  # issue #1754: 20 reads failed sends on a loaded host
    waits = []

    def slow_publisher(_sec):
        waits.append(_sec)
        if len(waits) == 25:
            _write(tmp_path, "queue.alive", {"generation": "g1", "ts": OLD_TS, "heartbeats": {}})

    monkeypatch.setattr(queue_observation, "_sleep", slow_publisher)
    assert _reserve(tmp_path, CHILD).attempt_id and len(waits) == 25


@pytest.mark.parametrize("damage", ["malformed", "non_object", "missing", "no_generation", "unreadable"])
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


@pytest.mark.parametrize("operation", ["reserve", "dispatch"])
@pytest.mark.parametrize("sliced", [False, True])
@pytest.mark.parametrize("publication", ["stable", "before_lock", "under_lock"])
def test_publication_waits_only_outside_the_money_lock(tmp_path, monkeypatch, operation, sliced, publication):
    from ouroboros import _usage_wait

    _publish(tmp_path)
    reservation = _reserve(tmp_path, CHILD) if operation == "dispatch" else None
    stamp, held, waits, acquisitions = tmp_path / "state/queue.alive", [], [], []
    original = ua._locked

    @contextlib.contextmanager
    def locked(*args, **kwargs):
        with original(*args, **kwargs) as view:
            if publication == "under_lock" and not acquisitions:
                stamp.unlink()
            acquisitions.append(True)
            held.append(True)
            try:
                yield view
            finally:
                held.pop()

    def finish_publication(seconds):
        assert not held, "queue publication wait held the money lock"
        waits.append(seconds)
        _publish(tmp_path)

    monkeypatch.setattr(ua, "_locked", locked)
    monkeypatch.setattr(queue_observation, "_sleep", finish_publication)
    if publication == "before_lock":
        stamp.unlink()
    token = _usage_wait._CANCEL.set(threading.Event() if sliced else None)
    try:
        def act():
            return _reserve(tmp_path, CHILD) if operation == "reserve" else ua.mark_dispatched(reservation)
        if operation == "reserve":
            assert act().attempt_id
        else:
            act()
    finally:
        _usage_wait._CANCEL.reset(token)
    assert len(waits) == int(publication != "stable")
    if reservation:
        with ua.usage_store.read(tmp_path) as view:
            assert view.attempt(reservation.attempt_id)["state"] == "dispatched"


@pytest.fixture(params=[1, 2], ids=["reserve", "dispatch"])
def reservation_race(tmp_path, monkeypatch, request):
    """Revoke on the chosen real acquisition, before that operation's write."""
    _publish(tmp_path, alive_generation=None if request.param == 0 else "g1")
    held, acquisitions = [], []
    original = ua._locked

    @contextlib.contextmanager
    def locked(*args, **kwargs):
        with original(*args, **kwargs) as view:
            acquisitions.append(True)
            if len(acquisitions) == request.param:
                (tmp_path / "state/queue.alive").unlink()
            held.append(True)
            try:
                yield view
            finally:
                held.pop()

    monkeypatch.setattr(ua, "_locked", locked)
    return held, acquisitions, request.param


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("provider_fails", [False, True])
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


@pytest.mark.parametrize("alive_generation", [None, "g0"])
@pytest.mark.parametrize("sliced", [False, True])
def test_reservation_race_with_persistent_absence_still_refuses(tmp_path, monkeypatch, reservation_race, alive_generation, sliced):
    from ouroboros import _usage_wait

    held, acquisitions, revoke_on = reservation_race
    waits, sent = [], []
    monkeypatch.setattr(queue_observation, "DISPATCH_PUBLICATION_WAIT_SEC", 0.2)
    clock = [0.0]
    monkeypatch.setattr(_usage_wait, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(queue_observation, "_monotonic", lambda: clock[0])

    def unavailable(seconds):
        assert not held
        waits.append(seconds)
        clock[0] += seconds
        _publish(tmp_path, alive_generation=alive_generation)

    monkeypatch.setattr(queue_observation, "_sleep", unavailable)
    token = _usage_wait._CANCEL.set(threading.Event() if sliced else None)
    try:
        with pytest.raises(ua.ControlAuthorityUnavailable, match="root budget fence authority unavailable") as refused:
            with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=CHILD, root_task_id=ROOT)):
                ua.execute_physical_attempt(ua.AttemptRequest(model="m", provider="p"), lambda: sent.append(True))
        assert refused.value.cause == "dispatch_publication_unconfirmed"
    finally:
        _usage_wait._CANCEL.reset(token)
    assert sum(waits) == pytest.approx(0.2)
    assert revoke_on <= len(acquisitions) <= revoke_on + len(waits) + 1
    rows = ua.read_usage_records(tmp_path)
    assert sent == [] and [row["state"] for row in rows] == (["released"] if revoke_on == 2 else [])
    if rows:
        assert rows[0]["reason"] == "before_dispatch_failed:ControlAuthorityUnavailable"


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


@pytest.mark.parametrize("reservation_race", [0, 1, 2], indirect=True, ids=["initial", "reserve", "dispatch"])
def test_async_cancellation_joins_the_publication_wait_without_a_send(tmp_path, monkeypatch, reservation_race):
    from ouroboros import _usage_wait

    held, acquisitions, revoke_on = reservation_race
    waiting, release = threading.Event(), threading.Event()
    waits, cancellations = [], []

    def pause(_seconds):
        assert not held
        waits.append(True)
        cancellations.append(_usage_wait._CANCEL.get())
        waiting.set()
        assert release.wait(3)

    async def send():
        pytest.fail("cancelled publication wait sent a provider call")

    async def exercise():
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id=CHILD, root_task_id=ROOT)):
            task = asyncio.create_task(ua.execute_physical_attempt_async(ua.AttemptRequest(model="m", provider="p"), send))
            try:
                # Preparation (first store creation, earlier locks) has no test clock:
                # the attempt either enters its publication wait or ends on its own.
                while not waiting.is_set() and not task.done():
                    await asyncio.sleep(0.01)
                assert waiting.is_set(), task
                task.cancel()
                assert await asyncio.to_thread(cancellations[0].wait, 3)
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    monkeypatch.setattr(queue_observation, "_sleep", pause)
    asyncio.run(exercise())
    assert len(waits) == 1 and len(acquisitions) == {0: 0, 1: 1, 2: 3}[revoke_on]
    assert [row["state"] for row in ua.read_usage_records(tmp_path)] == (["released"] if revoke_on == 2 else [])


@pytest.mark.parametrize("control", ["deadline", "owner_pause"])
@pytest.mark.parametrize("recertify", [False, True])
def test_initial_publication_gap_checks_controls_without_a_wait_owner(tmp_path, monkeypatch, control, recertify):
    from ouroboros import model_wait, owner_pause
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted

    _publish(tmp_path, alive_generation=None)
    remaining, waits = [60.0], []
    monkeypatch.setattr(model_wait, "dispatch_deadline_remaining_sec", lambda: remaining[0])

    def interrupt(seconds):
        waits.append(seconds)
        assert len(waits) == 1, "initial publication wait ignored control"
        if control == "deadline":
            remaining[0] = 0.0
        else:
            owner_pause.install_fence(tmp_path, ROOT, request_id="initial-gap-pause")
        if recertify:  # Both a continuing gap and a returned certificate observe the control.
            _publish(tmp_path)

    monkeypatch.setattr(queue_observation, "_sleep", interrupt)
    with pytest.raises(PhysicalDispatchInterrupted) as error:
        _reserve(tmp_path, CHILD)
    assert error.value.control_reason == control
    assert len(waits) == 1 and ua.read_usage_records(tmp_path) == []


@pytest.mark.parametrize("operation", ["reserve", "dispatch"])
@pytest.mark.parametrize("stage", ["write", "commit", "exit"])
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


@pytest.mark.parametrize("operation", ["reserve", "dispatch"])
def test_repeated_publication_races_share_one_wait_budget(tmp_path, monkeypatch, operation):
    from ouroboros import _usage_wait

    _publish(tmp_path)
    reservation = _reserve(tmp_path, CHILD) if operation == "dispatch" else None
    clock, held, acquisitions, waits = [0.0], [], [], []
    original = ua._locked
    original_read = queue_observation.load_queue_document
    monkeypatch.setattr(queue_observation, "DISPATCH_PUBLICATION_WAIT_SEC", 0.2)
    timer = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(_usage_wait, "time", timer)
    monkeypatch.setattr(queue_observation, "_monotonic", timer.monotonic)

    @contextlib.contextmanager
    def locked(*args, **kwargs):
        with original(*args, **kwargs) as view:
            acquisitions.append(True)
            assert len(acquisitions) <= 3, "publication retries reset their deadline"
            (tmp_path / "state/queue.alive").unlink()
            held.append(True)
            try:
                yield view
            finally:
                held.pop()

    def read_publication(*args, **kwargs):
        if held:
            clock[0] += 0.05  # Charge the failed publication probe, not store acquisition.
        return original_read(*args, **kwargs)

    def publish(seconds):
        assert not held
        waits.append(seconds)
        clock[0] += seconds
        _publish(tmp_path)

    monkeypatch.setattr(ua, "_locked", locked)
    monkeypatch.setattr(queue_observation, "load_queue_document", read_publication)
    monkeypatch.setattr(queue_observation, "_sleep", publish)
    with pytest.raises(ua.ControlAuthorityUnavailable, match="root budget fence authority unavailable") as refused:
        ua.mark_dispatched(reservation) if reservation else _reserve(tmp_path, CHILD)
    assert refused.value.cause == "dispatch_publication_unconfirmed"
    assert len(acquisitions) == 3 and len(waits) == 2
    assert [row["state"] for row in ua.read_usage_records(tmp_path)] == (["reserved"] if reservation else [])


@pytest.mark.parametrize("operation", ["reserve", "dispatch"])
def test_nested_precheck_near_expiry_keeps_the_outer_publication_deadline(tmp_path, monkeypatch, operation):
    from ouroboros import _usage_wait

    _publish(tmp_path)
    reservation = _reserve(tmp_path, CHILD) if operation == "dispatch" else None
    clock, held, acquisitions, waits, precheck_gaps = [0.0], [], [], [], []
    original_lock, original_read = ua._locked, queue_observation.load_queue_document
    monkeypatch.setattr(queue_observation, "DISPATCH_PUBLICATION_WAIT_SEC", 0.2)
    monkeypatch.setattr(_usage_wait, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(queue_observation, "_monotonic", lambda: clock[0])

    @contextlib.contextmanager
    def locked(*args, **kwargs):
        with original_lock(*args, **kwargs) as view:
            acquisitions.append(True)
            (tmp_path / "state/queue.alive").unlink(missing_ok=True)
            held.append(True)
            try:
                yield view
            finally:
                held.pop()

    def read_publication(root, name):
        if waits and not held:
            # Publication returns, then is revoked at the next pre-lock check.
            precheck_gaps.append(clock[0])
            clock[0] += 0.001
            (tmp_path / "state/queue.alive").unlink(missing_ok=True)
        return original_read(root, name)

    def publish(seconds):
        assert not held
        waits.append(seconds)
        assert len(waits) < 8, "nested prechecks started another wait episode"
        # The first wake is near expiry; scheduler time counts, not just sleep argv.
        clock[0] += 0.19 if len(waits) == 1 else seconds
        _publish(tmp_path)

    monkeypatch.setattr(ua, "_locked", locked)
    monkeypatch.setattr(queue_observation, "load_queue_document", read_publication)
    monkeypatch.setattr(queue_observation, "_sleep", publish)
    token = _usage_wait._CANCEL.set(threading.Event())
    try:
        with pytest.raises(ua.ControlAuthorityUnavailable, match="root budget fence authority unavailable") as refused:
            ua.mark_dispatched(reservation) if reservation else _reserve(tmp_path, CHILD)
        assert refused.value.cause == "dispatch_publication_unconfirmed"
    finally:
        _usage_wait._CANCEL.reset(token)
    assert precheck_gaps and len(acquisitions) == 1
    assert clock[0] <= 0.202, "nested precheck reset the elapsed publication budget"
    assert waits[1] == pytest.approx(0.009)
    assert [row["state"] for row in ua.read_usage_records(tmp_path)] == (["reserved"] if reservation else [])


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
