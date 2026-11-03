"""Absence is a fact of the owning engine store, not a process generation."""
from concurrent.futures import Future
from types import SimpleNamespace
import threading

import pytest

from ouroboros import delegate_collection as body, delegate_custody as c, delegate_custody_current as current
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests.test_delegate_body_collection import started, terminal


def runtime(root, factory):
    stop = threading.Event()
    stop.set()
    result = body.BodyRuntime(root, stop_event=stop, gateway_factory=factory, max_workers=2)
    result.thread.join(10)
    assert not result.thread.is_alive()
    return result


@pytest.mark.parametrize("kind", ["same_store_restart", "replaced_store", "legacy", "timeout", "forbidden"])
def test_body_absence_requires_owning_store_and_closes_once(tmp_path, monkeypatch, kind):
    entry = started(tmp_path, daemon_store_id="" if kind == "legacy" else "store-a")
    error = ClaudexorUnavailable("daemon_unreachable" if kind == "timeout" else "run_not_found",
        "test observation", status_code=0 if kind == "timeout" else 403 if kind == "forbidden" else 404,
        observation_timeout=kind == "timeout")
    reads, closed = [], []
    gateway = SimpleNamespace(store_identity="store-b" if kind == "replaced_store" else "store-a",
                              close=lambda: closed.append(True))

    def read(*_args, **_kwargs):
        reads.append(True)
        raise error

    gateway.get_run = read
    observer = runtime(tmp_path, lambda: gateway)
    if kind in {"same_store_restart", "legacy"}:  # an unbound (older) row keeps the daemon's 404 contract
        observer._observe(entry)
        observer._observe(entry)  # stale caller's copy cannot duplicate the closing fact
        assert current.run(tmp_path, entry.run_id).settled
        assert not current.run(tmp_path, entry.run_id).ledger_recorded
        assert sum(row["type"] == c.CLOSED_ABSENT for row in c.custody_rows(tmp_path)) == 1
    else:
        with pytest.raises(ClaudexorUnavailable):
            observer._observe(entry)
        fresh = current.run(tmp_path, entry.run_id)
        assert not fresh.settled and not fresh.terminal_source
        assert not any(row["type"] == c.CLOSED_ABSENT for row in c.custody_rows(tmp_path))
        if kind == "replaced_store":
            assert fresh.containment_fault["reason"] == "run_absence_identity_unproven"
            monkeypatch.setattr(body.BodyRuntime, "active", property(lambda _self: True))
            submits = []
            def submit(*_args):
                submits.append(True)
                pending = Future()
                pending.set_result(None)
                return pending
            observer.observers = SimpleNamespace(submit=submit)
            observer.tick()
            observer.tick()
            assert not submits, "unchanged unknown 404 must not poll every tick"
            observer._drop_observer_gateway()
            observer.tick()
            assert submits == [True], "a changed transport can recheck the gap"
    assert bool(closed) is (kind in {"timeout", "replaced_store"})
    assert observer.observation(entry.run_id)["answered"] is (kind != "timeout")
    observer._drop_observer_gateway()


def test_successful_read_binds_store_and_same_store_restart_can_close(tmp_path):
    entry = started(tmp_path)
    gateway = SimpleNamespace(store_identity="store-a", close=lambda: None,
        get_run=lambda *_a, **_kw: {"summary": {"runId": entry.run_id, "state": "running"},
                                  "lastSeq": 1, "pendingInteractions": []})
    observer = runtime(tmp_path, lambda: gateway)
    observer._observe(entry)
    assert current.run(tmp_path, entry.run_id).daemon_store_id == "store-a"
    # A duplicate STARTED fact and replay preserve the original binding.
    c.record_started(tmp_path, c.RunCustody(run_id=entry.run_id, task_id=entry.task_id, daemon_store_id="foreign"))
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.daemon_store_id == "store-a"
    assert current.fold(list(c.custody_rows(tmp_path)))["runs"][entry.run_id][0]["daemon_store_id"] == "store-a"
    replacement = SimpleNamespace(store_identity="store-a", close=lambda: None)
    def absent(*_a, **_kw):
        raise ClaudexorUnavailable("run_not_found", "gone", status_code=404)
    replacement.get_run = absent
    observer._drop_observer_gateway()
    observer.gateway_factory = lambda: replacement
    observer._observe(fresh)
    assert current.run(tmp_path, entry.run_id).settled
    observer._drop_observer_gateway()


@pytest.mark.parametrize("newer_fault", [False, True])
def test_successful_read_resolves_only_its_own_absence_gap(tmp_path, newer_fault):
    entry = started(tmp_path)
    c.record_containment_fault(tmp_path, entry, "run_absence_identity_unproven")
    observed = current.run(tmp_path, entry.run_id)
    if newer_fault:
        c.record_containment_fault(tmp_path, entry, "cancel_unverified")
    gateway = SimpleNamespace(store_identity="store-a", close=lambda: None,
        get_run=lambda *_a, **_kw: {"summary": {"runId": entry.run_id, "state": "running"},
                                  "lastSeq": 1, "pendingInteractions": []})
    observer = runtime(tmp_path, lambda: gateway)
    observer._observe(observed)
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.daemon_store_id == "store-a" and not fresh.settled
    assert fresh.containment_fault.get("reason") == ("cancel_unverified" if newer_fault else None)
    observer._drop_observer_gateway()


@pytest.mark.parametrize("identity", ["store-a", "store-b", ""])
def test_reconcile_uses_same_store_guard(tmp_path, identity):
    entry = started(tmp_path, daemon_store_id="store-a")
    def absent(*_a, **_kw):
        raise ClaudexorUnavailable("run_not_found", "gone", status_code=404)
    gateway = SimpleNamespace(store_identity=identity, get_run=absent)
    answer = c._reconcile_one(tmp_path, gateway, entry)
    assert answer["action"] == ("absent" if identity == "store-a" else "unreadable")
    assert current.run(tmp_path, entry.run_id).settled is (identity == "store-a")


@pytest.mark.parametrize("kind", ["same_store", "replaced_store", "legacy", "missing_identity", "timeout", "forbidden"])
def test_requested_external_stop_requires_owning_store_absence(tmp_path, monkeypatch, kind):
    from ouroboros import external_runs
    from ouroboros.gateways import claudexor

    entry = started(tmp_path, daemon_store_id="" if kind == "legacy" else "store-a")
    status = 0 if kind == "timeout" else 403 if kind == "forbidden" else 404
    def read(*_args, **_kwargs):
        raise ClaudexorUnavailable("daemon_unreachable" if kind == "timeout" else "run_unavailable",
                                   "controlled observation", status_code=status)
    gateway = SimpleNamespace(store_identity="store-b" if kind == "replaced_store" else
                              "" if kind == "missing_identity" else "store-a",
                              get_run=read, handshake=lambda: None, close=lambda: None,
                              cancel_run=lambda *_a, **_kw: pytest.fail("the stop was already issued"))
    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda: gateway)
    prior = {"runs": [{"run_id": entry.run_id, "stop_policy": "request_stop", "stop_outcome": "requested"}]}
    observed = external_runs.observe_task_runs(tmp_path, entry.task_id, prior=prior)
    row, = observed["runs"]
    confirmed = kind in {"same_store", "legacy"}  # an unbound (older) row keeps the daemon's 404 contract
    assert row["stop_outcome"] == (c.CANCEL_CONFIRMED if confirmed else "unverified")
    assert row["state"] == (external_runs.EXTERNAL_STOP_CONFIRMED if confirmed else external_runs.EXTERNAL_STOP_UNKNOWN)
    assert external_runs.task_owned_runs_open(observed) is not confirmed
    assert current.run(tmp_path, entry.run_id).settled is confirmed
    assert sum(event["type"] == c.CLOSED_ABSENT for event in c.custody_rows(tmp_path)) == int(confirmed)


def test_absence_never_replaces_retained_terminal_evidence(tmp_path):
    entry = started(tmp_path, daemon_store_id="store-a")
    body.observe_run(tmp_path, entry, terminal())
    fresh = current.run(tmp_path, entry.run_id)
    with pytest.raises(ClaudexorUnavailable, match="owning store"):
        c.close_absent_run(tmp_path, SimpleNamespace(store_identity="store-a"), fresh, "test")
    assert current.run(tmp_path, entry.run_id).terminal_source == fresh.terminal_source
    assert not any(row["type"] == c.CLOSED_ABSENT for row in c.custody_rows(tmp_path))


def test_observer_handshake_leaves_runtime_lock_available(tmp_path):
    entered, release = threading.Event(), threading.Event()
    gateway = SimpleNamespace(close=lambda: None)
    def connect():
        entered.set()
        assert release.wait(10)
        return gateway
    observer = runtime(tmp_path, connect)
    thread = threading.Thread(target=observer._observer_gateway)
    thread.start()
    assert entered.wait(10)
    try:
        assert observer.lock.acquire(timeout=1), "handshake blocked the body's control lock"
        observer.lock.release()
    finally:
        release.set()
        thread.join(10)
        observer._drop_observer_gateway()
    assert not thread.is_alive()
