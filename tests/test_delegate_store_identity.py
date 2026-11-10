"""Run identity belongs to the engine store that admitted it: admission, recovery and absence.

The engine double answers in Claudexor 5.1's own wire shapes (``packages/control-api``): a
start receipt names its store only to a request sending ``X-Claudexor-Store-Identity: 1``, a
run-detail 404 names the store it read (``context.storeId``), a control-route 404 names none,
and idempotency keys live in each store's own database.
"""
from concurrent.futures import Future
from types import SimpleNamespace
import json
import threading

import httpx
import pytest

from ouroboros import delegate_collection as body, delegate_custody as c, delegate_custody_current as current
from ouroboros.config import CLAUDEXOR_MIN_VERSION, CLAUDEXOR_PROTOCOL_MAJOR
from ouroboros.gateways import claudexor as cx, claudexor_metadata as metadata
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)
from tests.test_builtin_refusal_results import _delegate_registry
from tests.test_delegate_body_collection import started, terminal

_REAL_GATEWAY = cx.ClaudexorGateway


class Engine:
    """One serving store (``store=""``: an engine that names none) and the runs it holds."""

    def __init__(self, tmp_path, store="store-a"):
        descriptor = tmp_path / "control-api.json"
        descriptor.write_text("{}", encoding="utf-8")
        self.endpoint = cx.DaemonEndpoint("127.0.0.1", 1, "secret-token", str(descriptor),
                                          metadata.descriptor_stamp(descriptor))
        self.root, self.store, self.runs, self.posts, self.lose = str(tmp_path), store, {}, [], 0
        self.control_404 = False  # a control route that answers 404 while the run lives (no store named)
        self.replace_at_post = ""  # the store changes after the proving handshake, before the POST lands

    def _named(self, request):
        opted = request.headers.get("x-claudexor-store-identity") == "1"
        return {"storeId": self.store} if opted and self.store else {}

    def __call__(self, request):
        path = request.url.path
        if path == "/v2/handshake":
            return httpx.Response(200, json={"protocolMajor": CLAUDEXOR_PROTOCOL_MAJOR, "compatible": True,
                                             "engine": {"version": CLAUDEXOR_MIN_VERSION}, **self._named(request)})
        if path == "/v2/agent-capabilities":
            return httpx.Response(200, json={"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                                            "accessProfilesSupported": ["readonly"]}]})
        if path == "/v2/quota":
            return httpx.Response(200, json={"snapshots": []})
        if path == "/v2/operations":  # no compact wait: each run's detail is read
            return httpx.Response(200, json={"operations": []})
        if path == "/v2/projects":
            return httpx.Response(200, json={"projects": [{"id": "prj-existing", "root": self.root}]})
        if path == "/v2/runs":
            self.store, self.replace_at_post = self.replace_at_post or self.store, ""
            key = request.headers.get("Idempotency-Key")
            self.posts.append((key, self.store, request.headers.get("x-claudexor-store-identity"),
                               json.loads(request.read())))
            run_id = next((rid for rid, (store, k) in self.runs.items() if (store, k) == (self.store, key)),
                          f"run-{len(self.posts)}")  # the store's own idempotency table
            self.runs[run_id] = (self.store, key)  # accepted: only its reply below may be lost
            if self.lose:
                self.lose -= 1
                raise httpx.ConnectError("the reply was lost after the POST")
            return httpx.Response(200, json={"runId": run_id, "runDir": "/tmp/run", **self._named(request)})
        run_id = path.split("/")[3]
        held = run_id in self.runs and self.runs[run_id][0] == self.store
        if path.endswith("/control"):
            if held and not self.control_404:
                return httpx.Response(200, json={"accepted": True, "status": "cancelling"})
            return httpx.Response(404, json={"error": "no such run"})
        assert request.method == "GET" and path == f"/v2/runs/{run_id}", path
        if held:
            return httpx.Response(200, json={"summary": {"runId": run_id, "state": "running"},
                                             "lastSeq": 1, "pendingInteractions": []})
        return httpx.Response(404, json={"error": "no such run", "context": {"storeId": self.store} if self.store else {}})

    def gateway(self):
        gateway = _REAL_GATEWAY(self.endpoint)
        gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(self),
                                       headers=dict(gateway._client.headers))
        gateway.handshake()  # the serving store, as ensure_owned_gateway proves it before every start
        return gateway

    def absent(self, store):
        """The exact 404 a run-detail read of ``store`` raises through the production gateway."""
        self.store, previous = store, self.store
        try:
            with pytest.raises(ClaudexorUnavailable) as caught:
                self.gateway().get_run("run-unknown")
        finally:
            self.store = previous
        return caught.value


def runtime(root, factory):
    stop = threading.Event()
    stop.set()
    result = body.BodyRuntime(root, stop_event=stop, gateway_factory=factory)
    result.thread.join(10)
    assert not result.thread.is_alive()
    return result


@pytest.mark.parametrize("kind", ["same_store", "legacy", "other_store", "unnamed", "forbidden"])
def test_body_absence_requires_owning_store_and_closes_once(tmp_path, monkeypatch, kind):
    engine = Engine(tmp_path, store={"other_store": "store-b", "unnamed": ""}.get(kind, "store-a"))
    entry = started(tmp_path, daemon_store_id="" if kind == "legacy" else "store-a")
    gateway, closes = engine.gateway(), []
    gateway.close = lambda: closes.append(True)
    if kind == "forbidden":
        def refuse(*_args, **_kwargs):
            raise ClaudexorUnavailable("forbidden", "test observation", status_code=403)
        gateway.get_run = refuse
    observer = runtime(tmp_path, lambda: gateway)
    if kind in {"same_store", "legacy"}:  # an unbound (older) row keeps the daemon's 404 contract
        observer._observe(entry)
        observer._observe(entry)  # stale caller's copy cannot duplicate the closing fact
        assert current.run(tmp_path, entry.run_id).settled
        assert not current.run(tmp_path, entry.run_id).ledger_recorded
        closed, = [row for row in c.custody_rows(tmp_path) if row["type"] == c.CLOSED_ABSENT]
        assert closed["daemon_store_id"] == "store-a"  # the store whose answer closed it
    else:
        with pytest.raises(ClaudexorUnavailable):
            observer._observe(entry)
        fresh = current.run(tmp_path, entry.run_id)
        assert not fresh.settled and not fresh.terminal_source
        assert not any(row["type"] == c.CLOSED_ABSENT for row in c.custody_rows(tmp_path))
        if kind in {"other_store", "unnamed"}:
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
    assert bool(closes) is (kind in {"other_store", "unnamed"}), "an answered refusal keeps its transport"
    assert observer.observation(entry.run_id)["answered"]
    observer._drop_observer_gateway()


def test_a_stale_handshake_cannot_prove_absence_for_another_store(tmp_path):
    """#2649: the store is replaced inside one descriptor generation, so the cached handshake
    still names the run's store; only the 404's own ``storeId`` decides, here unknown."""
    engine = Engine(tmp_path)
    gateway = engine.gateway()
    entry = started(tmp_path, daemon_store_id="store-a")
    engine.store = "store-b"
    assert gateway.store_identity == "store-a"
    answer = c._reconcile_one(tmp_path, gateway, entry)
    assert answer["action"] == "unreadable"
    assert not current.run(tmp_path, entry.run_id).settled


def test_successful_read_binds_store_and_same_store_restart_can_close(tmp_path):
    entry = started(tmp_path)
    gateway = SimpleNamespace(store_identity="store-a", close=lambda: None,
        get_run=lambda *_a, **_kw: {"summary": {"runId": entry.run_id, "state": "running",
                                                "effectiveAccess": "readonly"},
                                  "lastSeq": 1, "pendingInteractions": []})
    observer = runtime(tmp_path, lambda: gateway)
    observer._observe(entry)
    assert current.run(tmp_path, entry.run_id).daemon_store_id == "store-a"
    # A duplicate STARTED fact and replay preserve the original binding.
    c.record_started(tmp_path, c.RunCustody(run_id=entry.run_id, task_id=entry.task_id, daemon_store_id="foreign"))
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.daemon_store_id == "store-a"
    assert current.fold(list(c.custody_rows(tmp_path)))["runs"][entry.run_id][0]["daemon_store_id"] == "store-a"
    replacement = Engine(tmp_path).gateway()  # the same store after a restart: its 404 names it
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
        get_run=lambda *_a, **_kw: {"summary": {"runId": entry.run_id, "state": "running",
                                                "effectiveAccess": "readonly"},
                                  "lastSeq": 1, "pendingInteractions": []})
    observer = runtime(tmp_path, lambda: gateway)
    observer._observe(observed)
    fresh = current.run(tmp_path, entry.run_id)
    assert fresh.daemon_store_id == "store-a" and not fresh.settled
    assert fresh.containment_fault.get("reason") == ("cancel_unverified" if newer_fault else None)
    observer._drop_observer_gateway()


@pytest.mark.parametrize("named", ["store-a", "store-b", ""])
def test_reconcile_uses_same_store_guard(tmp_path, named):
    entry = started(tmp_path, daemon_store_id="store-a")
    gateway = Engine(tmp_path, store=named).gateway()
    answer = c._reconcile_one(tmp_path, gateway, entry)
    assert answer["action"] == ("absent" if named == "store-a" else "unreadable")
    assert current.run(tmp_path, entry.run_id).settled is (named == "store-a")


@pytest.mark.parametrize("kind", ["same_store", "legacy", "missing_identity"])
def test_requested_external_stop_requires_owning_store_absence(tmp_path, monkeypatch, kind):
    from ouroboros import external_runs
    from ouroboros.gateways import claudexor

    entry = started(tmp_path, daemon_store_id="" if kind == "legacy" else "store-a")
    gateway = Engine(tmp_path, store="" if kind == "missing_identity" else "store-a").gateway()
    gateway.cancel_run = lambda *_a, **_kw: pytest.fail("the stop was already issued")
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


@pytest.mark.parametrize("kind", ["live", "absent", "other_store", "legacy_absent"])
def test_cancel_route_404_never_closes_a_run_by_itself(tmp_path, kind):
    """The control route's 404 names no store; the run-detail read decides absence."""
    engine = Engine(tmp_path, store="store-b" if kind == "other_store" else "store-a")
    entry = started(tmp_path, daemon_store_id="" if kind == "legacy_absent" else "store-a")
    engine.control_404 = True
    if kind == "live":
        engine.runs[entry.run_id] = ("store-a", entry.invocation_id)
    result = c.cancel_and_verify(tmp_path, engine.gateway(), entry, "test")
    closed = kind in {"absent", "legacy_absent"}
    assert result["outcome"] == (c.CANCEL_CONFIRMED if closed else c.CANCEL_CONTAINMENT_FAULT)
    assert current.run(tmp_path, entry.run_id).settled is closed
    if not closed:
        assert current.run(tmp_path, entry.run_id).containment_fault["reason"] == "cancel_unreachable"


def test_absence_never_replaces_retained_terminal_evidence(tmp_path):
    entry = started(tmp_path, daemon_store_id="store-a")
    body.observe_run(tmp_path, entry, terminal())
    fresh = current.run(tmp_path, entry.run_id)
    with pytest.raises(ClaudexorUnavailable, match="owning store"):
        c.close_absent_run(tmp_path, SimpleNamespace(), fresh, "test", absence=Engine(tmp_path).absent("store-a"))
    assert current.run(tmp_path, entry.run_id).terminal_source == fresh.terminal_source
    assert not any(row["type"] == c.CLOSED_ABSENT for row in c.custody_rows(tmp_path))


def _delegating(tmp_path, monkeypatch, store="store-a"):
    engine = Engine(tmp_path, store=store)
    _select_transport_access(monkeypatch, "readonly")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(cx, "ClaudexorGateway", lambda *_a, **_k: engine.gateway())
    return _delegate_registry(tmp_path, monkeypatch), engine


def _start(registry, retry_of=""):
    """A fresh start as the selected row makes it, or the public replay of a recorded one."""
    from ouroboros.tools import delegate

    if retry_of:
        return json.loads(registry.execute_result("delegate_start", {"retry_of": retry_of}).text)
    return json.loads(delegate._delegate_start(registry._ctx, "Summarize the layout.", max_seconds=120).text)


@pytest.mark.parametrize("receipt", ["names_its_store", "names_another_store", "names_none"])
def test_a_start_binds_the_store_its_receipt_names(tmp_path, monkeypatch, receipt):
    registry, engine = _delegating(tmp_path, monkeypatch, store="" if receipt == "names_none" else "store-a")
    engine.replace_at_post = "store-b" if receipt == "names_another_store" else ""
    payload = _start(registry)
    assert payload["status"] == "started", payload
    assert engine.posts[-1][2] == "1", "every start opts into the store identity"
    bound = {"names_its_store": "store-a", "names_another_store": "store-b", "names_none": ""}[receipt]
    assert current.run(tmp_path, payload["run_id"]).daemon_store_id == bound
    requested = c.invocation_record(tmp_path, payload["invocation_id"])
    assert requested["daemon_store_id"] == ""
    assert requested["store_identity_unproven"] is True


@pytest.mark.parametrize("serving", ["store-a", "store-b", ""])
def test_a_retry_replays_its_key_only_into_the_recording_store(tmp_path, monkeypatch, serving):
    registry, engine = _delegating(tmp_path, monkeypatch)
    engine.lose = 1
    lost = _start(registry)
    token = lost["pending_invocation_id"]
    requested = c.invocation_record(tmp_path, token)
    assert requested["daemon_store_id"] == ""
    assert requested["store_identity_unproven"] is True
    (accepted, owner), = engine.runs.items()  # the first POST was accepted; only its reply was lost
    assert owner == ("store-a", token)
    engine.store = serving
    payload = _start(registry, retry_of=token)
    assert list(engine.runs) == [accepted], "exactly one run was ever accepted"
    assert payload["reason"] == "invocation_store_unproven", payload
    assert len(engine.posts) == 1, "nothing was re-sent before the POST receipt proves its store"
    assert c.invocation_record(tmp_path, token)["state"] == "pending"


@pytest.mark.parametrize("recorded,serving", [("store-a", "store-a"), ("store-a", "store-b"), ("", "store-b"),
                                             ("store-a", "")])  # a replacement that names no store
def test_the_recovery_sweep_replays_only_into_the_recording_store(tmp_path, monkeypatch, recorded, serving):
    registry, engine = _delegating(tmp_path, monkeypatch, store=recorded)
    engine.lose = 1
    token = _start(registry)["pending_invocation_id"]
    (accepted, _owner), = engine.runs.items()  # accepted in the recording store; only its reply was lost
    engine.store = serving
    outcomes = c.reconcile_orphaned_runs(tmp_path, set(), gateway_factory=engine.gateway, replay_pending=lambda: True)
    assert len(engine.posts) == 1
    assert list(engine.runs) == [accepted], "the unknown POST outcome is retained without replay"
    outcome, = [row for row in outcomes if row.get("invocation_id") == token]
    assert (outcome["action"], outcome["reason"]) == ("invocation_retained", "invocation_store_unproven")
    assert [record["invocation_id"] for record in c.pending_invocations(tmp_path)] == [token]


def test_cached_pending_projection_keeps_store_facts_from_its_source_row(tmp_path):
    from ouroboros.delegate_pending import current_rows, pending_invocations, replay_store_unproven

    source = {"type": c.START_REQUESTED, "invocation_id": "cached-start", "request": {"prompt": "hello"},
              "handshake_store_id": "store-a", "store_identity_unproven": True, "daemon_store_id": ""}
    old = pending_invocations(tmp_path, [source], materialize=False)
    for key in ("handshake_store_id", "store_identity_unproven", "daemon_store_id"):
        old[0].pop(key, None)  # cached by a body predating store identity, but its source row kept it
    projected, = current_rows(tmp_path, [{"request": source, "pending": old}])
    assert projected["handshake_store_id"] == "store-a" and projected["store_identity_unproven"] is True
    assert replay_store_unproven(projected, ""), "an unnamed replacement cannot masquerade as legacy admission"
    assert "handshake_store_id" not in old[0], "a reader doesn't mutate the cached source"


def test_identityless_engine_keeps_exact_legacy_replay(tmp_path, monkeypatch):
    registry, engine = _delegating(tmp_path, monkeypatch, store="")
    engine.lose = 1
    token = _start(registry)["pending_invocation_id"]
    payload = _start(registry, retry_of=token)
    assert payload["status"] == "started" and payload["idempotent_recovery"] is True
    assert len(engine.runs) == 1 and len(engine.posts) == 2
    assert engine.posts[0][0] == engine.posts[1][0] == token
    assert engine.posts[0][3] == engine.posts[1][3]


def test_receipt_bound_same_store_replay_remains_supported(tmp_path, monkeypatch):
    """A request with a recorded admitting store may still replay only there."""
    registry, engine = _delegating(tmp_path, monkeypatch)
    engine.lose = 1
    token = _start(registry)["pending_invocation_id"]
    row = c.invocation_record(tmp_path, token)
    row["store_identity_unproven"] = False
    row["daemon_store_id"] = "store-a"
    # Test the replay guard's receipt-bound positive path without changing the
    # first POST's historical row in the custody log.
    assert c.replay_store_unproven(row, "store-a") == ""
    assert c.replay_store_unproven(row, "store-b")


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
