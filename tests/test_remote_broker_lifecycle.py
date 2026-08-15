"""Broker lifecycle: one owner per generation, and four ways it must not fail.

Every case here is a failure mode that would be invisible in a happy-path test
and expensive in production (RWS v2 §3.1/§3.4):

* a respawned worker must REPLACE its channel, not add a second live one;
* a call from a stale generation must be REFUSED with a typed code, never served
  and never left hanging;
* a broker shutdown during a pending call must resolve that call with an honest
  completion state instead of stranding the caller on its deadline;
* panic must never wait for the ordinary broker lock — a software delay in the
  panic path is exactly what the emergency-stop invariant forbids.

The transport is a fake throughout: what is under test is broker/worker
lifecycle, not SSH.
"""

from __future__ import annotations

import threading
import time

import pytest

from ouroboros.remote_worker_proxy import BROKER_GENERATION_STALE
from ouroboros.remote_workspace import RemoteSessionBroker
from ouroboros.workspace_diagnostics import RemoteWorkspaceError

pytestmark = pytest.mark.serial

_MANIFEST = {
    "schema_version": 1,
    "manifest_sha256": "a" * 64,
    "public_schema_sha256": "c" * 64,
    "native_operations": [{"name": "read_file"}],
    "native_kernel_modules": ["ouroboros.workspace_native"],
    "native_import_modules": ["ouroboros.workspace_native"],
    "native_import_edges": {},
}


class _FakeTransport:
    """A transport that records what the broker asked of it."""

    def __init__(self, request, *, home_importer=None):
        self.request = request
        self.home_importer = home_importer
        self.panicked = threading.Event()
        self.closed = threading.Event()
        self.block = threading.Event()
        self.entered = threading.Event()

    def handshake(self):
        return {
            "host_id": "host-1",
            "workspace_id": "workspace-1",
            "canonical_root": self.request.remote_root.rstrip("/"),
            "capability_hash": _MANIFEST["manifest_sha256"],
        }

    def artifact_identity(self):
        return {}

    def prepare(self, message, blobs):
        del blobs
        self.entered.set()
        # Held open so a test can shut the broker down mid-call.
        self.block.wait(10)
        return {
            "request_id": message["request_id"],
            "operation_id": message["operation_id"],
            "tool": message["tool"],
            "prepared_token": "token",
            "prepared_hash": "b" * 64,
            "expires_at_ms": int(time.time() * 1000) + 60_000,
            "execution_args": {},
            "native_facts": {},
        }

    def cancel(self, _message):
        return True

    def task_lease(self, _task_id, forget=False):
        del forget
        return False

    def health(self):
        return {"status": "ready", "phase": "ready"}

    def panic(self):
        self.panicked.set()

    def close(self):
        self.closed.set()


def _broker(tmp_path, generation="generation-1", transports=None):
    def factory(request, *, home_importer=None):
        transport = _FakeTransport(request, home_importer=home_importer)
        if transports is not None:
            transports.append(transport)
        return transport

    broker = RemoteSessionBroker(
        tmp_path,
        generation,
        _MANIFEST,
        transport_factory=factory,
    )
    broker.start()
    return broker


def _admit(broker):
    return broker.admit_workspace(
        {"id": "connection-1", "ssh_alias": "build"},
        remote_root="/srv/project",
        project_id="project-1",
        workspace_id="workspace-1",
        task_id="task-1",
    )


def test_session_admission_returns_identity_and_nothing_else(tmp_path):
    """The contract RWS-109 asserts against, checked without Docker.

    Session admission returns identities, target-native facts and the DERIVED
    placement descriptor — never a transport handle, never a connection secret,
    and never Home task state such as a staged attachment manifest.
    """

    transports: list[_FakeTransport] = []
    broker = _broker(tmp_path, transports=transports)
    try:
        admitted = _admit(broker)

        assert admitted["ok"] is True
        assert admitted["workspace_ref"] == {
            "kind": "ssh",
            "connection_id": "connection-1",
            "remote_root": "/srv/project",
            "workspace_id": "workspace-1",
        }
        assert admitted["canonical_root"] == "/srv/project"
        assert admitted["capability_hash"] == _MANIFEST["manifest_sha256"]
        assert admitted["server_generation"] == broker.server_generation
        # Home admission policy is somebody else's job.
        assert "attachment_manifest" not in admitted
        rendered = repr(admitted)
        assert "ssh_alias" not in rendered
        assert "transport" not in rendered
    finally:
        for transport in transports:
            transport.block.set()
        broker.close(timeout_sec=1)


def test_an_admission_in_flight_already_counts_as_an_active_lease(tmp_path):
    """The pre-queue window: committed to a connection, holding nothing yet.

    An owner retire/retrust between "admission started" and "session bound" would
    find no task session and no service lease, read the connection as idle, and
    pull it out from under a task that is already committed to it. So an admission
    in flight counts, and the count clears when it finishes.
    """

    transports: list[_FakeTransport] = []
    broker = _broker(tmp_path, transports=transports)
    try:
        assert broker.has_active_lease("connection-1") is False

        entered = threading.Event()
        released = threading.Event()
        original = broker._admit_on_broker

        def blocking_admit(payload):
            entered.set()
            released.wait(timeout=5)
            return original(payload)

        broker._admit_on_broker = blocking_admit
        worker = threading.Thread(target=lambda: _admit(broker), daemon=True)
        worker.start()
        assert entered.wait(timeout=5)
        # Nothing is bound yet — this is precisely the window a lease check misses.
        assert broker.has_active_lease("connection-1") is True

        released.set()
        worker.join(timeout=5)
        # Still busy, but now for the ordinary reason: the task session is bound.
        assert broker.has_active_lease("connection-1") is True
        assert broker.finish_task(
            {
                "kind": "ssh",
                "connection_id": "connection-1",
                "remote_root": "/srv/project",
                "workspace_id": "workspace-1",
            },
            task_id="task-1",
        )
        assert broker.has_active_lease("connection-1") is False
    finally:
        for transport in transports:
            transport.block.set()
        broker.close(timeout_sec=1)


# ── worker channel lifecycle ────────────────────────────────────────────


def test_respawn_replaces_a_worker_channel_instead_of_adding_one(tmp_path):
    broker = _broker(tmp_path)
    try:
        first = broker.create_worker_pipe_proxy("worker:0")
        second = broker.create_worker_pipe_proxy("worker:0")

        # Exactly one live broker endpoint for this worker id.
        assert len(broker._worker_channels.live()) == 1
        # The replaced worker's channel is dead, so its calls fail typed
        # instead of being answered on behalf of a process that no longer exists.
        with pytest.raises(RemoteWorkspaceError) as raised:
            first.cancel({"kind": "ssh"}, task_id="task-1")
        assert raised.value.code == "broker_pipe_closed"
        assert second.server_generation == broker.server_generation
    finally:
        broker.close(timeout_sec=1)


def test_distinct_workers_keep_distinct_channels(tmp_path):
    broker = _broker(tmp_path)
    try:
        broker.create_worker_pipe_proxy("worker:0")
        broker.create_worker_pipe_proxy("worker:1")

        assert len(broker._worker_channels.live()) == 2

        assert broker.close_worker_pipe_proxy("worker:0") is True
        assert len(broker._worker_channels.live()) == 1
        # Closing an unknown owner is not an error, just nothing to close.
        assert broker.close_worker_pipe_proxy("worker:0") is False
    finally:
        broker.close(timeout_sec=1)


def test_closing_the_pool_keeps_the_broker_alive(tmp_path):
    broker = _broker(tmp_path)
    try:
        broker.create_worker_pipe_proxy("worker:0")
        broker.create_worker_pipe_proxy("worker:1")

        assert broker.close_worker_pipe_proxies() == 2

        # The broker belongs to the server generation, not to a worker pool.
        assert broker._worker_channels.live() == []
        assert broker.create_worker_pipe_proxy("worker:0") is not None
    finally:
        broker.close(timeout_sec=1)


# ── stale generation ────────────────────────────────────────────────────


def test_a_stale_generation_call_is_refused_and_does_not_hang(tmp_path):
    broker = _broker(tmp_path)
    try:
        proxy = broker.create_worker_pipe_proxy("worker:0")
        # Simulate a proxy that survived into a NEW server generation.
        proxy._server_generation = "generation-0"

        started = time.monotonic()
        with pytest.raises(RemoteWorkspaceError) as raised:
            proxy.cancel({"kind": "ssh"}, task_id="task-1")
        elapsed = time.monotonic() - started

        assert raised.value.code == BROKER_GENERATION_STALE
        assert raised.value.completion == "not_started"
        assert raised.value.details["expected_generation"] == broker.server_generation
        # A refusal, not a timeout: nowhere near the 120s pipe deadline.
        assert elapsed < 10
    finally:
        broker.close(timeout_sec=1)


def test_a_matching_generation_call_still_reaches_the_broker(tmp_path):
    broker = _broker(tmp_path)
    try:
        proxy = broker.create_worker_pipe_proxy("worker:0")

        # No session exists, so this must fail with the SESSION error — proving
        # the generation check let the call through rather than short-circuiting.
        with pytest.raises(RemoteWorkspaceError) as raised:
            proxy.cancel(
                {
                    "kind": "ssh",
                    "connection_id": "connection-1",
                    "workspace_id": "workspace-1",
                    "remote_root": "/srv/project",
                },
                task_id="task-1",
            )
        assert raised.value.code != BROKER_GENERATION_STALE
    finally:
        broker.close(timeout_sec=1)


# ── shutdown during a pending call ──────────────────────────────────────


def test_broker_shutdown_during_a_pending_call_resolves_it(tmp_path):
    transports: list[_FakeTransport] = []
    broker = _broker(tmp_path, transports=transports)
    outcome: list[object] = []
    try:
        _admit(broker)
        transport = transports[-1]

        def call():
            try:
                outcome.append(
                    broker.prepare(
                        {
                            "kind": "ssh",
                            "connection_id": "connection-1",
                            "workspace_id": "workspace-1",
                            "remote_root": "/srv/project",
                        },
                        request_id="request-1",
                        operation_id="operation-1",
                        tool="read_file",
                        args={},
                        task_id="task-1",
                    )
                )
            except BaseException as exc:  # noqa: BLE001 — the outcome IS the assertion
                outcome.append(exc)

        caller = threading.Thread(target=call, daemon=True)
        caller.start()
        assert transport.entered.wait(5), "the call never reached the transport"

        broker.close(timeout_sec=1)
        transport.block.set()
        caller.join(timeout=15)

        assert not caller.is_alive(), "a pending call outlived broker shutdown"
        assert outcome, "the pending call neither returned nor raised"
        # Whatever the answer is, it is an ANSWER — the caller is never stranded.
        assert transport.panicked.is_set()
    finally:
        transport = transports[-1] if transports else None
        if transport is not None:
            transport.block.set()
        broker.close(timeout_sec=1)


def test_a_call_after_close_is_refused_immediately(tmp_path):
    """Closed means closed — typed and immediate, on both call paths.

    Queued methods go through `_submit`, which refuses with `broker_closed`.
    `cancel` deliberately bypasses the queue (it must not sit behind a blocked
    ordinary request), so it fails at session lookup instead — still typed, still
    immediate, never a wait.
    """

    broker = _broker(tmp_path)
    broker.close(timeout_sec=1)
    ref = {
        "kind": "ssh",
        "connection_id": "connection-1",
        "workspace_id": "workspace-1",
        "remote_root": "/srv/project",
    }

    with pytest.raises(RemoteWorkspaceError) as queued:
        broker.close_project_session(ref, project_id="project-1")
    assert queued.value.code == "broker_closed"

    started = time.monotonic()
    with pytest.raises(RemoteWorkspaceError) as direct:
        broker.cancel(ref, task_id="task-1")
    assert direct.value.code in {
        "task_session_unbound",
        "remote_session_disconnected",
    }
    assert time.monotonic() - started < 5


# ── panic never waits ───────────────────────────────────────────────────


def test_panic_does_not_wait_for_the_broker_state_lock(tmp_path):
    """The point of the append-only panic snapshots.

    A thread holds `_state_lock` for the whole test. Panic must still kill every
    transport, because panic reads `_panic_transports` directly and takes the
    lock only opportunistically (`blocking=False`). If it ever blocked here, a
    stuck broker operation would delay a kill — forbidden.
    """

    transports: list[_FakeTransport] = []
    broker = _broker(tmp_path, transports=transports)
    holding = threading.Event()
    release = threading.Event()

    def hold_lock():
        with broker._state_lock:
            holding.set()
            release.wait(20)

    holder = threading.Thread(target=hold_lock, daemon=True)
    try:
        _admit(broker)
        assert transports, "admission created no transport"
        holder.start()
        assert holding.wait(5)

        started = time.monotonic()
        broker.panic()
        elapsed = time.monotonic() - started

        assert elapsed < 5, "panic waited on the broker state lock"
        assert all(transport.panicked.is_set() for transport in transports)
    finally:
        release.set()
        holder.join(timeout=5)
        for transport in transports:
            transport.block.set()
        broker.close(timeout_sec=1)


def test_panic_close_all_reaches_every_live_broker(tmp_path):
    first_transports: list[_FakeTransport] = []
    second_transports: list[_FakeTransport] = []
    first = _broker(tmp_path / "a", "generation-a", first_transports)
    second = _broker(tmp_path / "b", "generation-b", second_transports)
    try:
        _admit(first)
        _admit(second)

        RemoteSessionBroker.panic_close_all()

        assert all(t.panicked.is_set() for t in first_transports)
        assert all(t.panicked.is_set() for t in second_transports)
    finally:
        for transport in (*first_transports, *second_transports):
            transport.block.set()
        first.close(timeout_sec=1)
        second.close(timeout_sec=1)


def test_the_broker_injects_the_home_importer_into_every_transport(tmp_path):
    """The one Home seam, proven to be injected rather than imported."""

    transports: list[_FakeTransport] = []
    sentinel = object()

    def factory(request, *, home_importer=None):
        transport = _FakeTransport(request, home_importer=home_importer)
        transports.append(transport)
        return transport

    broker = RemoteSessionBroker(
        tmp_path,
        "generation-1",
        _MANIFEST,
        transport_factory=factory,
        home_importer=sentinel,
    )
    broker.start()
    try:
        _admit(broker)
        assert transports and transports[0].home_importer is sentinel
    finally:
        for transport in transports:
            transport.block.set()
        broker.close(timeout_sec=1)


def test_a_pending_scope_whose_connection_is_gone_is_retained_and_reported(tmp_path):
    """An unreconciled scope must never look like "all clean".

    The owner-visible case: durable operations belong to a connection that is no
    longer in the store (retired, or the store was replaced). Silently reconciling
    against a host the owner revoked is worse than an outstanding claim, so the
    records stay on disk and the scope is reported with the reason.
    """

    from types import SimpleNamespace

    from ouroboros.remote_pending_operations import write_pending_operation

    write_pending_operation(
        SimpleNamespace(
            connection={"id": "connection-1"},
            project_id="project-1",
            workspace_id="workspace-1",
            remote_root="/srv/project",
            drive_root=tmp_path,
        ),
        task_id="task-1",
        request_id="request-1",
        operation_id="operation-1",
        prepared_hash="a" * 64,
        tool="write_file",
        import_kind="task_result_v1",
        import_context={},
    )
    broker = _broker(tmp_path)
    try:
        rows = broker.recover()

        assert [row["status"] for row in rows] == ["scope_retired"]
        assert rows[0]["pending_count"] == 1
        assert rows[0]["error"]["code"] == "connection_retired"
        assert rows[0]["error"]["retryable"] is False
    finally:
        broker.close(timeout_sec=1)
    # Retained, not consumed: the claim is still on disk for the owner to resolve.
    from ouroboros.remote_pending_operations import pending_operation_groups

    assert len(pending_operation_groups(tmp_path)) == 1


def test_an_injected_recovery_hook_owns_reconciliation(tmp_path):
    calls: list[object] = []
    broker = RemoteSessionBroker(
        tmp_path,
        "generation-1",
        _MANIFEST,
        transport_factory=lambda request, **_kw: _FakeTransport(request),
        pending_recovery=lambda broker: (calls.append(broker), [{"status": "ok"}])[1],
    )
    broker.start()
    try:
        assert broker.recover() == [{"status": "ok"}]
        assert calls == [broker]
    finally:
        broker.close(timeout_sec=1)


# ── panic custody ENDS when custody ends ─────────────────────────────────────


def test_panic_custody_shrinks_on_every_ordinary_exit(tmp_path):
    """A register that only grows is not a custody register, it is a log.

    `_panic_transports` and `_panic_events` were append-only lists whose single
    reset was `_detach_after_fork_child`, so a long-lived server accumulated one
    dead transport per session AND per Test/Bootstrap/directory-listing probe (each
    holding a `subprocess.Popen` and its stderr buffer) plus one `threading.Event`
    per admission — for the whole life of the generation. Counted, not read: a
    repeated admit/finish cycle must return the registers to where it found them.
    """
    transports: list[_FakeTransport] = []
    broker = _broker(tmp_path, transports=transports)
    try:
        baseline_transports = len(broker._panic_transports)
        # Admissions: every one takes an Event into custody and must give it back.
        for _ in range(4):
            _admit(broker)
            assert len(broker._panic_events) == 0, (
                "an admission that returned still holds its cancel event in custody"
            )
        # The session's own transport IS still in custody — it is live.
        assert len(broker._panic_transports) == baseline_transports + 1
        # Probe-shaped calls open and close a transport of their own.
        for _ in range(3):
            broker.test_connection({"id": "connection-1", "ssh_alias": "build"})
        assert len(broker._panic_transports) == baseline_transports + 1, (
            "a probe transport stayed in panic custody after it was closed"
        )
        assert sum(1 for t in transports if t.closed.is_set()) == 3
        # Closing the project session releases the last one.
        assert broker.close_project_session(
            {
                "kind": "ssh",
                "connection_id": "connection-1",
                "workspace_id": "workspace-1",
                "remote_root": "/srv/project",
            },
            project_id="project-1",
        )
        assert len(broker._panic_transports) == baseline_transports
    finally:
        broker.close(timeout_sec=2)


def test_panic_still_reaches_every_live_transport_and_then_releases_it(tmp_path):
    """Pruning must not cost panic its reach, and panic must not leak either."""

    transports: list[_FakeTransport] = []
    broker = _broker(tmp_path, transports=transports)
    try:
        _admit(broker)
        assert len(broker._panic_transports) == 1
        broker.panic()
        assert transports[0].panicked.is_set()
        # Terminal for this broker, so custody goes with everything else.
        assert broker._panic_transports == {}
        assert broker._panic_events == {}
    finally:
        broker.close(timeout_sec=2)


def test_a_closed_ssh_transport_stops_holding_its_dead_child(tmp_path):
    """`close()` killed the child and kept pointing at the corpse.

    `panic()` and `_reset_wire_state()` both drop `_process`; `close()` did not, so
    a closed transport still held a `subprocess.Popen` (and its buffered stderr).
    """
    from ouroboros.remote_ssh import OpenSSHExecdTransport
    from ouroboros.remote_session_admission import session_request

    broker = _broker(tmp_path, generation="generation-9")
    try:
        request = session_request(
            broker, {"id": "connection-1", "ssh_alias": "build"},
            "/srv/project", "project-1", "workspace-1",
        )
    finally:
        broker.close(timeout_sec=2)
    transport = OpenSSHExecdTransport(request)
    transport._active_tasks.add("task-1")
    transport.close()
    assert transport._process is None
    assert transport._helper_process is None
    assert transport._active_tasks == set()


# ── one shared channel, ONE lock ──────────────────────────────────────────────


def test_every_write_to_a_worker_channel_takes_the_channel_lock(tmp_path):
    """The overloaded reply wrote to a shared pipe with no lock at all.

    `_complete_pipe` wrapped its write in `WorkerChannels.send_lock(endpoint)`; the
    `broker_overloaded` reply in `_poll_worker_endpoints` called `endpoint.send`
    bare. Both run against the SAME durable per-worker endpoint from different
    threads (the broker thread and the `remote-broker-io` pool), and a
    `multiprocessing.Connection` writes the length header and the payload
    separately once the payload passes 16 KiB — so the small unlocked reply fits
    inside a large frame and desyncs that worker's channel for the rest of its life.

    Asserted by making the lock OBSERVABLE: it is held for the duration of every
    send, and the send never happens outside it.
    """
    broker = _broker(tmp_path)
    try:
        endpoint, _proxy = broker._worker_channels.mint("worker:1")
        real = broker._worker_channels.send_lock(endpoint)
        assert real is not None
        held_during_send: list[bool] = []

        class _WatchedEndpoint:
            def send(self, _payload):
                held_during_send.append(real.locked())

        broker._send_to_worker(_WatchedEndpoint(), {"ok": True})
        assert held_during_send == [], (
            "an unknown endpoint has no lock, so nothing may be written to it"
        )
        # The real endpoint's lock is taken, and taken for the write itself.
        original_send = endpoint.send
        try:
            endpoint.send = lambda payload: held_during_send.append(real.locked())
            broker._send_to_worker(endpoint, {"ok": True})
        finally:
            endpoint.send = original_send
        assert held_during_send == [True]
        assert not real.locked(), "the lock outlived the write"
    finally:
        broker.close(timeout_sec=2)


def test_the_overloaded_reply_is_the_locked_write_and_names_its_action(tmp_path):
    """The refusal the poll loop writes when the semaphore is exhausted.

    Two things at once, because they were one defect each in the same dict: it goes
    through `_send_to_worker` (so it cannot interleave with a `_complete_pipe`
    frame), and it is projected by `_error_dict` (so it carries the code's derived
    owner action instead of the hand-written `details: {}` it used to ship).
    """
    broker = _broker(tmp_path)
    try:
        endpoint, proxy = broker._worker_channels.mint("worker:1")
        sent: list[dict] = []
        original_send = endpoint.send
        try:
            endpoint.send = sent.append
            # Exhaust the in-flight semaphore so the poll loop must refuse.
            acquired = 0
            while broker._inflight.acquire(blocking=False):
                acquired += 1
            try:
                proxy._endpoint.send({"correlation_id": "corr-1", "method": "health"})
                for _ in range(200):
                    broker._poll_worker_endpoints()
                    if sent:
                        break
                    time.sleep(0.01)
            finally:
                for _ in range(acquired):
                    broker._inflight.release()
        finally:
            endpoint.send = original_send
        assert sent, "the poll loop never answered the overloaded request"
        error = sent[0]["error"]
        assert sent[0]["correlation_id"] == "corr-1"
        assert error["code"] == "broker_overloaded"
        assert error["action"] == "retry"
        assert error["details"]["action"] == "retry"
    finally:
        broker.close(timeout_sec=2)
