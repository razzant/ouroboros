"""Cold-start admission: the owned daemon's recovery-only window (Phase A).

A 3.4+ claudexord serves its authenticated /v2/handshake BEFORE the admission
gate — always 200, body carrying ``servingMode: normal | recovery_only`` —
while every product route answers 503 ``{"code": "daemon_recovery_only",
"retryable": true}`` until journal recovery completes. The 2026-08-17 incident:
the spawn-wait predicate (reachable = authenticated handshake) exits inside
that window, the next product call gets the 503, and the probe resolved the
harness away over a ~273 ms miss.

The fix under test: the spawn wait KEEPS reachability as its exit predicate and
only RECORDS the explicit servingMode; the single bounded admission wait lives
in ``ensure_owned_gateway``, outside the manager lock, uniform for spawn and
attach (~150 ms handshake polls under a wall-clock deadline), expiring into the
SAME typed ``daemon_recovery_only`` refusal the 503 produces (D28: bounded wait
then typed refusal — never a silent indefinite wait, never a kill of a
recovering daemon). The provisioning rotation patch is deferred past the window
(sent into it, the daemon 503s it and the patch is silently lost) and completed
by the first caller that observes normal admission.

The daemon here is the real in-process loopback pattern from
test_claudexor_owned_daemon.py: authenticated handshake with an explicit
servingMode flipping recovery_only -> normal after N handshakes, product routes
503 typed while recovering.
"""
import http.server
import json
import pathlib
import subprocess
import sys
import threading
import time

import pytest

# Real loopback ports (ThreadingHTTPServer) and a real spawned child process:
# not parallel-safe under the xdist lane (DEVELOPMENT.md "Parallel CI and the
# serial marker") — the whole module runs in the serial pass.
pytestmark = pytest.mark.serial

from ouroboros import claudexor_daemon as owned
from ouroboros.gateways.claudexor import ClaudexorUnavailable


# ---------------------------------------------------------------------------
# Fixture pieces
# ---------------------------------------------------------------------------


def _point_owned_home(monkeypatch, config_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
    monkeypatch.setattr(owned, "owned_config_dir", lambda: config_dir)
    monkeypatch.setattr(owned, "owned_descriptor_path",
                        lambda: config_dir / "daemon" / "control-api.json")
    monkeypatch.setattr(owned, "owned_daemon_provisioned",
                        lambda: (config_dir / "daemon" / "control-api.json").is_file())
    import ouroboros.config as config_mod
    monkeypatch.setattr(config_mod, "DATA_DIR", data_dir)


def _recovery_server(token: str, *, normal_after, with_mode_field: bool = True):
    """An in-process claudexord in its recovery-only admission window.

    ``normal_after``: the daemon admits normal work once it has answered that
    many handshakes (None = it recovers forever — blocked journal partitions).
    Product routes 503 typed while recovering, exactly the 3.4.2 wire shape.
    Threading server on purpose: the admission poll loop keeps ITS gateway's
    keep-alive connection open while the rotation reconcile's settings reads
    use a second one, which would deadlock a single-threaded handler.
    """
    state = {"handshakes": 0, "normal_after": normal_after,
             "mode_field": with_mode_field}

    def _admitted() -> bool:
        after = state["normal_after"]
        return after is not None and state["handshakes"] >= after

    class _Daemon(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _product(self) -> None:
            if not _admitted():
                self._send(503, {"code": "daemon_recovery_only", "retryable": True,
                                 "message": "journal recovery in progress"})
                return
            if self.path == "/v2/agent-capabilities":
                self._send(200, {"harnesses": [{
                    "id": "codex", "enabled": True, "status": "ok",
                    "accessProfilesSupported": ["readonly", "external_sandbox_full"],
                }]})
            elif self.path == "/v2/quota":
                self._send(200, {"snapshots": [], "absences": []})
            else:
                self._send(200, {})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, {})
                return
            if self.path != "/v2/handshake":
                self._product()
                return
            state["handshakes"] += 1
            body = {"compatible": True, "protocolMajor": 3,
                    "engine": {"version": "9.9.9"}}
            if state["mode_field"]:
                body["servingMode"] = "normal" if _admitted() else "recovery_only"
            self._send(200, body)

        def do_GET(self):
            if self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, {})
                return
            self._product()

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Daemon)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def _provision_home(config_dir: pathlib.Path, port: int, token: str) -> None:
    daemon_dir = config_dir / "daemon"
    daemon_dir.mkdir(parents=True, exist_ok=True)
    (daemon_dir / "token").write_text(token, encoding="utf-8")
    (daemon_dir / "control-api.json").write_text(json.dumps({
        "host": "127.0.0.1", "port": port,
        "tokenPath": str(daemon_dir / "token"),
    }), encoding="utf-8")


def _spawning_recovery_daemon(monkeypatch, servers: list, states: list, *,
                              normal_after, with_mode_field: bool = True,
                              child_factory=None):
    """Route ensure_running's spawn chokepoint to an in-process recovery daemon.

    Acts like claudexord: mints a token, starts serving, rewrites the discovery
    descriptor — then hands back a supervised child (a real detached sleeper by
    default, so stop() group-kill semantics stay real)."""
    import ouroboros.process_custody as custody_mod

    def fake_spawn(cmd, **kwargs):
        home = pathlib.Path(kwargs["env"]["CLAUDEXOR_CONFIG_DIR"])
        token = "tok-recovering"
        server, state = _recovery_server(token, normal_after=normal_after,
                                         with_mode_field=with_mode_field)
        servers.append(server)
        states.append(state)
        _provision_home(home, server.server_address[1], token)
        if child_factory is not None:
            return child_factory()
        from ouroboros.platform_layer import subprocess_new_group_kwargs
        return subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, **subprocess_new_group_kwargs())

    monkeypatch.setattr(custody_mod, "spawn_supervised", fake_spawn)


def _fresh_manager(monkeypatch, tmp_path, *, spawn_wait: float = 3.0):
    """A non-singleton manager wired into the seams under test, fast polls."""
    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", sys.executable)
    monkeypatch.setattr(owned, "_SPAWN_WAIT_SEC", spawn_wait)
    monkeypatch.setattr(owned, "_SPAWN_POLL_SEC", 0.05)
    monkeypatch.setattr(owned, "_ADMISSION_WAIT_SEC", 2.0)
    monkeypatch.setattr(owned, "_ADMISSION_POLL_SEC", 0.02)
    manager = owned.OwnedClaudexorDaemon()
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: manager)
    return manager, config_dir


def _rotation_recorder(monkeypatch, manager, states: list):
    """Record each rotation call as "was the daemon admitted at that moment?".

    The incident's exact failure was rotation fired INTO the recovery window
    (503'd, silently lost), so the fact worth pinning is the timing, not the
    call count alone. Post-merge the rotation surface is the mainline's
    ``reconcile_rotation`` riding every ensure — the timing pin holds: it must
    run only after the admission loop admitted, never into the window.
    """
    rotations = []

    def _rotate(_gateway):
        state = states[-1] if states else None
        admitted = bool(state and state["normal_after"] is not None
                        and state["handshakes"] >= state["normal_after"])
        rotations.append(admitted)

    monkeypatch.setattr(manager, "reconcile_rotation", _rotate)
    return rotations


# ---------------------------------------------------------------------------
# A6 (i): spawn lands inside the recovery window -> bounded admission wait ->
# normal -> handshaken gateway returned, rotation fired only after admission.
# ---------------------------------------------------------------------------


def test_spawn_landing_in_recovery_window_waits_for_admission(monkeypatch, tmp_path):
    servers, states = [], []
    manager, _ = _fresh_manager(monkeypatch, tmp_path)
    # Handshake #1 is the spawn loop's reachability probe (recovering), #2 is
    # ensure_owned_gateway's own (still recovering), #3 is the admission poll.
    _spawning_recovery_daemon(monkeypatch, servers, states, normal_after=3)
    rotations = _rotation_recorder(monkeypatch, manager, states)

    try:
        gateway = owned.ensure_owned_gateway()
        try:
            # The gateway is really admitted: a PRODUCT route answers, where the
            # incident got the 503 that killed the probe.
            catalog = gateway.agent_capabilities()
            assert [row["id"] for row in catalog["harnesses"]] == ["codex"]
        finally:
            gateway.close()
        assert states[0]["handshakes"] >= 3
        # Rotation was deferred out of the window and fired exactly once, after
        # normal admission — never 503'd into the void.
        assert rotations == [True]
        # The spawned child is retained: recovering was never treated as dead.
        assert manager._proc is not None and manager._proc.poll() is None
    finally:
        manager.stop()
        for server in servers:
            server.shutdown()


# ---------------------------------------------------------------------------
# A6 (ii): indefinite recovery -> bounded expiry -> typed refusal, and the
# recovering child is NOT terminated (the dead-spawn terminate tail must fork).
# ---------------------------------------------------------------------------


def test_indefinite_recovery_expires_typed_and_never_kills_the_child(monkeypatch, tmp_path):
    servers, states = [], []
    manager, _ = _fresh_manager(monkeypatch, tmp_path)

    class _RecoveringChild:
        pid = 424242

        def __init__(self):
            self.terminated = 0

        def poll(self):
            return None

        def terminate(self):
            self.terminated += 1

    child = _RecoveringChild()
    import ouroboros.platform_layer as platform_layer
    monkeypatch.setattr(platform_layer, "process_group_id", lambda _pid: 0)
    _spawning_recovery_daemon(monkeypatch, servers, states, normal_after=None,
                              child_factory=lambda: child)
    rotations = _rotation_recorder(monkeypatch, manager, states)

    try:
        with pytest.raises(ClaudexorUnavailable) as err:
            owned.ensure_owned_gateway(admission_wait_sec=0.3)
        # The SAME typed code the daemon's own 503 carries: the dispatch table
        # already classifies it (auto -> native marker, pin -> blocked).
        assert err.value.code == "daemon_recovery_only"
        # ATTACH-IF-ALIVE: a recovering daemon is alive and ours — the expiry
        # tail forked away from the dead-spawn terminate path.
        assert child.terminated == 0, "a recovering child was killed"
        assert manager._proc is child, "the self-started handle was dropped"
        assert rotations == []

        # A5 end-to-end: once the daemon admits normal work, the next caller
        # attaches, and the DEFERRED provisioning rotation finally lands.
        states[0]["normal_after"] = 0
        gateway = owned.ensure_owned_gateway()
        gateway.close()
        assert rotations == [True]
        assert child.terminated == 0
    finally:
        for server in servers:
            server.shutdown()


# ---------------------------------------------------------------------------
# A6 (iii): attach to an already-live recovering daemon -> same bounded wait,
# same success — the admission seam is uniform for spawn and attach.
# ---------------------------------------------------------------------------


def test_attach_to_recovering_daemon_waits_then_succeeds(monkeypatch, tmp_path):
    manager, config_dir = _fresh_manager(monkeypatch, tmp_path)
    token = "tok-recovering"
    server, state = _recovery_server(token, normal_after=3)
    _provision_home(config_dir, server.server_address[1], token)

    import ouroboros.process_custody as custody_mod

    def _no_spawn(*_a, **_k):
        raise AssertionError("attach to a live daemon must never spawn")

    monkeypatch.setattr(custody_mod, "spawn_supervised", _no_spawn)
    rotations = _rotation_recorder(monkeypatch, manager, [state])

    try:
        gateway = owned.ensure_owned_gateway()
        try:
            assert gateway.agent_capabilities()["harnesses"]
        finally:
            gateway.close()
        assert state["handshakes"] >= 3
        # Attached, not spawned. Post-merge the mainline reconcile rides EVERY
        # ensure — attach included (that is its point) — and the timing pin
        # holds: it ran only after admission, never into the window.
        assert manager._proc is None
        assert rotations == [True]
    finally:
        server.shutdown()


def test_zero_admission_wait_refuses_immediately_when_recovering(monkeypatch, tmp_path):
    """``admission_wait_sec=0`` is the no-stall variant (the supervisor sweep's
    contract): a recovering daemon is one handshake and an immediate typed
    refusal — zero admission polls, no sleep."""
    manager, config_dir = _fresh_manager(monkeypatch, tmp_path)
    token = "tok-recovering"
    server, state = _recovery_server(token, normal_after=None)
    _provision_home(config_dir, server.server_address[1], token)
    _rotation_recorder(monkeypatch, manager, [state])

    try:
        started = time.monotonic()
        with pytest.raises(ClaudexorUnavailable) as err:
            owned.ensure_owned_gateway(admission_wait_sec=0)
        assert err.value.code == "daemon_recovery_only"
        assert time.monotonic() - started < 1.0
        # The reachability handshake (ensure_running) IS the admission read: its body says
        # recovery_only and a zero wait refuses at once, with no second handshake.
        assert state["handshakes"] == 1
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# A6 (v): an engine that never says servingMode (pre-3.4) keeps today's exact
# path — no admission polls; the reconcile rides the ensure as everywhere.
# ---------------------------------------------------------------------------


def test_handshake_without_serving_mode_keeps_todays_path(monkeypatch, tmp_path):
    servers, states = [], []
    manager, _ = _fresh_manager(monkeypatch, tmp_path)
    _spawning_recovery_daemon(monkeypatch, servers, states, normal_after=0,
                              with_mode_field=False)
    rotation_at = []
    monkeypatch.setattr(manager, "reconcile_rotation",
                        lambda _gateway: rotation_at.append(states[0]["handshakes"]))

    try:
        gateway = owned.ensure_owned_gateway()
        gateway.close()
        # The spawn loop's reachability handshake is the start's only handshake (no
        # servingMode field ⇒ normal admission, no admission polls), and the mainline
        # reconcile ran once on that same gateway.
        assert states[0]["handshakes"] == 1
        assert rotation_at == [1]
    finally:
        manager.stop()
        for server in servers:
            server.shutdown()


# ---------------------------------------------------------------------------
# A6 (i)/(vii): the real executor probe resolves the harness when recovery
# completes within the admission window — the incident's exact miss.
# ---------------------------------------------------------------------------


def test_probe_resolves_harness_after_recovery_completes_in_window(monkeypatch, tmp_path):
    from ouroboros.subagents import probe_subagent_executor

    servers, states = [], []
    manager, _ = _fresh_manager(monkeypatch, tmp_path)
    _spawning_recovery_daemon(monkeypatch, servers, states, normal_after=3)
    _rotation_recorder(monkeypatch, manager, states)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "codex")

    try:
        resolution = probe_subagent_executor("auto")
        assert resolution.executor == "harness"
        assert resolution.reason == "harness_ready"
        assert resolution.route is not None and resolution.route.route_id == "codex"
    finally:
        manager.stop()
        for server in servers:
            server.shutdown()


def test_probe_refuses_typed_when_recovery_outlives_the_window(monkeypatch, tmp_path):
    """The bounded window expired: D28 lands at the dispatch table — `auto`
    falls back native WITH the typed marker, a pin blocks. The recovering
    daemon is not killed and the next probe attaches to it."""
    from ouroboros.subagents import probe_subagent_executor

    servers, states = [], []
    manager, _ = _fresh_manager(monkeypatch, tmp_path)
    monkeypatch.setattr(owned, "_ADMISSION_WAIT_SEC", 0.2)
    _spawning_recovery_daemon(monkeypatch, servers, states, normal_after=None)
    _rotation_recorder(monkeypatch, manager, states)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "codex")

    try:
        auto = probe_subagent_executor("auto")
        assert (auto.executor, auto.reason) == ("native", "daemon_recovery_only")
        pinned = probe_subagent_executor("harness")
        assert (pinned.executor, pinned.reason) == ("blocked", "daemon_recovery_only")
        assert manager._proc is not None and manager._proc.poll() is None
        assert len(servers) == 1, "the second probe attached instead of respawning"
    finally:
        manager.stop()
        for server in servers:
            server.shutdown()


def test_initial_handshake_is_read_bounded_by_the_admission_window(monkeypatch):
    """A daemon that accepts the socket but WITHHOLDS the handshake reply must
    not hold a zero/small-wait caller for the transport's 60s default read
    (final-gate finding on 01466781): the first handshake is read-bounded by
    the admission window, floored at the short-poll ceiling."""
    from types import SimpleNamespace

    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros.gateways import claudexor as gw_mod

    seen: list = []

    class _Gateway:
        def __init__(self, endpoint):
            pass

        def handshake(self, *, timeout_sec=None):
            seen.append(timeout_sec)
            return {"servingMode": "normal"}

        def close(self):
            pass

    stub = SimpleNamespace(ensure_running=lambda: object(),
                           reconcile_rotation=lambda gateway: None)
    monkeypatch.setattr(daemon_mod, "get_owned_daemon", lambda: stub)
    monkeypatch.setattr(gw_mod, "ClaudexorGateway", _Gateway)
    owned.ensure_owned_gateway(admission_wait_sec=0)
    assert seen == [gw_mod.SHORT_POLL_TIMEOUT_SEC]  # bounded, never the 60s default


def test_supervisor_sweep_wiring_passes_a_zero_wait_factory(monkeypatch):
    """The server tick opts OUT of the admission wait at its own call site.

    EXECUTED, not text-matched (proton0 review, both panels): the call rides
    inside a broad try/except on the supervisor path, so dead wiring would
    silently disable orphan reconciliation on every tick. Run the real
    ``_reconcile_delegated_runs``, capture the factory it passes, invoke it,
    and require the zero-wait call to reach ``ensure_owned_gateway`` — the
    property at stake is "the sweep never stalls the supervisor thread".
    """
    import server as server_mod
    from ouroboros import claudexor_daemon as daemon_mod
    from ouroboros import delegate_custody as custody_mod
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    captured: dict = {}

    def fake_reconcile(
        drive_root, *, running_task_ids, gateway_factory, recoverable_handles, replay_pending,
    ):
        captured["factory"] = gateway_factory
        captured["recoverable_handles"] = set(recoverable_handles)
        captured["replay_pending"] = replay_pending
        return []

    ensure_calls: list = []

    def fake_ensure(*, admission_wait_sec=None):
        ensure_calls.append(admission_wait_sec)
        raise ClaudexorUnavailable("daemon_recovery_only", "still recovering")

    monkeypatch.setattr(custody_mod, "reconcile_orphaned_runs", fake_reconcile)
    monkeypatch.setattr(daemon_mod, "ensure_owned_gateway", fake_ensure)
    server_mod._reconcile_delegated_runs(set())
    assert "factory" in captured, "the sweep never reached the reconciler"
    assert captured["recoverable_handles"] == set()
    with pytest.raises(ClaudexorUnavailable):
        captured["factory"]()
    assert ensure_calls == [0]  # the sweep's posture: skip until the next tick
    assert captured["replay_pending"]() is True  # an admitted sweep may replay an ended task's pending start
