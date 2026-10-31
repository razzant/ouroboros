"""START reuses the handshake that proved the daemon live; a silent listener is waited for within the
caller's own allowance, never respawned; no phase of the start (probe, startup wait, admission) outlives it."""
import http.server
import json
import os
import pathlib
import sys
import threading
import time

import pytest

from ouroboros import claudexor_daemon as owned
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.model_wait import execution_deadline_scope, monotonic_now

pytestmark = pytest.mark.serial  # real loopback ports, as in test_claudexor_admission_wait.py


def _point_owned_home(monkeypatch, config_dir: pathlib.Path, data_dir: pathlib.Path) -> None:
    monkeypatch.setattr(owned, "owned_config_dir", lambda: config_dir)
    monkeypatch.setattr(owned, "owned_descriptor_path", lambda: config_dir / "daemon" / "control-api.json")
    monkeypatch.setattr(owned, "owned_daemon_provisioned",
                        lambda: (config_dir / "daemon" / "control-api.json").is_file())
    import ouroboros.config as config_mod
    monkeypatch.setattr(config_mod, "DATA_DIR", data_dir)


def _daemon(token: str, *, silent_for: int, silence_sec: float = 30.0, serving_mode: str = "normal",
            answer_delay_sec: float = 0.0):
    """A daemon that accepts the socket but answers nothing for its first ``silent_for`` handshakes:
    each holds the socket for ``silence_sec`` (or until ``hold`` is set), then drops it unanswered.
    Later handshakes answer after ``answer_delay_sec`` (a busy daemon that answers late). Every
    handshake's arrival is stamped on the test's clock (``arrivals``)."""
    state = {"handshakes": 0, "silent_for": silent_for, "hold": threading.Event(), "arrivals": []}

    class _Daemon(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, {})
                return
            if self.path != "/v2/handshake":
                self._send(200, {})
                return
            state["handshakes"] += 1
            state["arrivals"].append(time.monotonic())
            if state["handshakes"] <= state["silent_for"]:
                state["hold"].wait(silence_sec)  # an accepted socket with no answer: a busy loop, not an absent daemon
                self.close_connection = True
                return
            if answer_delay_sec:
                state["hold"].wait(answer_delay_sec)
            self._send(200, {"compatible": True, "protocolMajor": 3, "servingMode": serving_mode,
                             "engine": {"version": "9.9.9", "sha": "live"}})

        def do_GET(self):
            self._send(200, {})

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Daemon)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def _manager(monkeypatch, tmp_path, server, token, *, silent_wait: float, startup_pids=None):
    data_dir = tmp_path / "data"
    config_dir = data_dir / "claudexor"
    _point_owned_home(monkeypatch, config_dir, data_dir)
    daemon_dir = config_dir / "daemon"
    daemon_dir.mkdir(parents=True)
    (daemon_dir / "token").write_text(token, encoding="utf-8")
    (daemon_dir / "control-api.json").write_text(json.dumps({
        "host": "127.0.0.1", "port": server.server_address[1], "tokenPath": str(daemon_dir / "token")}),
        encoding="utf-8")
    monkeypatch.setattr(owned, "_SPAWN_WAIT_SEC", silent_wait)
    monkeypatch.setattr(owned, "_SPAWN_POLL_SEC", 0.05)
    monkeypatch.setattr(owned, "_ADMISSION_POLL_SEC", 0.05)
    manager = owned.OwnedClaudexorDaemon()
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: manager)
    monkeypatch.setattr(owned, "verify_owned_home", lambda **_kw: "")
    monkeypatch.setattr(owned, "_write_ownership_marker", lambda: None)
    monkeypatch.setattr(manager, "_startup_pids", startup_pids or (lambda: {os.getpid()}))  # the daemon's pid is in custody
    monkeypatch.setattr(manager, "reconcile_rotation", lambda _gateway: None)

    def no_spawn(*_a, **_kw):
        raise AssertionError("a daemon that holds its socket must never be spawned twice")

    monkeypatch.setattr(manager, "_spawn", no_spawn)
    monkeypatch.setenv("OUROBOROS_CLAUDEXOR_BIN", sys.executable)  # no runtime download while attaching
    return manager


def _peer_appears_during_preparation():
    """``_startup_pids`` as the preparation path meets it: no known startup at the first probe, a
    peer's custodied startup from then on (the race ``ensure_running`` names before its re-probe)."""
    calls = []

    def pids():
        calls.append(1)
        return set() if len(calls) == 1 else {os.getpid()}

    return pids


def _preparation(monkeypatch, *, takes_sec: float):
    """Runtime preparation as a double: no pin, a command after ``takes_sec`` (not a probe: outside the clamp)."""
    from types import SimpleNamespace

    from ouroboros import claudexor_runtime

    def ensure():
        time.sleep(takes_sec)
        return [sys.executable]

    monkeypatch.setattr(claudexor_runtime, "get_runtime_manager", lambda: SimpleNamespace(pin=None, ensure=ensure))


def _recorded_probes(monkeypatch) -> list:
    """Every handshake this process starts, as ``(started_at, read_bound)``: the real bound, uncapped."""
    from ouroboros.gateways import claudexor as gateway_module
    real = gateway_module.ClaudexorGateway.handshake
    probes: list = []

    def recording(self, *, timeout_sec=None):
        probes.append((time.monotonic(), timeout_sec))
        return real(self, timeout_sec=timeout_sec)

    monkeypatch.setattr(gateway_module.ClaudexorGateway, "handshake", recording)
    return probes


def test_start_reuses_the_proving_handshake_without_a_second_one(monkeypatch, tmp_path):
    server, state = _daemon("tok", silent_for=0)
    try:
        _manager(monkeypatch, tmp_path, server, "tok", silent_wait=2.0)
        gateway = owned.ensure_owned_gateway()
        try:
            assert gateway.engine_version == "9.9.9"
            assert state["handshakes"] == 1, "the probe that proved the daemon live is the start's handshake"
        finally:
            gateway.close()
    finally:
        server.shutdown()


def test_the_first_probe_is_bounded_by_the_callers_remaining_allowance(monkeypatch, tmp_path):
    server, state = _daemon("tok", silent_for=0)
    try:
        _manager(monkeypatch, tmp_path, server, "tok", silent_wait=2.0)
        probes = _recorded_probes(monkeypatch)
        with execution_deadline_scope(monotonic_now() + 1.5):
            gateway = owned.ensure_owned_gateway()
        gateway.close()
        assert len(probes) == 1 and probes[0][1] is not None, "the proving probe reads under the caller's clock"
        assert 0 < probes[0][1] <= 1.5, "its bound is the caller's remainder, not the gateway's 60 s default"
    finally:
        server.shutdown()


def test_a_transient_silence_is_retried_within_the_callers_allowance_never_respawned(monkeypatch, tmp_path):
    server, state = _daemon("tok", silent_for=2, silence_sec=0.3)
    try:
        manager = _manager(monkeypatch, tmp_path, server, "tok", silent_wait=0.5)
        probes = _recorded_probes(monkeypatch)
        started = time.monotonic()
        with execution_deadline_scope(monotonic_now() + 20.0):  # the review slot's own window
            gateway = owned.ensure_owned_gateway()
        try:
            assert gateway.engine_version == "9.9.9"
            assert state["handshakes"] == 3 and time.monotonic() - started < 5.0
            assert manager._proc is None
            assert all(bound is not None and bound <= 20.0 for _at, bound in probes), "every probe under the allowance"
        finally:
            gateway.close()
            state["hold"].set()
    finally:
        server.shutdown()


def test_a_peer_ready_within_the_allowance_is_attached_by_the_re_probe_after_preparation(monkeypatch, tmp_path):
    server, state = _daemon("tok", silent_for=1, silence_sec=0.0)  # the first handshake is dropped at once
    try:
        manager = _manager(monkeypatch, tmp_path, server, "tok", silent_wait=0.5,
                           startup_pids=_peer_appears_during_preparation())
        _preparation(monkeypatch, takes_sec=0.0)
        probes = _recorded_probes(monkeypatch)
        started = time.monotonic()
        with execution_deadline_scope(monotonic_now() + 6.0):
            gateway = owned.ensure_owned_gateway()
        try:
            assert gateway.engine_version == "9.9.9" and state["handshakes"] == 2 and manager._proc is None
            assert time.monotonic() - started < 2.0
            assert all(bound is not None and bound <= 6.0 for _at, bound in probes), "both probes under the allowance"
        finally:
            gateway.close()
    finally:
        server.shutdown()


def test_the_re_probe_after_preparation_is_bounded_by_the_remainder_not_its_own_five_seconds(monkeypatch, tmp_path):
    server, state = _daemon("tok", silent_for=1, silence_sec=0.0, answer_delay_sec=0.8)
    try:
        _manager(monkeypatch, tmp_path, server, "tok", silent_wait=0.5, startup_pids=_peer_appears_during_preparation())
        _preparation(monkeypatch, takes_sec=0.0)
        probes = _recorded_probes(monkeypatch)
        deadline = monotonic_now() + 0.2
        with execution_deadline_scope(deadline):
            with pytest.raises(ClaudexorUnavailable) as caught:
                owned.ensure_owned_gateway()
        ended = time.monotonic()
        assert caught.value.code == "daemon_not_answering", "the typed custody of a live, late daemon"
        assert len(probes) == 2 and probes[1][1] <= 0.2, f"the re-probe reads the remainder, not 5 s: {probes}"
        assert ended - deadline < 0.5, "the start ended at the allowance, not when the daemon chose to answer"
    finally:
        state["hold"].set()
        server.shutdown()


def test_no_request_starts_after_the_allowance_even_when_preparation_outlasts_it(monkeypatch, tmp_path):
    server, state = _daemon("tok", silent_for=1, silence_sec=0.0)
    try:
        _manager(monkeypatch, tmp_path, server, "tok", silent_wait=0.5, startup_pids=_peer_appears_during_preparation())
        _preparation(monkeypatch, takes_sec=0.3)  # preparation is not a probe: it may outlast the allowance
        deadline = monotonic_now() + 0.2
        with execution_deadline_scope(deadline):
            with pytest.raises(ClaudexorUnavailable) as caught:
                owned.ensure_owned_gateway()
        assert caught.value.code == "daemon_not_answering"
        assert state["handshakes"] == 1, "no request after the allowance: the re-probe never opened"
        assert all(at < deadline for at in state["arrivals"])
    finally:
        server.shutdown()


def test_an_allowance_that_ends_unanswered_keeps_the_typed_custody_and_starts_no_probe_after_it(monkeypatch, tmp_path):
    server, state = _daemon("tok", silent_for=100)
    try:
        _manager(monkeypatch, tmp_path, server, "tok", silent_wait=0.3)
        probes = _recorded_probes(monkeypatch)
        deadline = monotonic_now() + 1.0
        with execution_deadline_scope(deadline):
            with pytest.raises(ClaudexorUnavailable) as caught:
                owned.ensure_owned_gateway()
        ended = time.monotonic()
        assert caught.value.code == "daemon_not_answering", "alive and silent, never daemon_unreachable"
        assert caught.value.observation_reason == "observation_read_timeout"
        assert probes and all(at < deadline for at, _bound in probes), "no probe starts after the allowance ends"
        assert ended - deadline < 0.5, "the last probe read only the allowance's residue, not the 60 s default"
    finally:
        state["hold"].set()
        server.shutdown()


def test_a_recovering_daemon_ends_at_the_original_deadline_with_the_typed_refusal(monkeypatch, tmp_path):
    server, state = _daemon("tok", silent_for=0, serving_mode="recovery_only")
    try:
        _manager(monkeypatch, tmp_path, server, "tok", silent_wait=2.0)
        deadline = monotonic_now() + 1.0  # shorter than the default admission window the start would open
        with execution_deadline_scope(deadline):
            with pytest.raises(ClaudexorUnavailable) as caught:
                owned.ensure_owned_gateway()
        ended = time.monotonic()
        assert caught.value.code == "daemon_recovery_only"
        assert state["handshakes"] > 1, "admission was re-asked on the proving gateway"
        assert all(at < deadline for at in state["arrivals"]), "no admission poll starts after the allowance ends"
        assert ended - deadline < 0.5, "admission ends at the caller's deadline, not at its own window's end"
    finally:
        server.shutdown()


def test_a_passive_sweep_keeps_its_short_bound_and_does_not_retry(monkeypatch, tmp_path):
    server, state = _daemon("tok", silent_for=100)
    try:
        _manager(monkeypatch, tmp_path, server, "tok", silent_wait=0.3)
        from ouroboros.gateways import claudexor as gateway_module
        monkeypatch.setattr(gateway_module, "SHORT_POLL_TIMEOUT_SEC", 0.3)
        probes = _recorded_probes(monkeypatch)
        started = time.monotonic()
        with execution_deadline_scope(monotonic_now() + 60.0):
            with pytest.raises(ClaudexorUnavailable) as caught:
                owned.ensure_owned_gateway(admission_wait_sec=0, startup_wait_sec=0.3)
        assert caught.value.code == "daemon_not_answering"
        assert time.monotonic() - started < 5.0, "a sweep skips and comes back next tick"
        assert probes and all(bound is not None and bound <= 0.3 for _at, bound in probes), "the sweep's short bound"
    finally:
        state["hold"].set()
        server.shutdown()
