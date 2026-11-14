"""Real-socket witness for the owner's wait-first gate (13.11): a silent own daemon types itself through the real path.

``test_claudexor_engine_wait_first`` scripts the typed failure row in process. Here a task-owned loopback listener
plays the install's own control daemon: it first answers a real handshake and route catalog (the descriptor
generation a model call reuses, ``gateways.claudexor_metadata``), then stops answering the way a stalled daemon does:
a bound port that no longer listens (macOS drops the SYN: a real connect timeout, the 13.11 shape), a closed port,
an accepted upload connection closed without a reply, or an accepted upload left unanswered. The real
``LLMClient`` → ``chat_claudexor`` → ``_ModelInvocation.prepare`` → ``ClaudexorGateway`` path then meets that socket,
``call_llm_with_retry`` spends its same-route burst and records the failure row, and that recorded row alone must
close the configured-route walk and open the engine wait. A live daemon's received refusal on the same socket stays a
refusal the walk may follow. No daemon is started or dialed: the descriptor names only this test's port, and
``ensure_owned_gateway`` (start/probe) refuses if reached. Connect and read bounds are scaled to a fraction of a
second; the production bounds (5 s connect, 60 s read) and the stalls they meet are not reproduced here.

Prepare's window is read from the transport's own clock (``_PrepareClock``), so how long a runner takes to build a
client never decides which code a socket failure records; a window spent before the wire is its own case below.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import claudexor_daemon as owned
from ouroboros import config
from ouroboros import llm_claudexor as transport
from ouroboros import loop_llm_call
from ouroboros import usage_accounting as ua
from ouroboros.gateways import claudexor as wire
from ouroboros.llm import LLMClient
from ouroboros.llm_claudexor import daemon_not_delivered
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.loop_transport import engine_unanswered, fallback_chain_allowed, reconcile_transport_wait
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY
from tests._usage_store_testing import ledger_rows

MODEL = "claudexor::codex=exact-model"
TOKEN = "fixture-loopback-token"
SCALED_BOUND_SEC = 0.3  # stands in for the gateway's 5 s connect and 60 s read bounds
CONNECT_FAILED = {"ConnectError", "ConnectTimeout"}  # what a stopped listener answers a connect, by platform
METHOD = {"/v2/uploads": "POST", "/v2/operations": "GET"}


class _PrepareClock:
    """``llm_claudexor``'s own clock binding (``transport.time``), the one clock replaced here: the ``time`` module,
    the gateway's timing and every socket bound stay real. Prepare's call window reads it; it stands still while a
    client is built, and prepare's poll after an unanswered gateway failure ends the window, so each prepare meets the
    socket once and re-raises what the socket returned. ``spend_on_rebuild`` instead lets the client rebuilt after
    that poll take what the window has left: the deadline spent before the next wire."""

    def __init__(self):
        self.now, self.rebuild_spends, self.polled = 0.0, False, False

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds if self.rebuild_spends else seconds + SCALED_BOUND_SEC
        self.polled = True

    def spend_on_rebuild(self, monkeypatch):
        build = transport.model_gateway

        def rebuilt(factory):
            gateway = build(factory)
            if self.polled:
                self.now += SCALED_BOUND_SEC
            self.polled = False
            return gateway

        self.rebuild_spends = True
        monkeypatch.setattr(transport, "model_gateway", rebuilt)


class _LoopbackDaemon:
    """A task-owned listener on an ephemeral loopback port; ``mode`` decides how it treats each request."""

    def __init__(self):
        self.mode, self.seen, self.release = "serve", [], threading.Event()
        daemon = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self):
                daemon.seen.append((self.command, self.path.split("?", 1)[0]))
                if daemon.mode == "close":  # the upload connection drops before any reply
                    self.close_connection = True
                    return
                if daemon.mode == "silent":  # accepted and never answered: the stalled daemon
                    daemon.release.wait(30)
                    self.close_connection = True
                    return
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    self._reply(401, {"code": "unauthorized"})
                elif daemon.mode == "busy":  # a live engine's received refusal
                    self._reply(503, {"code": "daemon_busy", "message": "busy", "retryable": True})
                elif self.path == "/v2/handshake":
                    self._reply(200, {"compatible": True, "protocolMajor": config.CLAUDEXOR_PROTOCOL_MAJOR,
                                      "servingMode": "normal", "engine": {"version": "9.9.9", "sha": "fixture"}})
                elif self.path == "/v2/operations":
                    self._reply(200, {"operations": []})
                else:
                    self._reply(404, {"code": "not_found", "message": self.path})

            do_GET = do_POST = _handle

            def log_message(self, *_args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.holder = None

    def stop_listening(self, *, keep_port: bool):
        """Stop serving. A port kept bound without listening makes macOS drop each SYN (the client's connect bound
        expires, as on 13.11) where Linux answers RST. A closed port is refused at once on Linux and macOS; on
        Windows CI (14.11) its connect was still unanswered at the scaled bound, so either connect failure is the
        stopped listener's answer."""
        self.server.shutdown()
        self.server.server_close()
        self.holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if keep_port:
            self.holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.holder.bind(("127.0.0.1", self.port))

    def close(self):
        self.release.set()
        if self.holder is not None:
            self.holder.close()
        else:
            self.server.shutdown()
            self.server.server_close()


@pytest.fixture
def stand(tmp_path, monkeypatch):
    root = tmp_path / "data"
    config_dir = root / "claudexor"
    (config_dir / "daemon").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.delenv(MODEL_ACCOUNTS_KEY, raising=False)
    monkeypatch.delenv("OUROBOROS_CLAUDEXOR_ATTACH_HOME", raising=False)
    monkeypatch.setattr(config, "DATA_DIR", root)
    monkeypatch.setattr(owned, "owned_config_dir", lambda: config_dir)
    monkeypatch.setattr(owned, "owned_descriptor_path", lambda: config_dir / "daemon" / "control-api.json")
    monkeypatch.setattr(owned, "owned_daemon_provisioned", lambda: True)
    rotations, starts = [], []
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: SimpleNamespace(
        reconcile_rotation=rotations.append, _proven_engine_version=""))

    def no_daemon_start(**_bounds):
        starts.append(_bounds)
        raise RuntimeError("this witness never starts, probes or repairs a daemon")

    monkeypatch.setattr(transport, "ensure_owned_gateway", no_daemon_start)
    monkeypatch.setattr(wire, "_CONNECT_TIMEOUT_SEC", SCALED_BOUND_SEC)
    monkeypatch.setattr(wire, "_READ_TIMEOUT_SEC", SCALED_BOUND_SEC)
    # The call window; operations(timeout_sec=remaining) then bounds the catalog read itself by it.
    monkeypatch.setattr(transport, "llm_transport_timeout_sec", lambda _timeout: SCALED_BOUND_SEC)
    monkeypatch.setattr(config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.001)
    from ouroboros import pricing
    monkeypatch.setattr(pricing, "_fetch_live_rows", lambda *_: pytest.fail("Claudexor must not fetch API pricing"))
    pauses, clock = [], _PrepareClock()
    # The burst's 4 s / 8 s pauses, recorded at their own seam instead of slept (it then returns as after a sleep).
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda sec, _deadline, **_wake: pauses.append(sec) or True)
    monkeypatch.setattr(transport, "time", clock)

    daemon = _LoopbackDaemon()
    (config_dir / "daemon" / "token").write_text(TOKEN, encoding="utf-8")
    (config_dir / "daemon" / "control-api.json").write_text(json.dumps({
        "host": "127.0.0.1", "port": daemon.port, "tokenPath": str(config_dir / "daemon" / "token")}), encoding="utf-8")
    try:
        yield SimpleNamespace(root=root, config_dir=config_dir, daemon=daemon, starts=starts, pauses=pauses, clock=clock)
    finally:
        daemon.close()


def _warm_generation(stand, *, catalog: bool):
    """A real handshake (and route catalog) proves this descriptor generation, as an earlier model call did."""
    gateway = wire.ClaudexorGateway(wire.discover_daemon_at(stand.config_dir))
    try:
        gateway.handshake()
        if catalog:
            gateway.operations()
    finally:
        gateway.close()
    expected = [("POST", "/v2/handshake")] + ([("GET", "/v2/operations")] if catalog else [])
    assert stand.daemon.seen == expected
    stand.daemon.seen.clear()


def _send(stand):
    usage = {}
    with ua.usage_scope(ua.UsageScope(drive_root=stand.root, task_id="t-sock", root_task_id="t-sock")):
        answer, _ = call_llm_with_retry(LLMClient(), [{"role": "user", "content": "hello"}], MODEL, None, "high", 3,
                                        stand.root / "logs", "t-sock", 1, None, usage)
    assert answer is None
    return usage


def _ctx(usage):
    return SimpleNamespace(exact_model_route=False, route_wait_on_primary=False, task_id="t-sock",
                           _accumulated_usage=usage, task_metadata={},
                           primary_route={"model": MODEL, "role": "main", "use_local": False})


def _opens_the_engine_wait(stand, usage, tmp_path, code):
    """The recorded row and what the wait-first gate makes of it: three released $0 sends of one request, no create
    and no daemon start, a typed not-delivered ``code``, the configured-route walk closed, the engine wait open."""
    # The real burst: three sends of one request, paused 4 s then 8 s, each released unsent ($0, no operation).
    assert stand.pauses == [4.0, 8.0]
    rows = ledger_rows(stand.root)
    assert len(rows) == 3 and {row["state"] for row in rows} == {"released"}
    assert stand.starts == []  # the proven generation was reused: nothing started or probed a daemon
    assert ("POST", "/v2/model-operations") not in stand.daemon.seen  # no create ever reached the daemon
    failure = usage["_last_llm_failure"]
    assert usage["_last_llm_error_kind"] == "transport_unavailable"
    assert (failure["not_dispatched_provider"], failure["operation_id"], failure["provider_code"]) == (
        "claudexor", "", code)
    assert failure["attempt_custody_state"] == "released"
    assert daemon_not_delivered(failure) and engine_unanswered(usage)

    # That row alone closes the configured-route walk and opens the engine wait on the same model.
    ctx = _ctx(usage)
    assert fallback_chain_allowed(ctx, "transport_unavailable", None, usage) is False
    notes = []
    episode = reconcile_transport_wait(None, ctx, msg_present=False, error_kind="transport_unavailable",
                                       drive_logs=tmp_path, task_id="t-sock", model=MODEL,
                                       emit_progress=lambda text, *, incident=None: notes.append(text))
    assert episode is not None and episode.engine_unanswered is True
    assert notes and notes[0].startswith("🌐 The local Claudexor engine is not answering, so this request was not sent")
    return failure


# Each row: the socket behaviour, whether the catalog was proven too (a cold catalog fails inside prepare's own
# deadline loop instead of at the upload), the request the daemon last saw, and the HTTPX failure the gateway met.
@pytest.mark.parametrize("mode,catalog,request_path,causes", [
    ("not_listening", True, None, CONNECT_FAILED),
    ("not_listening", False, None, CONNECT_FAILED),
    ("closed_port", True, None, CONNECT_FAILED),
    ("close", True, "/v2/uploads", {"RemoteProtocolError", "ReadError"}),
    ("silent", True, "/v2/uploads", {"ReadTimeout"}),
    ("silent", False, "/v2/operations", {"ReadTimeout"}),
    ("closed_port", False, None, CONNECT_FAILED),  # appended last, so the earlier rows keep their ids
])
def test_a_silent_own_daemon_on_a_real_socket_closes_the_walk_and_opens_the_engine_wait(
        stand, tmp_path, mode, catalog, request_path, causes):
    _warm_generation(stand, catalog=catalog)
    if mode in {"not_listening", "closed_port"}:
        stand.daemon.stop_listening(keep_port=mode == "not_listening")
    else:
        stand.daemon.mode = mode

    started = time.monotonic()
    usage = _send(stand)
    elapsed = time.monotonic() - started

    if request_path is None:
        assert stand.daemon.seen == []  # the stopped listener accepted nothing
    else:
        # Each send reused the proven generation (no handshake, no warm catalog) and met the failing request once.
        assert set(stand.daemon.seen) == {(METHOD[request_path], request_path)}, stand.daemon.seen
        assert len(stand.daemon.seen) <= 3, stand.daemon.seen
    assert elapsed < 20  # scaled bounds: no real 5 s connect or 60 s read was waited out

    # The typed failure row the wait-first gate reads, produced by the real socket failure.
    print(f"gateway_error[{mode}, catalog={catalog}]: {usage['_last_llm_failure']['error']}")  # the real HTTPX failure
    failure = _opens_the_engine_wait(stand, usage, tmp_path, "daemon_unreachable")
    assert failure["error"].rsplit(" ", 1)[-1] in causes, failure["error"]


def test_a_window_spent_rebuilding_the_client_records_the_same_unanswered_daemon(stand, tmp_path, monkeypatch):
    """The interleaving behind the red ``not_listening-False`` row on Linux CI (13.11), made deterministic: each
    prepare's first wire meets the stopped listener, and the client rebuilt for its next attempt takes the rest of
    the call's window before any wire. Prepare then raises its own deadline code, ``daemon_not_answering``, in place
    of the socket's ``daemon_unreachable``: the same unanswered own daemon, with every consequence of the first."""
    _warm_generation(stand, catalog=False)
    stand.daemon.stop_listening(keep_port=True)
    stand.clock.spend_on_rebuild(monkeypatch)

    usage = _send(stand)

    assert stand.daemon.seen == []
    print(f"gateway_error[rebuild_spends]: {usage['_last_llm_failure']['error']}")
    failure = _opens_the_engine_wait(stand, usage, tmp_path, "daemon_not_answering")
    assert "Call deadline elapsed before model creation" in failure["error"], failure["error"]
    assert not any(cause in failure["error"] for cause in CONNECT_FAILED), failure["error"]


def test_a_live_daemons_received_refusal_on_the_same_socket_keeps_the_walk(stand):
    """The engine answered (status 503, its own code): it is alive, so the configured routes stay open."""
    _warm_generation(stand, catalog=True)
    stand.daemon.mode = "busy"
    usage = _send(stand)

    failure = usage["_last_llm_failure"]
    assert stand.daemon.seen and ("POST", "/v2/model-operations") not in stand.daemon.seen
    # The not-dispatched receipt carries the engine's own code (its HTTP status stays on the gateway error).
    assert failure["operation_id"] == "" and failure["provider_code"] == "daemon_busy"
    assert not daemon_not_delivered(failure) and not engine_unanswered(usage)
    kind = usage["_last_llm_error_kind"]
    assert fallback_chain_allowed(_ctx(usage), kind, None, usage) is True
