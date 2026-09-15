"""The public supervision beat never becomes a five-second HTTP deadline."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from ouroboros import delegate_custody, delegate_supervision
from ouroboros.gateways import claudexor as gateway_module
from ouroboros.tools import delegate
from tests._delegated_transport_shared import _delegating_ctx


@pytest.mark.serial
@pytest.mark.parametrize("initially_queued", [False, True])
def test_public_wait_reads_response_slower_than_five_seconds(tmp_path, monkeypatch, initially_queued):
    requests = []
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def answer(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            requests.append((self.command, self.path))
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            assert self.path == "/v2/handshake"
            self.answer({"compatible": True, "protocolMajor": 3,
                         "engine": {"version": "3.10.2", "sha": "fixture"}})

        def do_GET(self):
            requests.append((self.command, self.path))
            assert self.path == "/v2/runs/run-slow"
            # Deliberate network-latency reproduction: the previous five-second
            # HTTP timeout fails before this real socket sends its headers.
            first_read = requests.count(("GET", "/v2/runs/run-slow")) == 1
            if first_read:
                threading.Event().wait(5.2)
            if initially_queued and first_read:
                self.answer({"lastSeq": 0, "summary": {"state": "queued", "runDir": str(run_dir)}})
                return
            self.answer({"lastSeq": 1, "summary": {
                "state": "succeeded", "effectiveAccess": "readonly", "runDir": str(run_dir),
            }, "primaryOutput": {"kind": "answer", "text": "complete result", "truncated": False}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway_type = gateway_module.ClaudexorGateway
    ctx = _delegating_ctx(tmp_path, acting=False)
    entry = delegate._RunCustody(task_id=ctx.task_id, route_id="fixture", model="fixture",
                                project_id="fixture", project_owned=False, access="readonly")
    monkeypatch.setitem(delegate_custody._CUSTODY, "run-slow", entry)
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda: gateway_type(
        gateway_module.DaemonEndpoint("127.0.0.1", server.server_port, "fixture-token")))
    try:
        started = time.monotonic()
        result = json.loads(delegate._delegate_wait_entry(ctx, "run-slow").text)
        assert time.monotonic() - started >= 5.0
        assert result["status"] == "terminal", result
        assert result["state"] == "succeeded"
        # One handshake per supervision loop, one GET per tick (S2): the loop holds the
        # transport across quiet ticks instead of rebuilding it every 3 s.
        assert requests == [("POST", "/v2/handshake")] + [("GET", "/v2/runs/run-slow")] * (2 if initially_queued else 1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_typed_observation_timeout_keeps_same_run_inside_supervision(tmp_path, monkeypatch):
    ctx = _delegating_ctx(tmp_path, acting=False)
    calls = []
    sleeps = []
    monkeypatch.setattr(delegate_supervision.time, "sleep", sleeps.append)

    def wait_once(_ctx, run_id, _window, _seq):
        calls.append(run_id)
        if len(calls) == 1:
            return json.dumps({"status": "observation_pending", "run_id": run_id,
                               "reason": "observation_read_timeout", "waited_sec": 5.0})
        return json.dumps({"status": "terminal", "run_id": run_id, "state": "succeeded"})

    result = json.loads(delegate_supervision.supervised_wait(ctx, "run-existing", wait_once=wait_once).text)
    assert result["status"] == "terminal"
    assert calls == ["run-existing", "run-existing"]
    assert sleeps == [delegate_supervision._TICK_SEC]


@pytest.mark.parametrize("status", [200, 401, 403, 503])
def test_received_auth_refusal_cannot_be_hidden_by_body_read_timeout(status):
    class BrokenBody(httpx.SyncByteStream):
        def __iter__(self):
            raise httpx.ReadTimeout("fixture read timeout")
            yield b""  # pragma: no cover -- makes this an iterator

    gateway = gateway_module.ClaudexorGateway(gateway_module.DaemonEndpoint("127.0.0.1", 1, "fixture"))
    gateway._client.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(
        lambda _request: httpx.Response(status, stream=BrokenBody())))
    try:
        with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
            gateway.get_run("run-existing")
        assert caught.value.status_code == status
        assert caught.value.observation_timeout is (status == 200)
        assert isinstance(caught.value.__cause__, httpx.ReadTimeout)
    finally:
        gateway.close()


def test_existing_control_returns_without_starting_another_observation(tmp_path, monkeypatch):
    ctx = _delegating_ctx(tmp_path, acting=False)
    calls = []
    monkeypatch.setattr(delegate_supervision, "_control_wakes", lambda _ctx: [{"type": "deadline"}])
    result = json.loads(delegate_supervision.supervised_wait(
        ctx, "run-existing", wait_once=lambda *_args: calls.append(1)).text)
    assert result["wake_events"] == [{"type": "deadline"}]
    assert calls == []


@pytest.mark.parametrize("failure,reason", [
    (httpx.ReadTimeout("read timed out"), "observation_read_timeout"),
    (httpx.ConnectError("connection refused"), "daemon_unreachable"),
    (httpx.ConnectTimeout("connect timed out"), "daemon_unreachable"),
    (httpx.PoolTimeout("pool exhausted"), "daemon_unreachable"),
    (httpx.ReadError("connection reset"), "daemon_unreachable"),
    (httpx.WriteError("broken pipe"), "daemon_unreachable"),
    (httpx.RemoteProtocolError("server disconnected"), "daemon_unreachable"),
])
def test_read_only_retryable_transport_failures_are_typed_observation_holes(failure, reason):
    """A socket that delivered no daemon answer is the same unresolved read as a read
    timeout: ``observation_timeout`` set, so the supervising wait renews quietly
    instead of waking the model on every 3 s beat (I1: 359 refusals on 2026-09-10,
    two of them ReadError). Classified by the exception TYPE, never by prose; a
    received status still wins (test above).

    The OBSERVATION reason separates the two halves of that class: our own read
    bound expiring says nothing about the daemon, while a socket that could not be
    opened or that broke mid-exchange did not carry an answer. The transport
    ``code`` stays ``daemon_unreachable`` for every other reader of it."""

    def _raise(_request):
        raise failure

    gateway = gateway_module.ClaudexorGateway(gateway_module.DaemonEndpoint("127.0.0.1", 1, "fixture"))
    gateway._client.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(_raise))
    try:
        with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
            gateway.get_run("run-existing")
        assert caught.value.code == "daemon_unreachable"
        assert caught.value.observation_timeout is True
        assert caught.value.observation_reason == reason
        assert caught.value.status_code == 0
        assert caught.value.__cause__ is failure
    finally:
        gateway.close()


def test_a_received_refusal_carries_no_observation_reason():
    """A 4xx the daemon actually sent is not an observation hole at all."""

    def _refuse(_request):
        return httpx.Response(401, json={"code": "http_401", "message": "unauthorized"})

    gateway = gateway_module.ClaudexorGateway(gateway_module.DaemonEndpoint("127.0.0.1", 1, "fixture"))
    gateway._client.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(_refuse))
    try:
        with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
            gateway.get_run("run-existing")
        assert caught.value.observation_timeout is False
        assert caught.value.observation_reason == ""
    finally:
        gateway.close()


def test_observation_read_failure_carries_the_gateway_typed_code(tmp_path, monkeypatch):
    """The observing wait relays the transport's own per-class observation reason
    (no hardcoded ``observation_read_timeout``), so the supervision loop can tell an
    unreachable daemon apart from a daemon that was merely slow; a received refusal
    keeps its refusal shape."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    entry = delegate._RunCustody(task_id=ctx.task_id, route_id="fixture", model="fixture",
                                project_id="fixture", project_owned=False, access="readonly")
    monkeypatch.setitem(delegate_custody._CUSTODY, "run-dead", entry)
    refusals = []

    class _Dead:
        def handshake(self, **_kw):
            raise refusals[-1]

        def close(self):
            pass

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda: _Dead())
    refusals.append(gateway_module.ClaudexorUnavailable(
        "daemon_unreachable", "ConnectError: [Errno 61]", observation_timeout=True,
        observation_reason="daemon_unreachable"))
    quiet = json.loads(delegate._delegate_wait(ctx, "run-dead", observation_only=True))
    assert quiet["status"] == "observation_pending" and quiet["run_id"] == "run-dead"
    assert quiet["reason"] == "daemon_unreachable"
    # A slow but LIVE daemon is the same quiet hole with a different typed reason,
    # so the supervision loop does not raise the outage line for it.
    refusals.append(gateway_module.ClaudexorUnavailable(
        "daemon_unreachable", "ReadTimeout", observation_timeout=True,
        observation_reason="observation_read_timeout"))
    slow = json.loads(delegate._delegate_wait(ctx, "run-dead", observation_only=True))
    assert slow["status"] == "observation_pending"
    assert slow["reason"] == "observation_read_timeout"
    refusals.append(gateway_module.ClaudexorUnavailable("http_401", "unauthorized", status_code=401))
    refused = json.loads(delegate._delegate_wait(ctx, "run-dead", observation_only=True))
    assert refused["status"] == "refused" and refused["reason"] == "http_401"


def _scripted_wait_once(script):
    steps = iter(script)

    def wait_once(_ctx, run_id, _window, _seq):
        status, reason = next(steps)
        if status == "terminal":
            return json.dumps({"status": "terminal", "run_id": run_id, "state": "succeeded"})
        body = {"status": status, "run_id": run_id}
        if reason:
            body.update(reason=reason, waited_sec=0.1)
        return json.dumps(body)

    return wait_once


def test_unreachable_daemon_episode_is_one_owner_line_each_way(tmp_path, monkeypatch):
    """N consecutive ``daemon_unreachable`` ticks tell the owner ONCE; the first read the
    daemon answers again tells the owner once more and re-arms the episode. The beat
    stays one ``_TICK_SEC`` per unreachable tick (no backoff), and the dedup is the
    client's ``toast_once`` on the existing typed pair, not new host state."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    notes = []
    ctx.emit_progress_fn = lambda text, *, incident=None: notes.append((text, incident))
    sleeps = []
    monkeypatch.setattr(delegate_supervision.time, "sleep", sleeps.append)
    unreachable = ("observation_pending", "daemon_unreachable")
    result = json.loads(delegate_supervision.supervised_wait(
        ctx, "run-existing", wait_once=_scripted_wait_once([
            unreachable, unreachable, unreachable, ("no_progress", None),
            unreachable, ("terminal", None),
        ])).text)
    assert result["status"] == "terminal"
    assert [text.startswith("Delegation daemon unreachable") for text, _ in notes] == [
        True, False, True, False]
    outage, recovered = notes[0][1], notes[1][1]
    assert outage["task_incident"] == recovered["task_incident"] == "delegation_daemon_unreachable"
    assert outage["toast_tone"] == "warn" and recovered["toast_tone"] == "ok"
    assert outage["toast_once"].startswith(f"{ctx.task_id}:delegation_daemon_unreachable:")
    assert recovered["toast_once"].startswith(f"{ctx.task_id}:delegation_daemon_recovered:")
    # The SECOND episode is a second line on both client surfaces: the toast set
    # dedupes on the key and the timeline on the rendered text, so an episode
    # discriminator has to reach both or outages 2..N are dropped, not repeated.
    keys = [incident["toast_once"] for _text, incident in notes]
    assert len(set(keys)) == 4, keys
    # The text names the episode it belongs to, so the timeline's text-keyed
    # dedup sees a second episode too (two episodes inside one second would
    # still collapse there; the toast key above never does).
    assert all(text.count(":") >= 2 for text, _ in notes), notes
    # One episode's own pair shares its stamp: the recovery names the outage it closes.
    assert outage["toast_once"].rsplit(":", 1)[1] == recovered["toast_once"].rsplit(":", 1)[1]
    assert notes[2][1]["toast_once"].rsplit(":", 1)[1] == notes[3][1]["toast_once"].rsplit(":", 1)[1]
    assert sleeps == [delegate_supervision._TICK_SEC] * 4


def test_other_typed_observation_reasons_say_nothing_to_the_owner(tmp_path, monkeypatch):
    """A daemon that was merely SLOW stays silent to the owner.

    ``observation_read_timeout`` is what the gateway mints for an httpx
    ReadTimeout and what the observing wait relays (both pinned above), so this
    is the production value of a live daemon that answered after our own read
    bound, not a hand-fed string: the run settles from that same daemon and the
    owner is never told it was unreachable.
    """
    ctx = _delegating_ctx(tmp_path, acting=False)
    notes = []
    ctx.emit_progress_fn = lambda text, *, incident=None: notes.append((text, incident))
    monkeypatch.setattr(delegate_supervision.time, "sleep", lambda _sec: None)
    slow = ("observation_pending", "observation_read_timeout")
    result = json.loads(delegate_supervision.supervised_wait(
        ctx, "run-existing", wait_once=_scripted_wait_once([slow, slow, ("terminal", None)])).text)
    assert result["status"] == "terminal"
    assert notes == []


def test_a_refusal_that_never_reached_the_daemon_closes_no_outage(tmp_path, monkeypatch):
    """Recovery is contact, not merely "something other than the outage".

    A refusal raised before any byte left the host (a missing descriptor, an
    unreadable token, an engine below the floor) used to satisfy the recovery
    gate: the owner was told the daemon was reachable again at the moment it
    became less reachable, and the constant recovery key was burned for the rest
    of the session. Only a payload the daemon itself produced closes the episode.
    """
    ctx = _delegating_ctx(tmp_path, acting=False)
    notes = []
    ctx.emit_progress_fn = lambda text, *, incident=None: notes.append((text, incident))
    monkeypatch.setattr(delegate_supervision.time, "sleep", lambda _sec: None)
    unreachable = ("observation_pending", "daemon_unreachable")
    result = json.loads(delegate_supervision.supervised_wait(
        ctx, "run-existing", wait_once=_scripted_wait_once([
            unreachable, unreachable, ("refused", "daemon_not_discovered"),
        ])).text)

    assert result["status"] == "refused" and result["reason"] == "daemon_not_discovered"
    assert [text.startswith("Delegation daemon unreachable") for text, _ in notes] == [True]
    assert not any("reachable again" in text for text, _ in notes)


@pytest.mark.parametrize("control", ["deadline", "cancellation_intent"])
def test_outer_controls_still_cut_a_long_unobserved_stretch(tmp_path, monkeypatch, control):
    """Quiet renewal on a dead socket never outlives the outer bounds: the deadline and a
    cancellation intent end the unobserved stretch on the next beat (budget and the
    absolute ceiling are the loop's own rails outside this function, pinned there)."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    ctx.emit_progress_fn = lambda text, *, incident=None: None
    monkeypatch.setattr(delegate_supervision.time, "sleep", lambda _sec: None)
    ticks = []
    monkeypatch.setattr(delegate_supervision, "_control_wakes",
                        lambda _ctx: [{"type": control}] if len(ticks) >= 3 else [])

    def wait_once(_ctx, run_id, _window, _seq):
        ticks.append(1)
        return json.dumps({"status": "observation_pending", "run_id": run_id,
                           "reason": "daemon_unreachable", "waited_sec": 0.1})

    result = json.loads(delegate_supervision.supervised_wait(ctx, "run-existing", wait_once=wait_once).text)
    assert result["wake_events"] == [{"type": control}]
    assert len(ticks) == 3, "the stretch ended on the control, not on a daemon answer"
