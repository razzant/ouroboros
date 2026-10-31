"""The public supervision beat never becomes a five-second HTTP deadline."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

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
            if self.path == "/v2/agent-capabilities":
                self.answer({"harnesses": [{"id": "fixture", "liveInput": "mid_turn"}]})
                return
            if self.path == "/v2/operations":
                # This engine lists no run event stream: the drained tail keeps its window view.
                self.answer({"operations": []})
                return
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
        # The route's live-input capability is read ONCE at entry on the same transport
        # and stamped on the wake; then one handshake per supervision loop, one catalog read
        # per engine identity (#1350: the first beat's question about the compact wait and the
        # terminal drain's about the run event route share it), one GET per tick (S2): the
        # loop holds the transport across quiet ticks.
        assert result["leaf_live_input"] == "mid_turn"
        assert requests == [("GET", "/v2/agent-capabilities"), ("POST", "/v2/handshake"), ("GET", "/v2/operations")] + [
            ("GET", "/v2/runs/run-slow")] * (2 if initially_queued else 1)
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


@pytest.mark.parametrize("method,path,family", [
    ("GET", "/v2/agent-capabilities?model=private-query", "/v2/agent-capabilities"),
    ("GET", "/v2/trust?repoRoot=%2Fprivate%2Frepo", "/v2/trust"),
    ("GET", "/v2/runs/private-id/artifacts/private-file?token=private-query", "/v2/runs/..."),
    ("POST", "/v2/threads/private-id/turns", "/v2/threads/..."),
])
def test_control_failure_names_the_request_without_private_values(monkeypatch, method, path, family):
    calls, failures = [], []

    def fail(request):
        calls.append(request)
        error = httpx.ReadTimeout(f"timed out on {request.url}; private-exception", request=request)
        failures.append(error)
        raise error

    gateway = gateway_module.ClaudexorGateway(gateway_module.DaemonEndpoint("127.0.0.1", 1, "private-token"))
    gateway.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(fail))
    clock = iter((10.0, 70.125))
    monkeypatch.setattr(gateway_module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    try:
        with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
            gateway._request(method, path, json_body={"text": "private-body"})
        error = caught.value
        assert str(error) == f"Claudexor control {method} {family} failed after 60.125s: ReadTimeout"
        assert error.code == "daemon_unreachable" and error.status_code == 0
        assert error.observation_reason == "observation_read_timeout" and not error.reported_cause
        assert error.__cause__ is failures[0] and len(calls) == 1
        assert not getattr(error, "delegated_start_unrequested", False), "the transport cannot prove a POST was unaccepted"
    finally:
        gateway.close()


@pytest.mark.parametrize("bound,read,connect", [(None, 60.0, 5.0), (0.0, 0.000001, 0.000001), (12.0, 12.0, 5.0)])
def test_control_and_live_message_keep_the_same_timeout_wire(bound, read, connect):
    calls = []

    def answer(request):
        calls.append(dict(request.extensions["timeout"]))
        return httpx.Response(200, json={"accepted": True, "outcome": "accepted"})

    gateway = gateway_module.ClaudexorGateway(gateway_module.DaemonEndpoint("127.0.0.1", 1, "fixture"))
    gateway.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(answer),
                                 timeout=httpx.Timeout(60.0, connect=5.0))
    try:
        assert gateway.get_run("run-existing", timeout_sec=bound)["accepted"] is True
        assert gateway.send_run_message("run-existing", "continue", idempotency_key="message", timeout_sec=bound)["accepted"] is True
        expected = {"connect": connect, "read": read, "write": read, "pool": read}
        assert calls == [expected, expected]
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


@pytest.mark.parametrize("raw", ["<html>proxy error</html>", '"a json string"', "[1, 2]", None])
def test_an_unreadable_tick_is_a_typed_host_failure_never_an_answer(tmp_path, monkeypatch, raw):
    """#947: a tick the host cannot read as an object is its own read failure. It publishes the
    family's refusal envelope (kept ``fault`` for the hold) with the whole text, keeps the last
    answered time, and closes no outage; its pending wake replays with the same class."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    notes = []
    ctx.emit_progress_fn = lambda text, *, incident=None: notes.append(text)
    monkeypatch.setattr(delegate_supervision.time, "sleep", lambda _sec: None)
    answered, ticks = {}, iter([("no_progress", None), ("observation_pending", "daemon_unreachable"), None])

    def wait_once(_ctx, run_id, _window, _seq):
        step = next(ticks)
        if step is None:
            answered.update(delegate_supervision._load_state(ctx, run_id))
            return raw
        return _scripted_wait_once([step])(_ctx, run_id, _window, _seq)

    result = delegate_supervision.supervised_wait(ctx, "run-existing", wait_once=wait_once)
    payload = json.loads(result.text)
    assert (result.status, result.code) == ("error", "TOOL_REPORTED_FAILURE")
    assert {key: payload[key] for key in ("status", "ok", "host_code", "reason", "run_id", "detail")} == {
        "status": "fault", "ok": False, "host_code": "TOOL_REPORTED_FAILURE", "reason": "observation_malformed",
        "run_id": "run-existing", "detail": raw if isinstance(raw, str) else repr(raw)}
    state = delegate_supervision._load_state(ctx, "run-existing")
    assert state["last_answered_observation_at"] == answered["last_answered_observation_at"]
    assert (state["observation"]["answered"], state["observation_failure"]) == (False, "observation_malformed")
    assert [text.startswith("Delegation daemon unreachable") for text in notes] == [True], notes
    replay = delegate_supervision.supervised_wait(ctx, "run-existing", wait_once=lambda *_a, **_k: pytest.fail("replayed"))
    assert replay.code == "TOOL_REPORTED_FAILURE" and json.loads(replay.text) == payload


def test_an_observed_failed_leaf_stays_an_answered_observation(tmp_path, monkeypatch):
    """The negative control: a daemon answer that the run FAILED is a successful read of a failed
    leaf, so it stays OK and closes an open outage (#947 changes only the host's unreadable tick)."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    notes = []
    ctx.emit_progress_fn = lambda text, *, incident=None: notes.append(text)
    monkeypatch.setattr(delegate_supervision.time, "sleep", lambda _sec: None)
    steps = iter([json.dumps({"status": "observation_pending", "run_id": "run-existing",
                              "reason": "daemon_unreachable", "waited_sec": 0.1}),
                  json.dumps({"status": "terminal", "run_id": "run-existing", "state": "failed"})])
    result = delegate_supervision.supervised_wait(ctx, "run-existing", wait_once=lambda *_a, **_k: next(steps))
    assert result.code == "OK" and json.loads(result.text)["state"] == "failed"
    assert any(text.startswith("Delegation daemon reachable again") for text in notes), notes


def test_an_oversized_unreadable_tick_spills_whole_and_keeps_its_class(tmp_path, monkeypatch):
    """The whole tick text survives an oversized wake through the existing exact source, while the
    fitted envelope still carries the refusal class and its reason."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tool_capabilities import tool_result_limit

    ctx = _delegating_ctx(tmp_path, acting=False)
    raw = "<html>" + "proxy page " * (tool_result_limit("delegate_wait") // 5) + "</html>"
    result = delegate_supervision.supervised_wait(ctx, "run-existing", wait_once=lambda *_a, **_k: raw)
    shown = json.loads(result.text)
    assert result.code == "TOOL_REPORTED_FAILURE" and len(result.text) <= tool_result_limit("delegate_wait")
    assert (shown["status"], shown["ok"], shown["reason"]) == ("fault", False, "observation_malformed")
    assert shown["wake_delivery"]["complete"] is False and "detail" not in shown
    full = json.loads(read_actor_source_bytes(tmp_path, ctx.task_id, shown["wake_delivery"]["source"]))
    assert full["detail"] == raw and full["ok"] is False


def test_a_recorded_legacy_fault_wake_replays_as_it_was_recorded(tmp_path):
    """Schema-1 pending wakes keep their recorded bytes and class: only ``refused`` was ever derived."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    state = delegate_supervision._load_state(ctx, "run-existing")
    legacy = {"status": "fault", "detail": "not json", "supervision_wake_id": "wake-legacy"}
    state["pending_wake"] = {"wake_id": "wake-legacy", "attempt_key": "1", "payload": legacy}
    delegate_supervision._save_state(ctx, state)
    replay = delegate_supervision.supervised_wait(ctx, "run-existing", wait_once=lambda *_a, **_k: pytest.fail("replayed"))
    assert replay.code == "OK" and json.loads(replay.text) == legacy


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


_REAL_GATEWAY = gateway_module.ClaudexorGateway  # tests below replace the module's constructor


def _scripted_gateway(handler):
    gateway = _REAL_GATEWAY(gateway_module.DaemonEndpoint("127.0.0.1", 1, "fixture"))
    gateway._client.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(handler))
    return gateway


def _busy(code="daemon_busy"):
    return httpx.Response(503, json={"code": code, "message": "daemon RPC timeout (claudexor.list)",
                                     "retryable": True})


@pytest.mark.parametrize("code", sorted(gateway_module.DAEMON_BUSY_CODES))
def test_a_busy_answer_to_a_read_is_an_observation_hole_and_a_post_keeps_its_refusal(code):
    """The engine's typed 503 that it cannot serve now is, for a GET, the same retryable
    hole as no answer (under the engine's own code, so no outage line); a POST keeps the
    refusal because a lost answer never proves a mutation was not accepted."""
    with _scripted_gateway(lambda _request: _busy(code)) as gateway:
        with pytest.raises(gateway_module.ClaudexorUnavailable) as read:
            gateway.get_run("run-existing")
        with pytest.raises(gateway_module.ClaudexorUnavailable) as write:
            gateway.handshake()
    assert (read.value.code, read.value.status_code) == (code, 503)
    assert read.value.observation_timeout is True and read.value.observation_reason == code
    assert read.value.observation_answered is True, "the engine itself answered: the transport is intact"
    assert (write.value.code, write.value.status_code) == (code, 503)
    assert write.value.observation_timeout is False and write.value.observation_reason == ""
    assert write.value.observation_answered is False


@pytest.mark.parametrize("status,code", [(503, "idempotency_status_unavailable"), (500, "daemon_busy")])
def test_other_received_answers_to_a_read_stay_refusals(status, code):
    with _scripted_gateway(lambda _request: httpx.Response(status, json={"code": code, "message": "m"})) as gateway:
        with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
            gateway.get_run("run-existing")
    assert caught.value.observation_timeout is False and caught.value.observation_reason == ""


def _supervised_busy(tmp_path, monkeypatch, *, busy_reads, busy_handshake=False, code="daemon_busy",
                     later_handshakes_busy=False, handshakes=None):
    """The real supervision loop over the real observing wait and gateway. ``later_handshakes_busy``:
    the engine was healthy when the wait began and then answers every route, the handshake
    included, with ``code`` (a recovery-only window)."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    notes, sleeps, reads = [], [], []
    ctx.emit_progress_fn = lambda text, *, incident=None: notes.append((text, incident))
    monkeypatch.setattr(delegate_supervision.time, "sleep", sleeps.append)
    entry = delegate._RunCustody(task_id=ctx.task_id, route_id="fixture", model="fixture",
                                project_id="fixture", project_owned=False, access="readonly")
    monkeypatch.setitem(delegate_custody._CUSTODY, "run-busy", entry)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    handshakes = [] if handshakes is None else handshakes

    def handler(request):
        if request.url.path == "/v2/handshake":
            handshakes.append(request.method)
            if busy_handshake or (later_handshakes_busy and len(handshakes) > 1):
                return _busy(code)
            return httpx.Response(
                200, json={"compatible": True, "protocolMajor": 3, "engine": {"version": "3.25.1", "sha": "f"}})
        if request.url.path == "/v2/agent-capabilities":
            return httpx.Response(200, json={"harnesses": [{"id": "fixture", "liveInput": "mid_turn"}]})
        if request.url.path == "/v2/operations":
            return httpx.Response(200, json={"operations": []})
        assert request.url.path == "/v2/runs/run-busy", request.url.path
        reads.append(request.method)
        if len(reads) <= busy_reads:
            return _busy(code)
        return httpx.Response(200, json={"lastSeq": 1, "summary": {
            "state": "succeeded", "effectiveAccess": "readonly", "runDir": str(run_dir),
        }, "primaryOutput": {"kind": "answer", "text": "complete result", "truncated": False}})

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda *a, **k: _scripted_gateway(handler))
    result = json.loads(delegate_supervision.supervised_wait(ctx, "run-busy").text)
    return result, sleeps, notes, reads


@pytest.mark.parametrize("code", sorted(gateway_module.DAEMON_BUSY_CODES))
def test_busy_reads_are_skipped_quietly_on_the_same_beat(tmp_path, monkeypatch, code):
    result, sleeps, notes, reads = _supervised_busy(tmp_path, monkeypatch, busy_reads=2, code=code)
    assert result["status"] == "terminal" and result["state"] == "succeeded", result
    assert reads == ["GET"] * 3, "two busy beats, then the answered read"
    assert sleeps == [delegate_supervision._TICK_SEC] * 2, "the cadence stays the three-second beat"
    assert notes == [], "the daemon itself answered: no outage line, even for its daemon_unreachable"


def test_a_recovery_only_window_keeps_the_transport_instead_of_rehandshaking_into_a_refusal(tmp_path, monkeypatch):
    """While the engine serves recovery only it refuses every route, the handshake included.
    The loop keeps the transport the engine answered on, so no rebuilt handshake turns the
    window into a refusal that wakes the model (reviewer repro, 10.10)."""
    handshakes = []
    result, sleeps, notes, reads = _supervised_busy(tmp_path, monkeypatch, busy_reads=2, code="daemon_recovery_only",
                                                    later_handshakes_busy=True, handshakes=handshakes)
    assert result["status"] == "terminal" and result["state"] == "succeeded", result
    assert handshakes == ["POST"], "one handshake: the answered transport was kept"
    assert reads == ["GET"] * 3 and sleeps == [delegate_supervision._TICK_SEC] * 2 and notes == []


def test_a_busy_answer_to_the_compact_wait_is_an_observation_hole_like_a_read():
    """The wait-many POST is a read in POST clothing: the engine's typed 503 carries the same
    observation facts a busy GET does, so the supervised beat keeps it quiet."""
    with _scripted_gateway(lambda _request: _busy("daemon_busy")) as gateway:
        with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
            gateway.wait_run_attention([{"runId": "run-existing"}], wait_ms=0, timeout_sec=1.0)
    assert (caught.value.code, caught.value.status_code) == ("daemon_busy", 503)
    assert caught.value.observation_timeout is True and caught.value.observation_reason == "daemon_busy"
    assert caught.value.observation_answered is True


@pytest.mark.parametrize("body,hole", [
    ({"code": "store_closed", "message": "closed", "retryable": True}, True),
    ({"code": "store_maintenance_failed", "message": "failed", "retryable": True}, True),
    ({"code": "idempotency_status_unavailable", "message": "not now"}, False),
], ids=["store_closed", "store_maintenance_failed", "plain-503"])
def test_a_retryable_503_on_the_compact_wait_is_an_observation_hole_and_a_plain_one_is_not(body, hole):
    with _scripted_gateway(lambda _request: httpx.Response(503, json=body)) as gateway:
        with pytest.raises(gateway_module.ClaudexorUnavailable) as caught:
            gateway.wait_run_attention([{"runId": "run-existing"}], wait_ms=0, timeout_sec=1.0)
    assert caught.value.status_code == 503 and caught.value.code == body["code"]
    assert caught.value.observation_timeout is hole and caught.value.observation_answered is hole
    assert caught.value.observation_reason == (body["code"] if hole else "")


def _repaired_engine(tmp_path, monkeypatch, *, events, wait_many=None, catalog=None, detail=None, repaired=True,
                     version="4.8.1"):
    """One observing beat of the real wait over the real gateway against a scripted engine that
    lists both observation routes and answers the contract probe as ``repaired`` says (the
    release number it proves is the same either way); ``events``/``wait_many``/``catalog``
    answer (or raise for) their route. Returns the beat's result and the owner's activity frames."""
    from ouroboros import delegate_activity

    delegate_activity.reset_process_memo()
    ctx = _delegating_ctx(tmp_path, acting=False)
    frames = []
    ctx.emit_progress_fn = lambda text, **meta: frames.append(meta)
    monkeypatch.setattr(delegate.time, "sleep", lambda _s: None)
    entry = delegate._RunCustody(task_id=ctx.task_id, route_id="fixture", model="fixture",
                                project_id="fixture", project_owned=False, access="readonly")
    monkeypatch.setitem(delegate_custody._CUSTODY, "run-x", entry)
    running = {"lastSeq": 1, "pendingInteractions": [],
               "summary": {"runId": "run-x", "state": "running", "effectiveAccess": "readonly"},
               "timeline": [{"type": "message", "title": "hi", "severity": "info"}]}
    quiet = {"wakeReason": "timeout", "attention": [],
             "summaries": [{"status": "running", "pending": False, "token": "A" * 43, "problem": None}]}

    def handler(request):
        path = request.url.path
        if path == "/v2/handshake":
            return httpx.Response(200, json={"compatible": True, "protocolMajor": 3,
                                             "engine": {"version": version, "sha": "f"}})
        if path == "/v2/agent-capabilities":
            return httpx.Response(200, json={"harnesses": [{"id": "fixture", "liveInput": "mid_turn"}]})
        if path == "/v2/operations":
            return catalog(request) if catalog else httpx.Response(200, json={"operations": [
                {"method": "POST", "path": "/v2/runs/wait-many"}, {"method": "GET", "path": "/v2/runs/:id/events"}]})
        if path == "/v2/runs/run-x/events":
            return events(request)
        if path == "/v2/runs/wait-many":
            if json.loads(request.content or b"{}").get("targets") == [{"runId": "capability-probe", "seen": None}]:
                return (httpx.Response(200, json={"wakeReason": "timeout", "attention": [], "summaries": [
                    {"status": None, "pending": None, "token": None, "problem": "not_found"}]}) if repaired
                    else httpx.Response(400, json={"code": "invalid_request",
                                                   "message": "targets[0].seen: expected string"}))
            return wait_many(request) if wait_many else httpx.Response(200, json=quiet)
        assert path == "/v2/runs/run-x", path
        return detail(request) if detail else httpx.Response(200, json=running)

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda *a, **k: _scripted_gateway(handler))
    result = json.loads(delegate._delegate_wait(ctx, "run-x", since_seq=0, observation_only=True))
    delegate_activity.reset_process_memo()
    return result, frames


def _raise(exc):
    def handler(_request):
        raise exc
    return handler


@pytest.mark.parametrize("repaired", [False, True])
@pytest.mark.parametrize("status,code", [(404, "not_found"), (503, "daemon_busy")])
def test_an_unavailable_stream_keeps_the_provisional_window_on_the_repaired_engine_too(
        tmp_path, monkeypatch, repaired, status, code):
    """A journal that cannot be opened hands the beat to the detail read and its degraded window,
    exactly as on the engine before the compact wait: never a refused wake."""
    result, frames = _repaired_engine(tmp_path, monkeypatch, repaired=repaired,
                                      events=lambda _r: httpx.Response(status, json={"code": code}))
    assert result["status"] == "progress", result
    sources = [meta["delegated_activity"]["source"] for meta in frames if "delegated_activity" in meta]
    assert sources == [{"kind": "timeline_window", "reason": f"stream_unavailable:{code}", "provisional": True, "rows": 1}]


def test_holes_on_the_repaired_engines_observation_routes_are_quiet(tmp_path, monkeypatch):
    stream = lambda _r: httpx.Response(200, headers={"content-type": "text/event-stream"}, text=": connected\n\n")  # noqa: E731
    busy, _frames = _repaired_engine(tmp_path, monkeypatch, events=stream,
                                     wait_many=lambda _r: _busy("daemon_busy"))
    assert busy["status"] == "observation_pending" and busy["reason"] == "daemon_busy", busy
    assert busy["answered"] is True, "the engine itself answered: no outage line"
    unreachable = _raise(httpx.ConnectError("refused"))
    gone, _frames = _repaired_engine(tmp_path, monkeypatch, events=stream, catalog=unreachable, detail=unreachable)
    assert gone["status"] == "observation_pending" and gone["reason"] == "daemon_unreachable", gone
    # A catalog the daemon cannot answer never refuses the beat by itself: the detail read decides,
    # and here it answers.
    decided, frames = _repaired_engine(tmp_path, monkeypatch, events=stream, catalog=unreachable)
    assert decided["status"] == "progress", decided


def test_a_busy_answer_to_a_post_still_wakes_the_model(tmp_path, monkeypatch):
    result, sleeps, notes, reads = _supervised_busy(tmp_path, monkeypatch, busy_reads=0, busy_handshake=True)
    assert result["status"] == "refused" and result["reason"] == "daemon_busy", result
    assert reads == [] and sleeps == [] and notes == []


def test_a_delivered_refused_wake_is_acknowledged_by_the_loop_and_the_next_wait_observes(tmp_path, monkeypatch):
    """A refusal relayed as a wake is a complete ERROR result. The loop still acknowledges the
    wake that result names (``acknowledge_pending_wake`` proves delivery itself), so the next
    ``delegate_wait`` observes the run instead of replaying the stale refusal."""
    from ouroboros import loop_tool_execution as execution
    from ouroboros.tools.registry import ToolRegistry

    ctx = _delegating_ctx(tmp_path, acting=False)
    monkeypatch.setattr(delegate_supervision.time, "sleep", lambda _sec: None)
    monkeypatch.setitem(delegate_custody._CUSTODY, "run-busy", delegate._RunCustody(
        task_id=ctx.task_id, route_id="fixture", model="fixture", project_id="fixture",
        project_owned=False, access="readonly"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    handshakes = []

    def handler(request):
        if request.url.path == "/v2/handshake":
            handshakes.append(request.method)
            return _busy() if len(handshakes) == 1 else httpx.Response(
                200, json={"compatible": True, "protocolMajor": 3, "engine": {"version": "3.25.1", "sha": "f"}})
        if request.url.path == "/v2/agent-capabilities":
            return httpx.Response(200, json={"harnesses": [{"id": "fixture", "liveInput": "mid_turn"}]})
        if request.url.path == "/v2/operations":
            return httpx.Response(200, json={"operations": []})
        assert request.url.path == "/v2/runs/run-busy", request.url.path
        return httpx.Response(200, json={"lastSeq": 1, "summary": {
            "state": "succeeded", "effectiveAccess": "readonly", "runDir": str(run_dir),
        }, "primaryOutput": {"kind": "answer", "text": "complete result", "truncated": False}})

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda *a, **k: _scripted_gateway(handler))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=tmp_path)
    registry._ctx = ctx
    executor = execution.StatefulToolExecutor()

    def wait(call_id):
        call = {"id": call_id, "type": "function", "function": {
            "name": "delegate_wait", "arguments": json.dumps({"run_id": "run-busy"})}}
        return execution._execute_with_timeout(registry, call, tmp_path / "logs", 30, ctx.task_id, executor)

    try:
        refused = wait("wait-refused")
        assert refused["is_error"] and json.loads(refused["result"])["reason"] == "daemon_busy", refused["result"]
        messages = []
        execution.process_tool_results([refused], messages, {"tool_calls": []}, lambda *_a, **_k: None,
                                       tools=registry)
        assert [m["tool_call_id"] for m in messages] == ["wait-refused"]
        from ouroboros import delegate_supervision as supervision
        from ouroboros.working_checkpoint import flush_content_acks

        assert supervision._load_state(ctx, "")["pending_wake"]
        flush_content_acks(ctx)  # The loop flushes only after its incorporated save.
        assert not supervision._load_state(ctx, "")["pending_wake"]
        observed = json.loads(wait("wait-next")["result"])
    finally:
        executor.shutdown()
    assert observed["status"] == "terminal" and observed["state"] == "succeeded", observed
    assert handshakes == ["POST", "POST"], "the next wait observed afresh instead of replaying"
