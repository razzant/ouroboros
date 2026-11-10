"""A wake's one read of its run never becomes a five-second HTTP deadline, and what it reads stays typed."""

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
from tests._cases import case_folders
from tests._run_wait_shared import held_run, wake_now


def test_public_wait_reads_response_slower_than_five_seconds(tmp_path, monkeypatch):
    """#1350: the public wait's wake reads its run through the ordinary transport bound, never a five-second HTTP
    deadline: a daemon answer slower than five seconds is still read. The route's live-input capability is read once
    on the same transport and stamped on the wake; one handshake and one GET of the run serve the publication."""
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
                self.answer({"operations": []})
                return
            assert self.path == "/v2/runs/run-slow"
            # Deliberate network-latency reproduction: the previous five-second
            # HTTP timeout fails before this real socket sends its headers.
            threading.Event().wait(5.2)
            self.answer({"lastSeq": 1, "summary": {
                "state": "succeeded", "effectiveAccess": "readonly", "runDir": str(run_dir),
            }, "primaryOutput": {"kind": "answer", "text": "complete result", "truncated": False}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway_type = gateway_module.ClaudexorGateway
    ctx = _delegating_ctx(tmp_path, acting=False)
    held_run(tmp_path, "run-slow", ctx.task_id, ended="succeeded", route="fixture", model="fixture")
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda: gateway_type(
        gateway_module.DaemonEndpoint("127.0.0.1", server.server_port, "fixture-token")))
    try:
        started = time.monotonic()
        result = json.loads(delegate._delegate_wait_entry(ctx, "run-slow").text)
        assert time.monotonic() - started >= 5.0
        assert result["status"] == "terminal", result
        assert result["state"] == "succeeded"
        assert result["leaf_live_input"] == "mid_turn"
        assert requests[:2] == [("GET", "/v2/agent-capabilities"), ("POST", "/v2/handshake")], requests
        assert requests.count(("POST", "/v2/handshake")) == 1 and requests.count(("GET", "/v2/runs/run-slow")) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_typed_observation_timeout_keeps_same_run_inside_supervision(tmp_path, monkeypatch):
    """A read timeout at the wake's one read is a quiet hole, never a refusal: the terminal the body recorded in
    custody answers the wake, with no second read and no beat."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    held_run(tmp_path, "run-existing", ctx.task_id, ended="succeeded")
    calls = []
    sleeps = []
    monkeypatch.setattr(delegate_supervision.time, "sleep", sleeps.append)

    def wait_once(_ctx, run_id, _window, _seq):
        calls.append(run_id)
        return json.dumps({"status": "observation_pending", "run_id": run_id,
                           "reason": "observation_read_timeout", "waited_sec": 5.0})

    result = json.loads(wake_now(ctx, "run-existing", wait_once=wait_once).text)
    assert result["status"] == "terminal_custody" and result["terminal_facts"], result
    assert calls == ["run-existing"] and sleeps == []


@pytest.mark.parametrize("status", [200, 401, 503])
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
    result = json.loads(wake_now(ctx, "run-existing", wait_once=lambda *_args: calls.append(1)).text)
    assert result["wake_events"] == [{"type": "deadline"}]
    assert calls == []


@pytest.mark.parametrize("failure,reason", [
    (httpx.ReadTimeout("read timed out"), "observation_read_timeout"),
    (httpx.ConnectError("connection refused"), "daemon_unreachable"),
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
    quiet = json.loads(delegate._read_run(ctx, "run-dead"))
    assert quiet["status"] == "observation_pending" and quiet["run_id"] == "run-dead"
    assert quiet["reason"] == "daemon_unreachable"
    # A slow but LIVE daemon is the same quiet hole with a different typed reason,
    # so the supervision loop does not raise the outage line for it.
    refusals.append(gateway_module.ClaudexorUnavailable(
        "daemon_unreachable", "ReadTimeout", observation_timeout=True,
        observation_reason="observation_read_timeout"))
    slow = json.loads(delegate._read_run(ctx, "run-dead"))
    assert slow["status"] == "observation_pending" and "primary" in slow["engine_pool"]
    assert slow["reason"] == "observation_read_timeout"
    refusals.append(gateway_module.ClaudexorUnavailable("http_401", "unauthorized", status_code=401))
    refused = json.loads(delegate._read_run(ctx, "run-dead"))
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


def test_a_refusal_that_never_reached_the_daemon_wakes_the_wait_with_its_reason(tmp_path, monkeypatch):
    """A refusal raised before any byte left the host (a missing descriptor, an unreadable token, an engine
    below the floor) is no quiet hole: the wake returns it typed, with its reason."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    result = json.loads(wake_now(ctx, "run-existing", wait_once=_scripted_wait_once([
        ("refused", "daemon_not_discovered")])).text)

    assert result["status"] == "refused" and result["reason"] == "daemon_not_discovered"


@pytest.mark.parametrize("raw", ["<html>proxy error</html>", '"a json string"', "[1, 2]", None])
def test_an_unreadable_tick_is_a_typed_host_failure_never_an_answer(tmp_path, monkeypatch, raw):
    """#947: a read the host cannot parse as an object is its own read failure. The wake publishes the
    family's refusal envelope (kept ``fault`` for the hold) with the whole text; the engine's reach is the
    body's line, never the wake's; its pending wake replays with the same class."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    notes = []
    ctx.emit_progress_fn = lambda text, *, incident=None: notes.append(text)
    result = wake_now(ctx, "run-existing", wait_once=lambda *_a, **_k: raw)
    payload = json.loads(result.text)
    assert (result.status, result.code) == ("error", "TOOL_REPORTED_FAILURE")
    assert {key: payload[key] for key in ("status", "ok", "host_code", "reason", "run_id", "detail")} == {
        "status": "fault", "ok": False, "host_code": "TOOL_REPORTED_FAILURE", "reason": "observation_malformed",
        "run_id": "run-existing", "detail": raw if isinstance(raw, str) else repr(raw)}
    assert notes == [], "the engine's reach is the body's line, never the wake's"
    replay = delegate_supervision.run_wait(ctx, "run-existing")
    shown = json.loads(replay.text)  # the recorded refusal, labelled as a replay (#2992)
    assert replay.code == "TOOL_REPORTED_FAILURE" and shown.pop("replay_of")["fresh_observation"] is False
    assert shown == payload


def test_an_observed_failed_leaf_stays_an_answered_observation(tmp_path, monkeypatch):
    """The negative control: a daemon answer that the run FAILED is a successful read of a failed
    leaf, so it stays OK (#947 changes only the host's unreadable read); an answer closing an outage
    is the body's (``tests/test_body_feed.py``)."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    notes = []
    ctx.emit_progress_fn = lambda text, *, incident=None: notes.append(text)
    result = wake_now(ctx, "run-existing", wait_once=lambda *_a, **_k: json.dumps(
        {"status": "terminal", "run_id": "run-existing", "state": "failed"}))
    assert result.code == "OK" and json.loads(result.text)["state"] == "failed"
    assert notes == []


def test_an_oversized_unreadable_tick_spills_whole_and_keeps_its_class(tmp_path, monkeypatch):
    """The whole read text survives an oversized wake through the existing exact source, while the
    fitted envelope still carries the refusal class and its reason."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.tool_capabilities import tool_result_limit

    ctx = _delegating_ctx(tmp_path, acting=False)
    raw = "<html>" + "proxy page " * (tool_result_limit("delegate_wait") // 5) + "</html>"
    result = wake_now(ctx, "run-existing", wait_once=lambda *_a, **_k: raw)
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
    replay = delegate_supervision.run_wait(ctx, "run-existing")  # replayed before any sleep is armed
    shown = json.loads(replay.text)
    marker = shown.pop("replay_of")  # a replay label, no invented time for a record that kept none (#2992)
    assert replay.code == "OK" and shown == legacy and marker["fresh_observation"] is False
    assert "published_at" not in marker


_REAL_GATEWAY = gateway_module.ClaudexorGateway  # tests below replace the module's constructor


def _scripted_gateway(handler):
    gateway = _REAL_GATEWAY(gateway_module.DaemonEndpoint("127.0.0.1", 1, "fixture"))
    gateway._client.close()
    gateway._client = httpx.Client(base_url="http://127.0.0.1:1", transport=httpx.MockTransport(handler))
    return gateway


def _busy(code="daemon_busy"):
    return httpx.Response(503, json={"code": code, "message": "daemon RPC timeout (claudexor.list)",
                                     "retryable": True})


def test_a_busy_answer_to_a_read_is_an_observation_hole_and_a_post_keeps_its_refusal(subtests):
    """The engine's typed 503 that it cannot serve now is, for a GET, the same retryable
    hole as no answer (under the engine's own code, so no outage line); a POST keeps the
    refusal because a lost answer never proves a mutation was not accepted."""
    for code in sorted(gateway_module.DAEMON_BUSY_CODES):
        with subtests.test(code=code):
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


def _supervised_busy(tmp_path, monkeypatch, *, busy_reads, busy_handshake=False, code="daemon_busy"):
    """The public wait's real wake over the real addressed read and gateway: the body recorded the run's terminal,
    so the wait (no continuation owner here: it waits in its slot) publishes its wake at once, reading the run once."""
    ctx = _delegating_ctx(tmp_path, acting=False)
    notes, sleeps, reads = [], [], []
    ctx.emit_progress_fn = lambda text, *, incident=None: notes.append((text, incident))
    monkeypatch.setattr(delegate_supervision.time, "sleep", sleeps.append)
    held_run(tmp_path, "run-busy", ctx.task_id, ended="succeeded", route="fixture", model="fixture")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def handler(request):
        if request.url.path == "/v2/handshake":
            if busy_handshake:
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
    result = json.loads(delegate_supervision.run_wait(ctx, "run-busy").text)
    return result, sleeps, notes, reads


def test_a_busy_read_at_the_wake_is_quiet_and_the_recorded_terminal_answers(subtests, tmp_path, monkeypatch):
    """The engine's typed busy answer to the wake's one read is a quiet hole, never a refusal and never an outage
    line: the terminal the body recorded answers the wake, with no second read and no beat."""
    case_paths = case_folders(tmp_path)
    for code in sorted(gateway_module.DAEMON_BUSY_CODES):
        tmp_path = next(case_paths)
        with subtests.test(code=code), monkeypatch.context() as monkeypatch:
            result, sleeps, notes, reads = _supervised_busy(tmp_path, monkeypatch, busy_reads=1, code=code)
            assert result["status"] == "terminal_custody" and result["terminal_facts"], result
            assert reads == ["GET"] and sleeps == [], "one read, no beat"
            assert notes == [], "the daemon itself answered: no outage line, even for its daemon_unreachable"


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
                     version="4.8.1", opt_in=False):
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
            body = json.loads(request.content or b"{}")
            if body.get("targets") == [{"runId": "capability-probe", "seen": None}]:
                if body.get("includeBlocking") is True and not opt_in:  # a strict input that predates it
                    return httpx.Response(400, json={"code": "invalid_request",
                                                     "message": "Unrecognized key(s) in object: 'includeBlocking'"})
                return (httpx.Response(200, json={"wakeReason": "timeout", "attention": [], "summaries": [
                    {"status": None, "pending": None, "token": None, "problem": "not_found"}]}) if repaired
                    else httpx.Response(400, json={"code": "invalid_request",
                                                   "message": "targets[0].seen: expected string"}))
            return wait_many(request) if wait_many else httpx.Response(200, json=quiet)
        assert path == "/v2/runs/run-x", path
        return detail(request) if detail else httpx.Response(200, json=running)

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda *a, **k: _scripted_gateway(handler))
    result = json.loads(delegate._read_run(ctx, "run-x", 0))
    delegate_activity.reset_process_memo()
    return result, frames


def _raise(exc):
    def handler(_request):
        raise exc
    return handler


@pytest.mark.parametrize(("status", "code", "repaired"), [(404, "not_found", True), (503, "daemon_busy", False)])
def test_an_unavailable_stream_hands_the_beat_to_the_detail_read_and_never_refuses_the_wake(
        tmp_path, monkeypatch, repaired, status, code):
    """A journal that cannot be opened hands the beat to the detail read, exactly as on the engine before the
    compact wait: never a refused wake. The run's words are the body's to publish once its journal opens
    (``delegate_feed``); the beat publishes no window of its own."""
    result, frames = _repaired_engine(tmp_path, monkeypatch, repaired=repaired,
                                      events=lambda _r: httpx.Response(status, json={"code": code}))
    assert result["status"] == "progress", result
    assert frames == []


def test_holes_on_the_repaired_engines_observation_routes_are_quiet(tmp_path, monkeypatch):
    stream = lambda _r: httpx.Response(200, headers={"content-type": "text/event-stream"}, text=": connected\n\n")  # noqa: E731
    unreachable = _raise(httpx.ConnectError("refused"))
    gone, _frames = _repaired_engine(tmp_path, monkeypatch, events=stream, catalog=unreachable, detail=unreachable)
    assert gone["status"] == "observation_pending" and gone["reason"] == "daemon_unreachable", gone
    # A catalog the daemon cannot answer never refuses the beat by itself: the detail read decides,
    # and here it answers.
    decided, frames = _repaired_engine(tmp_path, monkeypatch, events=stream, catalog=unreachable)
    assert decided["status"] == "progress", decided


def test_a_busy_answer_to_a_post_still_wakes_the_model(tmp_path, monkeypatch):
    """A POST the engine answers busy may have been accepted, so it stays a refusal, and the wake carries it."""
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
    held_run(tmp_path, "run-busy", ctx.task_id, ended="succeeded", route="fixture", model="fixture")
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


def test_an_opted_in_engine_marks_async_questions_end_to_end_over_http(tmp_path, monkeypatch):
    """FAKE ENGINE (scripted HTTP, no vendor or model) behind the real gateway and the wake's read
    (claudexor #628): a new async question is fetched and returned once as a running handoff, and a
    read of the same question is a quiet working state. The compact wait's opt-in and its quiet beat
    are the body's own (``test_delegate_collection_attention``)."""
    from ouroboros.gateways import claudexor_observation as obs
    from ouroboros.gateways import claudexor_run_events as run_events

    monkeypatch.setattr(obs, "_SUPPORT", {})
    monkeypatch.setattr(run_events, "_SUPPORT", {})
    reads, token = [], "A" * 43
    asked = {"lastSeq": 1, "pendingInteractions": [{
        "interactionId": "int-async", "delivery": "async", "harnessId": "codex",
        "questions": [{"id": "q", "question": "Also check the docs?", "options": [], "multi_select": False}]}],
        "summary": {"runId": "run-x", "state": "running", "effectiveAccess": "readonly",
                    "waitingOnUser": True, "asyncQuestionsOnly": True}}

    def wait_many(request):
        return httpx.Response(200, json={"wakeReason": "attention", "attention": [0], "summaries": [
            {"status": "running", "pending": True, "token": token, "problem": None, "blocking": False}]})

    def detail(request):
        reads.append(request.url.path)
        return httpx.Response(200, json=asked)

    stream = lambda _r: httpx.Response(200, headers={"content-type": "text/event-stream"}, text=": connected\n\n")  # noqa: E731
    first, _frames = _repaired_engine(tmp_path, monkeypatch, events=stream, wait_many=wait_many, detail=detail,
                                      version="9.9.9", opt_in=True)
    assert first["status"] == "waiting_on_user" and first["async_questions_only"] is True, first
    assert reads == ["/v2/runs/run-x"]
    second, _frames = _repaired_engine(tmp_path, monkeypatch, events=stream, wait_many=wait_many, detail=detail,
                                       version="9.9.9", opt_in=True)
    assert second["status"] == "progress" and second["waiting_on_user"] is True, second  # moved past the wake's cursor 0
    assert second["async_questions_only"] is True, "the question already shown stays quiet, never a second ask"
