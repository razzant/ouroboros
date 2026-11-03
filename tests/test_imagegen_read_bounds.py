"""A timed-out image call stays findable through the executor and recover_image().

Recovery-state permutations pause an in-process HTTP exchange and inject only the caller's
expired wait. The real worker, gateway, ledger and late adoption still run. Four socket cases
keep the physical deadline proof: an engine result, late headers, a trickling body and a URL
image download. Every call checks the real near-deadline clamp.
"""
from __future__ import annotations

import base64
import ipaddress
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from ouroboros import browser_policy, model_wait, usage_accounting, usage_store
from ouroboros import loop_tool_execution as execution
from ouroboros.gateways import claudexor as cx
from ouroboros.gateways import image_api
from ouroboros.task_pacing import effective_finalization_reserve_sec
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.usage_ledger import UsageAccountingError
from tests.test_imagegen_subscription import ImageWire
from tests.test_imagegen_tool import PNG, _summary, configured as configured, ctx as ctx
from tests.test_imagegen_wait_control import _setup

# Real ports and abandoned executor threads require the serial lane.
pytestmark = pytest.mark.serial

_REAL_SEND = httpx.HTTPTransport.handle_request  # taken before the autouse no-network patch
_CLAMP_SEC = 15  # the near-deadline executor bound every timed-out call here meets
_LATE_HEADERS_SEC = 8.0  # headers arrive inside the bound; the body then stops
_MODEL = "claudexor::codex=requested-image"
_RESULT = "/v2/model-operations/op-one/result"


def _loopback_only(transport, request):
    assert request.url.host == "127.0.0.1", request.url  # never a paid or public network call
    return _REAL_SEND(transport, request)


class _Peer(ThreadingHTTPServer):
    """A real HTTP peer: ``hold`` delays a whole reply until ``release``; ``shape`` sends
    headers late then stops the body ("late") or trickles the body ("trickle")."""

    daemon_threads = False  # server_close joins every handler: nothing outlives the test

    def __init__(self, app, hold=lambda _request: False, shape=lambda _request: None):
        self.app, self.hold, self.shape, self.release, self.requests = app, hold, shape, threading.Event(), []
        super().__init__(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self.server_address[1]}"
        self.loop = threading.Thread(target=self.serve_forever, name="image-peer")
        self.loop.start()

    def close(self):
        self.release.set()
        self.shutdown()
        self.server_close()
        self.loop.join()


class _Exchange:
    """Pause a transport exchange without waiting out another physical deadline."""

    url, loop = "http://127.0.0.1:1", None  # intercepted before any socket can be opened

    def __init__(self, app, hold=lambda _request: False):
        self.app, self.hold, self.requests = app, hold, []
        self.release, self.entered = threading.Event(), threading.Event()

    def send(self, request):
        assert request.url.host == "127.0.0.1", request.url
        self.requests.append((request.method, request.url.path))
        if self.hold(request):
            self.entered.set()
            assert self.release.wait(60), "test did not release its paused image exchange"
        return self.app(request)

    def expired_wait(self, future, _timeout):
        # This bounds a broken test's rendezvous, not a product timeout or outcome.
        assert self.entered.wait(10), "the image worker did not reach the selected exchange"
        assert not future.done(), "the exchange must still be held when the caller leaves"
        raise TimeoutError("injected expiry after the image exchange entered")

    def close(self):
        self.release.set()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def respond(self):
        peer = self.server
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        request = httpx.Request(self.command, peer.url + self.path, headers=list(self.headers.items()), content=body)
        peer.requests.append((request.method, request.url.path))
        if peer.hold(request):
            peer.release.wait(120)
        shape = peer.shape(request)
        if shape == "late":
            time.sleep(_LATE_HEADERS_SEC)
        try:
            response = peer.app(request)
        except httpx.TransportError:
            self.close_connection = True  # the effect happened; its reply is lost
            return
        content, sent = response.read(), 0
        try:
            self.send_response(response.status_code)
            self.send_header("Content-Type", response.headers.get("content-type", "application/json"))
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Connection", "close")
            self.end_headers()
            if shape == "late":
                self.wfile.write(content[:1])
                self.wfile.flush()
                sent = 1
                peer.release.wait(120)
            # A progressing body: every byte arrives well inside one read phase.
            while shape == "trickle" and sent < len(content) - 1 and not peer.release.wait(0.25):
                self.wfile.write(content[sent:sent + 1])
                self.wfile.flush()
                sent += 1
            self.wfile.write(content[sent:])
        except OSError:
            pass  # the client already left

    do_GET = do_POST = do_PUT = respond

    def log_message(self, *_args):
        pass


@pytest.fixture
def peers():
    opened, before = [], set(threading.enumerate())
    def open_peer(*args, real_socket=True):
        opened.append((_Peer if real_socket else _Exchange)(*args))
        return opened[-1]

    yield open_peer
    for peer in opened:
        peer.release.set()
    for thread in set(threading.enumerate()) - before - {peer.loop for peer in opened}:
        thread.join(60)  # every abandoned worker settles before the peers close
    for peer in opened:
        peer.close()


def _nth(method, path, nth=1):
    seen = [0]

    def hold(request):
        if (request.method, request.url.path) != (method, path):
            return False
        seen[0] += 1
        return seen[0] == nth
    return hold


def _engine(monkeypatch, peers, wire, hold=lambda _request: False, *, real_socket=True):
    peer = peers(wire, hold, real_socket=real_socket)
    port = int(peer.url.rsplit(":", 1)[1])
    gateway = lambda: cx.ClaudexorGateway(cx.DaemonEndpoint("127.0.0.1", port, "test-control-token"))  # noqa: E731
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", gateway)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", gateway)
    monkeypatch.setattr("ouroboros.llm_claudexor.read_owned_gateway", gateway)
    return peer


def _direct(monkeypatch, peers, app, hold=lambda _request: False, shape=lambda _request: None,
            *, real_socket=True):
    peer = peers(app, hold, shape) if real_socket else peers(app, hold, real_socket=False)
    monkeypatch.setattr(image_api, "resolve_image_target", lambda _model: {
        "provider": "openai", "model": "gpt-image-2", "base_url": peer.url + "/v1",
        "api_key": "fixture-secret", "headers": {}})
    monkeypatch.setattr(image_api, "env_proxies_configured", lambda: False)
    # The loopback CDN reads as public DNS; the address classification itself still runs.
    monkeypatch.setattr(browser_policy, "_resolved_addresses", lambda _host: {ipaddress.ip_address("8.8.8.8")})
    return peer


def _run(ctx, monkeypatch, peer, arguments, *, name="generate_image", during=None,
         keep_blocked=False, expire_after_entry=False):
    """One call through the real clamp, executor, registry and wait owner. ``during`` runs
    after the executor answered and before the blocked peer is released."""
    if peer.release.is_set():
        peer.release = threading.Event()  # a spent release must not pre-release this call's hold
    if isinstance(peer, _Exchange):
        peer.entered.clear()
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _transport, request: peer.send(request))
    else:
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _loopback_only)
    registry, task = _setup(ctx, monkeypatch)
    # Stay inside the integer clamp bucket even when both reads share a Windows clock tick.
    deadline = datetime.now(timezone.utc) + timedelta(seconds=effective_finalization_reserve_sec(ctx) + _CLAMP_SEC + 0.5)
    ctx.task_metadata["deadline_at"] = task["deadline_at"] = deadline.isoformat()
    timeout = execution._get_tool_timeout(registry, name, arguments)
    assert timeout == _CLAMP_SEC, timeout  # the near-deadline clamp, not the 360-second entry timeout
    call = {"id": f"{name}-call", "function": {"name": name, "arguments": json.dumps(arguments)}}
    before, started = set(threading.enumerate()), time.monotonic()
    with model_wait.task_model_wait_scope(task=task, drive_root=ctx.drive_root,
                                         event_queue=None, worker_slot_held=False) as owner:
        with monkeypatch.context() as wait:
            if expire_after_entry:
                assert isinstance(peer, _Exchange)
                wait.setattr(execution, "future_result", peer.expired_wait)
            result = execution._execute_with_timeout(registry, call, ctx.drive_root / "logs", timeout, ctx.task_id)
        elapsed = time.monotonic() - started
        assert owner.control_reason() is None, "the tool's own bound must not end the task"
        seen = during(registry) if during else None
        if not keep_blocked:
            peer.release.set()
            # What the abandoned worker does after its executor left is part of the evidence.
            for thread in set(threading.enumerate()) - before - {peer.loop}:
                thread.join(60)
    return registry, result, elapsed, seen


def _model_sees(registry, result):
    messages, trace = [], {"reasoning_notes": [], "tool_calls": []}
    errors = execution.process_tool_results([result], messages, trace, lambda _text: None, tools=registry)
    assert errors == int(result["is_error"]) and messages[-1]["role"] == "tool"
    return messages[-1]["content"]


def _ledger(ctx):
    return [(row["attempt_id"], row["revision"], row["state"]) for row in usage_accounting.read_usage_records(ctx.drive_root)]


def _discover(registry, ctx, peer, after=""):
    """recover_image() through the real executor: no request, attempt, settlement or ACK."""
    frozen = (len(peer.requests), _ledger(ctx))
    call = {"id": "list-call", "function": {"name": "recover_image",
                                            "arguments": json.dumps({"after_request_id": after} if after else {})}}
    listed = execution._execute_with_timeout(registry, call, ctx.drive_root / "logs", 30, ctx.task_id)
    assert listed["result_meta"]["tool_result_code"] == "OK", listed["result"]
    facts = _summary(_model_sees(registry, listed))
    assert (len(peer.requests), _ledger(ctx)) == frozen
    return facts


def _honest_timeout(result, elapsed):
    observed = {"code": result["result_meta"]["tool_result_code"], "elapsed": round(elapsed, 2),
                "model_text": result["result"][:240]}
    assert observed["code"] == "TOOL_TIMEOUT" and elapsed < _CLAMP_SEC + 5, json.dumps(observed)
    # The executor names no attempt and claims no outcome; recover_image() finds the attempt.
    assert not any(word in result["result"].lower() for word in ("request_id", "settled", "acknowledg", "complete"))


@pytest.mark.parametrize("stage", ["none", "create-lost", "poll", "result", "ack", "recovery"])
def test_timed_out_engine_attempt_is_listed_then_recovered_by_its_original_id(ctx, monkeypatch, peers, stage):
    wire = ImageWire(lose={"create-lost": ("POST", "/v2/model-operations"), "recovery": ("GET", _RESULT)}.get(stage))
    wire.pending = stage == "poll"
    hold = {"create-lost": _nth("POST", "/v2/model-operations"), "poll": _nth("GET", "/v2/model-operations/op-one"),
            "result": _nth("GET", _RESULT), "ack": _nth("POST", "/v2/model-operations/op-one/ack"),
            "recovery": _nth("GET", _RESULT, 2)}.get(stage, lambda _request: False)
    # The result case retains the real blocked socket / 15-second executor boundary.
    # Other stages preserve all recovery assertions without repeating that wall wait.
    peer = _engine(monkeypatch, peers, wire, hold, real_socket=stage == "result")
    acks = lambda: [call for call in wire.calls if call.url.path.endswith("/ack")]  # noqa: E731
    arguments, name = {"prompt": "fixture", "model": _MODEL}, "generate_image"
    if stage == "recovery":
        # The original call loses its result reply: unknown with its id, receipt not retained.
        _registry, first, _elapsed, _ = _run(ctx, monkeypatch, peer, arguments)
        assert first["result_meta"]["tool_result_code"] == "IMAGE_OUTCOME_UNKNOWN"
        arguments, name = {"request_id": _summary(first["result"])["request_id"]}, "recover_image"
    found = {}
    registry, result, elapsed, _ = _run(ctx, monkeypatch, peer, arguments, name=name,
                                        during=lambda registry: found.update(_discover(registry, ctx, peer)),
                                        expire_after_entry=stage not in {"none", "result"})
    [listed] = found["attempts"]
    request_id = listed["request_id"]
    assert found["has_more"] is False and (listed["model"], listed["provider"]) == (_MODEL, "claudexor")
    if stage == "none":
        assert result["result_meta"]["tool_result_code"] == "OK" and listed["ledger_state"] == "settled"
    else:
        _honest_timeout(result, elapsed)
        assert listed["ledger_state"] == {"ack": "settled", "recovery": "unresolved"}.get(stage, "dispatched")
    # The abandoned worker finished; the existing late adoption stands (lost create: unknown, no ACK).
    states = [state for _id, _revision, state in _ledger(ctx)]
    assert states == (["unresolved"] if stage == "create-lost" else ["settled"]) and len(acks()) == int(stage != "create-lost")
    seen = len(peer.requests)
    recovered = registry.execute_result("recover_image", {"request_id": request_id})
    if stage == "create-lost":
        assert recovered.code == "IMAGE_OUTCOME_UNKNOWN" and not acks()  # no operation id: nothing to observe
    else:
        facts = _summary(recovered.text)
        assert recovered.code == "OK" and facts["request_id"] == request_id
        assert all(Path(image["absolute_path"]).read_bytes() == PNG for image in facts["images"])
        # The retained receipt is observed offline; only the existing whole-result ACK is sent.
        assert set(peer.requests[seen:]) == {("POST", "/v2/model-operations/op-one/ack")}
    assert wire.upload_count == wire.generation_count == 1 and len(_ledger(ctx)) == 1


@pytest.mark.parametrize("stage", ["none", "late-headers", "progressing", "download"])
def test_timed_out_direct_attempt_is_listed_then_recovered_by_its_original_id(ctx, monkeypatch, peers, stage):
    def app(request):
        if request.url.path == "/v1/images/generations":
            row = ({"url": peer.url + "/result.png"} if stage == "download"
                   else {"b64_json": base64.b64encode(PNG).decode()})
            return httpx.Response(200, json={"data": [row], "usage": {"cost": 0.04}, "padding": "." * 600})
        assert request.url.path == "/result.png" and "authorization" not in request.headers
        return httpx.Response(200, content=PNG, headers={"Content-Type": "image/png"})
    post = ("POST", "/v1/images/generations")
    peer = _direct(monkeypatch, peers, app, hold=_nth("GET", "/result.png") if stage == "download" else lambda _r: False,
                   shape=lambda request: {"late-headers": "late", "progressing": "trickle"}.get(stage)
                   if (request.method, request.url.path) == post else None)
    found = {}

    def during(registry):
        found.update(_discover(registry, ctx, peer))
        if stage in {"late-headers", "progressing"}:
            frozen = (len(peer.requests), _ledger(ctx))
            early = registry.execute_result("recover_image", {"request_id": found["attempts"][0]["request_id"]})
            found["early"] = (early.code, early.text)
            assert (len(peer.requests), _ledger(ctx)) == frozen  # nothing resent or settled
    registry, result, elapsed, _ = _run(ctx, monkeypatch, peer, {"prompt": "fixture", "model": "openai::gpt-image-2"},
                                        during=during)
    [listed] = found["attempts"]
    assert (listed["model"], listed["provider"]) == ("openai::gpt-image-2", "openai")
    if "early" in found:
        code, text = found["early"]
        assert code == "IMAGE_OUTCOME_UNKNOWN" and "in flight" in text and "Nothing was resent" in text
        assert _summary(text)["ledger_state"] == "dispatched"
    if stage == "none":
        assert result["result_meta"]["tool_result_code"] == "OK" and listed["ledger_state"] == "settled"
    else:
        _honest_timeout(result, elapsed)
        # The URL download starts after settlement; a blocked POST is still dispatched.
        assert listed["ledger_state"] == ("settled" if stage == "download" else "dispatched")
    assert [state for _id, _revision, state in _ledger(ctx)] == ["settled"]  # the late worker adopted the response
    seen = len(peer.requests)
    recovered = registry.execute_result("recover_image", {"request_id": listed["request_id"]})
    facts = _summary(recovered.text)
    assert recovered.code == "OK" and Path(facts["images"][0]["absolute_path"]).read_bytes() == PNG
    # The complete body that arrived after the executor left was adopted and is reused:
    # no second POST; a URL row is fetched again.
    assert peer.requests[seen:] == ([("GET", "/result.png")] if stage == "download" else [])
    assert peer.requests.count(post) == 1


def test_concurrent_same_task_attempts_are_listed_separately_and_never_across_tasks(ctx, monkeypatch, peers):
    def app(_request):
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(PNG).decode()}], "padding": "." * 600})
    peer = _direct(monkeypatch, peers, app, hold=lambda request: request.method == "POST", real_socket=False)
    other = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root, task_id="other-image-task",
                        workspace_root=ctx.workspace_root, workspace_mode="external")
    usage_accounting.reserve_attempt(usage_accounting.AttemptRequest(
        model="openai::gpt-image-2", provider="openai", force_unknown_reservation=True, drive_root=ctx.drive_root,
        task_id=other.task_id, category="image_generation", source="imagegen"))
    arguments = {"prompt": "fixture", "model": "openai::gpt-image-2"}
    _registry, first, elapsed, _ = _run(ctx, monkeypatch, peer, arguments, keep_blocked=True,
                                       expire_after_entry=True)
    _honest_timeout(first, elapsed)
    found = {}
    _registry, second, elapsed, _ = _run(ctx, monkeypatch, peer, arguments,
                                         during=lambda registry: found.update(_discover(registry, ctx, peer)),
                                         expire_after_entry=True)
    _honest_timeout(second, elapsed)
    ids = [row["request_id"] for row in found["attempts"]]
    # Both in-flight attempts, newest first; neither is chosen, the other task's is absent.
    assert len(set(ids)) == 2 and found["has_more"] is False
    assert found["attempts"][0]["reserved_at"] >= found["attempts"][1]["reserved_at"]
    assert {row["ledger_state"] for row in found["attempts"]} == {"dispatched"}
    other_registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    other_registry.set_context(other)
    [foreign] = _summary(other_registry.execute_result("recover_image", {}).text)["attempts"]
    assert foreign["request_id"] not in ids and foreign["ledger_state"] == "reserved"
    # Knowing another task's request_id grants nothing, as an id or as a cursor.
    for arguments in ({"request_id": ids[0]}, {"after_request_id": ids[0]}):
        assert other_registry.execute_result("recover_image", arguments).code == "TOOL_ARG_ERROR"


def test_listing_pages_terminal_rows_by_original_id_and_an_unreadable_store_is_not_empty(ctx, monkeypatch):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    empty = registry.execute_result("recover_image", {})
    assert empty.code == "OK" and _summary(empty.text)["attempts"] == [] and _summary(empty.text)["has_more"] is False

    def reserve(task_id, category, model="openai::gpt-image-2"):
        return usage_accounting.reserve_attempt(usage_accounting.AttemptRequest(
            model=model, provider="openai", force_unknown_reservation=True, drive_root=ctx.drive_root,
            task_id=task_id, category=category, source="imagegen"))
    expected, order = {}, []
    for index in range(45):
        held = reserve(ctx.task_id, "image_generation", f"openai::model-{index % 3}")
        if index % 4:
            usage_accounting.mark_dispatched(held)
        if index % 4 == 2:
            usage_accounting.mark_unresolved(held, "fixture unknown response")
        if index % 4 == 3:
            usage_accounting.settle_attempt(held, {}, cost_usd=0.04, cost_final=True)
        if index % 4 == 0:
            usage_accounting.release_attempt(held, "fixture never dispatched")
        expected[held.attempt_id] = ("released", "dispatched", "unresolved", "settled")[index % 4]
        order.append(held.attempt_id)
    for _ in range(5):  # the same task's text calls and other tasks' images are not listed
        reserve(ctx.task_id, "task")
        reserve("other-image-task", "image_generation")
    pages, after = [], ""
    while True:
        facts = _summary(registry.execute_result("recover_image", {"after_request_id": after} if after else {}).text)
        pages.append(facts)
        if not facts["has_more"]:
            break
        after = facts["next_after_request_id"]
    listed = [row for page in pages for row in page["attempts"]]
    assert [len(page["attempts"]) for page in pages] == [20, 20, 5]
    assert [row["request_id"] for row in listed] == order[::-1]  # newest first; older pages stay reachable
    assert {row["request_id"]: row["ledger_state"] for row in listed} == expected
    assert all(row["final_at"] for row in listed if row["ledger_state"] != "dispatched")
    refused = registry.execute_result("recover_image", {"request_id": listed[0]["request_id"], "after_request_id": after})
    assert refused.code == "TOOL_ARG_ERROR"
    assert registry.execute_result("recover_image", {"after_request_id": "unknown"}).code == "TOOL_ARG_ERROR"

    usage_store.forget(ctx.drive_root)
    store = ctx.drive_root / usage_store.STORE_REL
    for path in store.parent.glob(store.name + "*"):
        path.write_bytes(b"not a usage store" * 64)
    with pytest.raises(UsageAccountingError):
        registry.execute_result("recover_image", {})  # unknown, never an empty list


def test_registry_schemas_direct_listing_before_any_new_generation(ctx):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    exposed = {schema["function"]["name"]: schema["function"] for schema in registry.schemas()}
    for name in ("generate_image", "recover_image"):
        text = exposed[name]["description"]
        assert "TOOL_TIMEOUT" in text and "recover_image() without arguments BEFORE any new generation" in text, name
    recover = exposed["recover_image"]
    assert recover["parameters"]["required"] == [] and set(recover["parameters"]["properties"]) == {
        "request_id", "after_request_id"}
    assert "current task" in recover["description"] and "grants nothing" in recover["description"]
