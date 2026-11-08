"""#572: local and GigaChat sends get the caller's bounds left at each actual request.

Real ``LLMClient`` preparation, accounting and SDK request objects. The GigaChat
library builds its own HTTP clients and timeouts; only its network transport is
an in-memory fake (or, once, a loopback socket). The clock is controlled.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import deadline_utils, model_wait, observability
from ouroboros import usage_accounting as ua
from ouroboros.llm import LLMClient
from ouroboros.llm_attempt import PhysicalDispatchInterrupted
from tests._usage_store_testing import ledger_rows

MESSAGES = [{"role": "user", "content": "hello"}]
GIGA = "gigachat::GigaChat-2"
CREDENTIALS = "Y3JlZGVudGlhbHM="  # base64, so the library logs no format warning
PHASES = ("connect", "read", "write", "pool")


def phases(value):
    return dict.fromkeys(PHASES, value)


def completion(content="ok"):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "index": 0, "finish_reason": "stop"}],
            "created": 1, "model": "GigaChat-2", "object": "chat.completion",
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    import ouroboros.request_wire_contract as wire

    now = [100.0]
    monkeypatch.setattr(deadline_utils, "utc_now", lambda: datetime.fromtimestamp(now[0], timezone.utc))
    monkeypatch.setattr(deadline_utils, "time", SimpleNamespace(time=lambda: now[0]))  # epoch deadlines
    monkeypatch.setattr(model_wait, "monotonic_now", lambda slot=None: now[0])
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setattr(wire, "canonical_wire_evidence_root", lambda: tmp_path)
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **k: 0.0)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="send-task", root_task_id="send-task")):
        yield SimpleNamespace(root=tmp_path, now=now)


def spend_preparation(monkeypatch, env, seconds):
    """Preparation (the durable pre-send candidate record) takes ``seconds`` of the window."""
    original = observability.persist_physical_candidate

    def persist(*args, **kwargs):
        result = original(*args, **kwargs)
        env.now[0] += seconds
        return result

    monkeypatch.setattr(observability, "persist_physical_candidate", persist)


def gigachat_network(monkeypatch, env, chat_statuses=(), latency=None, on_chat=None, content="ok"):
    """Swap only the library's network transport; record every request it starts.

    ``env.now`` is the controlled clock; ``latency`` is simulated server time per
    request kind; ``on_chat`` may spend more or fail a chat request after it was sent.
    """
    import gigachat.client as library

    seen, statuses = [], iter(chat_statuses)

    def respond(request):
        kind = "auth" if request.url.path.endswith("/oauth") else "chat"
        seen.append(SimpleNamespace(kind=kind, timeout=request.extensions["timeout"], body=request.content))
        env.now[0] += (latency or {}).get(kind, 0)
        if kind == "auth":
            return httpx.Response(200, json={"access_token": "fresh", "expires_at": 4_102_444_800_000})
        if on_chat is not None:
            on_chat(request.extensions["timeout"]["read"])
        status = next(statuses, 200)
        return httpx.Response(status, json=completion(content) if status == 200 else {"status": status, "message": "no"})

    for name in ("_get_kwargs", "_get_auth_kwargs"):
        original = getattr(library, name)
        monkeypatch.setattr(library, name, lambda settings, original=original: {
            **original(settings), "transport": httpx.MockTransport(respond)})
    return seen


def gigachat_client(monkeypatch, *, stale_token=False):
    target = {"provider": "gigachat", "resolved_model": "GigaChat-2", "usage_model": GIGA,
              "api_key": CREDENTIALS, "base_url": "https://giga.test/api/v1"}
    if stale_token:
        target["access_token"] = "stale"  # believed usable: the first chat answers 401
    client = LLMClient(api_key="unused")
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: dict(target))
    return client


def call(client, asynchronous, **kwargs):
    kwargs = {"messages": MESSAGES, "model": GIGA, "max_tokens": 64, **kwargs}
    return asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)


def local_client(monkeypatch, create):
    """A local lane whose SDK ``create`` is ``create``; nothing is measured on a server."""
    from importlib import import_module

    monkeypatch.setattr(import_module("ouroboros.local_model"), "get_manager",
                        lambda: SimpleNamespace(serving_context_evidence=lambda: {}))
    client = LLMClient(api_key="unused")
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(client, "_get_local_client", lambda: sdk)
    return client


def local_answer(content="ok"):
    from ouroboros.llm_stream import AssembledResponse

    return AssembledResponse({"id": "local", "choices": [{"message": {"role": "assistant", "content": content},
                                                          "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 1, "cost": 0}})


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_each_library_request_gets_the_window_left_when_it_starts(env, monkeypatch, asynchronous):
    """The rejected chat, the re-authentication and the re-sent bytes each start later."""
    seen = gigachat_network(monkeypatch, env, chat_statuses=[401, 200], latency={"chat": 1, "auth": 2})
    message, _usage = call(gigachat_client(monkeypatch, stale_token=True), asynchronous, caller_deadline_ts=110)

    assert message["content"] == "ok"
    assert [(item.kind, item.timeout) for item in seen] == [
        ("chat", phases(10)), ("auth", phases(9)), ("chat", phases(7))]
    assert seen[0].body == seen[2].body  # the library re-sends the same sealed bytes
    assert [row["state"] for row in ledger_rows(env.root)] == ["settled"]  # one accounted attempt


@pytest.mark.parametrize("spent_during", ["chat", "auth"])
def test_a_window_spent_before_the_401_resend_refuses_it_before_its_bytes(env, monkeypatch, spent_during):
    seen = gigachat_network(monkeypatch, env, chat_statuses=[401, 200], latency={spent_during: 11})
    with pytest.raises(PhysicalDispatchInterrupted) as refused:
        call(gigachat_client(monkeypatch, stale_token=True), False, caller_deadline_ts=110)

    assert refused.value.control_reason == "deadline"
    # The only chat bytes sent were answered 401: generation never started, nothing is resent.
    assert [item.kind for item in seen] == (["chat", "auth"] if spent_during == "auth" else ["chat"])
    assert [row["state"] for row in ledger_rows(env.root)] == ["released"]


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_without_a_bound_the_callers_timeout_and_the_library_default_are_unchanged(env, monkeypatch,
                                                                                  asynchronous):
    """The caller's timeout replaces the library's 30 s at construction, as before; async used to drop it."""
    seen = gigachat_network(monkeypatch, env)
    client = gigachat_client(monkeypatch)
    for timeout in (45.0, None):
        call(client, asynchronous, timeout=timeout)

    assert [(item.kind, item.timeout) for item in seen] == [
        ("auth", phases(45.0)), ("chat", phases(45.0)), ("auth", phases(30.0)), ("chat", phases(30.0))]
    assert [row["state"] for row in ledger_rows(env.root)] == ["settled"] * 2


def test_library_requests_outside_an_accounted_send_keep_their_timeout(env, monkeypatch):
    seen = gigachat_network(monkeypatch, env)
    library = gigachat_client(monkeypatch)._get_gigachat_client(
        {"api_key": CREDENTIALS, "base_url": "https://giga.test/api/v1"})
    with model_wait.calendar_scope(datetime.fromtimestamp(102, timezone.utc).isoformat()):
        library.get_token()
    assert [(item.kind, item.timeout) for item in seen] == [("auth", phases(30.0))]


def test_concurrent_calls_through_one_cached_client_keep_their_own_bounds(env, monkeypatch):
    """A bounded call (its 401, re-authentication and re-send) and an unbounded sibling run at once through one
    cached library client and token: each request keeps its own caller's bound and no client is added."""
    import contextvars

    import gigachat.client as library

    barrier, seen, first = threading.Barrier(2, timeout=10), [], set()

    def respond(request):
        if request.url.path.endswith("/oauth"):
            seen.append(("auth", request.extensions["timeout"]))
            env.now[0] += 2 if first else 0  # the bounded call's re-authentication takes 2 s
            return httpx.Response(200, json={"access_token": "fresh", "expires_at": 4_102_444_800_000})
        who = json.loads(request.content)["messages"][-1]["content"]
        seen.append((who, request.extensions["timeout"]))
        if who != "warm-up" and who not in first:
            first.add(who)
            barrier.wait()  # both calls' first requests are in flight on the shared client together
            if who == "bounded":
                return httpx.Response(401, json={"status": 401, "message": "token expired"})
        return httpx.Response(200, json=completion())

    for name in ("_get_kwargs", "_get_auth_kwargs"):
        original = getattr(library, name)
        monkeypatch.setattr(library, name, lambda settings, original=original: {
            **original(settings), "transport": httpx.MockTransport(respond)})
    client, answers = gigachat_client(monkeypatch), {}
    client.chat([{"role": "user", "content": "warm-up"}], GIGA, max_tokens=64)  # one OAuth token for both

    def ask(text, **bounds):
        answers[text] = client.chat([{"role": "user", "content": text}], GIGA, max_tokens=64, **bounds)[0]["content"]

    threads = [threading.Thread(target=contextvars.copy_context().run, args=(ask, "bounded"),
                                kwargs={"caller_deadline_ts": 110}),
               threading.Thread(target=contextvars.copy_context().run, args=(ask, "unbounded"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert answers == {"bounded": "ok", "unbounded": "ok"}
    assert seen[:2] == [("auth", phases(30.0)), ("warm-up", phases(30.0))]
    assert sorted(seen[2:4], key=lambda item: item[0]) == [("bounded", phases(10)), ("unbounded", phases(30.0))]
    assert seen[4:] == [("auth", phases(10)), ("bounded", phases(8))]  # re-auth and re-send see what is left
    assert len(client._gigachat_clients) == 1
    assert [row["state"] for row in ledger_rows(env.root)] == ["settled"] * 3


def test_the_pinned_library_exposes_both_http_clients_to_the_hook(env, monkeypatch):
    """The hook rides private attributes, consumer-proven on the pinned 0.2.3 only: a lock bump that moves them
    must fail here instead of silently leaving library socket timeouts un-narrowed."""
    from ouroboros.llm_gigachat import _bound_library_request

    gigachat_network(monkeypatch, env)
    library = gigachat_client(monkeypatch)._get_gigachat_client(
        {"api_key": CREDENTIALS, "base_url": "https://giga.test/api/v1"})
    assert all(_bound_library_request in http.event_hooks["request"]
               for http in (library._client, library._auth_client))


def test_a_library_client_without_its_http_clients_is_reported_and_a_stand_in_is_not(caplog):
    from ouroboros.llm_gigachat import _bind_library_requests

    future = type("GigaChat", (), {"__module__": "gigachat.client"})()
    stand_in = type("FakeGigaChat", (), {"__module__": "tests.provider_fakes"})()
    with caplog.at_level("WARNING", logger="ouroboros.llm_gigachat"):
        assert _bind_library_requests(stand_in) is stand_in
        assert not caplog.records
        assert _bind_library_requests(future) is future
    assert "exposes 0 of 2 HTTP clients" in caplog.records[-1].getMessage()


@pytest.mark.parametrize(("override", "asynchronous"), [
    pytest.param(33.0, False, id='override-sync'),
    pytest.param(None, True, id='client-default-async'),
])
def test_local_send_narrows_its_timeout_to_the_window_left_after_preparation(env, monkeypatch, asynchronous,
                                                                            override):
    """The override, else the client's own phases, never a wider default (connect stays 5 s)."""
    from importlib import import_module

    from openai import OpenAI

    timeouts = []

    def respond(request):
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={"id": "local", "object": "chat.completion", "created": 1,
                                         "model": "local-model", "choices": completion("local")["choices"],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}})

    monkeypatch.setattr(import_module("ouroboros.local_model"), "get_manager",
                        lambda: SimpleNamespace(serving_context_evidence=lambda: {}))
    spend_preparation(monkeypatch, env, 3)
    client = LLMClient(api_key="unused")
    with OpenAI(base_url="http://127.0.0.1:8799/v1", api_key="local", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(respond))) as local:
        monkeypatch.setattr(client, "_get_local_client", lambda: local)
        kwargs = dict(messages=MESSAGES, model="local", use_local=True, max_tokens=64, timeout=override,
                      caller_deadline_ts=110)
        message, _ = asyncio.run(client.chat_async(**kwargs)) if asynchronous else client.chat(**kwargs)

    assert message["content"] == "local"
    assert timeouts == [phases(7.0) if override else {**phases(7.0), "connect": 5.0}]


@pytest.mark.parametrize("lane", ["local", "gigachat"])
def test_repeated_calls_under_one_deadline_get_shrinking_windows_and_keep_prior_unknowns(
        env, monkeypatch, lane):
    """A caller's retry spends the same window; a refused late call relabels no earlier attempt."""
    sent = []

    def answer(timeout):
        sent.append(timeout)
        env.now[0] += 4 if len(sent) == 1 else 7  # the second answer ends at 111, past the bound
        if len(sent) == 1:
            raise httpx.ReadTimeout("no answer")  # dispatched: the outcome is unknown

    if lane == "local":
        client = local_client(monkeypatch, lambda **kw: answer(kw["timeout"]) or local_answer())
        kwargs = dict(model="local", use_local=True)
    else:
        gigachat_network(monkeypatch, env, on_chat=answer)
        client = gigachat_client(monkeypatch)
        kwargs = dict(model=GIGA)
    outcomes = []
    for _ in range(3):
        try:
            client.chat(MESSAGES, max_tokens=64, timeout=10, caller_deadline_ts=110, **kwargs)
            outcomes.append("answered")
        except PhysicalDispatchInterrupted as refused:
            outcomes.append(refused.control_reason)
        except httpx.ReadTimeout:
            outcomes.append("timed out")

    assert outcomes == ["timed out", "answered", "deadline"]
    assert sent == [10.0, 6.0]
    # The late call is refused before any reservation; the unknown first attempt stays unknown.
    assert [row["state"] for row in ledger_rows(env.root)] == ["unresolved", "settled"]


@pytest.mark.parametrize("lane", ["local", "gigachat"])
def test_main_projection_subtracts_the_finalization_reserve_once(env, monkeypatch, lane):
    """Main passes deadline − reserve as its calendar bound and a socket timeout from it."""
    from ouroboros.deadline_utils import main_transport_timeout_sec

    seen = gigachat_network(monkeypatch, env)
    spend_preparation(monkeypatch, env, 3)
    deadline, reserve = 120.0, 10.0
    model = GIGA if lane == "gigachat" else "local"
    kwargs = dict(caller_deadline_ts=deadline - reserve,
                  timeout=main_transport_timeout_sec(model, deadline, reserve_sec=reserve))
    assert kwargs["timeout"] == 10.0  # projected before preparation
    if lane == "gigachat":
        call(gigachat_client(monkeypatch), False, **kwargs)
        timeouts = [item.timeout["read"] for item in seen if item.kind == "chat"]
    else:
        timeouts = []
        client = local_client(monkeypatch, lambda **kw: timeouts.append(kw["timeout"]) or local_answer())
        client.chat(MESSAGES, model, use_local=True, max_tokens=64, **kwargs)
    # 120 − 10 − 103: neither the stale 10 s projection nor a second subtraction (4 s).
    assert timeouts == [7.0]


def test_local_safety_check_inside_a_tool_gets_the_tool_window_left(env, monkeypatch):
    """A real caller with only a relative timeout: the tool body's execution bound now reaches it."""
    from ouroboros import safety
    from ouroboros.config import get_safety_call_timeout_sec

    sent = []
    client = local_client(monkeypatch, lambda **kw: sent.append(kw["timeout"]) or local_answer(
        '{"status":"SAFE","reason":"ok"}'))
    ctx = SimpleNamespace(drive_root=env.root, task_id="send-task", task_metadata={})
    with model_wait.execution_deadline_scope(108):  # the tool worker's own window (loop_tool_execution)
        message, _ = safety._safety_model_call(
            client=client, ctx=ctx, tool_name="run_shell", light_model="local-model", use_local=True,
            call_type="safety_supervisor", user_prompt="ls", on_usage=lambda usage: None, account="")
    assert "SAFE" in message["content"]
    assert get_safety_call_timeout_sec() > 8.0 and sent == [8.0]  # was the whole safety transport ceiling


class _SlowChat(BaseHTTPRequestHandler):
    release = threading.Event()
    received = []

    def do_POST(self):  # noqa: N802 - http.server naming
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).received.append(self.path)
        type(self).release.wait(8)
        body = json.dumps(completion()).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass  # the client already left at its deadline

    def log_message(self, *args):
        pass


def test_a_real_loopback_socket_wait_ends_at_the_callers_deadline(tmp_path, monkeypatch):
    """The library's own 30 s socket would wait out the slow answer; the caller's bound does not."""
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    monkeypatch.setattr(ua, "estimate_cost_optional", lambda *a, **k: 0.0)
    _SlowChat.release, _SlowChat.received = threading.Event(), []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowChat)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = {"provider": "gigachat", "resolved_model": "GigaChat-2", "usage_model": GIGA, "access_token": "t",
              "base_url": f"http://127.0.0.1:{server.server_port}/api/v1", "verify_ssl_certs": False}
    client = LLMClient(api_key="unused")
    monkeypatch.setattr(client, "_resolve_remote_target", lambda model: dict(target))
    started = time.monotonic()
    try:
        with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="loopback", root_task_id="loopback")):
            with pytest.raises((httpx.TimeoutException, PhysicalDispatchInterrupted)) as ended:
                client.chat(MESSAGES, GIGA, max_tokens=64, caller_deadline_ts=time.time() + 3.0)
        elapsed = time.monotonic() - started
    finally:
        _SlowChat.release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)
    assert elapsed < 6.0  # the old path answered only after the server's 8 s
    states = [row["state"] for row in ledger_rows(tmp_path)]
    if _SlowChat.received:  # dispatched: the cut socket leaves the outcome honestly unknown
        assert isinstance(ended.value, httpx.TimeoutException) and states == ["unresolved"]
    else:  # a heavily loaded machine spent the window before the bytes
        assert states == ["released"]
