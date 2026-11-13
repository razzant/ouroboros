"""The browser-policy callback of an MCP task session: the existing policy owner, reached over loopback.

The bridge carries request facts to ``browser_policy`` with the session context's real
actor facts; it owns no predicate. The policy owner keeps ordinary bodiless and empty
requests ordinary and refuses an owner-shaped request whose body existed but could not
be read, for its Playwright consumer and this bridge alike.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import browser_policy, config, mcp_browser_guard, mcp_task_session
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.server_process import record_service_binding
from ouroboros.tools import browser
from ouroboros.tools.registry import ToolContext
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import owner
from tests.test_mcp_task_session import _private_mcp, body, call, fixture_server, rows, session_of, wire  # noqa: F401


OWNER_BODY = '{"OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS":"true"}'


@pytest.fixture
def services(tmp_path, monkeypatch):
    """A live recorded Ouroboros main endpoint and an unrelated loopback service."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    with socket.create_server(("127.0.0.1", 0)) as main, socket.create_server(("127.0.0.1", 0)) as other:
        record_service_binding(tmp_path, "main", "127.0.0.1", main.getsockname()[1], pid=os.getpid())
        yield main.getsockname()[1], other.getsockname()[1]


class _Playwright:
    """A Playwright request's body surface: ``post_data`` decodes ``post_data_buffer`` strictly."""

    def __init__(self, method, url, buffer):
        self.method, self.url, self.post_data_buffer = method, url, buffer

    @property
    def post_data(self):
        return self.post_data_buffer.decode() if self.post_data_buffer else None


def _decide(request, ctx=None, mode="advanced"):
    return browser_policy.browser_request_block_reason(request, ctx, restricted=False, runtime_mode=mode)


def test_policy_owner_keeps_empty_bodies_ordinary_and_refuses_unavailable_owner_bodies(services):
    main, other = services
    ours = f"http://127.0.0.1:{main}"
    allowed = [
        SimpleNamespace(method="GET", url=f"{ours}/api/settings", post_data=None),
        SimpleNamespace(method="POST", url=f"{ours}/api/settings", post_data=None, body_present=False),
        SimpleNamespace(method="POST", url=f"{ours}/api/decisions", post_data=None),
        SimpleNamespace(method="POST", url=f"{ours}/api/settings", post_data=""),
        SimpleNamespace(method="POST", url=f"{ours}/api/decisions", post_data="", body_present=False),
        _Playwright("POST", f"{ours}/api/settings", None),  # a genuinely bodyless Playwright POST
        _Playwright("POST", f"{ours}/api/decisions", b""),
        SimpleNamespace(method="POST", url=f"{ours}/api/tasks", post_data=None, body_present=True),  # no owner shape
        SimpleNamespace(method="POST", url=f"http://127.0.0.1:{other}/api/settings", post_data=None,
                        body_present=True),  # not Ouroboros
    ]
    for request in allowed:
        assert _decide(request) == "", vars(request)
    blocked = [
        SimpleNamespace(method="POST", url=f"{ours}/api/settings", post_data=OWNER_BODY),
        SimpleNamespace(method="POST", url=f"{ours}/api/settings", post_data=None, body_present=True),
        SimpleNamespace(method="POST", url=f"{ours}/api/decisions", post_data=None, body_present=True),
        _Playwright("POST", f"{ours}/api/settings", b"\xff\xfe binary"),  # post_data raises on decode
        _Playwright("POST", f"{ours}/api/decisions", b"\xff\xfe binary"),
        SimpleNamespace(method="POST", url=f"{ours}/api/decisions", post_data=None, post_data_buffer=b"{}"),
    ]
    for request in blocked:
        assert _decide(request) == "BROWSER_OWNER_CONTROL_BLOCKED: this operation belongs to the owner", vars(request)
        assert _decide(request, mode="cyber_pro") == ""  # Cyber keeps its policy


def test_script_rights_have_one_owner_for_every_browser_surface(tmp_path):
    readonly = ToolContext(repo_dir=tmp_path, drive_root=tmp_path,
                           task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False))
    acting = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, workspace_root=str(tmp_path),
                         task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree"))
    words = "BROWSER_LOCAL_READONLY_BLOCKED: local-readonly subagents cannot run arbitrary browser JavaScript."
    assert browser_policy.browser_script_block_reason(readonly) == words
    assert browser_policy.browser_script_block_reason(acting) == ""
    assert browser_policy.browser_script_block_reason(ToolContext(repo_dir=tmp_path, drive_root=tmp_path)) == ""
    assert browser._browser_action(readonly, "evaluate", value="1+1") == "⚠️ " + words  # unchanged sentence
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry.set_context(readonly)
    typed = registry.execute_result("browser_action", {"action": "evaluate", "value": "1+1"})
    assert (typed.status, typed.code, typed.text) == ("blocked", "ACCESS_BLOCKED", "⚠️ " + words)  # published typed


@contextmanager
def _guard(ctx):
    guard = mcp_browser_guard.BrowserGuard(ctx)
    try:
        yield guard
    finally:
        guard.close()


def _ask(guard, facts, *, raw=None):
    host, port = guard.address.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=20) as conn:
        conn.sendall(raw if raw is not None else json.dumps({"token": guard.token, **facts}).encode() + b"\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data)


def test_guard_decides_with_the_sessions_actual_actor_facts(tmp_path, services, monkeypatch):
    main, other = services
    readonly = ToolContext(repo_dir=tmp_path, drive_root=tmp_path,
                           task_constraint=TaskConstraint(mode="local_readonly_subagent", allow_enable=False))
    acting = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, workspace_root=str(tmp_path),
                         task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree"))
    owner_post = {"url": f"http://127.0.0.1:{main}/api/settings", "method": "POST", "post_data": OWNER_BODY}
    with _guard(readonly) as guard:
        assert "BROWSER_LOCAL_READONLY_BLOCKED: local-readonly" in _ask(guard, {"action": "evaluate"})["block"]
        assert "Ouroboros control-service" in _ask(guard, {"url": f"http://127.0.0.1:{main}/", "method": "GET"})["block"]
        assert _ask(guard, {"url": f"http://127.0.0.1:{other}/", "method": "GET", "post_data": None})["block"] == ""
    with _guard(acting) as guard:
        assert _ask(guard, {"action": "evaluate", "url": f"http://127.0.0.1:{other}/"}) == {"block": ""}
        assert "Ouroboros control-service" in _ask(guard, owner_post)["block"]
    root = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    with _guard(root) as guard:
        assert "BROWSER_OWNER_CONTROL_BLOCKED" in _ask(guard, owner_post)["block"]
        monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "cyber_pro")  # the boot-pinned mode, read per request
        assert _ask(guard, owner_post) == {"block": ""}
        assert _ask(guard, {"url": "http://169.254.169.254/latest/meta-data", "method": "GET"}) == {"block": ""}


def test_guard_refuses_what_it_cannot_judge_and_stops_with_its_session(tmp_path, monkeypatch):
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    monkeypatch.setattr(mcp_browser_guard, "MCP_BROWSER_GUARD_LINE_BYTES", 1024)
    with _guard(ctx) as guard:
        assert _ask(guard, {"armed": True}) == {"armed": True} and guard.armed
        wrong = _ask(guard, {"armed": True, "token": "0" * 64})
        assert wrong["armed"] is False and "guard token" in wrong["block"]
        for facts in ({"url": 5, "method": "GET"}, {"action": "screenshot"},
                      {"url": "http://example.test/", "method": "POST", "post_data": 7}):
            assert _ask(guard, facts)["block"].startswith("BROWSER_POLICY_UNAVAILABLE"), facts
        oversized = json.dumps({"token": guard.token, "url": "http://example.test/", "method": "POST",
                                "post_data": "x" * 4096}).encode() + b"\n"
        assert "unreadable request facts (ValueError)" in _ask(guard, {}, raw=oversized)["block"]
        assert "unreadable request facts" in _ask(guard, {}, raw=b"POST / HTTP/1.1\r\n\r\n")["block"]
        guard.closing()
        assert _ask(guard, {"url": "http://example.test/", "method": "GET"})["block"].startswith("BROWSER_SESSION_CLOSED")
        assert _ask(guard, {"armed": True})["armed"] is False
        host, port = guard.address.rsplit(":", 1)
    with pytest.raises(OSError):
        socket.create_connection((host, int(port)), timeout=2).close()


def test_guard_answers_a_page_load_burst_on_its_own_threads(tmp_path, monkeypatch):
    seen, real = [], browser_policy.browser_request_block_reason

    def slow(request, ctx, **facts):
        seen.append(threading.current_thread().name)
        time.sleep(0.3)  # a DNS-bound decision
        return real(request, ctx, **facts)
    monkeypatch.setattr(browser_policy, "browser_request_block_reason", slow)
    with _guard(ToolContext(repo_dir=tmp_path, drive_root=tmp_path)) as guard:
        answers = [None] * 32
        def ask(index):
            answers[index] = _ask(guard, {"url": f"http://127.0.0.1:9/{index}", "method": "GET", "post_data": None})
        started = time.monotonic()
        askers = [threading.Thread(target=ask, args=(index,)) for index in range(32)]
        for asker in askers:
            asker.start()
        for asker in askers:
            asker.join(20)
        elapsed = time.monotonic() - started
    assert answers == [{"block": ""}] * 32
    assert len(seen) == 32 and all(name.startswith("mcp-browser-guard") for name in seen)
    assert elapsed < 0.3 * 32 / 2  # decided in parallel, far below one-at-a-time


def test_bridge_reaches_the_policy_owner_through_the_real_server_process(tmp_path, monkeypatch, services):
    main, other = services
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True, browser_policy=True)
    wire(monkeypatch, registry, raw)
    ours = f"http://127.0.0.1:{main}"
    questions = [
        {"armed": True},
        {"armed": True, "token": "f" * 64},
        {"url": f"{ours}/", "method": "GET", "post_data": None, "resource_type": "document", "body_present": False},
        {"url": f"{ours}/api/settings", "method": "POST", "post_data": OWNER_BODY, "body_present": True},
        {"url": f"http://localhost:{main}/api/settings", "method": "POST", "post_data": OWNER_BODY},
        {"url": f"{ours}/api/settings", "method": "POST", "post_data": None, "body_present": True},
        {"url": f"{ours}/api/decisions", "method": "POST", "post_data": None, "body_present": False},
        {"url": f"http://127.0.0.1:{other}/api/settings", "method": "POST", "post_data": OWNER_BODY},
        {"url": "http://169.254.169.254/latest/meta-data", "method": "GET", "post_data": None},
        {"action": "evaluate", "url": f"http://127.0.0.1:{other}/"},
        {"action": "evaluate", "url": "http://169.254.169.254/"},
    ]
    with owner(registry):
        result = call(registry, "guard", {"questions": questions})
        guard = session_of(registry).guard
        armed, address = guard.armed, guard.address
    answers = body(result)
    owner_block = "BROWSER_OWNER_CONTROL_BLOCKED: this operation belongs to the owner"
    assert answers[0] == {"armed": True} and armed is True
    assert answers[1]["armed"] is False and "guard token" in answers[1]["block"]
    assert [answer["block"] for answer in answers[2:]] == [
        "", owner_block, owner_block, owner_block, "", "",
        "BROWSER_METADATA_BLOCKED: link-local/cloud metadata target", "",
        "BROWSER_METADATA_BLOCKED: link-local/cloud metadata target"]
    discovery, session = rows(log, "start")
    assert discovery["guard"] == "" and discovery["token"] is False  # listing never gets the guard
    assert session["guard"] == address and address.startswith("127.0.0.1:") and session["token"] is True
    token = guard.token
    assert token not in log.read_text(encoding="utf-8") and token not in result.text
    mcp_task_session.close_context_sessions(registry._ctx)
    host, port = address.rsplit(":", 1)
    with pytest.raises(OSError):
        socket.create_connection((host, int(port)), timeout=2).close()  # the guard ended with its session


def test_a_slow_policy_decision_never_stalls_the_session_loop(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True, browser_policy=True)
    wire(monkeypatch, registry, raw)
    real = browser_policy.browser_request_block_reason

    def dns_bound(request, ctx, **facts):
        time.sleep(1.5)
        return real(request, ctx, **facts)
    with owner(registry):
        body(call(registry, "bump"))  # the server starts pinging the client
        monkeypatch.setattr(browser_policy, "browser_request_block_reason", dns_bound)
        answers = body(call(registry, "guard", {"questions": [{"url": "http://127.0.0.1:9/", "method": "GET"}]}))
    assert answers == [{"block": ""}]
    asked, answered = rows(log, "ask")[0]["at"], rows(log, "answered")[0]["at"]
    pongs = [row for row in rows(log, "pong") if asked < row["at"] < answered]
    assert answered - asked >= 1.5 and len(pongs) >= 4, pongs  # the SDK loop answered pings meanwhile
    mcp_task_session.close_context_sessions(registry._ctx)


def _held(guard, monkeypatch):
    """The guard's request decision waits for this test: a stand-in for a policy read stuck in DNS."""
    entered, release, decided = threading.Event(), threading.Event(), []

    def held(facts):
        decided.append(facts.get("url"))
        entered.set()
        release.wait(20)
        return ""
    monkeypatch.setattr(guard, "_request_reason", held)
    return entered, release, decided


def _question(guard, n):
    return json.dumps({"token": guard.token, "url": f"http://127.0.0.1:9/{n}", "method": "GET"}).encode() + b"\n"


def _send(guard, data):
    host, port = guard.address.rsplit(":", 1)
    conn = socket.create_connection((host, int(port)), timeout=10)
    conn.sendall(data)
    return conn


def _reply(conn, timeout):
    """The guard's one answer line, or None when none arrived within ``timeout``."""
    conn.settimeout(timeout)
    data = b""
    try:
        while not data.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
    except socket.timeout:
        return None
    finally:
        conn.close()
    return json.loads(data) if data else None


def _queued(guard, count):
    deadline = time.monotonic() + 5
    while guard._pool._work_queue.qsize() < count and time.monotonic() < deadline:
        time.sleep(0.01)
    return guard._pool._work_queue.qsize() == count


def test_a_queued_question_reads_against_the_deadline_of_its_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_browser_guard, "MCP_BROWSER_GUARD_WORKERS", 1)
    monkeypatch.setattr(mcp_browser_guard, "MCP_BROWSER_GUARD_READ_SEC", 1.0)
    with _guard(ToolContext(repo_dir=tmp_path, drive_root=tmp_path)) as guard:
        entered, release, decided = _held(guard, monkeypatch)
        try:
            first = _send(guard, _question(guard, 1))
            assert entered.wait(5)
            unfinished = _send(guard, b"{")  # never completes its line; waits behind the busy worker
            assert _queued(guard, 1)
            time.sleep(1.2)  # its whole read allowance passes in the queue
        finally:
            release.set()
        released = time.monotonic()
        late = _reply(unfinished, 5)
        waited = time.monotonic() - released
        assert _reply(first, 5) == {"block": ""}
    assert late and "unreadable request facts (TimeoutError)" in late["block"], late
    assert waited < 0.6, waited  # refused when dequeued, not after a fresh allowance
    assert decided == ["http://127.0.0.1:9/1"]


def test_questions_beyond_the_pending_bound_are_refused_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_browser_guard, "MCP_BROWSER_GUARD_WORKERS", 1)
    monkeypatch.setattr(mcp_browser_guard, "MCP_BROWSER_GUARD_PENDING", 2, raising=False)
    with _guard(ToolContext(repo_dir=tmp_path, drive_root=tmp_path)) as guard:
        entered, release, decided = _held(guard, monkeypatch)
        try:
            first = _send(guard, _question(guard, 1))
            assert entered.wait(5)
            second = _send(guard, _question(guard, 2))  # the bound's last place: it waits for the worker
            assert _queued(guard, 1)
            started = time.monotonic()
            overflow = _reply(_send(guard, _question(guard, 3)), 2)
            refused_in = time.monotonic() - started
        finally:
            release.set()
        assert _reply(first, 5) == {"block": ""} and _reply(second, 5) == {"block": ""}
    assert overflow and overflow["block"].startswith("BROWSER_POLICY_UNAVAILABLE"), overflow
    assert refused_in < 1.0 and decided == ["http://127.0.0.1:9/1", "http://127.0.0.1:9/2"]  # never decided


def _line(conn, timeout):
    conn.settimeout(timeout)
    data = b""
    while not data.endswith(b"\n"):
        chunk = conn.recv(65536)
        assert chunk, data
        data += chunk
    return json.loads(data)


def test_a_refused_asker_whose_line_is_still_on_its_way_keeps_its_block(tmp_path, monkeypatch):
    """The refusal above, with the asker's line arriving after its block. A line landing on a closed
    socket resets the connection, and Windows then discards the delivered block (WinError 10053), so the
    block goes out at once while the close waits for the line, but only within the refusal's bound."""
    monkeypatch.setattr(mcp_browser_guard, "MCP_BROWSER_GUARD_WORKERS", 1)
    monkeypatch.setattr(mcp_browser_guard, "MCP_BROWSER_GUARD_PENDING", 1, raising=False)
    with _guard(ToolContext(repo_dir=tmp_path, drive_root=tmp_path)) as guard:
        entered, release, decided = _held(guard, monkeypatch)
        try:
            first = _send(guard, _question(guard, 1))
            assert entered.wait(5)
            late = _send(guard, b"")  # connected; its line is not sent yet
            assert _line(late, 2)["block"].startswith("BROWSER_POLICY_UNAVAILABLE")  # the block does not wait
            late.settimeout(0.3)
            with pytest.raises(socket.timeout):
                late.recv(65536)  # the connection stays open for the promised line
            late.sendall(_question(guard, 2))
            late.settimeout(2)
            assert late.recv(65536) == b""  # the line is taken, then a clean close: no reset
            late.close()
            silent = _send(guard, b"")  # never sends its line
            assert _line(silent, 2)["block"].startswith("BROWSER_POLICY_UNAVAILABLE")
            silent.settimeout(3)
            waited = time.monotonic()
            assert silent.recv(65536) == b"" and time.monotonic() - waited < 1.5  # closed at the bound
            silent.close()
        finally:
            release.set()
        assert _reply(first, 5) == {"block": ""}
    assert decided == ["http://127.0.0.1:9/1"]


def test_retiring_the_guard_answers_its_queued_questions_and_runs_none_of_them(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_browser_guard, "MCP_BROWSER_GUARD_WORKERS", 1)
    guard = mcp_browser_guard.BrowserGuard(ToolContext(repo_dir=tmp_path, drive_root=tmp_path))
    entered, release, decided = _held(guard, monkeypatch)
    try:
        first = _send(guard, _question(guard, 1))
        assert entered.wait(5)
        waiting = [_send(guard, _question(guard, n)) for n in (2, 3)]
        assert _queued(guard, 2)
        closed = time.monotonic()
        guard.close()  # returns while the first decision still runs: a policy thread cannot be interrupted
        answers = [_reply(conn, 3) for conn in waiting]
        assert not release.is_set() and time.monotonic() - closed < 4
        assert all(answer and answer["block"].startswith("BROWSER_SESSION_CLOSED") for answer in answers), answers
    finally:
        release.set()
    assert _reply(first, 5)["block"].startswith("BROWSER_SESSION_CLOSED")  # it ended while that ran
    guard._pool.shutdown(wait=True)
    assert decided == ["http://127.0.0.1:9/1"]  # no queued question reached the policy
