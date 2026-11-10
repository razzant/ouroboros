"""The browser-policy callback of an MCP task session: the existing policy owner, reached over loopback.

The bridge carries request facts to ``browser_policy`` with the session context's real
actor facts; it owns no predicate. The policy owner keeps ordinary bodiless and empty
requests ordinary and refuses an owner-shaped request whose body existed but could not
be read, for its Playwright consumer and this bridge alike.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ouroboros import artifacts, browser_policy, config, mcp_browser_guard, mcp_task_session
from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.server_process import record_service_binding
from ouroboros.tools import browser
from ouroboros.tools.registry import ToolContext
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import owner
from tests.test_mcp_task_session import _private_mcp, body, call, fixture_server, rows, session_of, wire  # noqa: F401


OWNER_BODY = '{"OUROBOROS_REVIEW_ENFORCEMENT":"advisory"}'


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
        {"action": "file_intake", "files": [{"path": "upload.bin"}]},
    ]
    upload = b"\x00real session\xff\r\n"
    (pathlib.Path(registry._ctx.workspace_root) / "upload.bin").write_bytes(upload)
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
        "BROWSER_METADATA_BLOCKED: link-local/cloud metadata target", ""]
    [retained] = answers[-1]["files"]  # the session's own task keeps the bytes; the server gets their path
    assert pathlib.Path(retained["path"]).read_bytes() == upload
    assert pathlib.Path(retained["path"]).is_relative_to(task_artifact_dir_path(registry._ctx.drive_root, "root"))
    assert {**retained, "path": ""} == {"path": "", "name": "upload.bin", "size": len(upload),
                                        "sha256": hashlib.sha256(upload).hexdigest()}
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


# -- file_intake: local files chosen for a page's file input, retained exactly for the session's task --------------

def _intake_world(tmp_path):
    """A top-level task context whose active workspace (its repository) holds ``upload/``."""
    repo, data = tmp_path / "repo", tmp_path / "data"
    (repo / "upload").mkdir(parents=True)
    data.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=data)
    ctx.task_id = "t-upload"
    return ctx, repo / "upload"


def _intake(guard, *files):
    return _ask(guard, {"action": "file_intake", "files": list(files)})


def _retained(ctx):
    home = task_artifact_dir_path(ctx.drive_root, ctx.task_id) / "source_handles" / "tool_results"
    return sorted(path.name for path in home.iterdir()) if home.exists() else []


def test_file_intake_retains_the_exact_bytes_and_names_of_every_selected_file(tmp_path):
    ctx, folder = _intake_world(tmp_path)
    names = ["отчёт 📄.bin", "été.TXT", "plain"]  # Cyrillic and emoji; a decomposed (NFD) spelling
    contents = [bytes(range(256)) * 3 + b"\r\n\x00\r", "é text\r\n".encode(), b""]
    for name, data in zip(names, contents):
        (folder / name).write_bytes(data)
    (folder / "chosen.pdf").symlink_to(folder / names[0])
    with _guard(ctx) as guard:
        answer = _intake(guard, {"path": f"upload/{names[0]}"},
                         {"path": str(folder / names[1])},  # absolute: read_file's root inference
                         {"root": "active_workspace", "path": "upload/plain"}, {"path": "upload/chosen.pdf"})
    assert answer["block"] == "" and set(answer) == {"block", "files"}, answer
    rows = answer["files"]
    assert [row["name"] for row in rows] == [*names, "chosen.pdf"]  # the selected spelling, byte for byte
    store = task_artifact_dir_path(ctx.drive_root, ctx.task_id)
    for row, data in zip(rows, [*contents, contents[0]]):
        path = pathlib.Path(row["path"])
        assert path.is_absolute() and path.is_relative_to(store / "source_handles" / "tool_results"), row
        assert path.read_bytes() == data
        assert (row["size"], row["sha256"]) == (len(data), hashlib.sha256(data).hexdigest())
    assert rows[3]["sha256"] == rows[0]["sha256"]  # the link's resolved target, under the link's own name
    assert [pathlib.PurePath(row["path"]).suffix for row in rows] == [".bin", ".TXT", ".bin", ".pdf"]


def test_file_intake_answers_no_file_when_any_selected_file_is_refused(tmp_path):
    from ouroboros.tools.core_file_tools import read_admission
    from ouroboros.tools.tool_resolution import _root_containing_absolute_path
    from tests.test_delegate_input_files import _presence

    ctx, folder = _intake_world(tmp_path)
    (folder / "ok.bin").write_bytes(b"ok")
    (folder / "inner").mkdir()
    (folder / "oracle.bin").write_bytes(b"black box")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not in any root")
    (folder / "escape.bin").symlink_to(outside)  # admission judges the link's resolved target
    ctx.task_contract = {"resource_policy": {"protected_artifacts": [{
        "id": "oracle", "role": "black_box_reference", "paths": [str(folder / "oracle.bin")]}]}}
    ok = {"path": "upload/ok.bin"}
    refused = [{"path": str(outside)}, {"path": "../outside.bin"}, {"path": "upload/escape.bin"},
               {"root": "bogus", "path": "upload/ok.bin"}, {"root": "skill_payload", "path": "x.bin"},
               {"path": "upload/oracle.bin"}]
    unreadable = folder / "locked.bin"
    unreadable.write_bytes(b"locked")
    unreadable.chmod(0)
    misses = {"does not exist": {"path": "upload/absent.bin"}, "is a folder": {"path": "upload/inner"}}
    if not os.access(unreadable, os.R_OK):  # a superuser reads it anyway
        misses["cannot be read (PermissionError"] = {"path": "upload/locked.bin"}
    if hasattr(os, "mkfifo"):
        os.mkfifo(folder / "pipe")
        misses["is not a regular file"] = {"path": "upload/pipe"}  # opened without waiting for a writer
    try:
        with _guard(ctx) as guard:
            for entry in refused:  # read_file's own refusal, in its own words, with read_file's root inference
                root = (entry.get("root") or _root_containing_absolute_path(ctx, "read_file", entry["path"])
                        or "active_workspace")
                words = read_admission(ctx, root, entry["path"])[2]
                answer = _intake(guard, ok, entry)
                assert answer == {"block": "BROWSER_FILE_INTAKE_BLOCKED: file 2 of 2 is refused: "
                                           + str(getattr(words, "text", words)).removeprefix("⚠️ ")}, (entry, answer)
            for words, entry in misses.items():
                answer = _intake(guard, ok, entry)
                assert set(answer) == {"block"} and words in answer["block"], (entry, answer)
                assert answer["block"].startswith("BROWSER_FILE_INTAKE_BLOCKED: file 2 of 2 "), answer
            _presence(ctx, ("system_repo", "unrelated"))  # a Presence ceiling granting nothing here
            answer = _intake(guard, ok)
            assert "PRESENCE_RESOURCE_BLOCKED" in answer["block"] and "files" not in answer, answer
            shapes = (None, [], "upload/ok.bin", [ok] * 17, [{"path": ""}], [{"path": "a\x00b"}], [{"path": 3}],
                      [{**ok, "extra": 1}], [{**ok, "root": 3}], [ok, "upload/ok.bin"])
            for files in shapes:
                answer = _ask(guard, {"action": "file_intake", "files": files})
                assert answer == {"block": "BROWSER_FILE_INTAKE_BLOCKED: files is 1 to 16 {path, root?} objects "
                                           "with a nonempty path"}, (files, answer)
    finally:
        unreadable.chmod(0o600)
    assert _retained(ctx) == []  # every refusal came before any retention
    taskless = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    with _guard(taskless) as guard:
        assert "no task owns this browser session" in _intake(guard, ok)["block"]


def test_file_intake_reads_no_more_than_its_declared_bounds(tmp_path, monkeypatch):
    ctx, folder = _intake_world(tmp_path)
    monkeypatch.setattr(mcp_browser_guard, "FILE_INTAKE_BYTES", 10)
    monkeypatch.setattr(mcp_browser_guard, "FILE_INTAKE_FILES", 2)
    for name, size in (("six", 6), ("five", 5), ("ten", 10), ("eleven", 11), ("grows", 1 << 20)):
        (folder / name).write_bytes(b"x" * size)
    selected = {(folder / name).stat().st_ino: name for name in ("ten", "grows")}
    grower, real, real_open, reads = (folder / "grows").stat().st_ino, os.fstat, os.fdopen, []

    def stale(fd):  # the file grew after its size was read: st_size still says it is empty
        observed = real(fd)
        return os.stat_result((*observed[:6], 0, *observed[7:10])) if observed.st_ino == grower else observed

    class Counted:  # every read of a selected file: the size asked for and the bytes it gave
        def __init__(self, handle, name):
            self._handle, self._name = handle, name

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

        def __getattr__(self, attr):
            return getattr(self._handle, attr)

        def read(self, size=-1):
            data = self._handle.read(size)
            reads.append((self._name, size, len(data)))
            return data

    def counted(fd, *args, **kwargs):
        name = selected.get(real(fd).st_ino)
        handle = real_open(fd, *args, **kwargs)
        return Counted(handle, name) if name else handle
    monkeypatch.setattr(os, "fstat", stale)
    monkeypatch.setattr(os, "fdopen", counted)
    with _guard(ctx) as guard:
        assert _intake(guard, {"path": "upload/ten"})["files"][0]["size"] == 10  # exactly the bound
        over = "exceeds the 10 bytes the selection may hold together"
        assert _intake(guard, {"path": "upload/eleven"})["block"].endswith("file 1 of 1 " + over)
        assert _intake(guard, {"path": "upload/six"}, {"path": "upload/five"})["block"].endswith("file 2 of 2 " + over)
        assert _intake(guard, {"path": "upload/grows"})["block"].endswith("file 1 of 1 " + over)  # the read stops
        assert "files is 1 to 2 " in _intake(guard, *[{"path": "upload/six"}] * 3)["block"]
    # One read each, asked for one byte past the bound: a mebibyte file reported empty gave 11 bytes, no more.
    assert reads == [("ten", 11, 10), ("grows", 11, 11)], reads
    assert len(_retained(ctx)) == 1


def test_file_intake_answers_no_file_once_its_session_closes_or_without_its_token(tmp_path, monkeypatch):
    from ouroboros.tools import core_file_tools

    ctx, folder = _intake_world(tmp_path)
    (folder / "a.bin").write_bytes(b"a")
    (folder / "b.bin").write_bytes(b"b")
    both, closed = [{"path": "upload/a.bin"}, {"path": "upload/b.bin"}], {"block": mcp_browser_guard._CLOSED}
    real_store, real_admission, stored = artifacts.store_actor_source_bytes, core_file_tools.read_admission, []

    def closing_after(guard, count, real, *, fail=False):
        def step(*args, **kwargs):
            if fail and len(stored) == count:
                raise OSError("disk full")
            result = real(*args, **kwargs)
            stored.append(result)
            if not fail and len(stored) == count:
                guard.closing()
            return result
        return step
    with _guard(ctx) as guard:
        wrong = _ask(guard, {"action": "file_intake", "files": both, "token": "0" * 64})
        assert set(wrong) == {"block"} and "guard token" in wrong["block"]
        guard.closing()
        assert _intake(guard, *both) == closed
    assert _retained(ctx) == []
    for count, real, target, kept in ((1, real_admission, core_file_tools, 0),  # closed while admitting
                                      (1, real_store, artifacts, 1),  # closed between two retentions
                                      (2, real_store, artifacts, 2)):  # closed as the last file was retained
        stored.clear()
        with _guard(ctx) as guard:
            name = "read_admission" if target is core_file_tools else "store_actor_source_bytes"
            monkeypatch.setattr(target, name, closing_after(guard, count, real))
            assert _intake(guard, *both) == closed
        monkeypatch.setattr(target, name, real)
        assert len(_retained(ctx)) == kept, (name, count)
    stored.clear()
    with _guard(ctx) as guard:  # a failed retention answers no file either; what was retained stays unanswered
        monkeypatch.setattr(artifacts, "store_actor_source_bytes", closing_after(guard, 1, real_store, fail=True))
        failed = _intake(guard, {"path": "upload/b.bin"}, {"path": "upload/a.bin"})
    assert failed == {"block": "BROWSER_FILE_INTAKE_BLOCKED: the selected files could not be retained (OSError)"}
    assert "disk full" not in failed["block"]
