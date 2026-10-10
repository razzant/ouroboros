"""Task browser bridge: upstream page state, invocation authority and custody.

The synthetic server speaks only what the pinned ``@playwright/mcp@0.0.82``
speaks: ``browser_tabs`` answers ``### Result`` with
``renderTabsMarkdown`` lines and page tools answer ``### Page`` sections
(playwright-core ``tools/backend/response.ts``). Like upstream, it starts its
"browser" lazily as a detached process group. The opt-in last tests drive the
real upstream package headless.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from ouroboros import mcp_client, mcp_task_sessions
from ouroboros.process_containment import CONTAINMENT_ENV_PREFIX, pid_is_zombie, pids_with_env_marker
from tests._cancel_intents_shared import _LiveProc, _reap_spawned_live_procs  # noqa: F401
from tests._cancel_intents_shared import qenv as _qenv

qenv = _qenv
pytestmark = pytest.mark.serial  # Real subprocesses (tests/conftest lane policy).
posix_bridge = pytest.mark.skipif(sys.platform == "win32", reason="Bridge transport and marker custody are POSIX-only")


SERVER = r'''
import json, os, socket, subprocess, sys, time
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Playwright")
tabs = [{"title": "Start", "url": "http://localhost:8765/"}]
state = {"current": 0, "browser": None, "redirect_in": 0, "snapshots": 0, "clicks": 0}

def guard(facts):
    # The host's init page, as upstream's context route would ask it.
    with socket.socket(socket.AF_UNIX) as conn:
        conn.connect(os.environ["OUROBOROS_BROWSER_GUARD"])
        conn.sendall(json.dumps(facts).encode() + b"\n")
        conn.shutdown(socket.SHUT_WR)
        return json.loads(b"".join(iter(lambda: conn.recv(65536), b"")))

def ensure_browser():
    # Upstream launches its browser on first use, detached into its own group,
    # and runs every init page on the tab before any tool acts on it.
    if state["browser"] is None:
        state["browser"] = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        if os.environ.get("PLAYWRIGHT_MCP_INIT_PAGE", "").endswith("guard.cjs") and not os.environ.get("FAKE_OWN_INIT_PAGE"):
            assert guard({"armed": True}) == {"armed": True}

def tabs_markdown():
    return "\n".join(f"- {i}:{' (current)' if i == state['current'] else ''} [{t['title']}]({t['url']})"
                     for i, t in enumerate(tabs))

def page_section():
    tab = tabs[state["current"]]
    return f"### Page\n- Page URL: {tab['url']}\n- Page Title: {tab['title']}"

@mcp.tool(structured_output=False)
def browser_tabs(action: str, index: int | None = None, url: str | None = None) -> str:
    ensure_browser()
    if action == "select":
        state["current"] = index
    text = "### Result\n" + tabs_markdown()
    if state["redirect_in"]:
        state["redirect_in"] -= 1
        if not state["redirect_in"]:  # The page script navigates on its own.
            tabs[state["current"]]["url"] = "http://localhost:8765/landed"
    return text

@mcp.tool(structured_output=False)
def browser_navigate(url: str) -> str:
    ensure_browser()
    tabs[state["current"]].update(url=url, title="Page")
    state["redirect_in"] = 2 if url.endswith("/redirecting") else 0
    return f"### Ran Playwright code\n```js\nawait page.goto({url!r});\n```\n" + page_section()

@mcp.tool(structured_output=False)
def browser_snapshot() -> str:
    state["snapshots"] += 1
    return page_section() + (f"\n### Snapshot\n```yaml\n- generic [ref=e1]: snapshot {state['snapshots']}"
                             f" clicks {state['clicks']}\n```")

@mcp.tool(structured_output=False)
def browser_click(target: str, element: str | None = None) -> str:
    state["clicks"] += 1
    return f"### Ran Playwright code\n```js\nawait page.click({target!r});\n```\n" + page_section()

@mcp.tool(structured_output=False)
def browser_page_request(url: str, method: str = "GET", post_data: str | dict | None = None) -> str:
    # What the page's own iframe, fetch or form does: the context route pauses it on the guard.
    # (FastMCP hands a JSON-looking string argument over already parsed.)
    body = json.dumps(post_data) if isinstance(post_data, dict) else post_data
    answer = guard({"url": url, "method": method, "post_data": body})
    return "### Result\n" + ("sent" if answer.get("block") == "" else "aborted")

@mcp.tool(structured_output=False)
def browser_run_code_unsafe(code: str) -> str:
    return "### Result\nran"

@mcp.tool(structured_output=False)
def browser_route(pattern: str, status: int | None = None, body: str | None = None) -> str:
    return "### Result\nrouted"

@mcp.tool(structured_output=False)
def browser_slow_click() -> str:
    state["clicks"] += 1
    time.sleep(3)
    return page_section()

mcp.run(transport="stdio")
'''


def _server(path):
    script = path / "server.py"
    script.write_text(textwrap.dedent(SERVER))
    return script


def _entry(script):
    return {"id": "browser", "enabled": True, "transport": "stdio",
            "command": sys.executable, "args": [str(script)], "browser_bridge": True}


def _ctx(root, attempt=1):
    return SimpleNamespace(task_id=f"synthetic-{root.name}", task_attempt=attempt,
                           task_lifecycle_bound=True, drive_root=root,
                           task_metadata={}, task_contract={}, messages=[])


def _live(marker):
    return [pid for pid in pids_with_env_marker(marker) or [] if not pid_is_zombie(pid)]


def _wait_gone(marker, seconds=4.0):
    deadline = time.monotonic() + seconds
    while _live(marker) and time.monotonic() < deadline:
        time.sleep(.05)
    return _live(marker)


@pytest.fixture
def policy(monkeypatch):
    """Keep URL policy independent of DNS and the installation's own ports."""
    blocked = {"words": ("forbidden",)}
    monkeypatch.setattr(mcp_task_sessions.browser_policy, "browser_url_block_reason",
                        lambda url, *_a, **_k: "BLOCKED_BY_TEST" if any(w in url for w in blocked["words"]) else "")
    return blocked


@pytest.fixture
def safety(monkeypatch):
    calls = []
    verdict = {"allowed": True, "advice": ""}

    def check(name, args, messages=None, ctx=None, python_resolution=None, resolved_binding=None):
        calls.append({"name": name, "args": dict(args), "page": (resolved_binding or {}).get("browser_page")})
        return verdict["allowed"], verdict["advice"]

    monkeypatch.setattr("ouroboros.safety.check_safety", check)
    return SimpleNamespace(calls=calls, verdict=verdict)


@pytest.fixture
def manager(tmp_path, monkeypatch, policy, safety):
    if sys.platform == "win32":
        pytest.skip("Bridge transport and marker custody are POSIX-only")
    pytest.importorskip("mcp")
    from ouroboros import owner_pause

    monkeypatch.setattr(owner_pause, "submit_async_preparation", lambda factory, **_kw: factory())
    instance = mcp_client.MCPManager()
    instance.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [_entry(_server(tmp_path))]})
    monkeypatch.setattr(mcp_client, "get_manager", lambda: instance)
    ctx = _ctx(tmp_path)
    yield SimpleNamespace(instance=instance, ctx=ctx)
    mcp_task_sessions.stop_task(ctx)


def _dispatch(ctx, tool, args):
    from ouroboros.tools.extension_dispatch import _dispatch_mcp_tool_result

    return _dispatch_mcp_tool_result(ctx, mcp_client.make_tool_name("browser", tool), args)


def _session(ctx):
    return mcp_task_sessions._sessions[(ctx.task_id, "browser")]


# -- upstream state reader -------------------------------------------------------


def _listing(text, *, error=False):
    return SimpleNamespace(isError=error, content=[SimpleNamespace(type="text", text=text)])


def test_page_state_reads_the_upstream_current_tab_line():
    text = ("### Result\n- 0: [Docs](https://example.com/)\n"
            "- 1: (current) [Foo (bar)](https://en.wikipedia.org/wiki/Foo_(bar))\n"
            "### Page\n- Page URL: https://ignored.example/")
    assert mcp_task_sessions._page_state(_listing(text)) == {
        "url": "https://en.wikipedia.org/wiki/Foo_(bar)", "tab_index": 1, "open_tabs": 2}


@pytest.mark.parametrize("text", [
    "### Result\n- 0: (current) [a](b](https://x.test/)",           # title holds "]("
    "### Result\n- 0: (current) [t](https://x.test/a](b)",          # url holds "]("
    "### Result\n- 0: (current) [t](https://x.test/) [crashed]",    # crashed tab
    "### Result\n- 0: [t](https://x.test/)",                        # no current tab
    "### Result\n- 0: (current) [t](a)\n- 1: (current) [u](b)",     # two current tabs
    "### Result\nNo open tabs. Navigate to a URL to create one.",
    "### Error\nTool \"browser_tabs\" not found",
])
def test_page_state_refuses_unknown_or_ambiguous_answers(text):
    with pytest.raises(RuntimeError, match="BROWSER_STATE_UNKNOWN"):
        mcp_task_sessions._page_state(_listing(text))
    with pytest.raises(RuntimeError, match="BROWSER_STATE_UNKNOWN"):
        mcp_task_sessions._page_state(_listing("### Result\n- 0: (current) [t](u)", error=True))


# -- one connection, actual state, policy and Safety ------------------------------


def test_one_connection_carries_actual_page_into_policy_and_safety(manager, safety, policy):
    ctx = manager.ctx
    assert manager.instance.refresh_server("browser", authority=ctx)["ok"] is True
    first = _dispatch(ctx, "browser_snapshot", {})
    second = _dispatch(ctx, "browser_snapshot", {})
    assert first.status == second.status == "ok"
    assert "snapshot 1" in first.text and "snapshot 2" in second.text  # one live server
    # Safety ran once per call, after the page read, with the actual page.
    assert [call["name"] for call in safety.calls] == [mcp_client.make_tool_name("browser", "browser_snapshot")] * 2
    assert safety.calls[0]["page"]["url"] == "http://localhost:8765/"
    assert safety.calls[0]["page"]["tab_index"] == 0

    refused = _dispatch(ctx, "browser_navigate", {"url": "https://forbidden.test/"})
    assert (refused.status, refused.code) == ("blocked", "LEGACY_BLOCKED")
    assert "BLOCKED_BY_TEST" in refused.text and "External MCP" not in refused.text
    moved = _dispatch(ctx, "browser_navigate", {"url": "https://allowed.test/"})
    assert moved.status == "ok"
    assert safety.calls[-1]["page"]["url"] == "http://localhost:8765/"  # the page it replaces

    policy["words"] = ("allowed.test",)  # The actual page is now outside policy.
    blocked = _dispatch(ctx, "browser_click", {"target": "e1"})
    assert blocked.code == "LEGACY_BLOCKED" and "BLOCKED_BY_TEST" in blocked.text
    away = _dispatch(ctx, "browser_navigate", {"url": "https://ok.test/"})
    assert away.status == "ok"  # A navigation is judged by its destination.
    policy["words"] = ("forbidden",)
    assert "snapshot 3 clicks 0" in _dispatch(ctx, "browser_snapshot", {}).text
    tabs = _dispatch(ctx, "browser_tabs", {"action": "select", "index": 0})
    assert tabs.code == "LEGACY_BLOCKED"


def test_safety_refusal_and_page_moving_during_assessment_dispatch_nothing(manager, safety):
    ctx = manager.ctx
    manager.instance.refresh_server("browser", authority=ctx)
    safety.verdict.update(allowed=False, advice="⚠️ SAFETY: not this")
    denied = _dispatch(ctx, "browser_click", {"target": "e1"})
    assert (denied.status, denied.code) == ("blocked", "SAFETY_VIOLATION")
    safety.verdict.update(allowed=True, advice="⚠️ SAFETY_WARNING: unchecked")
    assert _dispatch(ctx, "browser_navigate", {"url": "http://localhost:8765/redirecting"}).meta["safety_warning"]
    # The page script navigates between the assessed read and the pre-dispatch read.
    changed = _dispatch(ctx, "browser_snapshot", {})
    assert changed.code == "LEGACY_BLOCKED" and "BROWSER_STATE_CHANGED" in changed.text
    after = _dispatch(ctx, "browser_snapshot", {})
    assert "snapshot 1 clicks 0" in after.text  # neither refused call reached the server


def test_ambiguous_page_after_dispatch_withholds_result_and_refuses_next(manager):
    ctx = manager.ctx
    manager.instance.refresh_server("browser", authority=ctx)
    result = _dispatch(ctx, "browser_navigate", {"url": "http://localhost:8765/a](b"})
    assert result.code == "MCP_ERROR" and "BROWSER_POLICY_POST_DISPATCH" in result.text
    assert "was dispatched" in result.text and "External MCP" not in result.text
    refused = _dispatch(ctx, "browser_snapshot", {})
    assert refused.code == "LEGACY_BLOCKED" and "BROWSER_STATE_UNKNOWN" in refused.text


def test_owner_pause_refuses_the_action_itself(manager, monkeypatch):
    from ouroboros import owner_pause

    ctx = manager.ctx
    manager.instance.refresh_server("browser", authority=ctx)

    def refused():
        raise owner_pause.OwnerPauseRefused("owner_pause")

    monkeypatch.setattr(owner_pause, "operation_start", refused)
    with pytest.raises(owner_pause.OwnerPauseRefused):
        _dispatch(ctx, "browser_click", {"target": "e1"})
    monkeypatch.setattr(owner_pause, "operation_start", __import__("contextlib").nullcontext)
    assert "snapshot 1 clicks 0" in _dispatch(ctx, "browser_snapshot", {}).text


def test_settings_change_closes_the_held_bridge(manager, tmp_path):
    ctx = manager.ctx
    manager.instance.refresh_server("browser", authority=ctx)
    _dispatch(ctx, "browser_snapshot", {})
    marker = _session(ctx).token
    assert _live(marker)
    entry = _entry(tmp_path / "server.py")
    manager.instance.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [{**entry, "enabled": False}]})
    assert not _live(marker)
    assert (ctx.task_id, "browser") not in mcp_task_sessions._sessions


def test_dead_server_revokes_ready_session_and_same_attempt_discovery(manager):
    ctx = manager.ctx
    assert manager.instance.refresh_server("browser", authority=ctx)["ok"] is True
    session = _session(ctx)
    pids = _live(session.token)
    assert len(pids) == 1
    os.kill(pids[0], signal.SIGKILL)
    session.thread.join(timeout=5)
    assert not session.thread.is_alive()
    assert session.revoked
    assert manager.instance.refresh_server("browser", authority=ctx)["ok"] is False
    with pytest.raises(RuntimeError, match="did not open in this task attempt"):
        mcp_task_sessions.discover(session.cfg, ctx, 5)


def test_dispatched_bridge_timeout_retains_effect_unknown_and_close_receipt(manager):
    ctx = manager.ctx
    assert manager.instance.refresh_server("browser", authority=ctx)["ok"] is True
    manager.instance._tool_timeout_sec = 1
    result = _dispatch(ctx, "browser_slow_click", {})
    assert (result.status, result.code) == ("timeout", "MCP_TIMEOUT")
    assert "effect is unknown" in result.text
    assert "session was closed: confirmed" in result.text
    assert _session(ctx).revoked


def test_ownerless_discovery_and_settings_probe_do_not_spawn(tmp_path):
    script = tmp_path / "server.py"
    script.write_text("raise AssertionError('spawned')")
    manager = mcp_client.MCPManager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [_entry(script)]})
    assert manager.refresh_server("browser")["not_started"] is True
    assert manager.test_server(_entry(script))["code"] == "MCP_TASK_OWNER_REQUIRED"


def test_windows_bridge_refuses_before_process_or_socket_creation(tmp_path, monkeypatch):
    from ouroboros import platform_layer

    if sys.platform == "win32":
        assert platform_layer.IS_WINDOWS  # Exercise the actual platform refusal on Windows CI.
    else:
        monkeypatch.setattr(platform_layer, "IS_WINDOWS", True)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Windows refusal must precede process/container or socket creation")

    # Discovery may read prior scope membership; this constructs no process or socket.
    forbidden.for_scope = mcp_task_sessions.ProcessContainer.for_scope
    monkeypatch.setattr(mcp_task_sessions, "ProcessContainer", forbidden)
    monkeypatch.setattr(mcp_task_sessions, "_private_socket_dir", forbidden)
    cfg = mcp_client.normalize_server_config(_entry(tmp_path / "must-not-run.py"))
    with pytest.raises(RuntimeError, match="not yet available on Windows"):
        mcp_task_sessions.discover(cfg, _ctx(tmp_path), 15)


# -- the request guard: the requests Playwright routes ----------------------------


@pytest.fixture
def guarded(tmp_path, monkeypatch, safety):
    """The actual request policy (no URL stub) and a proven Ouroboros endpoint."""
    if sys.platform == "win32":
        pytest.skip("Bridge transport and marker custody are POSIX-only")
    pytest.importorskip("mcp")
    from ouroboros import config, owner_pause
    from ouroboros.server_process import record_service_binding

    monkeypatch.setattr(owner_pause, "submit_async_preparation", lambda factory, **_kw: factory())
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "pro")
    record_service_binding(tmp_path, "main", "127.0.0.1", 49159, pid=os.getpid())
    instance = mcp_client.MCPManager()
    entry = _entry(_server(tmp_path))
    instance.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [entry]})
    monkeypatch.setattr(mcp_client, "get_manager", lambda: instance)
    ctx = _ctx(tmp_path)
    state = SimpleNamespace(instance=instance, ctx=ctx, entry=entry, config=config,
                            ours="http://127.0.0.1:49159", other="http://127.0.0.1:49160")
    yield state
    mcp_task_sessions.stop_task(state.ctx)


def _page_request(ctx, url, method="GET", post_data=None):
    return _dispatch(ctx, "browser_page_request", {"url": url, "method": method, "post_data": post_data})


def test_guard_judges_each_page_request_with_the_existing_request_policy(guarded, monkeypatch):
    ctx = guarded.ctx
    assert guarded.instance.refresh_server("browser", authority=ctx)["ok"] is True
    owner_post = f"{guarded.ours}/api/owner/safety-mode"
    settings = '{"OUROBOROS_REVIEW_ENFORCEMENT": "advisory"}'
    blocked = _page_request(ctx, owner_post, "POST", "{}")
    assert blocked.status == "ok" and "aborted" in blocked.text, blocked.text  # The action ran; its request did not.
    assert "BROWSER_REQUEST_BLOCKED" in blocked.text and "BROWSER_OWNER_CONTROL_BLOCKED" in blocked.text
    assert blocked.meta["route_note"] and "External MCP" not in blocked.text.split("BROWSER_REQUEST_BLOCKED")[1]
    assert "aborted" in _page_request(ctx, f"{guarded.ours}/api/settings", "POST", settings).text
    assert "aborted" in _page_request(ctx, "http://169.254.169.254/latest/meta-data").text
    # An unrelated application reusing the pathname, and ordinary reads, go on.
    sent = _page_request(ctx, f"{guarded.other}/api/settings", "POST", settings)
    assert "sent" in sent.text and "BROWSER_REQUEST_BLOCKED" not in sent.text
    assert "sent" in _page_request(ctx, f"{guarded.ours}/", "GET").text
    monkeypatch.setattr(guarded.config, "_BOOT_RUNTIME_MODE", "cyber_pro")  # read per request
    assert "sent" in _page_request(ctx, owner_post, "POST", "{}").text


def test_restricted_child_requests_to_an_ouroboros_endpoint_are_aborted(guarded):
    ctx = guarded.ctx
    ctx.task_constraint = {"mode": "local_readonly_subagent"}
    assert guarded.instance.refresh_server("browser", authority=ctx)["ok"] is True
    framed = _page_request(ctx, f"{guarded.ours}/frame")
    assert "aborted" in framed.text and "BROWSER_LOCAL_READONLY_BLOCKED" in framed.text
    assert "sent" in _page_request(ctx, f"{guarded.other}/frame").text  # literal loopback keeps its reach


def test_unacknowledged_guard_refuses_every_action_undispatched(guarded):
    ctx = guarded.ctx
    # An init page of the owner's own replaces the host's: upstream never runs the guard.
    guarded.instance.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [
        {**guarded.entry, "env": {"FAKE_OWN_INIT_PAGE": "1"}}]})
    assert guarded.instance.refresh_server("browser", authority=ctx)["ok"] is True
    refused = _dispatch(ctx, "browser_navigate", {"url": "http://localhost:8765/next"})
    assert (refused.status, refused.code) == ("blocked", "LEGACY_BLOCKED")
    assert "BROWSER_REQUEST_GUARD_UNAVAILABLE" in refused.text
    assert _dispatch(ctx, "browser_click", {"target": "e1"}).code == "LEGACY_BLOCKED"
    raw = _session(ctx)._request("browser_snapshot", {}, 10)
    assert "localhost:8765/ " not in raw.content[0].text and "clicks 0" in raw.content[0].text


def test_server_side_code_beside_the_guard_is_refused_where_policy_binds(guarded, monkeypatch):
    ctx = guarded.ctx
    guarded.instance.refresh_server("browser", authority=ctx)
    code = {"code": "async (page) => page.context().unrouteAll()"}
    refused = _dispatch(ctx, "browser_run_code_unsafe", code)
    assert refused.code == "LEGACY_BLOCKED" and "BROWSER_REQUEST_GUARD_BYPASS" in refused.text
    assert "BROWSER_REQUEST_GUARD_BYPASS" in _dispatch(ctx, "browser_route", {"pattern": "**"}).text
    assert "routed" in _dispatch(ctx, "browser_route", {"pattern": "**/x", "body": "mock"}).text  # fulfils
    monkeypatch.setattr(guarded.config, "_BOOT_RUNTIME_MODE", "cyber_pro")
    assert "ran" in _dispatch(ctx, "browser_run_code_unsafe", code).text
    ctx.task_constraint = {"mode": "local_readonly_subagent"}  # Still binding for a read-only child.
    assert _dispatch(ctx, "browser_route", {"pattern": "**"}).code == "LEGACY_BLOCKED"


def _ask_guard(path, payload):
    import socket

    with socket.socket(socket.AF_UNIX) as conn:
        conn.connect(path)
        conn.sendall(payload)
        conn.shutdown(socket.SHUT_WR)
        return b"".join(iter(lambda: conn.recv(65536), b""))


@pytest.mark.parametrize("revoke", [False, True])
def test_guard_policy_runs_off_loop_and_rechecks_revocation(guarded, monkeypatch, revoke):
    import asyncio
    import json
    import threading

    guarded.instance.refresh_server("browser", authority=guarded.ctx)
    session = _session(guarded.ctx)
    loop_thread = threading.get_ident()
    started, released = threading.Event(), threading.Event()
    policy_threads = []

    def policy(_ctx, _facts):
        policy_threads.append(threading.get_ident())
        started.set()
        assert released.wait(2)
        return ""

    monkeypatch.setattr(mcp_task_sessions, "_request_block_reason", policy)

    class Writer:
        data = b""
        def write(self, data):
            self.data += data
        async def drain(self):
            pass
        def close(self):
            pass

    async def check():
        reader, writer = asyncio.StreamReader(), Writer()
        reader.feed_data(b'{"url":"http://example.test/","method":"GET"}\n')
        reader.feed_eof()
        pending = asyncio.create_task(session._answer_guard(reader, writer))
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            # This coroutine must advance while the DNS/policy thread waits.
            if revoke:
                session.revoked = True
            released.set()
            await pending
            return json.loads(writer.data)
        finally:
            released.set()

    try:
        answer = asyncio.run(check())
        assert policy_threads and all(t != loop_thread for t in policy_threads)
        assert answer["block"].startswith("BROWSER_SESSION_REVOKED") if revoke else answer == {"block": ""}
    finally:
        session.revoked = False  # Let fixture close the real transport normally.


def test_guard_diagnostics_disclose_overflow_and_truncated_fields(guarded, monkeypatch):
    import json

    ctx = guarded.ctx
    guarded.instance.refresh_server("browser", authority=ctx)
    session = _session(ctx)
    original, socket_path = session._request, os.path.join(session.guard_dir.name, "g.sock")
    facts = {"url": guarded.ours + "/api/owner/safety-mode?long=" + "x" * 800, "method": "POST", "post_data": "{}"}

    def request(name, args, timeout, *, dispatch=False):
        for _ in range(25 if dispatch else 0):  # Blocked during the action: 20 kept, 5 omitted.
            assert json.loads(_ask_guard(socket_path, (json.dumps(facts) + "\n").encode()))["block"]
        return original(name, args, timeout, dispatch=dispatch)

    monkeypatch.setattr(session, "_request", request)
    result = _dispatch(ctx, "browser_snapshot", {})
    assert "Earlier blocked-request diagnostics omitted: 5" in result.text
    assert "OMISSION NOTE" in result.text


def test_guard_answers_fail_closed_and_close_with_the_session(guarded):
    import json

    ctx = guarded.ctx
    guarded.instance.refresh_server("browser", authority=ctx)
    session = _session(ctx)
    path = os.path.join(session.guard_dir.name, "g.sock")
    for payload in (b"not json\n", b"[1]\n", b'{"url": 5, "method": "GET"}\n', b'{"url": "http://x.test/"}\n', b""):
        assert json.loads(_ask_guard(path, payload))["block"].startswith("BROWSER_POLICY_UNAVAILABLE")
    assert mcp_task_sessions.stop_task(ctx) == [{"server": "browser", "closure": "confirmed"}]
    assert not os.path.exists(session.guard_dir.name)  # No answer remains: the guard aborts.


def test_guard_script_routes_each_request_on_the_host_verdict(guarded, tmp_path):
    """The shipped guard under a fake Playwright surface, against the live host socket."""
    import json
    import shutil

    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    ctx = guarded.ctx
    guarded.instance.refresh_server("browser", authority=ctx)
    script = tmp_path / "guard.cjs"
    script.write_text(mcp_task_sessions._GUARD_JS)
    harness = tmp_path / "harness.cjs"
    harness.write_text(textwrap.dedent("""
        const guard = require(process.argv[2]);
        const routes = [];
        const context = { route: async (pattern, handler) => { routes.push([pattern, handler]); } };
        const page = { context: () => context };
        (async () => {
          let armed = 'armed';
          try { await guard.default({ page }); await guard.default({ page }); } catch (e) { armed = String(e); }
          const outcomes = [];
          for (const [url, method, body] of JSON.parse(process.argv[3])) {
            const outcome = [];
            await routes[0][1]({
              request: () => ({ url: () => url, method: () => method, postData: () => body }),
              fallback: async () => { outcome.push('fallback'); },
              abort: async code => { outcome.push('abort:' + code); },
            });
            outcomes.push(outcome.join());
          }
          console.log(JSON.stringify({ armed, patterns: routes.map(r => r[0]), outcomes }));
        })();
    """))
    cases = json.dumps([[f"{guarded.ours}/api/owner/safety-mode", "POST", "{}"],
                        [f"{guarded.other}/page", "GET", None]])
    socket_path = os.path.join(_session(ctx).guard_dir.name, "g.sock")

    def run(path):
        done = subprocess.run([node, str(harness), str(script), cases], env={**os.environ, "OUROBOROS_BROWSER_GUARD": path},
                              capture_output=True, text=True, timeout=60, check=True)
        return json.loads(done.stdout)

    assert run(socket_path) == {"armed": "armed", "patterns": ["**/*"],  # one route per context
                                "outcomes": ["abort:blockedbyclient", "fallback"]}
    assert _session(ctx).guard_armed
    unreachable = run(str(tmp_path / "gone.sock"))
    assert unreachable["armed"] != "armed"  # upstream's tab initialization fails with it
    assert unreachable["outcomes"] == ["abort:blockedbyclient"] * 2


# -- custody: task end, reopen, Panic and cancel ----------------------------------


@posix_bridge
def test_task_end_reaps_detached_descendant_and_ends_the_attempt(tmp_path, monkeypatch, policy, safety):
    pytest.importorskip("mcp")
    from ouroboros import loop_budget
    from ouroboros.tools import services

    cfg, ctx = mcp_client.normalize_server_config(_entry(_server(tmp_path))), _ctx(tmp_path)
    events = []
    monkeypatch.setattr(services, "stop_task_services", lambda _ctx: [])
    finalize_services = loop_budget._finalize_task_services
    monkeypatch.setattr(loop_budget, "_loop", lambda: SimpleNamespace(
        _emit_checkpoint_event=lambda *args: events.append(args[-1]),
        _finalize_task_services=finalize_services))
    exit_ctx = loop_budget._LoopExitContext(
        tools=SimpleNamespace(_ctx=ctx), drive_root=tmp_path, task_id=ctx.task_id,
        event_queue=None, drive_logs=tmp_path / "logs", accumulated_usage={}, llm_trace={},
    )
    try:
        assert "browser_tabs" in {tool["name"] for tool in mcp_task_sessions.discover(cfg, ctx, 15)}
        mcp_task_sessions.call(cfg, "mcp_browser__browser_snapshot", "browser_snapshot", {}, ctx, 10)
        marker = _session(ctx).token
        assert len(_live(marker)) >= 2  # the server and its detached "browser"
        # Pre-acceptance service finalization cannot terminate an active bridge.
        assert not finalize_services(exit_ctx)
        assert _live(marker)
    finally:
        loop_budget._cleanup_loop_resources(None, exit_ctx)
    assert events[0]["bridges"] == [{"server": "browser", "closure": "confirmed"}]
    assert not _live(marker)
    with pytest.raises(RuntimeError, match="ended"):
        mcp_task_sessions.discover(cfg, ctx, 15)


@posix_bridge
def test_unconfirmed_close_refuses_reopen_by_a_later_attempt(tmp_path, monkeypatch, policy, safety):
    pytest.importorskip("mcp")
    cfg, ctx = mcp_client.normalize_server_config(_entry(_server(tmp_path))), _ctx(tmp_path)
    mcp_task_sessions.discover(cfg, ctx, 15)
    session = _session(ctx)
    actual_reap = mcp_task_sessions.ProcessContainer.reap

    def uncertain_reap(container):
        actual_reap(container)  # keep the fixture clean while reporting no proof
        return "process identity could not be confirmed"

    # Every scan, the worker's own and each rescan on a fresh container, stays uncertain.
    monkeypatch.setattr(mcp_task_sessions.ProcessContainer, "reap", uncertain_reap)
    assert mcp_task_sessions.stop_task(ctx)[0]["closure"].startswith("unconfirmed:")
    assert _session(ctx) is session
    retry = _ctx(tmp_path, attempt=2)
    with pytest.raises(RuntimeError, match="before its earlier closure is confirmed"):
        mcp_task_sessions.discover(cfg, retry, 15)
    with pytest.raises(PermissionError, match="not discovered by this task attempt"):
        mcp_task_sessions.call(cfg, "mcp_browser__browser_snapshot", "browser_snapshot", {}, retry, 10)


@posix_bridge
def test_live_member_of_an_earlier_process_refuses_reopen(tmp_path, policy, safety):
    pytest.importorskip("mcp")
    cfg, ctx = mcp_client.normalize_server_config(_entry(_server(tmp_path))), _ctx(tmp_path)
    scope = mcp_task_sessions._scope(tmp_path, ctx.task_id)
    # What a killed worker's bridge leaves: marked, live, unknown to this process.
    leftover = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                                env={**os.environ, f"{CONTAINMENT_ENV_PREFIX}{scope}deadbeef": "1"})
    retry = _ctx(tmp_path, attempt=2)
    try:
        with pytest.raises(RuntimeError, match="earlier bridge processes are still live"):
            mcp_task_sessions.discover(cfg, ctx, 15)
        with pytest.raises(RuntimeError, match="did not open in this task attempt"):
            mcp_task_sessions.discover(cfg, ctx, 15)  # every round's listing: no rescan
        assert mcp_task_sessions.settle_dead_task(tmp_path, ctx.task_id) == {"closure": "confirmed"}
        leftover.wait(timeout=5)
        assert "browser_tabs" in {tool["name"] for tool in mcp_task_sessions.discover(cfg, retry, 15)}
    finally:
        if leftover.poll() is None:
            leftover.kill()
            leftover.wait(timeout=5)
        mcp_task_sessions.stop_task(retry)


@posix_bridge
def test_panic_finds_members_without_a_ledger_row(tmp_path, monkeypatch, policy, safety):
    pytest.importorskip("mcp")
    from ouroboros import server_control
    from ouroboros.process_custody import ledger_path
    from ouroboros.startup_historical_audit import audit

    cfg, ctx = mcp_client.normalize_server_config(_entry(_server(tmp_path))), _ctx(tmp_path)
    mcp_task_sessions.discover(cfg, ctx, 15)
    mcp_task_sessions.call(cfg, "mcp_browser__browser_snapshot", "browser_snapshot", {}, ctx, 10)
    marker = _session(ctx).token
    ledger_path(tmp_path).unlink()  # The spawn-to-record window: no durable row yet.
    assert _live(marker)

    monkeypatch.setattr(audit, "stop", lambda: None)
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda **_kw: None)
    monkeypatch.setattr("ouroboros.claudexor_daemon.get_owned_daemon", lambda **_kw: None)
    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda **_kw: [])
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *_a, **_kw: [])
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *_a, **_kw: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda **_kw: [])
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda *_a: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda *_a: None)
    monkeypatch.setattr(server_control, "_write_panic_flag", lambda *_a: None)

    class ExitCalled(Exception):
        pass

    monkeypatch.setattr(server_control.os, "_exit", lambda _code: (_ for _ in ()).throw(ExitCalled()))
    try:
        with pytest.raises(ExitCalled):
            server_control.execute_panic_stop(
                None, lambda **_kw: None, data_dir=tmp_path,
                panic_exit_code=120, log=SimpleNamespace(critical=lambda *_a: None),
            )
        assert not _wait_gone(marker)
    finally:
        mcp_task_sessions.stop_task(ctx)


@posix_bridge
def test_running_task_cancel_closes_bridge_after_worker_death(qenv, monkeypatch, policy, safety):
    pytest.importorskip("mcp")
    cfg, ctx = mcp_client.normalize_server_config(_entry(_server(qenv.drive))), _ctx(qenv.drive)
    mcp_task_sessions.discover(cfg, ctx, 15)
    mcp_task_sessions.call(cfg, "mcp_browser__browser_snapshot", "browser_snapshot", {}, ctx, 10)
    marker = _session(ctx).token
    proc = _LiveProc()
    qenv.workers.WORKERS[0] = SimpleNamespace(wid=0, proc=proc, busy_task_id=ctx.task_id, reaping=False)
    qenv.q.RUNNING[ctx.task_id] = {"task": {"id": ctx.task_id, "chat_id": 0}, "worker_id": 0}
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    try:
        assert qenv.tl.cancel_task_custody(ctx.task_id) == qenv.tl.CANCEL_CANCELLED
        assert not proc.is_alive()
        assert not _live(marker)
    finally:
        mcp_task_sessions.stop_task(ctx)


def test_unconfirmed_bridge_scan_never_undoes_a_confirmed_cancel(qenv, monkeypatch):
    task_id = f"synthetic-{qenv.drive.name}"
    monkeypatch.setattr(mcp_task_sessions, "stop_scope",
                        lambda *_a, **_k: {"closure": "unconfirmed: process table unreadable"})
    proc = _LiveProc()
    qenv.workers.WORKERS[0] = SimpleNamespace(wid=0, proc=proc, busy_task_id=task_id, reaping=False)
    qenv.q.RUNNING[task_id] = {"task": {"id": task_id, "chat_id": 0}, "worker_id": 0}
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_kw: None)
    assert qenv.tl.cancel_task_custody(task_id) == qenv.tl.CANCEL_CANCELLED
    assert not proc.is_alive()


# -- opt-in: the actual upstream package, headless --------------------------------


@posix_bridge
def test_actual_upstream_playwright_mcp_headless(tmp_path, monkeypatch, policy, safety):
    """Set OUROBOROS_PLAYWRIGHT_MCP_CLI (cli.js of @playwright/mcp@0.0.82) and
    OUROBOROS_PLAYWRIGHT_HEADLESS_SHELL (a chrome-headless-shell binary, which has
    no window). Never ``--extension``: that mode ignores headless and opens Chrome."""
    import http.server
    import shutil
    import threading

    cli = os.environ.get("OUROBOROS_PLAYWRIGHT_MCP_CLI", "")
    shell = os.environ.get("OUROBOROS_PLAYWRIGHT_HEADLESS_SHELL", "")
    node = shutil.which("node")
    if not (cli and shell and node and "headless-shell" in os.path.basename(shell)):
        pytest.skip("actual upstream consumer is opt-in")
    pages = tmp_path / "site"
    pages.mkdir()
    (pages / "a.html").write_text("<title>A</title><a href='b.html'>next</a>")
    (pages / "b.html").write_text("<title>B</title><p>done</p>")
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(pages), **k)  # noqa: E731
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    cfg = mcp_client.normalize_server_config({
        "id": "browser", "enabled": True, "transport": "stdio", "command": node, "browser_bridge": True,
        "args": [cli, "--headless", "--isolated", "--browser", "chromium", "--executable-path", shell],
    })
    ctx = _ctx(tmp_path)
    try:
        assert "browser_tabs" in {tool["name"] for tool in mcp_task_sessions.discover(cfg, ctx, 60)}
        marker = _session(ctx).token
        navigated, _advice, _note = mcp_task_sessions.call(
            cfg, "mcp_browser__browser_navigate", "browser_navigate", {"url": f"{base}/a.html"}, ctx, 60)
        assert navigated.status == "ok", navigated.text
        snap, _advice, _note = mcp_task_sessions.call(
            cfg, "mcp_browser__browser_snapshot", "browser_snapshot", {}, ctx, 60)
        assert snap.status == "ok" and "next" in snap.text
        assert safety.calls[-1]["page"]["url"] == f"{base}/a.html"
        policy["words"] = ("/a.html",)
        with pytest.raises(PermissionError, match="BLOCKED_BY_TEST"):
            mcp_task_sessions.call(cfg, "mcp_browser__browser_snapshot", "browser_snapshot", {}, ctx, 60)
        assert len(_live(marker)) >= 2  # node and its headless shell tree
        assert mcp_task_sessions.stop_task(ctx) == [{"server": "browser", "closure": "confirmed"}]
        assert not _live(marker)
    finally:
        mcp_task_sessions.stop_task(ctx)
        httpd.shutdown()


@pytest.fixture
def upstream(tmp_path, monkeypatch, safety):
    """The actual upstream package and real request policy against two local sites:
    ``plain`` is an ordinary loopback app, ``ours`` a proven Ouroboros endpoint
    recording every request it receives. Opt-in, as the test above."""
    if sys.platform == "win32":
        pytest.skip("Bridge transport and marker custody are POSIX-only")
    import http.server
    import shutil
    import threading
    from ouroboros import config, owner_pause
    from ouroboros.server_process import clear_service_binding, record_service_binding

    cli = os.environ.get("OUROBOROS_PLAYWRIGHT_MCP_CLI", "")
    shell = os.environ.get("OUROBOROS_PLAYWRIGHT_HEADLESS_SHELL", "")
    node = shutil.which("node")
    if not (cli and shell and node and "headless-shell" in os.path.basename(shell)):
        pytest.skip("actual upstream consumer is opt-in")
    hits = {"plain": [], "ours": []}
    pages, redirects = {}, {}

    def site(name):
        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                hits[name].append((self.command, self.path, self.rfile.read(length).decode()))
                if self.path in redirects:
                    self.send_response(307)
                    self.send_header("Location", redirects[self.path])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = pages.get(self.path, "<title>ok</title><p>ok</p>").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript" if self.path.endswith(".js") else "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")  # a page fetch reads the answer
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = _answer

            def log_message(self, *_args):
                pass

        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"

    monkeypatch.setattr(owner_pause, "submit_async_preparation", lambda factory, **_kw: factory())
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "pro")
    (plain, plain_url), (ours, ours_url) = site("plain"), site("ours")
    binding = record_service_binding(tmp_path, "main", "127.0.0.1", ours.server_address[1], pid=os.getpid())
    pages.update({
        "/form.html": f"<title>Form</title><form method='post' action='{ours_url}/api/settings'>"
                      "<input type='hidden' name='OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS' value='true'>"
                      "<button id='save'>Save</button></form>",
        "/frame.html": f"<title>Frame</title><iframe src='{ours_url}/frame'></iframe><p>framed</p>",
        "/popup.html": f"<title>Popup</title><button id='open' onclick=\"window.open('{ours_url}/popup')\">Open</button>",
    })
    entry = {"id": "browser", "enabled": True, "transport": "stdio", "command": node, "browser_bridge": True,
             "args": [cli, "--headless", "--isolated", "--browser", "chromium", "--executable-path", shell]}
    instance = mcp_client.MCPManager()
    instance.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [entry]})
    monkeypatch.setattr(mcp_client, "get_manager", lambda: instance)
    state = SimpleNamespace(instance=instance, entry=entry, ctx=_ctx(tmp_path), config=config, hits=hits,
                            plain=plain_url, ours=ours_url, pages=pages, redirects=redirects)
    try:
        yield state
    finally:
        mcp_task_sessions.stop_task(state.ctx)
        for httpd in (plain, ours):
            httpd.shutdown()
            httpd.server_close()
        clear_service_binding(tmp_path, "main", binding)


def _settle(seconds=1.0):
    time.sleep(seconds)  # page-initiated requests finish after the tool returns


def test_actual_upstream_guard_refuses_form_and_fetch_control_posts(upstream, monkeypatch):
    ctx, ours = upstream.ctx, upstream.ours
    assert upstream.instance.refresh_server("browser", authority=ctx)["ok"] is True
    session = _session(ctx)
    assert _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/form.html"}).status == "ok"
    clicked = _dispatch(ctx, "browser_click", {"target": "#save"})  # the page posts the form
    _settle()
    assert ("POST", "/api/settings") not in [hit[:2] for hit in upstream.hits["ours"]]
    assert "BROWSER_OWNER_CONTROL_BLOCKED" in clicked.text, clicked.text
    # The aborted post left Chromium's error page; a page of the site fetches next.
    assert _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/plain.html"}).status == "ok"
    fetch = (f"() => fetch('{ours}/api/owner/safety-mode', {{method: 'POST', body: 'mode=low'}})"
             ".then(r => 'sent ' + r.status, e => 'failed ' + e.message)")
    failed = _dispatch(ctx, "browser_evaluate", {"function": fetch})
    assert "failed" in failed.text and "BROWSER_OWNER_CONTROL_BLOCKED" in failed.text, failed.text
    assert upstream.hits["ours"] == []
    monkeypatch.setattr(upstream.config, "_BOOT_RUNTIME_MODE", "cyber_pro")  # Cyber keeps its reach.
    assert "sent 200" in _dispatch(ctx, "browser_evaluate", {"function": fetch}).text
    assert _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/form.html"}).status == "ok"
    _dispatch(ctx, "browser_click", {"target": "#save"})
    _settle()
    assert [hit[:2] for hit in upstream.hits["ours"]] == [("POST", "/api/owner/safety-mode"), ("POST", "/api/settings")]
    # All of it over the one retained connection, closed with a receipt.
    assert _session(ctx) is session and session.guard_armed and len(_live(session.token)) >= 2
    assert mcp_task_sessions.stop_task(ctx) == [{"server": "browser", "closure": "confirmed"}]
    assert not _live(session.token) and not os.path.exists(session.guard_dir.name)


def test_actual_upstream_guard_refuses_iframe_and_popup_of_a_restricted_child(upstream):
    ctx = upstream.ctx
    ctx.task_constraint = {"mode": "local_readonly_subagent"}
    assert upstream.instance.refresh_server("browser", authority=ctx)["ok"] is True
    framed = _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/frame.html"})
    assert framed.status == "ok" and "BROWSER_LOCAL_READONLY_BLOCKED" in framed.text, framed.text
    assert _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/popup.html"}).status == "ok"
    _dispatch(ctx, "browser_click", {"target": "#open"})  # the page opens a new tab on its own
    _settle()
    assert upstream.hits["ours"] == []  # neither the frame nor the popup's document was requested
    assert [hit[:2] for hit in upstream.hits["plain"]] == [("GET", "/frame.html"), ("GET", "/popup.html")]
    ctx.task_constraint = None  # The parent's reach: the same frame and popup do load.
    _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/frame.html"})
    _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/popup.html"})
    _dispatch(ctx, "browser_click", {"target": "#open"})
    _settle()
    assert [hit[:2] for hit in upstream.hits["ours"]] == [("GET", "/frame"), ("GET", "/popup")]


def test_actual_upstream_owner_init_page_leaves_no_unguarded_route(upstream, tmp_path):
    ctx = upstream.ctx
    own = tmp_path / "own.cjs"
    own.write_text("exports.default = async () => {};")
    upstream.instance.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [
        {**upstream.entry, "args": [*upstream.entry["args"], "--init-page", str(own)]}]})
    assert upstream.instance.refresh_server("browser", authority=ctx)["ok"] is True
    refused = _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/form.html"})
    assert refused.code == "LEGACY_BLOCKED" and "BROWSER_REQUEST_GUARD_UNAVAILABLE" in refused.text
    assert upstream.hits == {"plain": [], "ours": []}


def test_actual_upstream_guard_holds_worker_beacon_and_service_worker_posts(upstream):
    ctx, owner = upstream.ctx, f"{upstream.ours}/api/owner/safety-mode"
    post = f"fetch('{owner}', {{method: 'POST', body: 'mode=low'}})"
    upstream.pages.update({
        "/worker.html": "<title>Worker</title><script>new Worker('/w.js')</script>",
        "/w.js": f"{post}.catch(() => 0);",
        "/sw.html": "<title>SW</title><script>navigator.serviceWorker.register('/sw.js')"
                    ".then(() => navigator.serviceWorker.ready).then(() => document.title = 'ready')</script>",
        "/sw.js": "self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));"
                  f"self.addEventListener('fetch', e => {{ if (e.request.url.endsWith('/via-sw')) e.respondWith({post}); }});",
    })
    assert upstream.instance.refresh_server("browser", authority=ctx)["ok"] is True
    _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/worker.html"})
    _settle()
    beacon = _dispatch(ctx, "browser_evaluate", {"function": f"() => navigator.sendBeacon('{owner}', 'mode=low')"})
    assert "BROWSER_OWNER_CONTROL_BLOCKED" in beacon.text, beacon.text
    _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/sw.html"})
    _settle(2.0)  # the service worker installs and activates
    _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/sw.html"})  # now controlled by it
    answered = _dispatch(ctx, "browser_evaluate", {
        "function": f"() => fetch('{upstream.plain}/via-sw').then(r => 'sent ' + r.status, e => 'failed ' + e.message)"})
    assert "failed" in answered.text and "BROWSER_OWNER_CONTROL_BLOCKED" in answered.text, answered.text
    _settle()
    assert upstream.hits["ours"] == []  # the worker's, the beacon's and the service worker's posts
    assert "/w.js" in [hit[1] for hit in upstream.hits["plain"]]


def test_actual_upstream_redirect_hop_and_websocket_pass_unjudged(upstream):
    """The disclosed boundary, pinned so a change upstream shows up: Playwright
    routes no native redirect hop and no WebSocket, here or in tools/browser.py."""
    ctx, owner = upstream.ctx, f"{upstream.ours}/api/owner/safety-mode"
    upstream.redirects["/hop"] = owner
    upstream.pages["/hop.html"] = (f"<title>Hop</title><form method='post' action='{upstream.plain}/hop'>"
                                   "<input name='mode' value='low'><button id='go'>Go</button></form>")
    assert upstream.instance.refresh_server("browser", authority=ctx)["ok"] is True
    _dispatch(ctx, "browser_navigate", {"url": f"{upstream.plain}/hop.html"})
    hopped = _dispatch(ctx, "browser_click", {"target": "#go"})
    _settle()
    assert ("POST", "/api/owner/safety-mode", "mode=low") in upstream.hits["ours"]  # the 307 kept the body
    assert "BROWSER_REQUEST_BLOCKED" not in hopped.text
    socket_url = upstream.ours.replace("http://", "ws://") + "/ws"
    opened = _dispatch(ctx, "browser_evaluate", {"function": f"() => new Promise(done => {{ const w = new WebSocket("
                       f"'{socket_url}'); w.onerror = w.onopen = () => done('tried'); }})"})
    _settle()
    assert "tried" in opened.text and ("GET", "/ws", "") in upstream.hits["ours"]
