"""Task browser bridge: upstream page state, invocation authority and custody.

The synthetic server speaks only what the pinned ``@playwright/mcp@0.0.82``
speaks: ``browser_tabs`` answers ``### Result`` with
``renderTabsMarkdown`` lines and page tools answer ``### Page`` sections
(playwright-core ``tools/backend/response.ts``). Like upstream, it starts its
"browser" lazily as a detached process group. The opt-in last test drives the
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


SERVER = r'''
import subprocess, sys, time
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Playwright")
tabs = [{"title": "Start", "url": "http://localhost:8765/"}]
state = {"current": 0, "browser": None, "redirect_in": 0, "snapshots": 0, "clicks": 0}

def ensure_browser():
    # Upstream launches its browser on first use, detached into its own group.
    if state["browser"] is None:
        state["browser"] = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)

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


# -- custody: task end, reopen, Panic and cancel ----------------------------------


def test_task_end_reaps_detached_descendant_and_ends_the_attempt(tmp_path, monkeypatch, policy, safety):
    pytest.importorskip("mcp")
    from ouroboros import loop_budget
    from ouroboros.tools import services

    cfg, ctx = mcp_client.normalize_server_config(_entry(_server(tmp_path))), _ctx(tmp_path)
    events = []
    monkeypatch.setattr(services, "stop_task_services", lambda _ctx: [])
    monkeypatch.setattr(loop_budget, "_loop", lambda: SimpleNamespace(
        _emit_checkpoint_event=lambda *args: events.append(args[-1])))
    try:
        assert "browser_tabs" in {tool["name"] for tool in mcp_task_sessions.discover(cfg, ctx, 15)}
        mcp_task_sessions.call(cfg, "mcp_browser__browser_snapshot", "browser_snapshot", {}, ctx, 10)
        marker = _session(ctx).token
        assert len(_live(marker)) >= 2  # the server and its detached "browser"
    finally:
        loop_budget._finalize_task_services(loop_budget._LoopExitContext(
            tools=SimpleNamespace(_ctx=ctx), drive_root=tmp_path, task_id=ctx.task_id,
            event_queue=None, drive_logs=tmp_path / "logs", accumulated_usage={}, llm_trace={},
        ))
    assert events[0]["bridges"] == [{"server": "browser", "closure": "confirmed"}]
    assert not _live(marker)
    with pytest.raises(RuntimeError, match="ended"):
        mcp_task_sessions.discover(cfg, ctx, 15)


def test_unconfirmed_close_refuses_reopen_by_a_later_attempt(tmp_path, policy, safety):
    pytest.importorskip("mcp")
    cfg, ctx = mcp_client.normalize_server_config(_entry(_server(tmp_path))), _ctx(tmp_path)
    mcp_task_sessions.discover(cfg, ctx, 15)
    session = _session(ctx)
    actual_reap = session.container.reap

    def uncertain_reap():
        actual_reap()  # keep the fixture clean while reporting no proof
        return "process identity could not be confirmed"

    session.container.reap = uncertain_reap
    assert mcp_task_sessions.stop_task(ctx)[0]["closure"].startswith("unconfirmed:")
    assert _session(ctx) is session
    retry = _ctx(tmp_path, attempt=2)
    with pytest.raises(RuntimeError, match="before its earlier closure is confirmed"):
        mcp_task_sessions.discover(cfg, retry, 15)
    with pytest.raises(PermissionError, match="not discovered by this task attempt"):
        mcp_task_sessions.call(cfg, "mcp_browser__browser_snapshot", "browser_snapshot", {}, retry, 10)


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
        navigated, _advice = mcp_task_sessions.call(
            cfg, "mcp_browser__browser_navigate", "browser_navigate", {"url": f"{base}/a.html"}, ctx, 60)
        assert navigated.status == "ok", navigated.text
        snap, _advice = mcp_task_sessions.call(
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
