"""Task browser bridge: session lifetime, Settings generation and call outcome.

Each test forces one interleaving the source admits and checks the fact a
caller reads afterwards: whether a process launched and carried the session's
marker, what closure was reported and when, which catalog survived, and what a
call that lost its connection says about its effect. The synthetic server
records every launch and every action in files beside its script, so "never
launched" and "submitted once" are observations, not inferences.
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from ouroboros import mcp_client, mcp_task_sessions
from ouroboros.process_containment import ProcessContainer, pid_is_zombie, pids_with_env_marker

pytestmark = [
    pytest.mark.serial,  # Real subprocesses (tests/conftest lane policy).
    pytest.mark.skipif(sys.platform == "win32", reason="Bridge transport and marker custody are POSIX-only"),
]


SERVER = r'''
import json, os, pathlib, socket, subprocess, sys
from mcp.server.fastmcp import FastMCP

here = pathlib.Path(__file__).parent
with open(here / "launches.log", "a") as log:  # Every launch, with the markers it inherited.
    log.write(json.dumps(sorted(k for k in os.environ if k.startswith("OURO_PROC_CONTAINER_"))) + "\n")

mcp = FastMCP("Playwright")
state = {"browser": None}

def ensure_browser():
    # Like upstream: a detached "browser" on first use, then the host's init page.
    if state["browser"] is None:
        state["browser"] = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        with socket.socket(socket.AF_UNIX) as conn:
            conn.connect(os.environ["OUROBOROS_BROWSER_GUARD"])
            conn.sendall(b'{"armed": true}\n')
            conn.shutdown(socket.SHUT_WR)
            b"".join(iter(lambda: conn.recv(65536), b""))

@mcp.tool(structured_output=False)
def browser_tabs(action: str) -> str:
    if (here / "die-on-tabs").exists():
        os._exit(0)  # The connection closes before the host has read the page.
    ensure_browser()
    return "### Result\n- 0: (current) [Start](http://localhost:8765/)"

@mcp.tool(structured_output=False)
def browser_click(target: str) -> str:
    with open(here / "effects.log", "a") as effects:
        effects.write(target + "\n")
    if target == "then-exit":
        os._exit(0)  # The effect happened; the answer never leaves.
    if target == "stuck":
        raise RuntimeError("TimeoutError: locator.click: Timeout 5000ms exceeded. "
                           "waiting for element to be visible, enabled and stable")
    return "### Page\n- Page URL: http://localhost:8765/\n- Page Title: Start"

mcp.run(transport="stdio")
'''


def _server(path):
    script = path / "server.py"
    script.write_text(textwrap.dedent(SERVER))
    return script


def _entry(script, *extra):
    return {"id": "browser", "enabled": True, "transport": "stdio",
            "command": sys.executable, "args": [str(script), *extra], "browser_bridge": True}


def _ctx(root, attempt=1):
    return SimpleNamespace(task_id=f"lifecycle-{root.name}", task_attempt=attempt,
                           task_lifecycle_bound=True, drive_root=root,
                           task_metadata={}, task_contract={}, messages=[])


def _lines(path):
    return path.read_text().splitlines() if path.exists() else []


def _live(marker):
    return [pid for pid in pids_with_env_marker(marker) or [] if not pid_is_zombie(pid)]


def _wait_gone(marker, seconds=4.0):
    deadline = time.monotonic() + seconds
    while _live(marker) and time.monotonic() < deadline:
        time.sleep(.05)
    return _live(marker)


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    """A Settings-owned manager on the synthetic server; URL policy and Safety allow."""
    pytest.importorskip("mcp")
    from ouroboros import owner_pause

    monkeypatch.setattr(mcp_task_sessions.browser_policy, "browser_url_block_reason", lambda *_a, **_k: "")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(owner_pause, "submit_async_preparation", lambda factory, **_kw: factory())
    script = _server(tmp_path)
    instance = mcp_client.MCPManager()
    instance.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [_entry(script)]})
    monkeypatch.setattr(mcp_client, "get_manager", lambda: instance)
    ctx = _ctx(tmp_path)
    state = SimpleNamespace(instance=instance, ctx=ctx, script=script, root=tmp_path,
                            cfg=mcp_client.normalize_server_config(_entry(script)),
                            launches=tmp_path / "launches.log", effects=tmp_path / "effects.log")
    yield state
    mcp_task_sessions.stop_task(ctx)


def _dispatch(ctx, tool, args):
    from ouroboros.tools.extension_dispatch import _dispatch_mcp_tool_result

    return _dispatch_mcp_tool_result(ctx, mcp_client.make_tool_name("browser", tool), args)


# -- start and revocation share one admission -------------------------------------


def test_stop_before_start_never_launches_the_published_session(bridge, monkeypatch):
    """Discovery publishes before it starts; a Stop in that gap leaves nothing to launch."""
    session_type = mcp_task_sessions.TaskBrowserSession
    real_start = session_type.start
    seen = {}

    def stop_arrives_first(self, timeout):
        seen["session"] = self
        seen["closure"] = mcp_task_sessions.stop_task(bridge.ctx)
        return real_start(self, timeout)

    monkeypatch.setattr(session_type, "start", stop_arrives_first)
    with pytest.raises(RuntimeError, match="revoked before it started"):
        mcp_task_sessions.discover(bridge.cfg, bridge.ctx, 15)
    assert seen["closure"] == [{"server": "browser", "closure": "confirmed"}]
    assert seen["session"].thread.ident is None  # Its worker never started ...
    assert _lines(bridge.launches) == []  # ... so no server ever launched.
    assert not _live(seen["session"].token)


def test_stop_during_launch_confirms_only_after_the_worker_unwinds(bridge, monkeypatch):
    """A launch admitted before Stop is scanned again, on its own marker, once it has unwound."""
    real_factory = mcp_client._transport_factory
    entered, release = threading.Event(), threading.Event()

    @asynccontextmanager
    async def held_launch(cfg):
        import asyncio

        entered.set()
        await asyncio.to_thread(release.wait, 30)
        async with real_factory(cfg) as streams:
            yield streams

    monkeypatch.setattr(mcp_client, "_transport_factory", held_launch)
    monkeypatch.setattr(mcp_task_sessions, "_CLOSE_GRACE_SEC", 0.2)
    monkeypatch.setattr(mcp_task_sessions, "_EXIT_GRACE_SEC", 0.2, raising=False)
    opened = {}
    discovery = threading.Thread(target=lambda: opened.setdefault("error", _raises(
        lambda: mcp_task_sessions.discover(bridge.cfg, bridge.ctx, 30))), daemon=True)
    discovery.start()
    assert entered.wait(15)
    session = mcp_task_sessions._sessions[(bridge.ctx.task_id, "browser")]

    closure = mcp_task_sessions.stop_task(bridge.ctx)
    assert closure == [{"server": "browser", "closure": "unconfirmed: session worker did not exit"}]
    assert mcp_task_sessions._sessions.get(session.key) is session

    release.set()  # The admitted launch goes ahead, then sees the revocation and unwinds.
    session.thread.join(timeout=20)
    discovery.join(timeout=20)
    assert not session.thread.is_alive() and "revoked while starting" in repr(opened["error"])
    assert _lines(bridge.launches) == [json.dumps([session.token])]  # It carried the original marker.
    assert session.close_outcome == "confirmed"
    assert not _live(session.token)
    retry = _ctx(bridge.root, attempt=2)  # The confirmed closure no longer blocks a later attempt.
    assert "browser_tabs" in {tool["name"] for tool in mcp_task_sessions.discover(bridge.cfg, retry, 15)}
    mcp_task_sessions.stop_task(retry)


def _raises(fn):
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - the test reads which one
        return exc
    return None


def test_uncertain_scan_is_rescanned_on_the_original_marker(bridge, monkeypatch):
    """``reap`` consumes its marker: a later scan must be a fresh one, never an empty repeat."""
    scanned = []
    real_reap = ProcessContainer.reap

    def recorded(self):
        scanned.append(self._token)
        return real_reap(self)

    monkeypatch.setattr(ProcessContainer, "reap", recorded)
    mcp_task_sessions.discover(bridge.cfg, bridge.ctx, 15)
    mcp_task_sessions.call(bridge.cfg, "mcp_browser__browser_tabs", "browser_tabs", {"action": "list"},
                           bridge.ctx, 10)
    session = mcp_task_sessions._sessions[(bridge.ctx.task_id, "browser")]
    assert len(_live(session.token)) >= 2  # the server and its detached "browser"
    # The first scan cannot tell and signals nothing: the detached member lives on.
    session.container.reap = lambda: "process identity could not be confirmed"

    assert mcp_task_sessions.stop_task(bridge.ctx) == [{"server": "browser", "closure": "confirmed"}]
    assert not _wait_gone(session.token)
    assert scanned and set(scanned) == {session.token}


# -- which exit ends the attempt, and who owns it -----------------------------------


def test_direct_turn_without_an_attempt_is_the_initial_attempt():
    bound = SimpleNamespace(task_id="direct-1", task_attempt=None, task_lifecycle_bound=True)
    assert mcp_task_sessions._owner(bound) == ("direct-1", "1")
    assert mcp_task_sessions._owner(SimpleNamespace(**{**vars(bound), "task_attempt": 2})) == ("direct-1", "2")
    with pytest.raises(RuntimeError, match="bound task"):
        mcp_task_sessions._owner(SimpleNamespace(**{**vars(bound), "task_lifecycle_bound": False}))


@pytest.mark.parametrize("attempt", [1, None])
def test_registry_startup_discovers_and_reuses_task_bridge(bridge, monkeypatch, attempt):
    """The agent's public schema and tool path on a cold worker; a direct turn has no attempt number."""
    from contextlib import contextmanager, nullcontext
    from functools import wraps

    from ouroboros import owner_pause
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    @contextmanager
    def handoff(*_args):  # The synthetic task has no durable owner-pause result row.
        yield {}

    monkeypatch.setattr(owner_pause, "tool_handoff", handoff)
    monkeypatch.setattr(owner_pause, "operation_start", nullcontext)
    sdk_session, identities = mcp_client.ClientSession, []

    @wraps(sdk_session)
    def recorded_session(*args, **kwargs):
        identities.append(kwargs["client_info"])
        return sdk_session(*args, **kwargs)

    monkeypatch.setattr(mcp_client, "ClientSession", recorded_session)
    ctx = ToolContext(repo_dir=bridge.root, drive_root=bridge.root, task_id=bridge.ctx.task_id,
                      task_lifecycle_bound=True, messages=[])
    ctx.task_attempt, ctx.is_direct_chat = attempt, attempt is None
    registry = ToolRegistry(repo_dir=bridge.root, drive_root=bridge.root)
    registry.set_context(ctx)
    name = "mcp_browser__browser_click"
    assert name in {schema["function"]["name"] for schema in registry.schemas()}
    results = [registry.execute_result(name, {"target": target}) for target in ("e1", "e2")]
    assert [result.status for result in results] == ["ok", "ok"], [result.text for result in results]
    assert _lines(bridge.effects) == ["e1", "e2"] and len(_lines(bridge.launches)) == 1
    assert len(identities) == 1 and identities[0].name == "Ouroboros" and identities[0].version
    assert mcp_task_sessions.stop_task(ctx) == [{"server": "browser", "closure": "confirmed"}]


def test_pausing_cleanup_closes_bridge_but_reopens_same_attempt(bridge, monkeypatch):
    from ouroboros import loop_budget
    from ouroboros.tools import services

    ctx = bridge.ctx
    monkeypatch.setattr(services, "stop_task_services", lambda _ctx: [])
    assert bridge.instance.refresh_server("browser", authority=ctx)["ok"]
    assert _dispatch(ctx, "browser_click", {"target": "before"}).status == "ok"
    ctx._budget_pausing = True
    loop_budget._cleanup_loop_resources(None, loop_budget._LoopExitContext(
        tools=SimpleNamespace(_ctx=ctx), drive_root=bridge.root, task_id=ctx.task_id,
        event_queue=None, drive_logs=bridge.root / "logs", accumulated_usage={}, llm_trace={},
    ))
    assert (ctx.task_id, "browser") not in mcp_task_sessions._sessions
    assert (ctx.task_id, "1") not in mcp_task_sessions._ended_attempts
    assert bridge.instance.refresh_server("browser", authority=ctx)["ok"]
    assert _dispatch(ctx, "browser_click", {"target": "after"}).status == "ok"
    assert _lines(bridge.effects) == ["before", "after"] and len(_lines(bridge.launches)) == 2


WORKER = r'''
import json, pathlib, sys
from types import SimpleNamespace
from ouroboros import browser_policy, loop_budget, mcp_client, mcp_task_sessions, safety
from ouroboros.tools import services

root = pathlib.Path(sys.argv[1])
ctx = SimpleNamespace(task_id="cross-worker-pause", task_attempt=1, task_lifecycle_bound=True,
                      drive_root=root, task_metadata={}, task_contract={}, messages=[])
safety.check_safety = lambda *a, **k: (True, "")
browser_policy.browser_url_block_reason = lambda *a, **k: ""
cfg = mcp_client.normalize_server_config({"id": "browser", "enabled": True, "transport": "stdio",
    "command": sys.executable, "args": [str(root / "server.py")], "browser_bridge": True})
try:
    mcp_task_sessions.discover(cfg, ctx, 15)
    result, _, _ = mcp_task_sessions.call(cfg, "mcp_browser__browser_click", "browser_click",
                                          {"target": sys.argv[2]}, ctx, 10)
    session = mcp_task_sessions._sessions[(ctx.task_id, "browser")]
    if sys.argv[2] == "pause":
        ctx._budget_pausing = True
        services.stop_task_services = lambda *a: []
        finalize = loop_budget._finalize_task_services
        loop_budget._loop = lambda: SimpleNamespace(_finalize_task_services=finalize,
                                                    _emit_checkpoint_event=lambda *a: None)
        loop_budget._cleanup_loop_resources(None, loop_budget._LoopExitContext(
            tools=SimpleNamespace(_ctx=ctx), drive_root=root, task_id=ctx.task_id, event_queue=None,
            drive_logs=root / "logs", accumulated_usage={}, llm_trace={}))
        print(json.dumps({"marker": session.token, "closed": session.close_outcome}), flush=True)
        sys.stdin.readline()  # The parked worker stays alive across the successor's open.
    else:
        print(json.dumps({"status": result.status, "closure": mcp_task_sessions.stop_task(ctx)}), flush=True)
finally:
    mcp_task_sessions.stop_task(ctx)
'''


def test_paused_attempt_can_resume_in_another_process(bridge):
    """A parked worker remains alive while the same attempt opens in a successor process."""
    worker = bridge.root / "worker.py"
    worker.write_text(textwrap.dedent(WORKER))
    # Workers import this candidate, not a different installed package.
    env = {**os.environ, "PYTHONPATH": str(pathlib.Path(__file__).resolve().parents[1])}
    parked = subprocess.Popen([sys.executable, str(worker), str(bridge.root), "pause"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env)
    try:
        row = json.loads(parked.stdout.readline())
        assert row["closed"] == "confirmed"
        assert parked.poll() is None and not _live(row["marker"])
        successor = subprocess.run([sys.executable, str(worker), str(bridge.root), "resume"],
                                   capture_output=True, text=True, env=env, timeout=45, check=True)
        result = json.loads(successor.stdout)
        assert result == {"status": "ok", "closure": [{"server": "browser", "closure": "confirmed"}]}
        assert _lines(bridge.effects) == ["pause", "resume"] and len(_lines(bridge.launches)) == 2
        assert parked.poll() is None
    finally:
        parked.communicate("exit\n", timeout=15)
        assert parked.returncode == 0


# -- a Settings change racing a discovery's open -----------------------------------


@pytest.mark.parametrize("change", ["changed", "removed", "mcp_disabled"])
def test_refresh_opened_after_reconfigure_closes_exactly_its_stale_session(bridge, monkeypatch, change):
    """Settings land after the refresh read its entry, before its open: revocation saw nothing."""
    real_discover = mcp_task_sessions.discover
    replacement = _entry(bridge.script, "--replacement")
    settings = {
        "changed": {"MCP_ENABLED": True, "MCP_SERVERS": [replacement]},
        "removed": {"MCP_ENABLED": True, "MCP_SERVERS": []},
        "mcp_disabled": {"MCP_ENABLED": False, "MCP_SERVERS": [_entry(bridge.script)]},
    }[change]
    opened = {}

    def settings_saved_first(cfg, ctx, timeout):
        bridge.instance.reconfigure(settings)
        tools = real_discover(cfg, ctx, timeout)
        opened["token"] = mcp_task_sessions._sessions[(ctx.task_id, cfg.id)].token
        return tools

    monkeypatch.setattr(mcp_task_sessions, "discover", settings_saved_first)
    result = bridge.instance.refresh_server("browser", authority=bridge.ctx)
    assert result["ok"] is False and "stale MCP refresh discarded" in result["error"]
    assert result["bridge_closure"] == "confirmed"
    assert not _wait_gone(opened["token"])
    assert (bridge.ctx.task_id, "browser") not in mcp_task_sessions._sessions
    monkeypatch.setattr(mcp_task_sessions, "discover", real_discover)
    if change == "changed":  # The current entry opens in the same attempt.
        assert bridge.instance.refresh_server("browser", authority=bridge.ctx)["ok"] is True
        held = mcp_task_sessions._sessions[(bridge.ctx.task_id, "browser")]
        assert held.cfg.args[-1] == "--replacement"
    assert len(_lines(bridge.launches)) == (2 if change == "changed" else 1)


def test_stale_refresh_failure_keeps_the_replacement_catalog():
    manager = mcp_client.MCPManager()
    old = {"id": "plain", "enabled": True, "transport": "streamable_http", "url": "http://127.0.0.1:9/old"}
    new = {**old, "url": "http://127.0.0.1:9/new"}
    manager.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [old]})

    async def listing(cfg, _timeout):
        if cfg.url.endswith("/old"):
            # A replacement is saved and listed while this listing is still out.
            manager.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [new]})
            assert manager.refresh_server("plain")["ok"] is True
            raise ConnectionError("the old endpoint went away")
        return [{"name": "fresh", "description": "", "input_schema": {}}]

    manager._async_list_tools = listing
    assert manager.refresh_server("plain")["ok"] is False
    assert [tool["raw_name"] for tool in manager.list_tools_for_registry()] == ["fresh"]
    status = next(row for row in manager.status_payload()["servers"] if row["id"] == "plain")
    assert status["last_error"] == "" and status["last_refreshed"]


# -- what a call that lost its connection says --------------------------------------


def test_action_effect_then_eof_reports_unknown_effect_once_and_closes(bridge):
    ctx = bridge.ctx
    assert bridge.instance.refresh_server("browser", authority=ctx)["ok"] is True
    session = mcp_task_sessions._sessions[(ctx.task_id, "browser")]

    result = _dispatch(ctx, "browser_click", {"target": "then-exit"})
    assert (result.status, result.code) == ("error", "MCP_ERROR")
    assert "BROWSER_ACTION_OUTCOME_UNKNOWN" in result.text
    assert "was submitted" in result.text and "it was not resent" in result.text
    assert "Whether it took effect is unknown" in result.text
    assert "The task session was closed: confirmed" in result.text
    assert {key: result.meta[key] for key in ("dispatch", "effect", "transport", "bridge_closure")} == {
        "dispatch": "possible", "effect": "unknown", "transport": "closed", "bridge_closure": "confirmed"}
    assert _lines(bridge.effects) == ["then-exit"]
    assert not _wait_gone(session.token)
    # The server ended the session, not the host: this attempt does not respawn it.
    assert bridge.instance.refresh_server("browser", authority=ctx)["ok"] is False
    with pytest.raises(RuntimeError, match="did not open in this task attempt"):
        mcp_task_sessions.discover(session.cfg, ctx, 15)
    assert _lines(bridge.effects) == ["then-exit"] and len(_lines(bridge.launches)) == 1


def test_connection_closing_before_the_action_reports_it_not_dispatched(bridge):
    ctx = bridge.ctx
    assert bridge.instance.refresh_server("browser", authority=ctx)["ok"] is True
    session = mcp_task_sessions._sessions[(ctx.task_id, "browser")]
    (bridge.root / "die-on-tabs").write_text("")

    result = _dispatch(ctx, "browser_click", {"target": "e1"})
    assert (result.status, result.code) == ("blocked", "LEGACY_BLOCKED")
    assert "connection closed during 'browser_tabs'" in result.text
    assert "The task session was closed: confirmed; the action was not dispatched" in result.text
    assert _lines(bridge.effects) == []
    assert session.revoked and not _wait_gone(session.token)


def test_tool_error_answer_keeps_the_session_reusable(bridge):
    """A fast actionability failure is the server's answer, not a broken transport."""
    ctx = bridge.ctx
    assert bridge.instance.refresh_server("browser", authority=ctx)["ok"] is True
    session = mcp_task_sessions._sessions[(ctx.task_id, "browser")]

    failed = _dispatch(ctx, "browser_click", {"target": "stuck"})
    assert (failed.status, failed.code) == ("error", "MCP_ERROR")
    assert failed.meta["mcp_is_error"] is True and "enabled and stable" in failed.text
    assert "BROWSER_ACTION_OUTCOME_UNKNOWN" not in failed.text and "transport" not in failed.meta
    assert not session.revoked

    clicked = _dispatch(ctx, "browser_click", {"target": "e1"})
    assert clicked.status == "ok"
    assert mcp_task_sessions._sessions[(ctx.task_id, "browser")] is session
    assert _lines(bridge.effects) == ["stuck", "e1"] and len(_lines(bridge.launches)) == 1
