"""Task-session MCP over a real stdio server process: one initialized session per live task loop.

Every call goes through the real ToolRegistry, the loop's tool executor (thread hop,
custody rows) and the real MCP SDK client against a small SDK server process that
logs its own starts, calls and answered pings. Default servers and discovery keep
their per-call processes; a task session keeps state, closes on both loop-cleanup
paths, on a settings change and after an unanswered call, and never resends.
"""
from __future__ import annotations

import copy
import json
import sys
import time

import pytest

from ouroboros import mcp_catalog, mcp_client, mcp_task_session, owner_pause
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import claims, owner, run


SERVER = r'''
import asyncio, json, os, socket, sys, time
import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

LOG = sys.argv[1]
server = Server("task-session-fixture")
state = {"count": 0, "pinger": None, "pages": ["about:blank"], "selected": 0}
OBJECT = {"type": "object", "properties": {}}


def note(**row):
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "at": time.time(), **row}) + "\n")


# Lifecycle switches (tests/test_mcp_task_session_lifecycle.py): a file beside the log turns one on, and only for
# a task session's process (it carries the task-scope marker), so discovery and the other tests never see them.
IN_SESSION = any(key.startswith("OURO_TASK_SCOPE_") for key in os.environ)


def flag(name):
    return IN_SESSION and os.path.exists(os.path.join(os.path.dirname(LOG), name))


# Failed-test cleanup is addressed to this fixture through its private directory, never a PID/name scan.
TEARDOWN = os.path.join(os.path.dirname(LOG), "teardown")
def teardown_watch():
    while not os.path.exists(TEARDOWN):
        time.sleep(0.05)
    os._exit(0)
import threading
threading.Thread(target=teardown_watch, daemon=True).start()

STDOUT = os.fdopen(os.dup(1), "w")  # survives the SDK's stdio wrapper, for the farewell
if flag("hang_init"):  # never answers initialize
    note(event="hang_init")
    time.sleep(3600)
if flag("stubborn"):  # ignores SIGTERM and lingers after stdin EOF: only SIGKILL ends it
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if flag("child"):  # a descendant in the server's process group: only the SDK's group escalation reaches it
    import subprocess
    child_code = "import os,sys,time\nend=time.monotonic()+60\nwhile time.monotonic()<end and not os.path.exists(sys.argv[1]): time.sleep(0.05)"
    note(event="child", child=subprocess.Popen([sys.executable, "-c", child_code, TEARDOWN],
                                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL).pid)


@server.list_tools()
async def tools():
    note(event="list")
    return [types.Tool(name="bump", description="counts", inputSchema=OBJECT),
            types.Tool(name="hang", description="never answers", inputSchema=OBJECT),
            types.Tool(name="die", description="the server exits during the call", inputSchema=OBJECT),
            types.Tool(name="gate", description="answers once a file named release exists", inputSchema=OBJECT),
            types.Tool(name="fail", description="answers isError", inputSchema=OBJECT),
            types.Tool(name="garble", description="writes bytes no UTF-8 reader accepts", inputSchema=OBJECT),
            types.Tool(name="browser_navigate", description="sets the selected page's URL", inputSchema=OBJECT),
            types.Tool(name="browser_tabs", description="opens or selects a page", inputSchema=OBJECT),
            types.Tool(name="browser_snapshot", description="reads the selected page", inputSchema=OBJECT),
            types.Tool(name="guard", description="asks the host guard",
                       inputSchema={"type": "object", "properties": {"questions": {"type": "array"}}})]


async def pings(session):
    while True:
        await session.send_ping()
        note(event="pong")
        await anyio.sleep(0.2)


def ask(facts):
    host, port = os.environ["OUROBOROS_BROWSER_GUARD"].rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=20) as conn:
        conn.sendall(json.dumps(facts).encode("utf-8") + b"\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data)


@server.call_tool()
async def call(name, arguments):
    note(event="call", tool=name)
    if state["pinger"] is None and not flag("stubborn"):  # sends nothing unasked; tests pick late ones (farewell)
        state["pinger"] = asyncio.get_running_loop().create_task(pings(server.request_context.session))
    if name == "bump":
        state["count"] += 1
        return [types.TextContent(type="text", text=json.dumps({"pid": os.getpid(), "count": state["count"]}))]
    if name == "browser_navigate":
        state["pages"][state["selected"]] = arguments["url"]
    elif name == "browser_tabs":
        if arguments["action"] == "new":
            state["pages"].append("about:blank")
            state["selected"] = len(state["pages"]) - 1
        elif arguments["action"] == "select":
            state["selected"] = arguments["index"]
    if name in {"browser_navigate", "browser_tabs", "browser_snapshot"}:
        return [types.TextContent(type="text", text=json.dumps({
            "pid": os.getpid(), "selected": state["selected"],
            "url": state["pages"][state["selected"]], "pages": state["pages"]}))]
    if name == "hang":
        await anyio.sleep(3600)
    if name == "die":
        os._exit(3)
    if name == "garble":  # the client's reader fails on its own, not on a stream a shutdown closed
        os.write(1, b"\xff\xfe\n")
        await anyio.sleep(3600)
    if name == "gate":
        while not flag("release"):
            await anyio.sleep(0.05)
        return [types.TextContent(type="text", text=json.dumps({"pid": os.getpid(), "gate": "open"}))]
    if name == "fail":
        raise ValueError("fixture says no")
    if name == "guard":
        answers = []
        for question in arguments.get("questions") or []:
            note(event="ask")
            answers.append(await anyio.to_thread.run_sync(
                ask, {"token": os.environ["OUROBOROS_BROWSER_GUARD_TOKEN"], **question}))
            note(event="answered")
        return [types.TextContent(type="text", text=json.dumps(answers))]
    raise ValueError("unknown tool")


async def main():
    note(event="start", guard=os.environ.get("OUROBOROS_BROWSER_GUARD", ""),
         token=bool(os.environ.get("OUROBOROS_BROWSER_GUARD_TOKEN")),
         task_scope=any(key.startswith("OURO_TASK_SCOPE_") for key in os.environ))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


anyio.run(main)
if flag("farewell"):  # a message the client can only meet after its session ended
    STDOUT.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/message",
                             "params": {"level": "info", "data": "goodbye"}}) + "\n")
    STDOUT.flush()
if flag("stubborn"):
    time.sleep(30)
'''


@pytest.fixture(autouse=True)
def _private_mcp(tmp_path, monkeypatch):
    from ouroboros import config

    assert mcp_client._MCP_SDK_AVAILABLE, "the real MCP SDK session is the subject under test"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    mcp_client.reset_manager_for_tests()
    yield
    mcp_client.reset_manager_for_tests()


def fixture_server(tmp_path, server_id="fx", **extra):
    script = tmp_path / "task_session_server.py"
    script.write_text(SERVER, encoding="utf-8")
    log = tmp_path / f"{server_id}.jsonl"
    return {"id": server_id, "enabled": True, "transport": "stdio", "command": sys.executable,
            "args": [str(script), str(log)], "cwd": str(tmp_path), **extra}, log


def wire(monkeypatch, registry, *servers, timeout=20):
    from ouroboros import config

    settings = {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": max(20, timeout), "MCP_SERVERS": list(servers)}
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    manager = mcp_client.get_manager()
    manager.reconfigure(settings)
    for server in servers:
        assert manager.refresh_server(server["id"])["ok"]
    if settings["MCP_TOOL_TIMEOUT_SEC"] != timeout:
        settings["MCP_TOOL_TIMEOUT_SEC"] = timeout
        manager.reconfigure(settings)
    monkeypatch.setattr(registry, "_mcp_name_miss", lambda _name: None)
    return manager, settings


def call(registry, tool, args=None, *, server_id="fx", timeout=40):
    return run(registry, name=f"mcp_{server_id}__{tool}", args=args or {"n": 1}, timeout=timeout)["tool_result"]


def body(result):
    """The server's JSON answer: the last part of the producer text, before any trailing host note."""
    return json.loads((result.text if result.producer_text is None else result.producer_text).rsplit("\n\n", 1)[1])


RESTARTED = "was started anew for this call"  # the one host note of a replacement session's first answered call


def restarted(result):
    """That host note's ended-session reason, or None when the result carries no such note."""
    notes = [note for note in result.host_annotations if RESTARTED in note]
    assert len(notes) <= 1 and all(result.text.endswith(note) for note in notes), result
    return notes[0].split("ended (", 1)[1].split("), and this connection", 1)[0] if notes else None


def rows(log, event=None):
    found = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return [row for row in found if event is None or row["event"] == event]


def alive(pid):
    from ouroboros.platform_layer import pid_is_alive

    return pid_is_alive(pid)  # never os.kill(pid, 0): on Windows signal 0 is a console CTRL_C_EVENT


def eventually(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def session_of(registry, server_id="fx"):
    return vars(registry._ctx)[mcp_task_session._SLOT].sessions[server_id]


def test_one_session_per_task_loop_while_default_calls_and_listing_stay_per_call(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    session_raw, session_log = fixture_server(tmp_path, "fx", task_session=True)
    plain_raw, plain_log = fixture_server(tmp_path, "plain")
    wire(monkeypatch, registry, session_raw, plain_raw)
    discovery = rows(session_log, "start")
    assert len(discovery) == 1 and discovery[0]["guard"] == "" and discovery[0]["task_scope"] is False
    with owner(registry):
        answered = [call(registry, "bump") for _ in range(3)]
        plain = [body(call(registry, "bump", server_id="plain")) for _ in range(2)]
    counted = [body(result) for result in answered]
    assert not any(result.host_annotations for result in answered)  # a loop's first session replaced nothing
    assert [row["count"] for row in counted] == [1, 2, 3] and len({row["pid"] for row in counted}) == 1
    assert [row["count"] for row in plain] == [1, 1] and len({row["pid"] for row in plain}) == 2
    starts = rows(session_log, "start")
    assert len(starts) == 2 and starts[1]["pid"] == counted[0]["pid"]
    assert starts[1]["task_scope"] is True and starts[1]["guard"] == "" and starts[1]["token"] is False
    assert alive(counted[0]["pid"])
    assert not claims(tmp_path)  # every answered call discharged its MCP duty
    mcp_task_session.close_context_sessions(registry._ctx)
    assert session_of(registry).closure == "closed" and not alive(counted[0]["pid"])


def test_server_error_result_keeps_the_session(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw)
    with owner(registry):
        first = body(call(registry, "bump"))
        failed = call(registry, "fail")
        after = body(call(registry, "bump"))
    assert failed.status == "error" and failed.meta["mcp_is_error"] is True and "fixture says no" in failed.text
    assert after == {"pid": first["pid"], "count": 2}
    mcp_task_session.close_context_sessions(registry._ctx)


def test_timeout_after_send_closes_the_session_and_never_resends(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    manager, settings = wire(monkeypatch, registry, raw)
    with owner(registry):
        first = body(call(registry, "bump", timeout=15))  # a server's start is not the budget under test
        manager.reconfigure({**settings, "MCP_TOOL_TIMEOUT_SEC": 2})  # only the hang call's budget; same identity
        started = time.monotonic()
        hung = call(registry, "hang", timeout=15)
        waited = time.monotonic() - started
        assert hung.status == "timeout" and hung.meta["mcp_call_not_sent"] is False, hung
        assert "remote outcome is unknown" in hung.text and waited < 2 + 3  # inside the loop's outer margin
        assert eventually(lambda: not alive(first["pid"]))
        remaining = claims(tmp_path)
        assert len(remaining) == 1 and next(iter(remaining.values()))["external_pending"]  # effect unknown
        manager.reconfigure(settings)  # the replacement server's start gets the ordinary budget again
        replaced = call(registry, "bump", timeout=15)
        again = call(registry, "bump", timeout=15)
    second = body(replaced)
    assert second["pid"] != first["pid"] and second["count"] == 1
    assert restarted(replaced) == "a call did not answer in time"  # the mind learns the state did not carry over
    assert body(again) == {"pid": second["pid"], "count": 2} and restarted(again) is None  # said once
    assert [row["tool"] for row in rows(log, "call")].count("hang") == 1  # never resent
    mcp_task_session.close_context_sessions(registry._ctx)


def test_idle_session_answers_the_servers_pings(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw)
    with owner(registry):
        first = body(call(registry, "bump"))
        idle_from = time.time()
        time.sleep(1.5)  # no call: only the session's own reader can answer
        pongs = [row for row in rows(log, "pong") if row["at"] > idle_from]
        second = body(call(registry, "bump"))
    assert len(pongs) >= 4, pongs
    assert second == {"pid": first["pid"], "count": 2}
    mcp_task_session.close_context_sessions(registry._ctx)


@pytest.mark.parametrize("running", [False, True], ids=["before_start", "live_session"])
def test_pause_after_tool_admission_refuses_before_any_start_or_send(tmp_path, monkeypatch, running):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw)
    with owner(registry):
        first = body(call(registry, "bump")) if running else None
        bound = mcp_task_session._not_live_owner

        def pause_after_registry_admission(ctx):  # the owner's Pause lands inside this invocation
            owner_pause.install_fence(tmp_path, "root", request_id="task-session-pause")
            return bound(ctx)
        monkeypatch.setattr(mcp_task_session, "_not_live_owner", pause_after_registry_admission)
        refused = call(registry, "bump")
    assert refused.status == "blocked" and refused.meta["mcp_call_not_sent"] is True, refused
    assert len(rows(log, "call")) == (1 if running else 0)
    assert len(rows(log, "start")) == (2 if running else 1)  # discovery, plus the live session
    if running:
        assert alive(first["pid"])  # Pause fences new effects; it does not end the session
    assert not claims(tmp_path)
    mcp_task_session.close_context_sessions(registry._ctx)


def test_derived_context_cannot_start_or_use_a_session(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw)
    derived = copy.copy(registry)
    derived._ctx = copy.copy(registry._ctx)  # a review collector or voice view
    with owner(registry):
        refused = call(derived, "bump")
        owned = body(call(registry, "bump"))
        derived._ctx = copy.copy(registry._ctx)  # a copy made after the session exists
        refused_later = call(derived, "bump")
    for result in (refused, refused_later):
        assert result.status == "error" and result.meta["mcp_call_not_sent"] is True
        assert "MCP_TASK_SESSION_UNAVAILABLE" in result.text and "own live tool loop" in result.text
    assert owned["count"] == 1 and len(rows(log, "start")) == 2 and len(rows(log, "call")) == 1
    mcp_task_session.close_context_sessions(registry._ctx)


@pytest.mark.parametrize("cold_pause", [False, True], ids=["loop_exit", "cold_pause"])
def test_loop_cleanup_closes_on_both_paths_and_retires_the_context(tmp_path, monkeypatch, cold_pause):
    from ouroboros import loop_budget

    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    wire(monkeypatch, registry, raw)
    with owner(registry):
        pid = body(call(registry, "bump"))["pid"]
        registry._ctx._budget_pausing = cold_pause  # the exact pause takes the early return
        exit_ctx = loop_budget._LoopExitContext(registry, tmp_path, "root", None, tmp_path / "logs", {}, {})
        loop_budget._cleanup_loop_resources(None, exit_ctx)
        assert session_of(registry).closure == "closed" and not alive(pid)
        late = call(registry, "bump")  # a late thread of the ended loop
    assert late.meta["mcp_call_not_sent"] is True and "task loop has ended" in late.text
    assert len(rows(log, "start")) == 2


def test_settings_change_closes_the_live_session(tmp_path, monkeypatch):
    registry, _, _ = _registry(tmp_path, monkeypatch)
    raw, log = fixture_server(tmp_path, task_session=True)
    manager, settings = wire(monkeypatch, registry, raw)
    with owner(registry):
        first = body(call(registry, "bump"))
        manager.reconfigure({**settings, "MCP_SERVERS": [{**raw, "args": [*raw["args"], "--changed"]}]})
        assert eventually(lambda: not alive(first["pid"]))
        manager.reconfigure(settings)
        assert manager.refresh_server("fx")["ok"]
        replaced = call(registry, "bump")
        second = body(replaced)
        manager.reconfigure({**settings, "MCP_SERVERS": [{**raw, "enabled": False}]})
        assert eventually(lambda: not alive(second["pid"]))
    assert second["pid"] != first["pid"] and second["count"] == 1
    assert restarted(replaced) == "its MCP configuration changed"
    mcp_task_session.close_context_sessions(registry._ctx)


def test_flags_parse_validate_and_keep_the_catalog_identity():
    base = {"id": "fx", "enabled": True, "transport": "stdio", "command": "server"}
    plain = mcp_client.normalize_server_config(base)
    flagged = mcp_client.normalize_server_config({**base, "task_session": True, "browser_policy": "true"})
    assert (plain.task_session, plain.browser_policy) == (False, False)
    assert (flagged.task_session, flagged.browser_policy) == (True, True)
    assert not flagged.configuration_warnings
    assert mcp_catalog.identity(plain) == mcp_catalog.identity(flagged)  # listing evidence stays valid
    assert mcp_task_session.session_identity(plain) != mcp_task_session.session_identity(flagged)
    for raw, words in [
        ({"id": "web", "transport": "streamable_http", "url": "https://example.com/mcp", "task_session": True},
         "apply only to stdio"),
        ({**base, "browser_policy": True}, "requires task_session"),
        ({**base, "task_session": True, "env": {"OUROBOROS_BROWSER_GUARD": "127.0.0.1:1"}}, "reserved"),
        ({**base, "task_session": True, "env_from_settings": {"OUROBOROS_BROWSER_GUARD_TOKEN": "SOME_KEY"}},
         "reserved"),
    ]:
        errors = []
        assert mcp_client.normalize_server_config(raw, settings={"SOME_KEY": "x"}, errors=errors) is None
        assert words in errors[0], errors
    manager = mcp_client.MCPManager()
    manager.reconfigure({"MCP_ENABLED": True, "MCP_SERVERS": [{**base, "task_session": True}]})
    row = manager.status_payload()["servers"][0]
    assert (row["task_session"], row["browser_policy"]) == (True, False)


def test_agent_turn_finally_closes_the_session_its_loop_left_open(tmp_path, monkeypatch):
    """Separate executor calls keep the selected page in a real turn; its ``finally`` closes the session."""
    from ouroboros import agent as agent_module
    from ouroboros import agent_task_pipeline, config, safety
    from ouroboros.model_wait import task_model_wait_scope

    raw, log = fixture_server(tmp_path, task_session=True)
    settings = {**config.SETTINGS_DEFAULTS, "MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 20, "MCP_SERVERS": [raw]}
    monkeypatch.setattr(config, "load_settings", lambda: dict(settings))
    manager = mcp_client.get_manager()
    manager.reconfigure(settings)
    assert manager.refresh_server("fx")["ok"]
    monkeypatch.setenv("OUROBOROS_MODEL", "provider::main")
    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda _self: None)
    monkeypatch.setattr(agent_module, "build_llm_messages", lambda **_: ([], {}))
    monkeypatch.setattr(agent_task_pipeline, "_run_post_task_processing_async", lambda *_a, **_kw: None)
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_kw: (True, ""))
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *_a, **_kw: pytest.fail("no model round in this turn"))
    seen = {}

    def loop_without_its_own_cleanup(**kw):  # a return the loop's own finally never saw
        tools = kw["tools"]
        seen["result"] = call(tools, "bump")
        seen["pages"] = [call(tools, name, args) for name, args in (
            ("browser_navigate", {"url": "https://example.test/first"}),
            ("browser_tabs", {"action": "new"}),
            ("browser_navigate", {"url": "https://example.test/second"}),
            ("browser_tabs", {"action": "select", "index": 0}),
            ("browser_snapshot", {}),
        )]
        seen["session"] = session_of(tools)
        return "done", {}, {"reasoning_notes": [], "tool_calls": []}
    monkeypatch.setattr(agent_module, "run_llm_loop", loop_without_its_own_cleanup)
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    task = {"id": "turn", "type": "task", "chat_id": 1, "text": "use the fixture", "_attempt": 1}
    agent = agent_module.OuroborosAgent(agent_module.Env(repo_dir=repo, drive_root=drive))
    with task_model_wait_scope(task=task, drive_root=drive, event_queue=None, worker_slot_held=False):
        agent._handle_task_scoped(dict(task))
    pid = body(seen["result"])["pid"]
    assert all(result.status == "ok" for result in seen["pages"]), seen["pages"]
    pages = [body(result) for result in seen["pages"]]
    assert {page["pid"] for page in pages} == {pid}
    assert [page["selected"] for page in pages] == [0, 1, 1, 0, 0]
    assert pages[-1] == {"pid": pid, "selected": 0, "url": "https://example.test/first",
                         "pages": ["https://example.test/first", "https://example.test/second"]}
    assert seen["session"].closure == "closed" and not alive(pid)
    assert len(rows(log, "start")) == 2  # discovery, then the turn's one session
    assert [row["tool"] for row in rows(log, "call")] == [
        "bump", "browser_navigate", "browser_tabs", "browser_navigate", "browser_tabs", "browser_snapshot"]
