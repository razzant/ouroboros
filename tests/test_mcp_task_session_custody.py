"""Physical custody of task-held MCP stdio servers, proved with real processes.

The fixture server answers MCP over stdio through the public ``ClientSession``
and, on ``initialize``, double-forks a helper: the helper is reparented away
from the server's process tree but stays in its process group and keeps
writing heartbeats, the way a daemonized browser bridge would. Each test
proves who is still alive by pid, process group and heartbeat, never by a
thread or a receipt alone. A "worker" is a real child Python process that holds
a session and is then killed, as a hard Stop kills a pool worker. POSIX only:
nothing here claims Windows behaviour.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

pytest.importorskip("mcp")

from ouroboros import mcp_client, mcp_task_sessions, platform_layer, process_custody  # noqa: E402
from ouroboros.process_containment import pid_is_zombie  # noqa: E402

pytestmark = [
    pytest.mark.serial,
    pytest.mark.skipif(os.name == "nt", reason="process-group custody is POSIX-only and unverified on Windows"),
]

REPO = pathlib.Path(__file__).resolve().parents[1]

_SERVER = r'''import json, os, sys, time
witness, beats, mode = sys.argv[1], sys.argv[2], sys.argv[3]
def note(**row):
    with open(witness, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "pgid": os.getpgrp(), **row}) + "\n")
def detach():
    if os.fork() == 0:
        if os.fork() == 0:
            if mode == "devnull":
                null = os.open(os.devnull, os.O_RDWR)
                for fd in (0, 1, 2):
                    os.dup2(null, fd)
            note(event="detached")
            while True:
                with open(beats, "a", encoding="utf-8") as handle:
                    handle.write(f"{os.getpid()}\n")
                time.sleep(0.05)
        os._exit(0)
    os.wait()
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request.get("method")
    if method == "initialize":
        note(event="start", client=request["params"]["clientInfo"]["name"])
        detach()
        result = {"protocolVersion": request["params"]["protocolVersion"],
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "detaching", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "ping", "description": "synthetic", "inputSchema": {"type": "object"}}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "pong"}], "isError": False}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
note(event="eof")
'''

_WORKER = r'''import json, pathlib, sys, time
sys.path.insert(0, sys.argv[1])
from ouroboros import mcp_client, mcp_task_sessions, process_custody
root, server, witness, beats, session = sys.argv[2:7]
if session:
    process_custody.adopt_session_id(session)
mcp_task_sessions._custody_root = lambda: pathlib.Path(root)
cfg = mcp_client.normalize_server_config({"id": "held", "transport": "stdio", "command": sys.executable,
                                          "args": [server, witness, beats, "devnull"], "session_scope": "task"})
result = mcp_task_sessions.call_task_session(cfg, "task-w", "ping", {}, 30)
print(json.dumps({"text": result.text}), flush=True)
time.sleep(600)
'''


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "state").mkdir(parents=True)
    (tmp_path / "server.py").write_text(_SERVER, encoding="utf-8")
    monkeypatch.setattr(mcp_task_sessions, "_custody_root", lambda: root)
    monkeypatch.setattr(mcp_task_sessions, "_sessions", {})
    monkeypatch.setattr(mcp_task_sessions, "_terminal_tasks", set())
    monkeypatch.setattr(mcp_task_sessions, "_emergency", "")
    monkeypatch.setattr(mcp_task_sessions, "_EOF_GRACE_SEC", 1.0)
    monkeypatch.setattr(mcp_task_sessions, "_SETTLE_SEC", 1.5)
    state = {"root": root, "tmp": tmp_path, "witness": tmp_path / "witness.jsonl",
             "beats": tmp_path / "beats.txt", "children": []}
    yield state
    for task_id in {key[1] for key in list(mcp_task_sessions._sessions)}:
        mcp_task_sessions.close_task_sessions(task_id)
    for child in state["children"]:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
    # Never leave a helper behind, and never signal a pid that is no longer ours.
    for row in _rows(state["witness"]):
        if row["event"] == "detached" and _alive(row["pid"]) and platform_layer.process_group_id(row["pid"]) == row["pgid"]:
            os.kill(row["pid"], signal.SIGKILL)


def _cfg(world, mode="devnull"):
    return mcp_client.normalize_server_config({
        "id": "held", "transport": "stdio", "command": sys.executable, "session_scope": "task",
        "args": [str(world["tmp"] / "server.py"), str(world["witness"]), str(world["beats"]), mode]})


def _rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")[:-1]]


def _alive(pid):
    return platform_layer.pid_is_alive(pid) and not pid_is_zombie(pid)


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _one(world, event, predicate=lambda row: True):
    assert _wait_for(lambda: any(row["event"] == event and predicate(row) for row in _rows(world["witness"])))
    (row,) = [row for row in _rows(world["witness"]) if row["event"] == event and predicate(row)]
    return row


def _beating(world, pid):
    def count():
        return sum(1 for line in world["beats"].read_text().split("\n") if line == str(pid))

    before = count()
    time.sleep(0.4)
    return count() > before


def _ledger(world):
    path = process_custody.ledger_path(world["root"])
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _session_rows(world):
    live = {}
    for row in _ledger(world):
        live[row["pid"]] = row
    return [row for row in live.values() if row["purpose"].startswith(process_custody.TASK_SESSION_PURPOSE_PREFIX)]


def _supervisor_rows(world):
    path = world["root"] / "logs" / "supervisor.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _open(world, task_id="task-a", mode="devnull"):
    result = mcp_task_sessions.call_task_session(_cfg(world, mode), task_id, "ping", {}, 30)
    assert result.text == "pong"
    leader = _one(world, "start", lambda row: row["client"] == f"Ouroboros task {task_id}")
    helper = _one(world, "detached", lambda row: row["pgid"] == leader["pgid"])
    assert helper["pgid"] == leader["pid"] and helper["pid"] != leader["pid"]
    assert _beating(world, helper["pid"])
    return leader, helper


def _spawn_worker(world, *, session=""):
    script = world["tmp"] / "worker.py"
    script.write_text(_WORKER, encoding="utf-8")
    child = subprocess.Popen(
        [sys.executable, str(script), str(REPO), str(world["root"]), str(world["tmp"] / "server.py"),
         str(world["witness"]), str(world["beats"]), session],
        stdout=subprocess.PIPE, cwd=str(REPO),
    )
    world["children"].append(child)
    assert json.loads(child.stdout.readline()) == {"text": "pong"}
    leader = _one(world, "start", lambda row: row["client"] == "Ouroboros task task-w")
    helper = _one(world, "detached", lambda row: row["pgid"] == leader["pgid"])
    assert _beating(world, helper["pid"])
    return child, leader, helper


def test_sdk_stdio_close_leaves_the_detached_helper_running(world):
    """Control: the gap this custody closes is real for this very fixture."""
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=sys.executable, args=list(_cfg(world).args))

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                assert (await session.call_tool("ping", {})).content[0].text == "pong"

    anyio.run(run)
    leader = _one(world, "eof")
    helper = _one(world, "detached")
    assert not _alive(leader["pid"])
    assert _beating(world, helper["pid"])  # the SDK's normal close left it behind


@pytest.mark.parametrize("mode", ["devnull", "inherit"])
def test_normal_close_kills_the_whole_group_and_releases_the_durable_row(world, mode):
    leader, helper = _open(world, mode=mode)
    (row,) = _session_rows(world)
    assert (row["pid"], row["pgid"], row["scope"]) == (leader["pid"], leader["pid"], "task")
    assert (row["owner_task"], row["purpose"], row["spawner_pid"]) == ("task-a", "mcp_task_session:held", os.getpid())
    assert row["session_id"] == process_custody.current_custody_session_id()

    assert mcp_task_sessions.close_task_sessions("task-a") == [
        {"server_id": "held", "task_id": "task-a", "closed": True}]
    assert not _alive(leader["pid"]) and not _alive(helper["pid"])
    assert not _beating(world, helper["pid"])
    assert not platform_layer.process_group_is_alive(leader["pgid"])
    assert _session_rows(world) == []
    stopped = [r for r in _supervisor_rows(world) if r["type"] == "process_stopped"]
    assert [(r["pid"], r["reason"]) for r in stopped] == [(leader["pid"], "task_session_close")]
    assert mcp_task_sessions._sessions == {}


def test_close_kills_the_owned_group_even_when_its_custody_row_is_gone(world):
    """The owned Popen is the first authority; the ledger row is the durable second."""
    leader, helper = _open(world)
    process_custody.ledger_path(world["root"]).unlink()

    assert mcp_task_sessions.close_task_sessions("task-a")[0]["closed"] is True
    assert not _alive(leader["pid"]) and not _alive(helper["pid"])


def test_a_terminal_task_never_reopens_a_session(world):
    _open(world)
    mcp_task_sessions.close_task_sessions("task-a")
    starts = len([row for row in _rows(world["witness"]) if row["event"] == "start"])
    with pytest.raises(RuntimeError, match="cannot reopen"):
        mcp_task_sessions.call_task_session(_cfg(world), "task-a", "ping", {}, 30)
    time.sleep(0.3)
    assert len([row for row in _rows(world["witness"]) if row["event"] == "start"]) == starts
    assert mcp_task_sessions._sessions == {} and _session_rows(world) == []
    # Another task is not affected by the first one's terminal state.
    assert mcp_task_sessions.call_task_session(_cfg(world), "task-b", "ping", {}, 30).text == "pong"


def test_failed_group_kill_keeps_custody_and_refuses_replacement_after_leader_exit(world, monkeypatch):
    leader, helper = _open(world)
    originals = {owner: owner.kill_process_group_id for owner in (platform_layer, process_custody)}
    for owner in originals:
        monkeypatch.setattr(owner, "kill_process_group_id", lambda *_a, **_k: None)

    # A Settings change is not terminal for the task: the close is requested
    # without waiting, cannot be confirmed, and blocks any replacement.
    mcp_task_sessions.retain_sessions({})
    held = mcp_task_sessions._sessions[("held", "task-a")]
    assert _wait_for(lambda: not held._thread.is_alive(), timeout=15)
    assert not held.closed() and "unconfirmed" in held.custody.detail
    assert not _alive(leader["pid"]) and _beating(world, helper["pid"])
    assert [row["pid"] for row in _session_rows(world)] == [leader["pid"]]
    with pytest.raises(RuntimeError, match="not confirmed disconnection"):
        mcp_task_sessions.call_task_session(_cfg(world), "task-a", "ping", {}, 30)
    assert len([row for row in _rows(world["witness"]) if row["event"] == "start"]) == 1

    receipts = mcp_task_sessions.close_task_sessions("task-a", wait=3)
    assert [(r["closed"], "unconfirmed" in r["detail"]) for r in receipts] == [(False, True)]
    assert mcp_task_sessions._sessions[("held", "task-a")] is held and _beating(world, helper["pid"])

    for owner, original in originals.items():
        monkeypatch.setattr(owner, "kill_process_group_id", original)
    # Once the leader was reaped, its numeric pgid cannot authenticate the
    # original group. A later stop must not guess, even when signalling works.
    assert mcp_task_sessions.close_task_sessions("task-a", wait=5)[0]["closed"] is False
    assert _beating(world, helper["pid"])
    assert [row["pid"] for row in _session_rows(world)] == [leader["pid"]]


def test_ledger_release_cannot_close_session_while_captured_group_survives(world, monkeypatch):
    leader, helper = _open(world)
    monkeypatch.setattr(platform_layer, "kill_process_group_id", lambda *_a, **_kw: None)
    monkeypatch.setattr(process_custody, "stop_group_custody", lambda *_a, **_kw: [leader["pid"]])

    receipt = mcp_task_sessions.close_task_sessions("task-a", wait=5)[0]

    assert receipt["closed"] is False
    assert _beating(world, helper["pid"])
    assert mcp_task_sessions._sessions[("held", "task-a")].custody.released is False


def test_hard_worker_stop_kills_the_session_group_the_worker_spawned(world, monkeypatch):
    from supervisor import queue as supervisor_queue
    from supervisor import workers
    from supervisor.worker_pool_lifecycle import kill_worker_tree

    child, leader, helper = _spawn_worker(world)
    (row,) = _session_rows(world)
    assert row["spawner_pid"] == child.pid and row["owner_task"] == "task-w"
    monkeypatch.setattr(supervisor_queue, "DRIVE_ROOT", world["root"])
    monkeypatch.setattr(workers, "DRIVE_ROOT", world["root"])

    kill_worker_tree(child.pid, keep_services=True)

    child.wait(timeout=10)
    assert not _alive(leader["pid"]) and not _alive(helper["pid"])
    assert not _beating(world, helper["pid"])
    assert _session_rows(world) == []
    (receipt,) = [r for r in _supervisor_rows(world) if r["type"] == "worker_task_sessions_stopped"]
    assert (receipt["worker_pid"], receipt["stopped"], receipt["unconfirmed"]) == (child.pid, [leader["pid"]], [])


def _kill_worker_and_leader(child, leader):
    """A hard worker death: the leader sees stdin EOF and usually exits by itself."""
    child.kill()
    child.wait(timeout=10)
    try:
        os.kill(leader["pid"], signal.SIGKILL)
    except ProcessLookupError:
        pass
    assert _wait_for(lambda: not _alive(leader["pid"]))


def test_boot_reap_retains_a_group_that_outlived_its_dead_leader(world):
    child, leader, helper = _spawn_worker(world)  # the worker minted its own generation
    _kill_worker_and_leader(child, leader)
    assert _beating(world, helper["pid"])
    assert [row["pid"] for row in _session_rows(world)] == [leader["pid"]]

    reaped = process_custody.reap_orphaned_processes(world["root"], running_task_ids=set())

    assert reaped == []
    assert _beating(world, helper["pid"])
    assert [row["pid"] for row in _session_rows(world)] == [leader["pid"]]


def test_same_generation_reap_retains_unidentified_group_after_task_is_gone(world):
    child, leader, helper = _spawn_worker(world, session=process_custody.current_custody_session_id())
    _kill_worker_and_leader(child, leader)

    assert process_custody.reap_orphaned_processes(world["root"], running_task_ids={"task-w"}) == []
    assert _beating(world, helper["pid"])
    assert [row["pid"] for row in _session_rows(world)] == [leader["pid"]]  # dead leader, row kept

    assert process_custody.reap_orphaned_processes(world["root"], running_task_ids=set()) == []
    assert _beating(world, helper["pid"])
    assert [row["pid"] for row in _session_rows(world)] == [leader["pid"]]


def test_a_live_stranger_at_the_leader_pid_is_never_signalled(world, monkeypatch):
    """A recycled leader pid provides no authority to signal or close the row."""
    entry = {"pid": 4242, "pgid": 4242, "purpose": "mcp_task_session:held", "scope": "task",
             "owner_task": "gone", "session_id": "old", "fingerprint": {"start_time": "then", "cmd_sha256": "x"}}
    assert process_custody.append_jsonl(process_custody.ledger_path(world["root"]), entry)
    monkeypatch.setattr(process_custody, "pid_is_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(process_custody, "pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(process_custody, "process_group_has_live_members", lambda pgid: True)
    monkeypatch.setattr(process_custody, "_fingerprint_matches", lambda *_a, **_k: False)
    monkeypatch.setattr(process_custody, "kill_process_group_id",
                        lambda *a, **k: pytest.fail(f"a recycled group must not be signalled: {a}"))

    unconfirmed = []
    assert process_custody.stop_group_custody(world["root"], lambda row: True,
                                              timeout_sec=0, unconfirmed=unconfirmed) == []
    assert unconfirmed and _session_rows(world) == [entry]
    assert process_custody.reap_orphaned_processes(world["root"], running_task_ids=set()) == []
    assert _session_rows(world) == [entry]


def test_panic_request_signals_this_process_groups_at_once_and_refuses_new_sessions(world):
    leader, helper = _open(world)
    receipts = mcp_task_sessions.request_emergency_stop()
    assert [(r["requested"], r["scope"], r["pgid"]) for r in receipts] == [(True, "group", leader["pid"])]
    assert _wait_for(lambda: not _alive(helper["pid"]) and not _alive(leader["pid"]), timeout=3)
    with pytest.raises(RuntimeError, match="Emergency Stop requested"):
        mcp_task_sessions.call_task_session(_cfg(world), "task-b", "ping", {}, 30)
    assert not any(row["event"] == "start" and row["client"] == "Ouroboros task task-b"
                   for row in _rows(world["witness"]))


def test_execute_panic_stop_ends_direct_chat_and_worker_session_groups(world, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import server_control
    from ouroboros.startup_historical_audit import audit

    child, worker_leader, worker_helper = _spawn_worker(world)
    chat_leader, chat_helper = _open(world, task_id="direct-chat")

    class ExitCalled(RuntimeError):
        pass

    monkeypatch.setattr(audit, "stop", lambda: None)
    monkeypatch.setattr("ouroboros.tools.shell.kill_all_tracked_subprocesses", lambda **kw: [])
    monkeypatch.setattr("ouroboros.workspace_executor.kill_all_foreground", lambda *a, **kw: [])
    monkeypatch.setattr("ouroboros.tools.services.kill_all_services", lambda *a, **kw: [])
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda **kw: [])
    monkeypatch.setattr("ouroboros.local_model.get_manager", lambda **kw: None)
    monkeypatch.setattr("ouroboros.claudexor_daemon.get_owned_daemon", lambda **kw: SimpleNamespace(
        panic_stop=lambda **kw: [], stop_outcome=lambda: True))
    monkeypatch.setattr(server_control, "_persist_panic_controls", lambda _root: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    def checked_exit(code):
        # Panic's worker-held group must be dead BEFORE the real hard exit.
        assert all(not _alive(pid) for pid in (worker_leader["pid"], worker_helper["pid"]))
        assert _session_rows(world) == []
        raise ExitCalled(code)

    monkeypatch.setattr(server_control.os, "_exit", checked_exit)
    reports = []

    import threading

    before = set(threading.enumerate())
    with pytest.raises(ExitCalled):
        server_control.execute_panic_stop(
            consciousness=None, kill_workers_fn=lambda **kw: None, data_dir=world["root"], panic_exit_code=120,
            log=SimpleNamespace(critical=lambda message, *args: reports.append(args)))
    for thread in set(threading.enumerate()) - before:
        if thread.name.startswith("panic-"):
            thread.join(timeout=10)

    # The dicts were logged by reference; the joined settlement threads filled them.
    requests, settlements = reports[0][0], reports[0][1]
    assert [(r["task_id"], r["requested"], r["scope"]) for r in requests["mcp-task-sessions"]] == [
        ("direct-chat", True, "group")]  # the only session held in this process
    assert sorted(settlements["mcp-task-sessions"]["stopped"]) == sorted(
        [worker_leader["pid"], chat_leader["pid"]])
    assert settlements["mcp-task-sessions"]["unconfirmed"] == []
    for pid in (worker_leader["pid"], worker_helper["pid"], chat_leader["pid"], chat_helper["pid"]):
        assert _wait_for(lambda pid=pid: not _alive(pid), timeout=5), pid
    assert not _beating(world, worker_helper["pid"]) and not _beating(world, chat_helper["pid"])
    assert _session_rows(world) == []
    assert child.poll() is None  # a worker's session was ended without the worker's cooperation
