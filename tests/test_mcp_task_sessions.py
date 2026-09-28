"""Task-held MCP sessions (``session_scope: task``) through the real dispatch seam.

The fixture is a synthetic stateful stdio MCP server: ``remember`` stores a
value in the server process, ``recall`` returns it, ``hang`` never answers
(it writes a heartbeat instead, so a test can see the process stop).
It records every process start (with the MCP client name the task session
sends), call and stdin EOF in a witness file; process liveness proves which
server actually ended when its session closed.
No browser, network or real page content is involved.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import pytest

from ouroboros import mcp_client, mcp_task_sessions, platform_layer
from ouroboros.consciousness_authority import CONSCIOUSNESS_INITIATOR
from ouroboros.contracts.task_constraint import TaskConstraint
from ouroboros.tools.registry import ToolContext, ToolRegistry

pytest.importorskip("mcp")

pytestmark = pytest.mark.serial  # real server processes and module-global session state

_SERVER = r'''import json, os, sys, time
witness = sys.argv[1]
def note(**row):
    with open(witness, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"pid": os.getpid(), **row}) + "\n")
memory = {"value": ""}
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request.get("method")
    if method == "initialize":
        note(event="start", client=request["params"]["clientInfo"]["name"], cwd=os.getcwd())
        result = {"protocolVersion": request["params"]["protocolVersion"],
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "stateful", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": name, "description": "synthetic", "inputSchema": {"type": "object"}}
                            for name in ("remember", "recall", "hang")]}
    elif method == "tools/call":
        name, args = request["params"]["name"], request["params"].get("arguments") or {}
        note(event="call", tool=name)
        if name == "hang":
            for _ in range(600):
                note(event="beat")
                time.sleep(0.1)
        if name == "remember":
            memory["value"] = args.get("value", "")
        result = {"content": [{"type": "text", "text": "value=" + memory["value"]}], "isError": False}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
note(event="eof")
'''


@pytest.fixture
def stateful(tmp_path, monkeypatch):
    mcp_client.reset_manager_for_tests()
    script = tmp_path / "stateful_mcp.py"
    script.write_text(_SERVER, encoding="utf-8")
    witness = tmp_path / "witness.jsonl"
    server = {"id": "held", "enabled": True, "transport": "stdio", "command": sys.executable,
              "args": [str(script), str(witness)], "session_scope": "task"}
    settings = {"MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 10, "MCP_SERVERS": [server]}
    mcp_client.reconfigure_from_settings(settings)
    assert mcp_client.get_manager().refresh_server("held")["ok"]
    # Dispatch rechecks Settings on a hit; keep the test's configuration authoritative.
    monkeypatch.setattr(mcp_client, "ensure_configured_from_settings", lambda **_kw: None)
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **kw: (True, ""))
    # Custody rows land in this test's root; terminal task ids never leak between tests.
    monkeypatch.setattr(mcp_task_sessions, "_custody_root", lambda: tmp_path / "data")
    monkeypatch.setattr(mcp_task_sessions, "_terminal_tasks", set())
    monkeypatch.setattr(mcp_task_sessions, "_emergency", "")
    yield {"witness": witness, "server": server, "settings": settings, "tmp": tmp_path}
    for task_id in {key[1] for key in list(mcp_task_sessions._sessions)}:
        mcp_task_sessions.close_task_sessions(task_id)
    mcp_client.reset_manager_for_tests()


def _rows(witness):
    # Only complete lines: the server may be appending while a test reads.
    return [json.loads(line) for line in witness.read_text(encoding="utf-8").split("\n")[:-1]]


def _task_pids(witness, task_id):
    return {row["pid"] for row in _rows(witness)
            if row["event"] == "start" and row["client"] == f"Ouroboros task {task_id}"}


def _registry(tmp_path, task_id="", **fields):
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data", task_id=task_id or None, **fields)
    registry = ToolRegistry(ctx.repo_dir, ctx.drive_root)
    registry.set_context(ctx)
    return registry


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_session_scope_is_a_validated_known_field():
    base = {"id": "s", "transport": "stdio", "command": "npx"}
    assert mcp_client.normalize_server_config(base).session_scope == "call"
    held = mcp_client.normalize_server_config({**base, "session_scope": "task"})
    assert held.session_scope == "task" and held.configuration_warnings == ()
    errors = []
    assert mcp_client.normalize_server_config({**base, "session_scope": "tab"}, errors=errors) is None
    assert "session_scope" in errors[0]


def test_one_task_reuses_its_session_and_another_task_cannot_inherit_it(stateful):
    task_a = _registry(stateful["tmp"], "task-a")
    assert "value=alpha" in task_a.execute("mcp_held__remember", {"value": "alpha"})
    assert "value=alpha" in task_a.execute("mcp_held__recall", {})
    assert len(_task_pids(stateful["witness"], "task-a")) == 1  # one process served both calls

    task_b = _registry(stateful["tmp"], "task-b")
    result = task_b.execute("mcp_held__recall", {})
    assert "value=alpha" not in result and "value=" in result
    assert _task_pids(stateful["witness"], "task-b").isdisjoint(_task_pids(stateful["witness"], "task-a"))
    assert "value=alpha" in task_a.execute("mcp_held__recall", {})  # A's own state is untouched


@pytest.mark.parametrize("fields", [
    {"task_metadata": {"delegation_role": "subagent"}},
    {"task_contract": {"lineage": {"delegation_role": "subagent"}}},
    # An explicit parent grant does not turn a child into the session's owner.
    {"task_constraint": TaskConstraint(mode="acting_subagent", surface="self_worktree",
                                       external_tool_grants=("mcp_held__recall",))},
])
def test_delegated_child_neither_sees_nor_opens_a_task_session(stateful, fields):
    parent = _registry(stateful["tmp"], "task-parent")
    assert "value=secret-state" in parent.execute("mcp_held__remember", {"value": "secret-state"})
    starts = len([row for row in _rows(stateful["witness"]) if row["event"] == "start"])

    child = _registry(stateful["tmp"], "task-child", **fields)
    names = {schema["function"]["name"] for schema in child.schemas()}
    assert not any(name.startswith("mcp_held__") for name in names)
    assert child.get_schema_by_name("mcp_held__recall") is None
    assert "one session per task" in child.policy_hidden_reason("mcp_held__recall")
    assert any(item.get("reason") == "task_session_scope" for item in child.capability_omissions())
    result = child.execute_result("mcp_held__recall", {})
    assert (result.status, result.code) == ("blocked", "ACCESS_BLOCKED")
    assert "MCP_TOOL_DISALLOWED" in result.text and "one session per task" in result.text
    assert "secret-state" not in result.text
    assert len([row for row in _rows(stateful["witness"]) if row["event"] == "start"]) == starts


def test_callers_without_a_task_or_from_consciousness_are_refused_before_transport(stateful):
    starts = len(_rows(stateful["witness"]))
    anonymous = _registry(stateful["tmp"])
    assert anonymous.get_schema_by_name("mcp_held__recall") is None
    assert "one session per task" in anonymous.execute("mcp_held__recall", {})

    wake = _registry(stateful["tmp"], "task-wake", task_metadata={"initiator": CONSCIOUSNESS_INITIATOR})
    assert wake.get_schema_by_name("mcp_held__recall") is not None  # dispatch-only, like disabled_tools
    assert "one session per task" in wake.execute("mcp_held__recall", {})
    assert len(_rows(stateful["witness"])) == starts


def test_task_end_closes_its_session_and_ends_the_server_process(stateful):
    task_a, task_b = _registry(stateful["tmp"], "task-a"), _registry(stateful["tmp"], "task-b")
    task_a.execute("mcp_held__remember", {"value": "alpha"})
    task_b.execute("mcp_held__remember", {"value": "beta"})
    (pid_a,) = _task_pids(stateful["witness"], "task-a")

    (scratch,) = {row["cwd"] for row in _rows(stateful["witness"]) if row.get("client") == "Ouroboros task task-a"}
    assert "ouroboros-mcp-task-" in scratch and pathlib.Path(scratch).is_dir()  # private, not the host cwd

    receipts = mcp_task_sessions.close_task_sessions("task-a")
    assert receipts == [{"server_id": "held", "task_id": "task-a", "closed": True}]
    assert not platform_layer.pid_is_alive(pid_a)
    assert not pathlib.Path(scratch).exists()
    assert "value=beta" in task_b.execute("mcp_held__recall", {})  # another task's session survives
    late = task_a.execute_result("mcp_held__recall", {})  # e.g. a tool thread that outlived its round
    assert late.status == "error" and "cannot reopen" in late.text and "value=" not in late.text
    assert len(_task_pids(stateful["witness"], "task-a")) == 1  # no new connection was started


def test_loop_exit_cleanup_closes_the_tasks_sessions(stateful, monkeypatch):
    from types import SimpleNamespace

    from ouroboros import loop_budget

    task_a = _registry(stateful["tmp"], "task-a")
    task_a.execute("mcp_held__remember", {"value": "alpha"})
    (pid_a,) = _task_pids(stateful["witness"], "task-a")
    monkeypatch.setattr(loop_budget, "_loop", lambda: SimpleNamespace(_finalize_task_services=lambda _ctx: None))
    exit_ctx = SimpleNamespace(trace_ctx=None, tools=task_a, drive_root=None, task_id="task-a")
    loop_budget._cleanup_loop_resources(None, exit_ctx)
    assert not mcp_task_sessions._sessions
    assert not platform_layer.pid_is_alive(pid_a)


def test_disabling_the_server_in_settings_closes_held_sessions(stateful):
    task_a = _registry(stateful["tmp"], "task-a")
    task_a.execute("mcp_held__remember", {"value": "alpha"})
    (pid_a,) = _task_pids(stateful["witness"], "task-a")
    disabled = {**stateful["settings"], "MCP_SERVERS": [{**stateful["server"], "enabled": False}]}
    mcp_client.reconfigure_from_settings(disabled)
    assert mcp_task_sessions._sessions.get(("held", "task-a")) is None or not mcp_task_sessions._sessions[("held", "task-a")].alive()
    assert _wait_for(lambda: not platform_layer.pid_is_alive(pid_a))
    assert "MCP_DISABLED" in task_a.execute("mcp_held__recall", {})


def test_a_timed_out_call_closes_the_session_and_says_so(stateful):
    mcp_client.get_manager()._tool_timeout_sec = 1
    task_a = _registry(stateful["tmp"], "task-a")
    task_a.execute("mcp_held__remember", {"value": "alpha"})
    (pid_a,) = _task_pids(stateful["witness"], "task-a")
    result = task_a.execute_result("mcp_held__hang", {})
    assert (result.status, result.code) == ("timeout", "MCP_TIMEOUT")
    assert "closure was requested" in result.text
    # The close is asynchronous; no replacement may race the SDK's exit.
    # The server is mid-call, so stdin EOF alone cannot end it; the SDK terminates it.
    assert _wait_for(lambda: _beats_stopped(stateful["witness"], pid_a))
    assert _wait_for(lambda: not mcp_task_sessions._sessions[('held', 'task-a')]._thread.is_alive()
                     if ('held', 'task-a') in mcp_task_sessions._sessions else True)
    assert "value=alpha" not in task_a.execute("mcp_held__recall", {})


def _beats_stopped(witness, pid):
    def beats():
        return sum(1 for row in _rows(witness) if row["pid"] == pid and row["event"] == "beat")

    before = beats()
    time.sleep(0.5)
    return before > 0 and beats() == before
