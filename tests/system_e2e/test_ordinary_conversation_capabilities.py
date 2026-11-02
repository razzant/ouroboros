"""S27: ordinary Main/Project tools and concurrency through real stdio MCP.

One Main turn is held inside its model call while another completes MCP work.
Both start after a Project exists. The Project then uses MCP and the built-in
writer against its attached folder. Durable results, tool traces, wire schemas
and physical files jointly prove capability; a stub's final answer alone cannot.
The shared e2e_clone fixture runs committed HEAD, so an uncommitted candidate
must first be materialized in an isolated clone by its operator.
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid

import pytest

from devtools.benchmarks.common.server_runner import _api
from tests.system_e2e.harness import (
    LANE_MOCK, ArtifactOracle, ModelGate, ScriptedStubModel, body_text,
    keyless_settings, message_text, require_lane, start_server, wait_durable_result, wait_until, ws_url,
)
from tests.system_e2e.test_system_scenarios_w6 import _WsFrames, _direct_activities


MCP_SOURCE = '''from pathlib import Path
from mcp.server.fastmcp import FastMCP

server = FastMCP("ordinary-fixture")

@server.tool()
def read_note(name: str) -> str:
    """Read a note in this fixture's private folder."""
    return Path(name).read_text(encoding="utf-8")

@server.tool()
def write_note(name: str, text: str) -> str:
    """Write a note in this fixture's private folder and read it back."""
    Path(name).write_text(text, encoding="utf-8")
    return Path(name).read_text(encoding="utf-8")

server.run(transport="stdio")
'''


def _received_turn(oracle: ArtifactOracle, client_message_id: str) -> dict | None:
    for row in oracle.events("task_received"):
        task = row.get("task") or {}
        origin = (task.get("metadata") or {}).get("origin_message_ref") or {}
        if origin.get("client_message_id") == client_message_id:
            return task
    return None


@pytest.mark.integration
@pytest.mark.serial
def test_s27_ordinary_main_and_project_mcp_with_concurrent_native_turn(e2e_clone, tmp_path):
    require_lane(LANE_MOCK)
    room, mcp_dir = tmp_path / "attached room", tmp_path / "mcp fixture"
    room.mkdir()
    mcp_dir.mkdir()
    subprocess.run(["git", "init", "-q", str(room)], check=True, capture_output=True)
    script = mcp_dir / "server.py"
    script.write_text(MCP_SOURCE, encoding="utf-8")
    (mcp_dir / "input.txt").write_text("S27_READ_яё𐍈🚀", encoding="utf-8")
    read_tool, write_tool = "mcp_fixture__read_note", "mcp_fixture__write_note"
    steps = [
        {"tool": read_tool, "arguments": {"name": "input.txt"}},
        {"tool": write_tool, "arguments": {"name": "main.txt", "text": "S27_MAIN_WRITE"}},
        {"final": "S27_MAIN_COMPLETE"},
        {"final": "S27_HELD_COMPLETE"},
        {"tool": read_tool, "arguments": {"name": "input.txt"}},
        {"tool": write_tool, "arguments": {"name": "project.txt", "text": "S27_PROJECT_WRITE"}},
        {"tool": "write_file", "arguments": {
            "root": "active_workspace", "path": "room-result.txt", "content": "S27_ROOM_WRITE",
        }},
        {"final": "S27_PROJECT_COMPLETE"},
    ]
    gate = ModelGate(lambda body: bool(body.get("tools")) and "S27_HOLD" in body_text(body),
                     timeout=300)
    with ScriptedStubModel(steps, gate=gate) as stub:
        settings = keyless_settings(stub, MCP_ENABLED=True, MCP_SERVERS=[{
            "id": "fixture", "enabled": True, "transport": "stdio", "command": sys.executable,
            "args": [str(script)], "cwd": str(mcp_dir),
        }])
        server = start_server(e2e_clone, tmp_path / "server", settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            project = _api(server.base_url, "POST", "/api/projects", {
                "name": "S27 attached room", "path": str(room),
            }, timeout=60).get("project") or {}
            assert project.get("id") and project.get("chat_id"), project
            assert project.get("working_dir") == str(room), project
            from websockets.sync.client import connect

            with connect(ws_url(server), open_timeout=30, proxy=None) as ws, _WsFrames(ws) as frames:
                def send(content, *, in_project=False):
                    message_id = "e2e-s27-" + uuid.uuid4().hex
                    ws.send(json.dumps({
                        "type": "chat", "content": content, "client_message_id": message_id,
                        "chat_id": project["chat_id"] if in_project else 1,
                        **({"project_id": project["id"]} if in_project else {}),
                    }))
                    return message_id

                def finish(message_id, marker):
                    task = wait_until(lambda: _received_turn(oracle, message_id), 60)
                    assert task and task.get("_is_direct_chat") is True, task
                    assert not task.get("_ephemeral_turn"), task
                    task_id = task["id"]
                    stored = wait_durable_result(oracle, task_id, timeout=180)
                    assert stored.get("status") == "completed", stored
                    assert marker in str(stored.get("result") or ""), stored
                    assert wait_until(lambda: [f for f in frames.find(type="chat", task_id=task_id)
                                             if not f.get("is_progress") and marker in str(f.get("content"))], 60)
                    assert not oracle.child_task_ids(task_id), "ordinary work was delegated/promoted"
                    return task_id

                held_message = send("S27_HOLD: think until the model responds, then answer here.")
                assert gate.arrived.wait(180), "the first native turn never reached the model"
                held = wait_until(lambda: (_direct_activities(server, held_message) or [None])[0], 30)
                assert held, "the held Main turn is not addressable"
                held_id = held["activity_id"]
                assert held_id not in oracle.running_ids(), "native turn acquired a pool worker"
                main_id = finish(send("Read input.txt with the fixture, write main.txt, then answer here."),
                                 "S27_MAIN_COMPLETE")
                assert (mcp_dir / "main.txt").read_text() == "S27_MAIN_WRITE"
                assert not gate.release.is_set() and not gate.timed_out
                assert oracle.task_result(held_id).get("status") == "running"
                gate.release.set()
                assert finish(held_message, "S27_HELD_COMPLETE") == held_id
                project_id = finish(send("Read input.txt and write project.txt with the fixture; "
                                         "write room-result.txt in this Project and answer here.",
                                         in_project=True), "S27_PROJECT_COMPLETE")

            assert (mcp_dir / "project.txt").read_text() == "S27_PROJECT_WRITE"
            assert (room / "room-result.txt").read_text() == "S27_ROOM_WRITE"
            assert not (e2e_clone / "room-result.txt").exists(), "room write targeted the system repo"
            assert oracle.task_result(project_id).get("_is_direct_chat") is True
            assert not [row for row in oracle._jsonl("logs/chat.jsonl")
                        if row.get("type") == "project_completion_summary"
                        and row.get("task_id") == project_id], "ordinary Project reply leaked into Main"
            for task_id, expected in ((main_id, {read_tool, write_tool}),
                                      (project_id, {read_tool, write_tool, "write_file"})):
                rows = [r for r in oracle.tools_rows() if r.get("task_id") == task_id]
                assert {r.get("tool") for r in rows} == expected, rows
                assert "S27_READ_яё𐍈🚀" in json.dumps(rows, ensure_ascii=False), rows
                assert all(not r.get("is_error") for r in rows), rows
            agent_calls = [body for kind, body in stub.calls if kind == "agent"]
            assert len(agent_calls) == 5 and stub.script_consumed(), stub.kinds()
            # An unpinned server's call schemas load on demand (#2332): every request names its tools in
            # the connections map and carries the loader; the calls above ran without loading them.
            for body in agent_calls:
                names = {tool.get("function", {}).get("name") for tool in body.get("tools", [])}
                system = "\n".join(message_text(m) for m in body.get("messages") or []
                                   if isinstance(m, dict) and m.get("role") == "system")
                assert "enable_tools" in names and "### mcp_fixture" in system, names
                assert "- read_note" in system and "- write_note" in system, system[-3000:]
            assert gate.held == 1 and not gate.timed_out
        finally:
            gate.release.set()
            server.stop()
