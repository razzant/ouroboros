"""S42: a direct Project turn's saved work continues after the application stops and the owner resumes.

A keyless real server runs the committed checkout (``e2e_clone``). The owner creates a
Project and writes into its room over the SAME /ws the SPA opens, so the turn is the
direct lane's own. Its first round writes a file in the room; its second model round is
held by a ``ModelGate`` while the application stops (SIGTERM to the whole tree). The next
server on the same data root restores the saved turn, the owner presses Resume over
HTTP, and the same task id continues its saved loop on a worker and persists its answer.

Doubled, and only these: the model is the loopback stub and its held round. The stop,
restore, Resume, Project-hold revalidation, assignment and result persistence are the
shipped ones. Before #2547 Resume answered ``ok`` and the row stayed pending under
``project_routing_fence_lookup_failed``.
"""

from __future__ import annotations

import json
import subprocess
import uuid

import pytest

from devtools.benchmarks.common.server_runner import _api, _api_status
from tests.system_e2e.harness import (
    LANE_MOCK,
    ArtifactOracle,
    ModelGate,
    ScriptedStubModel,
    body_text,
    keyless_settings,
    require_lane,
    start_server,
    wait_durable_result,
    wait_until,
    ws_url,
)

MARK = "S2547_ROOM_TURN"
FINAL = "S2547_CONTINUED_AFTER_RESUME"
WORK = {"tool": "write_file", "arguments": {
    "root": "active_workspace", "path": "s2547-progress.txt", "content": "written before the stop"}}


class _RoomStub(ScriptedStubModel):
    """The room turn's first round works; every later round of it answers."""

    def __init__(self, gate: ModelGate) -> None:
        super().__init__(gate=gate)
        self.served: list = []

    def _next_step(self, body):
        if MARK not in body_text(body):
            return None
        step = dict(WORK) if not self.served else {"final": FINAL}
        self.served.append("work" if "tool" in step else "final")
        return step


def _worked(body: dict) -> bool:
    """A round of the room turn after its write landed: the one the stop catches."""
    return bool(body.get("tools")) and MARK in body_text(body) and "s2547-progress.txt" in json.dumps(
        [m for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "tool"])


def _received(oracle: ArtifactOracle, message_id: str) -> dict | None:
    for row in oracle.events("task_received"):
        task = row.get("task") or {}
        if ((task.get("metadata") or {}).get("origin_message_ref") or {}).get("client_message_id") == message_id:
            return task
    return None


def _pending(oracle: ArtifactOracle, task_id: str) -> dict:
    rows = oracle.queue_snapshot().get("pending") or []
    return next((dict(row.get("task") or {}) for row in rows if isinstance(row, dict) and row.get("id") == task_id), {})


def _saved(data_root, task_id: str) -> list:
    from ouroboros.working_checkpoint import _working_dir

    directory = _working_dir(data_root, task_id)
    return sorted(path.name for path in directory.glob("*.json")) if directory.is_dir() else []


@pytest.mark.integration
@pytest.mark.serial
def test_s42_direct_project_turn_continues_after_stop_restart_and_owner_resume(e2e_clone, tmp_path):
    require_lane(LANE_MOCK)
    room = tmp_path / "room"
    room.mkdir()
    subprocess.run(["git", "init", "-q", str(room)], check=True, capture_output=True)
    gate = ModelGate(_worked, timeout=900)
    root = tmp_path / "server"
    with _RoomStub(gate) as stub:
        settings = keyless_settings(stub, OUROBOROS_MAX_WORKERS=1)
        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            project = _api(server.base_url, "POST", "/api/projects",
                           {"name": "S2547 room", "path": str(room)}, timeout=60).get("project") or {}
            assert project.get("id") and project.get("chat_id"), project
            from websockets.sync.client import connect

            message_id = "e2e-s2547-" + uuid.uuid4().hex
            with connect(ws_url(server), open_timeout=30, proxy=None) as ws:
                ws.send(json.dumps({"type": "chat", "client_message_id": message_id,
                                    "content": f"{MARK}: write the progress note in this room, then report.",
                                    "chat_id": project["chat_id"], "project_id": project["id"]}))
                task = wait_until(lambda: _received(oracle, message_id), 60)
                assert task and task.get("_is_direct_chat") is True and task.get("project_id") == project["id"], task
                task_id = task["id"]
                assert gate.arrived.wait(240), "the room turn never reached its second round"
            assert (room / "s2547-progress.txt").read_text(encoding="utf-8") == "written before the stop"
            assert wait_until(lambda: _saved(server.data_root, task_id), 60), "no working state was saved"
            assert task_id not in oracle.running_ids(), "the direct turn acquired a pool worker"
            before_stop = oracle.task_result(task_id)
            assert before_stop.get("status") == "running" and before_stop.get("_is_direct_chat") is True
        finally:
            server.stop()  # the application stops with the turn inside its held round

        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            restored = wait_until(lambda: _pending(oracle, task_id), 60)
            assert restored and restored.get("project_id") == project["id"], restored
            assert restored.get("_working_recovery"), restored
            resumed = _api_status(server.base_url, "POST", f"/api/tasks/{task_id}/resume", {}, timeout=60)
            body = resumed.get("body") or {}
            assert resumed["status"] == 200 and body.get("ok") is True, resumed
            stored = wait_until(
                lambda: (lambda row: row if row.get("status") in {"completed", "failed", "cancelled"} else None)(
                    oracle.task_result(task_id)), 180)
            held = _pending(oracle, task_id)
            assert stored, {"resume": {key: body.get(key) for key in ("ok", "selection", "same_generation")},
                            "still_pending": bool(held), "project_hold": held.get("_project_admission_restore_hold"),
                            "row_has_basis": "_project_admission" in held,
                            "result_has_basis": "_project_admission" in oracle.task_result(task_id)}
            stored = wait_durable_result(oracle, task_id, timeout=60)
            assert stored.get("status") == "completed", stored
            assert FINAL in str(stored.get("result") or ""), stored
            assert stub.served[:2] == ["work", "final"], "one write before the stop, the answer after Resume"
            assert int(stored.get("task_attempt") or 0) >= 2, "the saved continuation ran as the next attempt"
            assert not _pending(oracle, task_id), "nothing of the task stays queued"
        finally:
            gate.release.set()
            server.stop()
