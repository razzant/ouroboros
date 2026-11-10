"""S42: a direct Project turn's saved work continues after the application stops and the owner resumes.

A keyless real server runs the committed checkout (``e2e_clone``). The owner creates a
Project and writes into its room over the SAME /ws the SPA opens, so the turn is the
owner's own: since #2654 a conversation-lane carrier process holds its published RUNNING
row. Its first round writes a file in the room; its second model round is held by a
``ModelGate`` while the application stops (SIGTERM to the whole tree). The next server on
the same data root restores the saved turn under exactly ONE queue row, the owner
presses Resume over HTTP, and the same task id continues its saved loop on a worker and
persists its answer.

Doubled, and only these: the model is the loopback stub and its held round. The stop,
restore, Resume, Project-hold revalidation, assignment and result persistence are the
shipped ones. Before #2547 Resume answered ``ok`` and the row stayed pending under
``project_routing_fence_lookup_failed``.
"""

from __future__ import annotations

import contextlib
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
    conversation_turn_row,
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


def _restored_once(oracle: ArtifactOracle, task_id: str) -> dict:
    """The restored turn's one PENDING row. The stop published the lane turn's RUNNING row and
    the next boot recovers its saved work from it; a second successor of the same id is a
    duplicate custody owner, which leaves recovery closed for both ("Another queue owner
    holds this task id") and Resume answering ok over a turn that never runs."""
    rows = oracle.pending_rows(task_id)
    assert len(rows) == 1, {"pending_rows_for_one_task": len(rows), "rows": [
        {key: (row.get("task") or {}).get(key) for key in (
            "_attempt", "_working_recovery", "_project_admission_restore_hold")} for row in rows]}
    return dict(rows[0].get("task") or {})


@contextlib.contextmanager
def _released(gate: ModelGate):
    """Release the held round on every exit path. A stub request thread blocked in the gate is
    the scenario's own thread: a failure before the final release must not leave it alive past
    the test (conftest thread hygiene named ``process_request_thread`` after such a failure)."""
    try:
        yield gate
    finally:
        gate.release.set()


def _saved(data_root, task_id: str) -> list:
    from ouroboros.working_checkpoint import _working_dir

    directory = _working_dir(data_root, task_id)
    return sorted(path.name for path in directory.glob("*.json")) if directory.is_dir() else []


@pytest.mark.integration
def test_s42_direct_project_turn_continues_after_stop_restart_and_owner_resume(e2e_clone, tmp_path):
    require_lane(LANE_MOCK)
    room = tmp_path / "room"
    room.mkdir()
    subprocess.run(["git", "init", "-q", str(room)], check=True, capture_output=True)
    gate = ModelGate(_worked, timeout=900)
    root = tmp_path / "server"
    with _RoomStub(gate) as stub, _released(gate):
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
            # #2654: the room turn runs on a conversation-lane carrier, outside the
            # MAX_WORKERS=1 project pool, its RUNNING row published (never PENDING).
            conversation_turn_row(oracle, task_id)
            before_stop = oracle.task_result(task_id)
            assert before_stop.get("status") == "running" and before_stop.get("_is_direct_chat") is True
        finally:
            server.stop()  # the application stops with the turn inside its held round

        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            assert wait_until(lambda: _pending(oracle, task_id), 60), "the saved turn was not restored"
            restored = _restored_once(oracle, task_id)
            assert restored.get("project_id") == project["id"], restored
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


TURN_A, TURN_B, SIBLING = "S42_FOLDERLESS_TURN", "S42_SIBLING_STARTER", "S42_SIBLING_TASK"
PROOF = "s42-resource-proof.txt"
CARRIED = "[Owner words quoted verbatim"


def _request(body: dict) -> str:
    """The current request: the last user message naming one of this scenario's turns.
    Host notes (the request clock) follow it, and room history may quote earlier turns.
    A room row the host carries to the room's other live roots (#2924) arrives as quoted
    owner words under the ``CARRIED`` line: context for that turn, never its request."""
    from tests.system_e2e.harness import message_text

    texts = [message_text(m) for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]
    return next((text for text in reversed(texts) if not text.startswith(CARRIED)
                 and any(mark in text for mark in (TURN_A, TURN_B, SIBLING))), "")


class _FolderStub(ScriptedStubModel):
    """Turn A writes before the stop and again after Resume; turn B starts a sibling
    task with an explicit folder; the sibling answers at once."""

    def __init__(self, gate: ModelGate, folder) -> None:
        super().__init__(gate=gate)
        self.folder, self.served = str(folder), []

    def _next_step(self, body):
        request = _request(body)
        if SIBLING in request:
            return {"final": "sibling done"}
        if TURN_B in request:
            if "B:start" in self.served:
                return {"final": "sibling started"}
            self.served.append("B:start")
            return {"tool": "start_task", "arguments": {
                "objective": f"{SIBLING}: work in the explicit folder", "title": "S42 sibling",
                "workspace_root": self.folder}}
        if TURN_A not in request:
            return None
        writes = [item for item in self.served if item.startswith("A:")]
        content = ("before stop", "after resume")[len(writes)] if len(writes) < 2 else None
        self.served.append(f"A:{content or 'final'}")
        return ({"tool": "write_file", "arguments": {"root": "active_workspace", "path": PROOF, "content": content}}
                if content else {"final": FINAL})


def _turn_a_second_round(body: dict) -> bool:
    return (bool(body.get("tools")) and TURN_A in _request(body)
            and PROOF in json.dumps([m for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "tool"]))


def _send(server, project, text):
    from websockets.sync.client import connect

    message_id = "e2e-s42-" + uuid.uuid4().hex
    with connect(ws_url(server), open_timeout=30, proxy=None) as ws:
        ws.send(json.dumps({"type": "chat", "client_message_id": message_id, "content": text,
                            "chat_id": project["chat_id"], "project_id": project["id"]}))
    return message_id


@pytest.mark.integration
def test_s42_resumed_folderless_work_keeps_its_task_drive_after_a_sibling_gives_the_room_a_folder(e2e_clone, tmp_path):
    require_lane(LANE_MOCK)
    folder = tmp_path / "sibling-folder"
    folder.mkdir()
    gate = ModelGate(_turn_a_second_round, timeout=900)
    root = tmp_path / "server"
    with _FolderStub(gate, folder) as stub, _released(gate):
        settings = keyless_settings(stub, OUROBOROS_MAX_WORKERS=1)
        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            project = _api(server.base_url, "POST", "/api/projects", {"name": "S42 folderless room"},
                           timeout=60).get("project") or {}
            assert project.get("id") and not project.get("working_dir"), project
            message_id = _send(server, project, f"{TURN_A}: write the proof note in this room, then report.")
            task = wait_until(lambda: _received(oracle, message_id), 60)
            assert task and task.get("_is_direct_chat") is True, task
            task_id = task["id"]
            assert gate.arrived.wait(240), "turn A never reached its second round"
            drive = server.data_root / "task_drives" / task_id
            assert (drive / PROOF).read_text(encoding="utf-8") == "before stop"
            assert wait_until(lambda: _saved(server.data_root, task_id), 60), "no working state was saved"
            conversation_turn_row(oracle, task_id)  # held by a conversation-lane carrier when the stop lands
        finally:
            server.stop()

        server = start_server(e2e_clone, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            assert wait_until(lambda: _pending(oracle, task_id), 60), "turn A was not restored"
            _restored_once(oracle, task_id)
            _send(server, project, f"{TURN_B}: start the sibling task in its explicit folder.")
            room = wait_until(lambda: next((p for p in _api(server.base_url, "GET", "/api/projects", timeout=30)
                                            .get("projects") or [] if p.get("id") == project["id"]
                                            and p.get("working_dir")), None), 120)
            assert room and room["working_dir"] == str(folder), room
            resumed = _api_status(server.base_url, "POST", f"/api/tasks/{task_id}/resume", {}, timeout=60)
            assert resumed["status"] == 200 and (resumed.get("body") or {}).get("ok") is True, resumed
            stored = wait_until(lambda: (lambda row: row if row.get("status") in {"completed", "failed", "cancelled"}
                                         else None)(oracle.task_result(task_id)), 180)
            assert stored and stored.get("status") == "completed", {
                "result": oracle.task_result(task_id), "pending": _pending(oracle, task_id), "served": stub.served}
            assert (drive / PROOF).read_text(encoding="utf-8") == "after resume"
            assert not (folder / PROOF).exists(), "resumed work wrote into the folder the room acquired later"
            assert stub.served[:3] == ["A:before stop", "B:start", "A:after resume"], stub.served
        finally:
            gate.release.set()
            server.stop()


TURN_R, BG = "S42_RETURNED_ROOM_TURN", "S42_BACKGROUND_HOLDER"
FINAL_R = "S42_RETURNED_TURN_CONTINUED_BESIDE_BACKGROUND_WORK"


class _ReturnStub(ScriptedStubModel):
    """The room turn writes before the stop and answers after Resume; the background work answers once released."""

    def __init__(self, gate) -> None:
        super().__init__(gate=gate)
        self.served: list = []

    def _next_step(self, body):
        request = _current(body)
        if BG in request:
            self.served.append("BG:final")
            return {"final": "background work done"}
        if TURN_R not in request:
            return None
        step = dict(WORK) if not any(item == "R:work" for item in self.served) else {"final": FINAL_R}
        self.served.append("R:work" if "tool" in step else "R:final")
        return step


def _current(body: dict) -> str:
    """The request a model call serves: the first user message naming this scenario's work, the work's own ask
    (later host notes and carried owner rows are context, never its request)."""
    from tests.system_e2e.harness import message_text

    texts = [message_text(m) for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "user"]
    return next((text for text in texts if not text.startswith(CARRIED)
                 and any(mark in text for mark in (TURN_R, BG))), "")


class _Gates:
    """Both holds on one loopback model: the room turn's second round before the stop, the background round after."""

    def __init__(self, *gates) -> None:
        self.gates = gates

    def __call__(self, body: dict) -> None:
        for gate in self.gates:
            gate(body)


@pytest.fixture(scope="module")
def s42_candidate(tmp_path_factory):
    """The working tree under test, uncommitted changes included (``candidate_checkout`` with origin proof), on a
    dependency-only interpreter; a clean tree may serve its HEAD clone instead, a dirty one never."""
    from tests.candidate_checkout import CandidateError, candidate_checkout, require_candidate_interpreter
    from tests.system_e2e.harness import REPO_ROOT, clone_repo

    require_lane(LANE_MOCK)
    try:
        require_candidate_interpreter()
    except CandidateError as exc:
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO_ROOT,
                               check=True, capture_output=True, text=True).stdout
        if dirty.strip():
            pytest.fail(f"{exc} — and the working tree is dirty, so a HEAD clone would not be the candidate")
        yield clone_repo(tmp_path_factory.mktemp("s42"))
        return
    with candidate_checkout(REPO_ROOT, tmp_path_factory.mktemp("s42") / "candidate", origin_proof=True) as candidate:
        yield candidate


@pytest.mark.integration
def test_s42_a_returned_room_turn_continues_on_its_lane_while_background_work_holds_the_project_slot(
        s42_candidate, tmp_path):
    """#2870 on a real server: after a stop and my human's Resume, the returned room turn needs neither the only
    project slot (``MAX_WORKERS`` 1) nor the room's one-writer lease, both held by background work in the same room;
    it continues on a conversation-lane carrier, its own origin and carrier recorded by its body."""
    require_lane(LANE_MOCK)
    room = tmp_path / "room"
    room.mkdir()
    subprocess.run(["git", "init", "-q", str(room)], check=True, capture_output=True)

    def worked(body: dict) -> bool:
        return (bool(body.get("tools")) and TURN_R in _current(body) and "s2547-progress.txt" in json.dumps(
            [m for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "tool"]))

    held_turn = ModelGate(worked, timeout=900)
    held_background = ModelGate(lambda body: bool(body.get("tools")) and BG in _current(body), timeout=900)
    root = tmp_path / "server"
    with _ReturnStub(_Gates(held_turn, held_background)) as stub, _released(held_turn), _released(held_background):
        settings = keyless_settings(stub, OUROBOROS_MAX_WORKERS=1)
        server = start_server(s42_candidate, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            project = _api(server.base_url, "POST", "/api/projects",
                           {"name": "S42 returned room", "path": str(room)}, timeout=60).get("project") or {}
            assert project.get("id") and project.get("chat_id"), project
            from websockets.sync.client import connect

            message_id = "e2e-s42r-" + uuid.uuid4().hex
            with connect(ws_url(server), open_timeout=30, proxy=None) as ws:
                ws.send(json.dumps({"type": "chat", "client_message_id": message_id,
                                    "content": f"{TURN_R}: write the progress note in this room, then report.",
                                    "chat_id": project["chat_id"], "project_id": project["id"]}))
                task = wait_until(lambda: _received(oracle, message_id), 60)
                assert task and task.get("_is_direct_chat") is True, task
                task_id = task["id"]
                assert held_turn.arrived.wait(240), "the room turn never reached its second round"
            assert wait_until(lambda: _saved(server.data_root, task_id), 60), "no working state was saved"
            conversation_turn_row(oracle, task_id)
        finally:
            server.stop()  # a quit: the saved turn waits for my human's Resume

        server = start_server(s42_candidate, root, settings)
        try:
            oracle = ArtifactOracle(server.data_root)
            assert wait_until(lambda: _pending(oracle, task_id), 60), "the saved turn was not restored"
            restored = _restored_once(oracle, task_id)
            assert (restored.get("_execution_owner") or {}).get("lane") == "conversation", restored
            created = _api(server.base_url, "POST", "/api/tasks", {
                "description": f"{BG}: background work in this room.", "project_id": project["id"],
                "chat_id": project["chat_id"], "memory_mode": "forked", "actor_id": "e2e-driver",
                "source": "e2e-driver", "metadata": {"source": "e2e-driver", "delegation_role": "root"}}, timeout=60)
            background = str(created.get("task_id") or "")
            assert background and held_background.arrived.wait(240), {"created": created, "served": stub.served}
            assert background in oracle.running_ids(), "the background work holds the only project slot"
            resumed = _api_status(server.base_url, "POST", f"/api/tasks/{task_id}/resume", {}, timeout=60)
            assert resumed["status"] == 200 and (resumed.get("body") or {}).get("ok") is True, resumed
            stored = wait_until(
                lambda: (lambda row: row if row.get("status") in {"completed", "failed", "cancelled"} else None)(
                    oracle.task_result(task_id)), 180)
            held = _pending(oracle, task_id)
            assert stored, {"still_pending": bool(held), "project_hold": held.get("_project_admission_restore_hold"),
                            "served": stub.served, "running": sorted(oracle.running_ids())}
            stored = wait_durable_result(oracle, task_id, timeout=60)
            assert stored.get("status") == "completed" and FINAL_R in str(stored.get("result") or ""), stored
            owner = stored.get("execution_owner") or {}
            assert (owner.get("kind"), owner.get("lane"), owner.get("executor")) == (
                "direct", "conversation", "worker"), owner
            assert int(stored.get("task_attempt") or 0) == 2 == int(stored.get("admitted_dispatch_attempt") or 0)
            assert str(oracle.task_result(background).get("status") or "") not in {"completed", "failed", "cancelled"}
            assert background in oracle.running_ids(), "the background work still held its slot throughout"
            held_background.release.set()
            assert wait_durable_result(oracle, background, timeout=180).get("status") == "completed"
            assert stub.served.count("R:work") == 1, stub.served
        finally:
            held_turn.release.set()
            held_background.release.set()
            server.stop()
