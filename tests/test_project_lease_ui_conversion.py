"""Only pending work changes lease scope at placement; running scope stays pinned."""

import asyncio
import json
from types import SimpleNamespace
import pytest

@pytest.fixture(autouse=True)
def _redirect_queue_snapshot(tmp_path, monkeypatch):
    """Every handler test here calls the real ``api_project_from_task``, which now
    persists the queue snapshot after its in-memory lease mark. Redirect
    ``supervisor.queue``'s snapshot writer to a throwaway temp path so no test can write
    (or silently swallow a failed write to) the live ``state/queue_snapshot.json`` under
    the real data root. A test that asserts on the snapshot re-points it explicitly."""
    import supervisor.queue as queue

    snap = tmp_path / "_queue_isolation" / "queue_snapshot.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", snap)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)


def _request(tmp_path, body):
    async def _json():
        return body

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)),
        json=_json,
    )


def test_ui_conversion_scopes_a_pending_task(tmp_path, monkeypatch):
    """codex finding: a task converted while still PENDING (not yet RUNNING) must get
    its project_id on the PENDING dict too — assign_tasks reads the candidate's own
    project_id and copies it into RUNNING, so without this the queued task starts
    unscoped and never holds its lane."""
    from ouroboros.gateway.projects import api_project_from_task
    import supervisor.workers as workers

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "chat.jsonl").write_text("", encoding="utf-8")

    pending_task = {"id": "tq", "project_id": "", "type": "task"}
    monkeypatch.setattr(workers, "PENDING", [pending_task])

    resp = asyncio.run(api_project_from_task(_request(
        tmp_path, {"task_id": "tq", "id": "task-tq", "objective_hint": "queued work"},
    )))
    pid = json.loads(resp.body.decode("utf-8"))["project"]["id"]

    # the PENDING task dict is now scoped -> it will occupy the lane once assigned
    assert pending_task["project_id"] == pid
    assert workers.PENDING[0]["project_id"] == pid


def test_ui_conversion_persists_pending_scope_across_restart(tmp_path, monkeypatch):
    """scope finding: marking a PENDING converted task only in memory is not enough.
    restore_pending_from_snapshot rebuilds PENDING from state/queue_snapshot.json on
    restart, and assignment reads task['project_id'] from THERE (never the durable
    bindings). So the convert path must persist the snapshot right after the in-memory
    mark — otherwise a restart in the window (the snapshot is only rewritten on the next
    queue event) restores the task UNSCOPED and it never holds its lane."""
    from ouroboros.gateway.projects import api_project_from_task
    import supervisor.queue as queue
    import supervisor.workers as workers

    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "chat.jsonl").write_text("", encoding="utf-8")
    snap = tmp_path / "state" / "queue_snapshot.json"
    snap.parent.mkdir(parents=True, exist_ok=True)

    pending_task = {"id": "tq", "project_id": "", "type": "task", "chat_id": 9}
    pending_list = [pending_task]
    # Point BOTH the gateway's view (workers.*) and the snapshot writer (queue.*) at the
    # SAME live list, as init_queue_refs does in production.
    monkeypatch.setattr(workers, "PENDING", pending_list)
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(queue, "PENDING", pending_list)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", snap)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})

    resp = asyncio.run(api_project_from_task(_request(
        tmp_path, {"task_id": "tq", "id": "task-tq", "objective_hint": "queued work"},
    )))
    pid = json.loads(resp.body.decode("utf-8"))["project"]["id"]

    # the mark reached the persisted snapshot, not just the in-memory list
    assert snap.exists()
    saved = json.loads(snap.read_text(encoding="utf-8"))
    assert saved["pending"][0]["task"]["project_id"] == pid

    # simulate a restart: empty live queue, restore from the snapshot -> STILL scoped
    pending_list.clear()
    assert queue.restore_pending_from_snapshot() == 1
    assert pending_list[0]["project_id"] == pid


def test_running_mark_is_noop_pending_mark_follows_binding():
    from ouroboros.project_lease import mark_task_project, running_project_ids

    running = {"one": {"task": {"id": "one", "project_id": "old"}}}
    pending = [{"id": "two", "project_id": "old"}]
    assert not mark_task_project(running, pending, "one", "new", authority="binding")
    assert running_project_ids(running.values()) == {"old"}
    assert not mark_task_project(running, pending, "two", "new")
    assert mark_task_project(running, pending, "two", "new", authority="binding")
    assert pending[0]["project_id"] == "new"


_OWNER_TEXT = "Publish and merge the seven pull requests"


def _origin_ref(client_message_id: str = "msg-owner", text: str = _OWNER_TEXT, chat_id: int = 1) -> dict:
    from ouroboros.project_dialogue import build_owner_message_ref

    return build_owner_message_ref(
        chat_id=chat_id, client_message_id=client_message_id,
        ts="2026-09-14T12:15:20+00:00", text=text,
    )


def _seed_origin_task(tmp_path, task_id: str, ref: dict, title: str = "Ship the queue") -> None:
    """Persist the ingress-captured origin the convert path reads by value. An empty
    title leaves the naming step to the (patchable) model call."""
    from ouroboros.task_results import write_task_result

    write_task_result(
        tmp_path, task_id, "running",
        origin_message_ref=ref, origin_message_text=_OWNER_TEXT, **({"title": title} if title else {}),
    )


def _live_queue(monkeypatch, tmp_path, running: dict, pending: list) -> None:
    import supervisor.queue as queue
    import supervisor.workers as workers

    for mod in (workers, queue):
        monkeypatch.setattr(mod, "RUNNING", running)
        monkeypatch.setattr(mod, "PENDING", pending)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})


def _root(task_id: str, ref: dict, project_id: str = "") -> dict:
    return {"id": task_id, "project_id": project_id, "type": "task",
            "delegation_role": "root", "chat_id": 1, "origin_message_ref": dict(ref)}


def _convert(tmp_path, task_id: str, **body):
    payload = {"task_id": task_id, "id": f"task-{task_id}", "objective_hint": _OWNER_TEXT}
    payload.update(body)
    from ouroboros.gateway.projects import api_project_from_task

    return asyncio.run(api_project_from_task(_request(tmp_path, payload)))


def _events(tmp_path, kind: str) -> list:
    path = tmp_path / "logs" / "events.jsonl"
    if not path.exists():
        return []
    return [row for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            if (row := json.loads(line)).get("type") == kind]
