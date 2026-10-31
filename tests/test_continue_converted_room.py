"""Continue of a task the owner turned into a Project stays in that room (Batch4 1A).

"Turn into project" writes only the immutable task<->Project binding; the
interrupted worker's own terminal result keeps the chat and project copy it
started with (``agent_task_pipeline`` publishes its local task). Driven through
the real conversion and Continue endpoints on an installed queue: a fresh
Continue admits its new root into the room the binding names (claim, queue row,
successor binding, live and late event addressing), keeping the predecessor's
original folder, exact owner words and money group; a claim frozen before a
later conversion keeps its destination; unreadable or closed room authority
refuses instead of falling back to Main.
"""

from __future__ import annotations

import copy
import json
import queue as stdqueue
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue

pytestmark = pytest.mark.serial  # installs the process-wide queue/state roots

NONCE = "press-0001-abcdef"
TEXT = "Write the Friday report"
BILLING = {"billing_group_id": "pred-1", "billing_group_limit_usd": 20.0,
           "billing_group_limit_source": "initial_task_admission", "billing_group_limit_revision": "admission-1"}


@pytest.fixture
def host(tmp_path, monkeypatch):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.gateway.task_continue import api_task_continue

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    # Terminal delivery's transport only; the durable outbox and routing stay real.
    monkeypatch.setattr(workers, "get_event_q", lambda: stdqueue.Queue())
    app = Starlette(routes=[Route("/api/projects/from-task", api_project_from_task, methods=["POST"]),
                            Route("/api/tasks/{task_id}/continue", api_task_continue, methods=["POST"])])
    app.state.drive_root = tmp_path
    folder = tmp_path / "original-folder"
    folder.mkdir()
    return SimpleNamespace(root=tmp_path, queue=queue, workers=workers, client=TestClient(app), folder=folder)


def _start(host, *, chat_id=1, project_id=""):
    """A running root as its worker sees it: the RUNNING lane and its own result."""
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    ref = build_owner_message_ref(chat_id=chat_id, client_message_id="m-1", ts="2026-10-01T00:00:00Z", text=TEXT)
    task = {"id": "pred-1", "type": "task", "chat_id": chat_id, "project_id": project_id,
            "root_task_id": "pred-1", "workspace_root": str(host.folder), "origin_message_ref": ref,
            "origin_message_text": TEXT, "title": "Friday report", "text": TEXT}
    write_task_result(host.root, "pred-1", STATUS_RUNNING, billing_group=dict(BILLING),
                      **{key: value for key, value in task.items() if key != "id"})
    host.workers.RUNNING["pred-1"] = {"task": copy.deepcopy(task)}
    return task


def _convert(host):
    response = host.client.post("/api/projects/from-task", json={"task_id": "pred-1", "name": "Friday report"})
    assert response.status_code == 200, response.text
    return response.json()["project"]


def _interrupt(host, worker_task):
    """The worker publishes ITS copy of the task; a conversion never reached it."""
    from ouroboros.task_results import write_task_result

    write_task_result(host.root, "pred-1", "failed", result="interrupted", reason_code="round_limit",
                      project_id=str(worker_task.get("project_id") or ""), root_task_id="pred-1",
                      workspace_root=worker_task["workspace_root"])
    host.workers.RUNNING.clear()


def _continue(host, nonce=NONCE):
    return host.client.post("/api/tasks/pred-1/continue", json={"action_nonce": nonce})


def _row(host, task_id):
    return next(row for row in host.workers.PENDING if row["id"] == task_id)


def _room_of(host, task_id, row=None):
    """Where the real addressing seams deliver this task's frames."""
    from ouroboros.projects_registry import project_chat_for_task_tree
    from supervisor.log_addressing import address_task_event
    from supervisor.terminal_delivery import lineage_chat_id

    running = {task_id: {"task": row}} if row is not None else {}
    return {
        "live": address_task_event(running, host.root, {"type": "tool_call_started", "task_id": task_id}).get("chat_id"),
        "late": address_task_event({}, host.root, {"type": "task_done", "task_id": task_id}).get("chat_id"),
        "child": address_task_event({}, host.root, {"type": "tool_call_started", "task_id": "helper-1",
                                                    "parent_task_id": task_id, "root_task_id": task_id}).get("chat_id"),
        "tree": project_chat_for_task_tree(host.root, task_id, "", task_id),
        "delivery": lineage_chat_id(host.root, {"root_task_id": task_id, "parent_task_id": task_id}, "helper-1"),
    }


@pytest.mark.parametrize("raw_project", ["", "proj_0123456789ab", "other"])
def test_continue_of_a_converted_task_admits_its_new_root_in_the_converted_room(host, raw_project):
    """The worker's stale copy says Main (empty), a derived workspace scope, or
    another registered room; the conversion's binding names the actual room."""
    from ouroboros.projects_registry import create_project, project_binding_for_task, update_project
    from ouroboros.task_results import load_task_result

    raw_chat = 1
    if raw_project == "other":
        raw_chat = create_project(host.root, "other", name="Another room")["chat_id"]
    worker_task = _start(host, chat_id=raw_chat, project_id=raw_project)
    project = _convert(host)
    pid, chat = project["id"], project["chat_id"]
    # Placement is the durable binding; the executing lane keeps its lease until its next
    # execution boundary (D5: no relabelling of an executing lease).
    assert project_binding_for_task(host.root, "pred-1")["project_id"] == pid
    assert host.workers.RUNNING["pred-1"]["task"]["project_id"] == raw_project, "the conversion relabelled the live lane"
    # The room's folder may be (re)attached later; Continue keeps the prepared one.
    elsewhere = host.root / "room-folder"
    elsewhere.mkdir()
    assert update_project(host.root, pid, working_dir=str(elsewhere))
    _interrupt(host, worker_task)
    before = load_task_result(host.root, "pred-1")
    assert (before["chat_id"], before["project_id"]) == (raw_chat, raw_project), "a stale copy, by design"

    response = _continue(host)

    assert response.status_code == 200, response.text
    successor = response.json()["successor_task_id"]
    row = _row(host, successor)
    assert (row["chat_id"], row["project_id"], row["metadata"]["project_id"]) == (chat, pid, pid)
    basis = row["_project_admission"]
    assert basis["project_id"] == pid and basis["project"]["chat_id"] == chat and not basis.get("legacy_basis")
    assert not row.get("_project_scope_none")
    claim = load_task_result(host.root, "pred-1")["continued_by"]
    assert (claim["binding"]["chat_id"], claim["binding"]["project_id"]) == (chat, pid)
    stored = load_task_result(host.root, successor)
    assert (stored["chat_id"], stored["project_id"]) == (chat, pid)
    # The successor is bound to the same room, by the existing writer, with the
    # ORIGINAL owner message (its own chat) as the binding's origin.
    predecessor_binding = project_binding_for_task(host.root, "pred-1")
    bound = project_binding_for_task(host.root, successor)
    assert (bound["project_id"], bound["project_chat_id"]) == (pid, chat)
    assert bound["source_ref"] == predecessor_binding["source_ref"] == worker_task["origin_message_ref"]
    assert bound["source_text"] == TEXT
    assert _room_of(host, successor, row) == {"live": chat, "late": chat, "child": chat, "tree": chat,
                                              "delivery": chat}
    # Only the room moved: folder, owner words/ref, money and identity are the predecessor's.
    assert row["workspace_root"] == str(host.folder) and claim["binding"]["workspace_root"] == str(host.folder)
    meta = row["metadata"]
    assert meta["origin_message_ref"] == worker_task["origin_message_ref"]
    assert meta["origin_message_ref"]["chat_id"] == raw_chat, "conversion never relabels authorship"
    assert meta["continuation"]["owner_sources"]["original"]["content"] == TEXT
    assert meta["owner_corpus"] == [{"source": "origin_message", "content": TEXT}]
    assert {key: meta["continuation"][key] for key in BILLING} == BILLING
    assert row["root_task_id"] == successor and not row.get("parent_task_id")
    after = load_task_result(host.root, "pred-1")
    assert (after["chat_id"], after["project_id"]) == (raw_chat, raw_project), "the predecessor is not rewritten"

    # The same press replays the same admission; no second root, no retarget.
    again = _continue(host)
    assert again.status_code == 200 and again.json()["replay"] is True
    assert again.json()["successor_task_id"] == successor
    assert [task["id"] for task in host.workers.PENDING] == [successor]
    assert load_task_result(host.root, "pred-1")["continued_by"]["binding"] == claim["binding"]
    assert project_binding_for_task(host.root, successor) == bound


@pytest.mark.parametrize("case", ["main", "native", "native-bound"])
def test_main_and_native_project_continue_keep_their_own_room(host, case):
    from ouroboros.projects_registry import bind_task_to_project, create_project, project_binding_for_task

    chat, pid = 1, ""
    if case != "main":
        project = create_project(host.root, "native", name="Native room", working_dir=str(host.folder))
        chat, pid = project["chat_id"], project["id"]
    worker_task = _start(host, chat_id=chat, project_id=pid)
    if case == "native-bound":  # a room-born root is bound to its own room
        bind_task_to_project(host.root, "pred-1", pid, chat, origin={"ref": worker_task["origin_message_ref"]})
    _interrupt(host, worker_task)

    response = _continue(host)

    assert response.status_code == 200, response.text
    successor = response.json()["successor_task_id"]
    row = _row(host, successor)
    assert (row["chat_id"], row["project_id"]) == (chat, pid)
    assert row["workspace_root"] == str(host.folder)
    assert row["metadata"]["origin_message_ref"] == worker_task["origin_message_ref"]
    bound = project_binding_for_task(host.root, successor)
    assert (bound or {}).get("project_id") == (pid if case == "native-bound" else None)
    rooms = _room_of(host, successor, row)
    assert rooms["live"] == chat
    if case == "main":
        assert rooms["tree"] == 0, "Main work gains no Project room"
    elif case == "native":  # a room-born root is addressed to its room unbound (the one room rule);
        # an id the store never admitted ("helper-1") has no room of its own yet: the caller keeps its address
        assert {key: rooms[key] for key in ("live", "late", "tree", "child")} == {"live": chat, "late": chat, "tree": chat, "child": None}
    else:
        assert rooms == {"live": chat, "late": chat, "child": chat, "tree": chat, "delivery": chat}


def test_a_claim_frozen_before_a_later_conversion_keeps_its_destination(host, monkeypatch):
    """Admitted or only claimed (its first snapshot failed): converting the
    predecessor afterwards never retargets the recorded Continue."""
    from ouroboros.projects_registry import project_binding_for_task
    from ouroboros.task_results import load_task_result

    worker_task = _start(host)
    _interrupt(host, worker_task)
    real_persist = host.queue.persist_queue_snapshot
    monkeypatch.setattr(host.queue, "persist_queue_snapshot", lambda reason="": False)
    failed = _continue(host)
    assert failed.status_code == 503 and failed.json()["reason_code"] == "queue_snapshot_persist_failed"
    frozen = load_task_result(host.root, "pred-1")["continued_by"]
    assert frozen["state"] == "bound" and (frozen["binding"]["chat_id"], frozen["binding"]["project_id"]) == (1, "")
    monkeypatch.setattr(host.queue, "persist_queue_snapshot", real_persist)
    project = _convert(host)
    assert project_binding_for_task(host.root, "pred-1")["project_id"] == project["id"]

    retried = _continue(host)

    assert retried.status_code == 200, retried.text
    successor = retried.json()["successor_task_id"]
    assert successor == frozen["successor_task_id"]
    row = _row(host, successor)
    assert (row["chat_id"], row["project_id"]) == (1, "") and row["_project_scope_none"] is True
    assert project_binding_for_task(host.root, successor) is None
    assert load_task_result(host.root, "pred-1")["continued_by"]["binding"] == frozen["binding"]
    assert _room_of(host, successor, row)["live"] == 1
    # Admitted now; a later replay is still the same Main admission.
    replay = _continue(host)
    assert replay.json()["replay"] is True and replay.json()["successor_task_id"] == successor
    assert [task["id"] for task in host.workers.PENDING] == [successor]
    assert project_binding_for_task(host.root, successor) is None


def test_an_admitted_continue_replays_unchanged_after_its_predecessor_is_converted(host):
    from ouroboros.projects_registry import project_binding_for_task
    from ouroboros.task_results import load_task_result

    worker_task = _start(host)
    _interrupt(host, worker_task)
    first = _continue(host)
    assert first.status_code == 200, first.text
    successor = first.json()["successor_task_id"]
    admitted = copy.deepcopy(load_task_result(host.root, successor)["continuation_admission"])
    _convert(host)

    replay = _continue(host)

    assert replay.status_code == 200 and replay.json()["replay"] is True
    assert replay.json()["successor_task_id"] == successor
    assert load_task_result(host.root, successor)["continuation_admission"] == admitted
    row = _row(host, successor)
    assert (row["chat_id"], row["project_id"]) == (1, "") and [t["id"] for t in host.workers.PENDING] == [successor]
    assert project_binding_for_task(host.root, successor) is None
    assert _room_of(host, successor, row)["live"] == 1


def test_a_failed_first_snapshot_retries_the_same_nonce_into_the_same_room(host, monkeypatch):
    from ouroboros.projects_registry import project_binding_for_task
    from ouroboros.task_results import load_task_result

    worker_task = _start(host)
    project = _convert(host)
    _interrupt(host, worker_task)
    real_persist = host.queue.persist_queue_snapshot
    monkeypatch.setattr(host.queue, "persist_queue_snapshot", lambda reason="": False)
    failed = _continue(host)
    assert failed.status_code == 503 and failed.json()["reason_code"] == "queue_snapshot_persist_failed"
    successor = failed.json()["successor_task_id"]
    assert host.workers.PENDING == [] and load_task_result(host.root, successor) is None
    claim = load_task_result(host.root, "pred-1")["continued_by"]
    assert claim["state"] == "bound"
    assert (claim["binding"]["chat_id"], claim["binding"]["project_id"]) == (project["chat_id"], project["id"])
    assert _continue(host, "another-press-99").json()["reason_code"] == "already_continued"
    monkeypatch.setattr(host.queue, "persist_queue_snapshot", real_persist)

    retried = _continue(host)

    assert retried.status_code == 200 and retried.json()["successor_task_id"] == successor
    row = _row(host, successor)
    assert (row["chat_id"], row["project_id"]) == (project["chat_id"], project["id"])
    assert [task["id"] for task in host.workers.PENDING] == [successor]
    assert project_binding_for_task(host.root, successor)["project_chat_id"] == project["chat_id"]
    assert load_task_result(host.root, "pred-1")["continued_by"]["state"] == "admitted"


@pytest.mark.parametrize("damage,error,claimed", [
    ("bindings_unreadable", "project_routing_fence_lookup_failed", False),
    ("binding_malformed", "project_routing_fence_lookup_failed", False),
    ("project_missing", "project_routing_fence_changed", True),
    ("project_closed", "project_routing_fence", True),
])
def test_unreadable_missing_or_closed_room_refuses_without_falling_back_to_main(host, damage, error, claimed):
    from ouroboros import projects_registry as registry
    from ouroboros.task_results import load_task_result

    worker_task = _start(host)
    project = _convert(host)
    _interrupt(host, worker_task)
    bindings = registry._bindings_path(host.root)
    if damage == "bindings_unreadable":
        bindings.write_text("{torn", encoding="utf-8")
    elif damage == "binding_malformed":
        data = json.loads(bindings.read_text(encoding="utf-8"))
        data["bindings"]["pred-1"]["project_chat_id"] = "not-a-room"
        bindings.write_text(json.dumps(data), encoding="utf-8")
    elif damage == "project_missing":
        path = registry._registry_path(host.root)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["projects"] = [row for row in data["projects"] if row["id"] != project["id"]]
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        assert registry.begin_project_deletion(host.root, project["id"])["lifecycle"] == "deleting"

    refused = _continue(host)

    assert refused.status_code == 409 and refused.json()["reason_code"] == error, refused.text
    assert host.workers.PENDING == [] and not host.queue.ADMISSION_RESERVATIONS
    claim = load_task_result(host.root, "pred-1").get("continued_by")
    assert bool(claim) is claimed
    if claimed:  # the claim keeps the actual room for the same nonce; it never becomes Main
        assert (claim["binding"]["chat_id"], claim["binding"]["project_id"]) == (project["chat_id"], project["id"])
        assert registry.project_binding_for_task(host.root, claim["successor_task_id"]) is None
