"""Chat→owner resolution goes through ONE seam (R3).

``server.py`` used to compare an inbound chat id against ``project["chat_id"]``
directly, which sees thread #0 ONLY — every message to any other thread of a
project would have been classified as Main and scoped to no project. These
tests pin the seam and the behaviour that depends on it.
"""

from __future__ import annotations

import types

from ouroboros.projects_registry import (
    begin_project_deletion,
    create_project,
    create_thread,
)


def _ctx(tmp_path):
    return types.SimpleNamespace(DRIVE_ROOT=tmp_path)


def test_every_thread_of_a_project_classifies_as_that_project(tmp_path):
    import server

    project = create_project(tmp_path, "racer", name="Cyber Racer")
    thread = create_thread(tmp_path, "racer", name="Tuning")
    ctx = _ctx(tmp_path)

    assert server._project_id_for_registered_chat(ctx, project["chat_id"]) == "racer"
    assert server._project_id_for_registered_chat(ctx, thread["chat_id"]) == "racer"
    # Main and unknown transport ids stay unscoped.
    assert server._project_id_for_registered_chat(ctx, 1) == ""
    assert server._project_id_for_registered_chat(ctx, 987654321) == ""


def test_reserved_lookup_answers_for_a_thread_of_a_fenced_project(tmp_path):
    import server

    create_project(tmp_path, "racer")
    thread = create_thread(tmp_path, "racer", name="Tuning")
    begin_project_deletion(tmp_path, "racer")
    ctx = _ctx(tmp_path)

    # Fenced: ordinary routing must NOT resurrect the room...
    assert server._project_id_for_registered_chat(ctx, thread["chat_id"]) == ""
    # ...but the reserved lookup still identifies it, so the caller can emit the
    # typed "project_unavailable" receipt instead of silently answering as Main.
    reserved = server._reserved_project_for_chat(ctx, thread["chat_id"])
    assert reserved.get("id") == "racer"
    assert reserved.get("lifecycle") == "deleting"


def test_owner_notices_from_a_thread_still_bind_to_main(tmp_path):
    """A WEB owner message in ANY project thread keeps binding owner notices to
    Main (1) — the behaviour that broke when a thread misclassified as Main."""
    import server

    create_project(tmp_path, "racer")
    thread = create_thread(tmp_path, "racer", name="Tuning")
    ctx = _ctx(tmp_path)

    assert server._owner_binding_chat_id(ctx, thread["chat_id"], False) == 1
    assert server._owner_binding_chat_id(ctx, thread["chat_id"], True) == thread["chat_id"]


class _PastTheGate(Exception):
    """Raised from the first thing `_route_owner_message` does AFTER admission."""


def _route(tmp_path, chat_id):
    """Send one owner message into `chat_id`. Returns the routing receipts.

    `inject_observation` is the first statement past the admission gate, so a
    `_PastTheGate` escaping means the message was ADMITTED; returning normally
    means it was refused, and the receipts say with what.
    """
    import server

    receipts = []

    class _Bridge:
        def broadcast(self, payload):
            receipts.append(payload)

    def _observe(*_args, **_kwargs):
        raise _PastTheGate()

    ctx = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        consciousness=types.SimpleNamespace(inject_observation=_observe),
    )
    admitted = False
    try:
        server._route_owner_message(_Bridge(), ctx, {
            "chat_id": int(chat_id), "text": "still here?", "client_message_id": "m-1",
        })
    except _PastTheGate:
        admitted = True
    return admitted, receipts


def test_a_thread_fenced_for_deletion_stops_admitting_owner_messages(tmp_path):
    """T3R2-B2: the fence must fence the ADMISSION GATE, not only the classifier.

    `begin_thread_deletion` and `api_thread_delete` both state in prose that
    marking `deleting` closes routing before cancellation starts. It did not:
    `_project_id_for_registered_chat` honours `thread_lifecycle`, but that helper
    CLASSIFIES — the gate is `_route_owner_message`, and it read the PROJECT's
    lifecycle alone. A message landed in a `deleting` thread carrying project
    scope, and a promotion could queue a task whose chat_id is a room that is
    already gone from every surface.
    """
    from ouroboros.projects_registry import begin_thread_deletion, complete_thread_deletion

    create_project(tmp_path, "racer")
    thread = create_thread(tmp_path, "racer", name="Tuning")
    cid = int(thread["chat_id"])

    # Control: an ACTIVE thread of an active project is admitted.
    admitted, _ = _route(tmp_path, cid)
    assert admitted is True

    begin_thread_deletion(tmp_path, "racer", thread["id"])
    admitted, receipts = _route(tmp_path, cid)
    assert admitted is False, "a fenced thread must not admit an owner message"
    assert [r["status"] for r in receipts] == ["project_unavailable"]
    assert receipts[0]["chat_id"] == cid

    # ...and a TOMBSTONED one, which is gone from every surface, likewise.
    complete_thread_deletion(tmp_path, "racer", thread["id"])
    admitted, receipts = _route(tmp_path, cid)
    assert admitted is False
    assert [r["status"] for r in receipts] == ["project_unavailable"]


def test_an_archived_thread_still_admits_owner_messages(tmp_path):
    """Archiving HIDES a thread; it does not close it. The fence is for the two
    states the owner has written off, and widening it would make an archived
    thread a room the owner can open and not speak in."""
    from ouroboros.projects_registry import archive_thread

    create_project(tmp_path, "racer")
    thread = create_thread(tmp_path, "racer", name="Tuning")
    archive_thread(tmp_path, "racer", thread["id"])

    admitted, _ = _route(tmp_path, int(thread["chat_id"]))

    assert admitted is True
