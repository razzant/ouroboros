"""A transferred Main conversation retains the ordinary Project completion mirror."""
from __future__ import annotations

import json

import pytest

from ouroboros.project_dialogue import build_owner_message_ref, enqueue_project_completion_summary
from ouroboros.projects_registry import bind_task_to_project, create_project
from supervisor.terminal_delivery import enqueue_terminal_delivery_outcome as _ORIGINAL_OUTCOME


class _Queue:
    def __init__(self, refuse=False):
        self.rows, self.refuse = [], refuse

    def put(self, event):
        if self.refuse:
            raise RuntimeError("queue closed")
        self.rows.append(event)


def _main_origin(client_message_id="first"):
    return {"ref": build_owner_message_ref(chat_id=1, client_message_id=client_message_id,
                                           ts="2026-09-21T12:00:00Z", text="Investigate"),
            "text": "Investigate"}


@pytest.mark.parametrize("source_chat", [1, "project", 0, -42, None])
@pytest.mark.parametrize("direct_carrier", ["event", "task", "result", "done"])
def test_direct_completion_requires_a_durable_main_origin(tmp_path, monkeypatch, source_chat, direct_carrier):
    project = create_project(tmp_path, "research", name="Research")
    chat = project["chat_id"] if source_chat == "project" else source_chat
    origin = {"absent": "system"} if chat is None else {
        "ref": build_owner_message_ref(chat_id=chat, client_message_id="request-one",
                                       ts="2026-09-21T12:00:00Z", text="Investigate"),
        "text": "Investigate",
    }
    bind_task_to_project(tmp_path, "turn-one", project["id"], project["chat_id"], origin=origin)
    rows = []
    monkeypatch.setattr("supervisor.terminal_delivery.enqueue_terminal_delivery",
                        lambda root, event, **_transport: rows.append(event) or True)
    event = {}
    task = {"id": "turn-one", "chat_id": project["chat_id"], "project_id": project["id"]}
    result = {"status": "completed", "result": "# The complete answer\n\nEvidence remains intact.",
              "terminal_origin": "model_final", "project_id": project["id"]}
    done = {"status": "completed", "outcome_axes": {"execution": {"status": "ok"}}}
    {"event": event, "task": task, "result": result, "done": done}[direct_carrier]["_is_direct_chat"] = True
    admitted = enqueue_project_completion_summary(tmp_path, event, "turn-one", task, result, done)
    assert admitted is (source_chat == 1)
    assert len(rows) == int(source_chat == 1)
    if rows:
        assert rows[0]["system_type"] == "project_completion_summary"
        assert rows[0]["delivery_id"] == "project-completion:turn-one"
        assert rows[0]["chat_id"] == 1
        assert rows[0]["progress_meta"]["completion_answer"] == result["result"]


def test_handoff_identity_follows_origin_not_retry_task_or_text():
    from ouroboros.project_handoff import handoff_identity
    a = {"chat_id": 1, "client_message_id": "first"}
    assert handoff_identity("p", "root", a) == handoff_identity("p", "retry", {**a, "text_sha256": "other"})
    assert handoff_identity("p", "root", a) != handoff_identity("p", "root", {**a, "client_message_id": "second"})
    assert handoff_identity("p", "root", None) != handoff_identity("p", "retry", None)
    assert handoff_identity("p", "root", a) != handoff_identity("q", "root", a)


def test_outbox_outcome_separates_queued_repeat_registry_gap_and_refusal(tmp_path, monkeypatch):
    """The typed word the receipt reads; the boolean projection keeps its old meaning."""
    from supervisor import terminal_delivery as td
    event = {"type": "send_message", "chat_id": 1, "task_id": "t", "text": "x", "delivery_id": "project-handoff:1"}
    queue = _Queue()
    assert td.enqueue_terminal_delivery_outcome(tmp_path, dict(event), event_queue=queue) == "queued"
    assert json.loads((tmp_path / "state" / "terminal_deliveries.json").read_text())["pending"]["project-handoff:1"]
    assert td.enqueue_terminal_delivery(tmp_path, dict(event), event_queue=queue) is True  # owed twice = once
    assert len(queue.rows) == 2
    td.register_delivery(tmp_path, "project-handoff:1")
    assert td.enqueue_terminal_delivery_outcome(tmp_path, dict(event), event_queue=queue) == "already_delivered"
    assert td.enqueue_terminal_delivery(tmp_path, dict(event), event_queue=queue) is False
    assert len(queue.rows) == 2
    other = {**event, "delivery_id": "project-handoff:2"}
    assert td.enqueue_terminal_delivery_outcome(tmp_path, dict(other), event_queue=_Queue(refuse=True)) == "unavailable"
    monkeypatch.setattr(td, "register_pending_delivery", lambda *_a, **_k: False)
    assert td.enqueue_terminal_delivery_outcome(tmp_path, dict(other), event_queue=queue) == "queued_unregistered"
    assert td.enqueue_terminal_delivery(tmp_path, dict(other), event_queue=queue) is True
    assert td.enqueue_terminal_delivery_outcome(tmp_path, {}, event_queue=queue) == "unavailable"


def test_handoff_receipt_words_follow_binding_origin_and_outbox(tmp_path, monkeypatch):
    from ouroboros import project_handoff as ph
    from supervisor import terminal_delivery as td
    project = create_project(tmp_path, "research", name="Research")
    queue = _Queue()
    # The real outbox, fed a fake supervisor queue: no supervisor process in a unit test.
    monkeypatch.setattr(td, "enqueue_terminal_delivery_outcome",
                        lambda root, event, **kw: _ORIGINAL_OUTCOME(root, event, event_queue=queue))
    assert ph.enqueue_project_handoff(tmp_path, "not-bound") == "unbound"
    bind_task_to_project(tmp_path, "root", project["id"], project["chat_id"], origin=_main_origin())
    first = ph.enqueue_project_handoff(tmp_path, "root")
    assert first == "durable"
    row = queue.rows[0]
    assert row["system_type"] == "project_handoff" and row["chat_id"] == 1
    assert row["delivery_id"] == row["progress_meta"]["handoff_id"] == ph.handoff_identity(project["id"], "root", _main_origin()["ref"])
    assert "status" not in row["progress_meta"]
    # Repeating the same conversion before the send retries the same owed row; after it, nothing is owed.
    assert ph.enqueue_project_handoff(tmp_path, "root") == "durable"
    td.register_delivery(tmp_path, row["delivery_id"])
    assert ph.enqueue_project_handoff(tmp_path, "root") == "already_delivered"
    assert len(queue.rows) == 2
    # A Project-room origin is never a Main transfer; a ref-less legacy binding gets no Main default.
    assert ph.enqueue_project_handoff(tmp_path, "root", source_ref={"chat_id": project["chat_id"], "client_message_id": "inside"}) == "origin_unproven"
    bind_task_to_project(tmp_path, "legacy", project["id"], project["chat_id"], origin={"absent": "system"})
    assert ph.enqueue_project_handoff(tmp_path, "legacy") == "origin_unproven"
    assert len(queue.rows) == 2


def test_handoff_receipt_reports_registry_gap_and_transport_refusal(tmp_path, monkeypatch):
    from ouroboros import project_handoff as ph
    from supervisor import terminal_delivery as td
    project = create_project(tmp_path, "research", name="Research")
    bind_task_to_project(tmp_path, "root", project["id"], project["chat_id"], origin=_main_origin())
    outcomes = iter(["queued_unregistered", "unavailable"])
    monkeypatch.setattr(td, "enqueue_terminal_delivery_outcome", lambda *a, **k: next(outcomes))
    assert ph.enqueue_project_handoff(tmp_path, "root") == "unregistered"
    assert ph.enqueue_project_handoff(tmp_path, "root") == "unavailable"
    monkeypatch.setattr(td, "enqueue_terminal_delivery_outcome", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    assert ph.enqueue_project_handoff(tmp_path, "root") == "unavailable"


def test_agent_producers_publish_one_main_receipt_and_none_for_project_origins(tmp_path, monkeypatch):
    """The promote/scope handlers hand only the task id to the receipt: its binding's own origin decides."""
    from supervisor import events_project_routing as routing
    from ouroboros import project_handoff as ph
    from supervisor import terminal_delivery as td
    project = create_project(tmp_path, "research", name="Research")
    bind_task_to_project(tmp_path, "from-main", project["id"], project["chat_id"], origin=_main_origin("m1"))
    bind_task_to_project(tmp_path, "from-room", project["id"], project["chat_id"], origin={
        "ref": build_owner_message_ref(chat_id=project["chat_id"], client_message_id="r1",
                                       ts="2026-09-21T12:00:00Z", text="Inside"), "text": "Inside"})
    queue = _Queue()
    monkeypatch.setattr(td, "enqueue_terminal_delivery_outcome",
                        lambda root, event, **kw: _ORIGINAL_OUTCOME(root, event, event_queue=queue))
    monkeypatch.setattr(routing, "_emit_routing_receipt", lambda *a, **k: None)
    monkeypatch.setattr("supervisor.workers.DRIVE_ROOT", tmp_path)
    ctx = type("Ctx", (), {"DRIVE_ROOT": tmp_path})()
    for task_id in ("from-main", "from-main", "from-room"):
        routing._handle_ensure_project_scope({"task_id": task_id, "project_id": project["id"]}, ctx)
    assert [r["task_id"] for r in queue.rows] == ["from-main", "from-main"]
    assert len({r["delivery_id"] for r in queue.rows}) == 1
    assert ph.enqueue_project_handoff(tmp_path, "from-room") == "origin_unproven"


def test_handoff_is_a_main_only_history_row(tmp_path):
    from ouroboros.project_dialogue import room_membership
    row = {"type": "project_handoff", "task_id": "root"}
    rooms = lambda task_id, parent="", root="": 22 if task_id == "root" else 0  # noqa: E731
    assert room_membership(1, {22}, [], rooms)(1, row)
    assert not room_membership(22, {22}, [], rooms)(1, row)
